"""Wrap Foley-Omni inference for single-job web requests.

Heavy ML imports are deferred until InferenceService is constructed so the
HTTP app can boot with SKIP_MODEL_LOAD=1 without torch/diffusers installed.

Clips longer than ~9.5s are FOG-style chunked, then audio is stitched.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

from omegaconf import DictConfig

from app.chunking import cut_clip, mute_video, plan_chunks, stitch_audio
from app.prompts import build_text_prompt
from app.video_utils import get_video_fps, validate_video_duration

logger = logging.getLogger(__name__)

MAX_VIDEO_DURATION_SEC = 30.0

__all__ = [
    "InferenceService",
    "build_text_prompt",
    "validate_video_duration",
    "MAX_VIDEO_DURATION_SEC",
]


class InferenceService:
    """Owns the FoleyOmniEngine singleton. Only the GPU worker should call run_job."""

    def __init__(self, engine_config: DictConfig, device: int = 0):
        import torch
        from foley_omni.distributed_comms.parallel_states import initialize_sequence_parallel_state
        from foley_omni.fusion_engine import FoleyOmniEngine

        _ensure_mmaudio_on_path()
        initialize_sequence_parallel_state(1)
        self.device = device
        self.target_dtype = torch.bfloat16
        self.config = engine_config
        logger.info("Loading FoleyOmniEngine (this can take several minutes)...")
        torch.cuda.set_device(device)
        self.engine = FoleyOmniEngine(
            config=engine_config,
            device=device,
            target_dtype=self.target_dtype,
        )
        logger.info("FoleyOmniEngine ready")

    def run_job(
        self,
        *,
        video_path: str | Path,
        text_prompt: str,
        output_path: str | Path,
        seed: int,
        sample_steps: int,
        duration_sec: Optional[float] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Path:
        import numpy as np

        _ensure_mmaudio_on_path()
        from inference_v2st import SkipShortVideoError, generate_audio_for_video

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        if duration_sec is None:
            progress("Measuring video duration")
            duration_sec = validate_video_duration(
                video_path,
                max_duration_sec=float(self.config.get("duration", MAX_VIDEO_DURATION_SEC)),
            )

        sample_rate = int(self.config.get("sample_rate", 16000))
        work = Path(tempfile.mkdtemp(prefix="foley_job_"))
        try:
            local = work / "input.mp4"
            shutil.copy(str(video_path), local)
            muted = mute_video(str(local), str(work / "input_muted.mp4"))
            fps = get_video_fps(muted)

            chunks, freeze_extra = plan_chunks(float(duration_sec))
            chunk_dir = work / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_paths: list[str] = []
            for i, ((s, e), fx) in enumerate(zip(chunks, freeze_extra)):
                p = chunk_dir / f"chunk_{i:03d}.mp4"
                cut_clip(muted, s, e, str(p), fps, fx)
                chunk_paths.append(str(p))

            n = len(chunk_paths)
            chunk_wavs: list[str | None] = [None] * n

            for idx, cpath in enumerate(chunk_paths):
                progress(f"Generating soundtrack chunk {idx + 1}/{n}")
                # Pass 10.0 so generate_audio_for_video re-reads the chunk's true duration
                try:
                    audio = generate_audio_for_video(
                        foley_omni_engine=self.engine,
                        video_path=cpath,
                        text_prompt=text_prompt,
                        seed=int(seed) + idx,
                        solver_name=str(self.config.get("solver_name", "unipc")),
                        sample_steps=sample_steps,
                        shift=float(self.config.get("shift", 5.0)),
                        audio_guidance_scale=float(self.config.get("audio_guidance_scale", 5.0)),
                        slg_layer=int(self.config.get("slg_layer", 11)),
                        audio_negative_prompt=str(self.config.get("audio_negative_prompt", "")),
                        duration=10.0,
                        cfg_apply_to_clip_features=bool(
                            self.config.get("cfg_apply_to_clip_features", True)
                        ),
                        cfg_apply_to_sync_features=bool(
                            self.config.get("cfg_apply_to_sync_features", True)
                        ),
                        device=self.device,
                        dtype=self.target_dtype,
                    )
                except SkipShortVideoError as exc:
                    logger.warning("Skipping short chunk %s: %s", idx, exc)
                    continue

                wav_path = work / f"chunk_{idx:03d}.wav"
                _write_wav_file(str(wav_path), np.asarray(audio), sample_rate)
                chunk_wavs[idx] = str(wav_path)

            if n == 1 and chunk_wavs[0]:
                progress("Muxing video and audio")
                import soundfile as sf

                audio_arr, _ = sf.read(chunk_wavs[0], dtype="float32")
                self._mux_video_audio(
                    video_path=Path(muted),
                    generated_audio=audio_arr,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )
            else:
                progress("Stitching chunk audio")
                full_wav = work / "full.wav"
                stitch_audio(chunks, chunk_wavs, float(duration_sec), str(full_wav), sample_rate)
                progress("Muxing video and audio")
                import soundfile as sf

                audio_arr, _ = sf.read(str(full_wav), dtype="float32")
                self._mux_video_audio(
                    video_path=Path(muted),
                    generated_audio=audio_arr,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )

            progress("Done")
            return output_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _mux_video_audio(
        *,
        video_path: Path,
        generated_audio,
        output_path: Path,
        sample_rate: int,
    ) -> None:
        import av
        import numpy as np
        import torch
        from fractions import Fraction

        from mmaudio.data.av_utils import VideoInfo
        from mmaudio.eval_utils import make_video

        with av.open(str(video_path)) as container:
            video_stream = container.streams.video[0]
            video_fps = video_stream.average_rate or video_stream.guessed_rate or 24.0
            if hasattr(video_fps, "numerator"):
                video_fps = float(video_fps.numerator) / float(video_fps.denominator)
            else:
                video_fps = float(video_fps)

            video_frames = []
            for frame in container.decode(video_stream):
                video_frames.append(frame.to_ndarray(format="rgb24"))

        if not video_frames:
            raise ValueError("Could not decode video frames for muxing")

        num_frames = len(video_frames)
        video_duration = num_frames / video_fps
        target_audio_length = int(video_duration * sample_rate)
        audio = np.asarray(generated_audio, dtype=np.float32)
        if len(audio) > target_audio_length:
            audio = audio[:target_audio_length]
        elif len(audio) < target_audio_length:
            audio = np.pad(audio, (0, target_audio_length - len(audio)), mode="constant")
        audio = np.clip(audio, -1.0, 1.0)

        fps_frac = (
            Fraction(int(round(video_fps * 1000)), 1000)
            if video_fps != int(video_fps)
            else Fraction(int(video_fps), 1)
        )
        video_info = VideoInfo(
            duration_sec=video_duration,
            fps=fps_frac,
            clip_frames=torch.zeros(1, 1, 3, 224, 224),
            sync_frames=torch.zeros(1, 1, 3, 224, 224),
            all_frames=video_frames,
        )
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        make_video(video_info, output_path, audio_tensor, sampling_rate=sample_rate)


def _write_wav_file(path: str, audio, sample_rate: int) -> None:
    import numpy as np

    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def _ensure_mmaudio_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    mmaudio_root = project_root / "mmaudio"
    if str(mmaudio_root) not in sys.path:
        sys.path.insert(0, str(mmaudio_root))
