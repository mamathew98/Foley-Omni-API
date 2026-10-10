"""Wrap Foley-Omni inference for single-job web requests.

Heavy ML imports are deferred until InferenceService is constructed so the
HTTP app can boot with SKIP_MODEL_LOAD=1 without torch/diffusers installed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

from omegaconf import DictConfig

from app.prompts import build_text_prompt
from app.video_utils import validate_video_duration

logger = logging.getLogger(__name__)

MAX_VIDEO_DURATION_SEC = 10.0

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
        import torch

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

        progress("Generating soundtrack (sampling)")
        try:
            generated_audio = generate_audio_for_video(
                foley_omni_engine=self.engine,
                video_path=str(video_path),
                text_prompt=text_prompt,
                seed=seed,
                solver_name=str(self.config.get("solver_name", "unipc")),
                sample_steps=sample_steps,
                shift=float(self.config.get("shift", 5.0)),
                audio_guidance_scale=float(self.config.get("audio_guidance_scale", 5.0)),
                slg_layer=int(self.config.get("slg_layer", 11)),
                audio_negative_prompt=str(self.config.get("audio_negative_prompt", "")),
                duration=float(duration_sec),
                cfg_apply_to_clip_features=bool(self.config.get("cfg_apply_to_clip_features", True)),
                cfg_apply_to_sync_features=bool(self.config.get("cfg_apply_to_sync_features", True)),
                device=self.device,
                dtype=self.target_dtype,
            )
        except SkipShortVideoError as exc:
            raise ValueError(str(exc)) from exc

        sample_rate = int(self.config.get("sample_rate", 16000))
        progress("Muxing video and audio")
        self._mux_video_audio(
            video_path=video_path,
            generated_audio=np.asarray(generated_audio),
            output_path=output_path,
            sample_rate=sample_rate,
        )
        progress("Done")
        return output_path

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


def _ensure_mmaudio_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    mmaudio_root = project_root / "mmaudio"
    if str(mmaudio_root) not in sys.path:
        sys.path.insert(0, str(mmaudio_root))
