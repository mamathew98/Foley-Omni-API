"""
ZeroGPU-aware soundtrack generation for the Soundstage Space.

ZeroGPU defaults: no qint8, cpu_offload OFF (A10G speed; falls back to ON on OOM),
@spaces.GPU around generate, single-flight lock. Override via FOLEY_CPU_OFFLOAD.
"""

from __future__ import annotations

import contextlib
import gc
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

from setup_weights import ensure_ready

REPO_DIR = ensure_ready()

MODE = os.getenv("FOLEY_MODE", "inprocess").strip().lower()
IS_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))
ZEROGPU_DURATION = int(os.getenv("FOLEY_GPU_SECONDS", "180"))
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "30"))
MAX_CHUNK = 9.5
MIN_DROP = 0.05
MIN_GEN_DURATION = 1.0
MIN_STRETCH_INPUT_MS = 300
MAX_REL_GAP_FOR_STRETCH = 0.08
SAMPLE_STEPS_DEFAULT = int(os.getenv("SAMPLE_STEPS", "8"))

# UI presets (label → steps). Default = Fast.
PRESETS = {
    "Fast": 8,
    "Balanced": 15,
    "Quality": 25,
}
DEFAULT_PRESET = "Fast"

_GPU_LOCK = threading.Lock()
_ENGINE: dict = {"inf": None, "engine": None, "cfg": None, "failed": False}


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def build_prompt(words: str, sfx: str, music: str) -> str:
    parts = []
    w = (words or "").strip()
    a = (sfx or "").strip()
    m = (music or "").strip()
    if w:
        parts.append(f"[WORDS]{w}[END_WORDS]")
    if a:
        parts.append(f"[AUDIO_CAPTION]{a}[END_AUDIO_CAPTION]")
    if m:
        parts.append(f"[MUSIC]{m}[END_MUSIC]")
    if not parts:
        # Unconditioned-ish prompt (FOG used spaces in all blocks)
        return "[WORDS] [END_WORDS][AUDIO_CAPTION] [END_AUDIO_CAPTION][MUSIC] [END_MUSIC]"
    return "".join(parts)


def probe_video(path: str) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
    )
    info = json.loads(r.stdout or "{}")
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise RuntimeError("No video stream found in the uploaded file.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(fmt.get("duration") or vs.get("duration") or 0)

    def parse_rate(s: str) -> float | None:
        try:
            num, den = s.split("/")
            num, den = float(num), float(den)
            return num / den if den != 0 else num
        except Exception:
            return None

    fps = (
        parse_rate(vs.get("avg_frame_rate", "") or "")
        or parse_rate(vs.get("r_frame_rate", "") or "")
        or 25.0
    )
    return {"duration": duration, "has_audio": has_audio, "fps": float(fps)}


def plan_chunks(full_duration: float) -> tuple[list[tuple[float, float]], list[float]]:
    boundaries = sorted({0.0, float(full_duration)})
    raw = (
        list(zip(boundaries[:-1], boundaries[1:]))
        if len(boundaries) > 1
        else [(0.0, float(full_duration))]
    )
    chunks: list[tuple[float, float]] = []
    for c_start, c_end in raw:
        dur = c_end - c_start
        if dur <= MIN_DROP:
            continue
        if dur <= MAX_CHUNK:
            chunks.append((c_start, c_end))
        else:
            n = math.ceil(dur / MAX_CHUNK)
            step = dur / n
            t = c_start
            for _ in range(n):
                chunks.append((t, min(t + step, c_end)))
                t += step
    freeze_extra = [max(0.0, MIN_GEN_DURATION - (e - s)) for s, e in chunks]
    return chunks, freeze_extra


def cut_clip(
    src: str,
    start: float,
    end: float,
    out_path: str,
    fps: float,
    freeze_extra_sec: float = 0.0,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
    ]
    if freeze_extra_sec > 0:
        cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={freeze_extra_sec:.3f}"]
    cmd += [
        "-r",
        f"{fps:.3f}",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg cut failed:\n{(r.stderr or '')[-800:]}")


def _time_stretch_to_length(seg_audio, target_ms: int):
    import numpy as np
    import librosa
    from pydub import AudioSegment

    if len(seg_audio) == 0 or len(seg_audio) == target_ms:
        return seg_audio
    sr = seg_audio.frame_rate
    samples = np.array(seg_audio.get_array_of_samples()).astype(np.float32) / 32768.0
    rate = (len(samples) / sr) / (target_ms / 1000.0)
    n_fft = 2048
    while n_fft > 256 and n_fft > len(samples):
        n_fft //= 2
    stretched = librosa.effects.time_stretch(samples, rate=rate, n_fft=n_fft, hop_length=n_fft // 4)
    i16 = np.clip(stretched * 32768.0, -32768, 32767).astype(np.int16)
    return AudioSegment(i16.tobytes(), frame_rate=sr, sample_width=2, channels=1)


def stitch_audio(chunks, chunk_wavs, full_duration: float, out_wav: str, sample_rate: int = 16000) -> None:
    from pydub import AudioSegment

    full_audio = AudioSegment.silent(duration=int(full_duration * 1000), frame_rate=sample_rate)
    for i, (s, e) in enumerate(chunks):
        wav_path = chunk_wavs[i] if i < len(chunk_wavs) else None
        if not wav_path or not os.path.exists(wav_path):
            continue
        seg = (
            AudioSegment.from_wav(wav_path)
            .set_frame_rate(sample_rate)
            .set_channels(1)
            .set_sample_width(2)
        )
        seg_len_ms = int(round((e - s) * 1000))
        gap_ms = seg_len_ms - len(seg)
        rel_gap = abs(gap_ms) / seg_len_ms if seg_len_ms > 0 else 0
        if gap_ms == 0:
            pass
        elif gap_ms < 0:
            seg = seg[:seg_len_ms]
        elif rel_gap > MAX_REL_GAP_FOR_STRETCH:
            pass
        elif len(seg) < MIN_STRETCH_INPUT_MS:
            seg = seg + AudioSegment.silent(duration=gap_ms, frame_rate=sample_rate)
        else:
            seg = _time_stretch_to_length(seg, seg_len_ms)
        full_audio = full_audio.overlay(seg, position=int(s * 1000))
    full_audio.export(out_wav, format="wav")


def _make_base_cfg(sample_steps: int | None = None, seed: int | None = None):
    from omegaconf import OmegaConf

    cfg_path = REPO_DIR / "inference_v2st.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    cfg.sp_size = 1
    cfg.audio_only = True
    # ZeroGPU A10G (~24GB): prefer no offload for speed; set FOLEY_CPU_OFFLOAD=1 to force.
    cfg.cpu_offload = _env_flag("FOLEY_CPU_OFFLOAD", False if IS_ZEROGPU else True)
    cfg.fp8 = False
    cfg.qint8 = _env_flag("FOLEY_QINT8", False)
    cfg.sample_steps = int(sample_steps if sample_steps is not None else SAMPLE_STEPS_DEFAULT)
    if seed is not None:
        cfg.seed = int(seed)
    cfg.duration = 10.0
    cfg.video_path = None
    cfg.json_file = None
    cfg.ckpt_dir = str(REPO_DIR / "ckpts")
    cfg.model_checkpoint = str(REPO_DIR / "ckpts" / "Foley-Omni" / "v2st.pth")
    return cfg


@contextlib.contextmanager
def _zerogpu_safe_cuda():
    """Prevent main-process CUDA init on ZeroGPU (fork worker requirement)."""
    if not IS_ZEROGPU:
        yield
        return
    import torch

    saved = {}
    for name in ("ipc_collect", "synchronize", "reset_peak_memory_stats"):
        if hasattr(torch.cuda, name):
            saved[name] = getattr(torch.cuda, name)
            setattr(torch.cuda, name, lambda *a, **k: None)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.cuda, name, fn)


def _load_engine(sample_steps: int | None = None, seed: int | None = None) -> None:
    import torch

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)

    with _zerogpu_safe_cuda():
        import inference_v2st as inf

        if not inf.MMAUDIO_AVAILABLE:
            raise RuntimeError("mmaudio modules could not be imported.")

        cfg = _make_base_cfg(sample_steps=sample_steps, seed=seed)
        inf._init_logging(0)
        if not IS_ZEROGPU:
            try:
                torch.cuda.set_device(0)
            except Exception as e:
                print(f"[engine] set_device skipped: {e}", flush=True)
        inf.initialize_sequence_parallel_state(1)

        print(
            f"[engine] loading engine cpu_offload={cfg.cpu_offload} steps={cfg.sample_steps} ...",
            flush=True,
        )
        try:
            engine = inf.FoleyOmniEngine(config=cfg, device=0, target_dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            if cfg.cpu_offload:
                raise
            print("[engine] OOM without offload — retrying with cpu_offload=True", flush=True)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            cfg.cpu_offload = True
            engine = inf.FoleyOmniEngine(config=cfg, device=0, target_dtype=torch.bfloat16)
        print(f"[engine] loaded cpu_offload={cfg.cpu_offload}", flush=True)
        _ENGINE.update(inf=inf, engine=engine, cfg=cfg)


def warmup() -> None:
    if MODE != "inprocess" or _ENGINE["engine"] is not None or _ENGINE["failed"]:
        return
    try:
        _load_engine()
    except Exception:
        import traceback

        traceback.print_exc()
        _ENGINE["failed"] = True
        print("[engine] in-process load failed", flush=True)


def _gpu(duration: int):
    if IS_ZEROGPU:
        import spaces

        return spaces.GPU(duration=duration)
    return lambda f: f


def _generate_one_impl(video_path: str, prompt: str, sample_steps: int, seed: int):
    import torch

    inf, engine, cfg = _ENGINE["inf"], _ENGINE["engine"], _ENGINE["cfg"]
    if engine is None:
        raise RuntimeError("Model engine is not available inside the GPU worker.")

    # Refresh tunable fields for this request
    cfg.sample_steps = int(sample_steps)
    cfg.seed = int(seed)

    accurate, _ = inf.get_video_duration_accurate(video_path)
    duration = float(accurate) if accurate and accurate > 0 else float(cfg.get("duration", 10.0))
    apply_clip, apply_sync = inf._resolve_negative_video_feature_cfg(cfg)

    try:
        return inf.generate_audio_for_video(
            foley_omni_engine=engine,
            video_path=video_path,
            text_prompt=prompt,
            seed=int(seed),
            solver_name=str(cfg.get("solver_name", "unipc")),
            sample_steps=int(sample_steps),
            shift=float(cfg.get("shift", 5.0)),
            audio_guidance_scale=float(cfg.get("audio_guidance_scale", 5.0)),
            slg_layer=int(cfg.get("slg_layer", 11)),
            audio_negative_prompt=str(cfg.get("audio_negative_prompt", "")),
            duration=duration,
            cfg_apply_to_clip_features=apply_clip,
            cfg_apply_to_sync_features=apply_sync,
            device=0,
            dtype=torch.bfloat16,
        )
    except inf.SkipShortVideoError as e:
        raise ValueError(str(e)) from e
    finally:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


@_gpu(ZEROGPU_DURATION)
def _generate_one(video_path: str, prompt: str, sample_steps: int, seed: int):
    """Runs on the ZeroGPU worker — load engine here so CUDA init stays in-worker."""
    import traceback

    try:
        if _ENGINE["engine"] is None:
            _load_engine(sample_steps=sample_steps, seed=seed)
        elif _ENGINE["cfg"] is not None:
            _ENGINE["cfg"].sample_steps = int(sample_steps)
            _ENGINE["cfg"].seed = int(seed)
        if _ENGINE["engine"] is None:
            raise RuntimeError("Model engine failed to load inside the GPU worker.")
        return ("ok", _generate_one_impl(video_path, prompt, sample_steps, seed))
    except Exception:
        tb = traceback.format_exc()
        print("[gpu-worker] ERROR:\n" + tb, flush=True)
        return ("err", tb)


def _write_wav(path: str, audio, sample_rate: int) -> None:
    import numpy as np

    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def ensure_faststart(video_path: str) -> str:
    """Remux MP4 with moov at the front so Gradio/browser can start playback immediately."""
    if not video_path:
        return video_path
    fd, out = tempfile.mkstemp(suffix=".mp4", prefix="soundstage_web_")
    os.close(fd)
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        # Fall back to original if remux fails (codec/container oddities)
        try:
            os.remove(out)
        except OSError:
            pass
        return video_path
    return out


def _mux(video_path: str, wav_path: str, out_mp4: str) -> None:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            out_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{(r.stderr or '')[-800:]}")


def _cleanup_old_results(max_age_s: int = 3600) -> None:
    import time

    patterns = ("soundstage_result_*.mp4", "soundstage_web_*.mp4")
    for pat in patterns:
        for f in glob.glob(os.path.join(tempfile.gettempdir(), pat)):
            try:
                if time.time() - os.path.getmtime(f) > max_age_s:
                    os.remove(f)
            except OSError:
                pass


def generate(
    video_path: str,
    words: str = "",
    sfx: str = "",
    music: str = "",
    seed: int = 103,
    sample_steps: int = 15,
    progress=None,
) -> str:
    if not video_path:
        raise ValueError("Please upload a video first.")

    info = probe_video(video_path)
    if info["duration"] <= 0:
        raise ValueError("Could not read video duration.")
    if info["duration"] > MAX_VIDEO_SECONDS + 0.05:
        raise ValueError(
            f"This preview accepts videos up to {MAX_VIDEO_SECONDS:.0f}s "
            f"(yours: {info['duration']:.1f}s)."
        )
    if info["duration"] < MIN_GEN_DURATION:
        raise ValueError(f"Video is too short (need at least ~{MIN_GEN_DURATION:.0f}s).")

    prompt = build_prompt(words, sfx, music)
    _cleanup_old_results()
    work = tempfile.mkdtemp(prefix="soundstage_")
    try:
        local_video = os.path.join(work, "input.mp4")
        shutil.copy(video_path, local_video)
        if info["has_audio"]:
            muted = os.path.join(work, "input_muted.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", local_video, "-c:v", "copy", "-an", muted],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                local_video = muted

        chunks, freeze_extra = plan_chunks(info["duration"])
        chunk_dir = os.path.join(work, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_paths: list[str] = []
        for i, ((s, e), fx) in enumerate(zip(chunks, freeze_extra)):
            p = os.path.join(chunk_dir, f"chunk_{i:03d}.mp4")
            cut_clip(local_video, s, e, p, info["fps"], fx)
            chunk_paths.append(p)

        n = len(chunk_paths)
        if progress:
            progress(0.05, desc=f"Preparing ({n} chunk{'s' if n != 1 else ''})…")

        sample_rate = 16000
        chunk_wavs: list[str | None] = [None] * n

        with _GPU_LOCK:
            if not IS_ZEROGPU:
                if _ENGINE["engine"] is None and not _ENGINE["failed"]:
                    _load_engine(sample_steps=sample_steps, seed=seed)
                elif _ENGINE["cfg"] is not None:
                    _ENGINE["cfg"].sample_steps = int(sample_steps)
                    _ENGINE["cfg"].seed = int(seed)
                if _ENGINE["engine"] is None:
                    raise RuntimeError(
                        "Model failed to load. Check Space logs. "
                        "ZeroGPU Spaces need hardware set to ZeroGPU / A10G."
                    )

            for idx, cpath in enumerate(chunk_paths):
                if progress:
                    progress(
                        (idx + 0.15) / (n + 1),
                        desc=f"Generating soundtrack {idx + 1}/{n}…",
                    )
                status, payload = _generate_one(
                    cpath, prompt, int(sample_steps), int(seed) + idx
                )
                if status == "err":
                    raise RuntimeError("GPU step failed:\n" + str(payload)[-1200:])
                audio = payload
                if audio is None:
                    continue
                if _ENGINE.get("cfg") is not None:
                    sample_rate = int(_ENGINE["cfg"].get("sample_rate", 16000))
                wav_path = os.path.join(work, f"chunk_{idx:03d}.wav")
                _write_wav(wav_path, audio, sample_rate)
                chunk_wavs[idx] = wav_path

        if progress:
            progress(0.9, desc="Stitching audio…")
        full_wav = os.path.join(work, "full.wav")
        stitch_audio(chunks, chunk_wavs, info["duration"], full_wav, sample_rate)

        if progress:
            progress(0.95, desc="Muxing video…")
        fd, final_path = tempfile.mkstemp(suffix=".mp4", prefix="soundstage_result_")
        os.close(fd)
        _mux(local_video, full_wav, final_path)

        if progress:
            progress(1.0, desc="Done")
        return final_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
