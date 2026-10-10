"""FOG-style video chunking + audio stitch for long clips (≤ ~30s)."""

from __future__ import annotations

import math
import os
import subprocess
from typing import Sequence

MAX_CHUNK = 9.5
MIN_DROP = 0.05
MIN_GEN_DURATION = 1.0
MIN_STRETCH_INPUT_MS = 300
MAX_REL_GAP_FOR_STRETCH = 0.08


def plan_chunks(full_duration: float) -> tuple[list[tuple[float, float]], list[float]]:
    """Split duration into ≤ MAX_CHUNK pieces; return (chunks, freeze_extra_per_chunk)."""
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


def mute_video(src: str, out_path: str) -> str:
    """Return muted path if mute succeeds, else src."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-c:v", "copy", "-an", out_path],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and os.path.exists(out_path):
        return out_path
    return src


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


def stitch_audio(
    chunks: Sequence[tuple[float, float]],
    chunk_wavs: Sequence[str | None],
    full_duration: float,
    out_wav: str,
    sample_rate: int = 16000,
) -> None:
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
