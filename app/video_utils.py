"""Lightweight video helpers (PyAV only)."""

from __future__ import annotations

from pathlib import Path


def get_video_duration_sec(video_path: str | Path) -> float:
    import av

    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        video_fps = video_stream.average_rate or video_stream.guessed_rate or 24.0
        if hasattr(video_fps, "numerator"):
            video_fps = float(video_fps.numerator) / float(video_fps.denominator)
        else:
            video_fps = float(video_fps)

        frame_count = 0
        for _ in container.decode(video_stream):
            frame_count += 1

        if frame_count == 0:
            raise ValueError("Could not decode video frames")
        return frame_count / float(video_fps)


def validate_video_duration(video_path: str | Path, max_duration_sec: float = 10.0) -> float:
    duration = get_video_duration_sec(video_path)
    if duration <= 0:
        raise ValueError("Could not read video duration")
    if duration > max_duration_sec + 0.05:
        raise ValueError(
            f"Video is {duration:.2f}s; maximum allowed is {max_duration_sec:.1f}s. "
            "Trim the clip and try again."
        )
    return float(duration)
