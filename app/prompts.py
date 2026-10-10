"""Prompt helpers with no ML dependencies."""

from __future__ import annotations


def build_text_prompt(words: str = "", audio_caption: str = "", music: str = "") -> str:
    parts: list[str] = []
    words = (words or "").strip()
    audio_caption = (audio_caption or "").strip()
    music = (music or "").strip()
    if words:
        parts.append(f"[WORDS]{words}[END_WORDS]")
    if audio_caption:
        parts.append(f"[AUDIO_CAPTION]{audio_caption}[END_AUDIO_CAPTION]")
    if music:
        parts.append(f"[MUSIC]{music}[END_MUSIC]")
    if not parts:
        raise ValueError("At least one of words, audio_caption, or music is required")
    return "".join(parts)
