#!/usr/bin/env python3
"""Cross-platform checkpoint downloader (Windows + Linux)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_RELATIVE_PATHS = [
    "ckpts/Foley-Omni/v2st.pth",
    "ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/special_tokens_map.json",
    "ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model",
    "ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json",
    "ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json",
    "ckpts/mmaudio/ext_weights/v1-16.pth",
    "ckpts/mmaudio/ext_weights/best_netG.pt",
    "ckpts/mmaudio/ext_weights/synchformer_state_dict.pth",
]


def all_present(root: Path) -> bool:
    return all((root / rel).is_file() for rel in REQUIRED_RELATIVE_PATHS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Foley-Omni release checkpoints from Hugging Face.")
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="CocoBro/Foley-Omni",
        help="Hugging Face repo id (default: CocoBro/Foley-Omni)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if checkpoints already exist",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/)",
    )
    args = parser.parse_args()

    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if all_present(root) and not args.force:
        print(f"[download] Checkpoints already present under {root / 'ckpts'} — skipping.")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            '[download] huggingface_hub is missing. Install with:\n'
            '  pip install -U "huggingface_hub[cli]>=0.30.0,<1.0"',
            file=sys.stderr,
        )
        return 1

    print(f"[download] Fetching from {args.repo_id} → {root}")
    for rel in REQUIRED_RELATIVE_PATHS:
        print(f"  - {rel}")
        hf_hub_download(
            repo_id=args.repo_id,
            filename=rel,
            local_dir=str(root),
        )

    if not all_present(root):
        missing = [rel for rel in REQUIRED_RELATIVE_PATHS if not (root / rel).is_file()]
        print("[download] ERROR: missing files after download:", file=sys.stderr)
        for rel in missing:
            print(f"  - {rel}", file=sys.stderr)
        return 1

    print(f"[download] Done. Checkpoints are under: {root / 'ckpts'}")
    print("[download] Note: CLIP weights for online feature extraction are fetched by open_clip on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
