"""
Prepare inference code + checkpoints for the Soundstage Space.

Modes:
1) Monorepo: if parent folder already has inference_v2st.py (this project), use it.
2) Standalone Space: clone Foley-Omni from GitHub, then download Hub weights.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

REPO_URL = os.getenv("FOLEY_REPO_URL", "https://github.com/SAIL-PRLab/Foley-Omni.git").strip()
REPO_COMMIT = os.getenv("FOLEY_COMMIT", "").strip()
WEIGHTS_REPO = os.getenv("FOLEY_WEIGHTS_REPO", "CocoBro/Foley-Omni").strip()

# When Space repo is only hf_space/*, code lives under vendor/
VENDOR_DIR = HERE / "vendor" / "Foley-Omni"

CKPT_FILES = [
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

REPO_DIR: Path | None = None


def resolve_repo_dir() -> Path:
    global REPO_DIR
    if REPO_DIR is not None:
        return REPO_DIR

    if (PARENT / "inference_v2st.py").is_file() and (PARENT / "foley_omni").is_dir():
        REPO_DIR = PARENT
        print(f"[setup] using monorepo at {REPO_DIR}")
        return REPO_DIR

    REPO_DIR = VENDOR_DIR
    print(f"[setup] using vendor dir {REPO_DIR}")
    return REPO_DIR


def _apply_patch(file_path: Path, old: str, new: str, label: str) -> None:
    content = file_path.read_text(encoding="utf-8")
    if new in content:
        print(f"[patch] {label}: already patched")
        return
    if old not in content:
        print(f"[patch] {label}: pattern not found — skipping (upstream may differ)")
        return
    file_path.write_text(content.replace(old, new), encoding="utf-8")
    print(f"[patch] {label}: patched")


def apply_zerogpu_patches(repo: Path) -> None:
    """Best-effort RAM / attention patches used by the working FOG Space."""
    m = repo / "foley_omni"
    if not m.is_dir():
        return

    _apply_patch(
        m / "modules" / "t5.py",
        old="""    # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device if not cpu_offload else "cpu").eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model""",
        new="""    # init model on the meta device (allocates no real memory yet)
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device="meta").eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, assign=True)
        del state_dict
        target_device = device if not cpu_offload else "cpu"
        model = model.to(target_device)
        self.model = model""",
        label="t5.py",
    )

    _apply_patch(
        m / "fusion_engine.py",
        old="""            logging.info(f"Loading model checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:""",
        new="""            logging.info(f"Loading model checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
            except Exception:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:""",
        label="fusion_engine.py",
    )

    imp_old = "from .attention import flash_attention"
    imp_new = "from .attention import attention as flash_attention"
    for rel in ("modules/model.py", "modules/clip.py", "modules/__init__.py"):
        p = m / rel
        if p.is_file():
            _apply_patch(p, imp_old, imp_new, f"{rel} attention import")


def clone_repo_if_needed(repo: Path) -> None:
    if (repo / "inference_v2st.py").is_file():
        print("[setup] inference code already present")
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    if REPO_COMMIT:
        subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", REPO_COMMIT], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)


def download_checkpoints(repo: Path) -> None:
    missing = [f for f in CKPT_FILES if not (repo / f).is_file()]
    if not missing:
        print("[setup] checkpoints already present")
        return
    from huggingface_hub import snapshot_download

    print(f"[setup] downloading {len(missing)} files from {WEIGHTS_REPO} ...")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        WEIGHTS_REPO,
        local_dir=str(repo),
        allow_patterns=CKPT_FILES,
        token=token,
    )
    still = [f for f in CKPT_FILES if not (repo / f).is_file()]
    if still:
        raise RuntimeError(f"Checkpoint download incomplete: {still}")


def ensure_ready() -> Path:
    repo = resolve_repo_dir()
    clone_repo_if_needed(repo)
    apply_zerogpu_patches(repo)
    download_checkpoints(repo)
    print("[setup] ready")
    return repo


if __name__ == "__main__":
    ensure_ready()
