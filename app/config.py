"""Runtime configuration for the preview webapp."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = PROJECT_ROOT / "inference_v2st.yaml"


@dataclass(frozen=True)
class AppSettings:
    access_token: str
    admin_token: str
    ckpt_dir: Path
    model_checkpoint: Path
    storage_dir: Path
    max_queue_size: int
    max_video_duration_sec: float
    cpu_offload: bool
    sample_rate: int
    default_sample_steps: int
    default_seed: int
    solver_name: str
    shift: float
    audio_guidance_scale: float
    audio_negative_prompt: str
    model_name: str
    skip_model_load: bool

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> AppSettings:
    yaml_cfg: DictConfig = OmegaConf.load(str(DEFAULT_YAML)) if DEFAULT_YAML.exists() else OmegaConf.create({})

    ckpt_dir = Path(os.environ.get("CKPT_DIR", yaml_cfg.get("ckpt_dir", "./ckpts"))).resolve()
    model_checkpoint = Path(
        os.environ.get(
            "MODEL_CHECKPOINT",
            yaml_cfg.get("model_checkpoint", str(ckpt_dir / "Foley-Omni" / "v2st.pth")),
        )
    ).resolve()
    storage_dir = Path(os.environ.get("STORAGE_DIR", str(PROJECT_ROOT / "storage"))).resolve()

    access_token = os.environ.get("ACCESS_TOKEN", "").strip()
    admin_token = os.environ.get("ADMIN_TOKEN", "").strip()
    if not access_token:
        raise RuntimeError("ACCESS_TOKEN environment variable is required")
    if not admin_token:
        raise RuntimeError("ADMIN_TOKEN environment variable is required")
    if access_token == admin_token:
        raise RuntimeError("ACCESS_TOKEN and ADMIN_TOKEN must be different")

    return AppSettings(
        access_token=access_token,
        admin_token=admin_token,
        ckpt_dir=ckpt_dir,
        model_checkpoint=model_checkpoint,
        storage_dir=storage_dir,
        max_queue_size=int(os.environ.get("MAX_QUEUE_SIZE", "10")),
        max_video_duration_sec=float(os.environ.get("MAX_VIDEO_DURATION_SEC", "30.0")),
        cpu_offload=_env_bool("CPU_OFFLOAD", bool(yaml_cfg.get("cpu_offload", False))),
        sample_rate=int(os.environ.get("SAMPLE_RATE", yaml_cfg.get("sample_rate", 16000))),
        default_sample_steps=int(os.environ.get("SAMPLE_STEPS", yaml_cfg.get("sample_steps", 50))),
        default_seed=int(os.environ.get("SEED", yaml_cfg.get("seed", 103))),
        solver_name=str(os.environ.get("SOLVER_NAME", yaml_cfg.get("solver_name", "unipc"))),
        shift=float(os.environ.get("SHIFT", yaml_cfg.get("shift", 5.0))),
        audio_guidance_scale=float(
            os.environ.get("AUDIO_GUIDANCE_SCALE", yaml_cfg.get("audio_guidance_scale", 5.0))
        ),
        audio_negative_prompt=str(
            os.environ.get(
                "AUDIO_NEGATIVE_PROMPT",
                yaml_cfg.get("audio_negative_prompt", "robotic, muffled, echo, distorted"),
            )
        ),
        model_name=str(os.environ.get("MODEL_NAME", yaml_cfg.get("model_name", "960x960_10s"))),
        skip_model_load=_env_bool("SKIP_MODEL_LOAD", False),
    )


def build_engine_config(settings: AppSettings) -> DictConfig:
    """OmegaConf config consumed by FoleyOmniEngine."""
    base = OmegaConf.load(str(DEFAULT_YAML)) if DEFAULT_YAML.exists() else OmegaConf.create({})
    overrides = OmegaConf.create(
        {
            "ckpt_dir": str(settings.ckpt_dir),
            "model_checkpoint": str(settings.model_checkpoint),
            "model_name": settings.model_name,
            "sample_rate": settings.sample_rate,
            "sample_steps": settings.default_sample_steps,
            "seed": settings.default_seed,
            "solver_name": settings.solver_name,
            "shift": settings.shift,
            "audio_guidance_scale": settings.audio_guidance_scale,
            "audio_negative_prompt": settings.audio_negative_prompt,
            "cpu_offload": settings.cpu_offload,
            "sp_size": 1,
            "fp8": False,
            "mode": "vt2a",
            "duration": settings.max_video_duration_sec,
            "audio_only": False,
            "cfg_zero_video_features_in_negative": True,
            "cfg_apply_to_clip_features": True,
            "cfg_apply_to_sync_features": True,
            "slg_layer": int(base.get("slg_layer", 11)),
        }
    )
    return OmegaConf.merge(base, overrides)
