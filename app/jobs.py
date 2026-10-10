"""In-memory + on-disk job store."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    owner_token_hash: str
    status: str  # queued | running | done | failed | cancelled
    video_path: str
    text_prompt: str
    seed: int
    sample_steps: int
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    progress: Optional[str] = None
    error: Optional[str] = None
    duration_sec: Optional[float] = None
    result_path: Optional[str] = None
    words: str = ""
    audio_caption: str = ""
    music: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class JobStore:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._load_existing()

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _load_existing(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job.from_dict(data)
                # Stale queued/running jobs from a previous process cannot be resumed.
                if job.status in {"queued", "running"}:
                    job.status = "failed"
                    job.error = "Server restarted before this job completed"
                    job.updated_at = _utc_now()
                    path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
                self._jobs[job.job_id] = job
            except Exception:
                continue

    def _persist(self, job: Job) -> None:
        self._path(job.job_id).write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def create(
        self,
        *,
        owner_token_hash: str,
        video_path: str,
        text_prompt: str,
        seed: int,
        sample_steps: int,
        words: str = "",
        audio_caption: str = "",
        music: str = "",
        duration_sec: Optional[float] = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            owner_token_hash=owner_token_hash,
            status="queued",
            video_path=video_path,
            text_prompt=text_prompt,
            seed=seed,
            sample_steps=sample_steps,
            words=words,
            audio_caption=audio_caption,
            music=music,
            duration_sec=duration_sec,
            progress="Queued",
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = _utc_now()
            self._persist(job)
            return job

    def list_by_status(self, status: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == status]

    def count_active(self) -> int:
        """Queued + running jobs (consume queue capacity)."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in {"queued", "running"})
