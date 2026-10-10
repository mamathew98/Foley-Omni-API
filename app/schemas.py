"""API request / response schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class MeResponse(BaseModel):
    role: Literal["guest", "admin"]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    gpu_busy: bool
    queue_depth: int
    max_queue_size: int


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    position: Optional[int] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    position: Optional[int] = None
    progress: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    seed: int
    sample_steps: int
    duration_sec: Optional[float] = None
    result_ready: bool = False


class AdminClearResponse(BaseModel):
    cancelled_count: int
    cancelled_job_ids: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
