"""FastAPI entrypoint for the Foley-Omni preview webapp."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import AuthContext, Role, require_admin, require_auth
from app.config import AppSettings, build_engine_config, load_settings
from app.inference_service import InferenceService
from app.jobs import JobStore
from app.prompts import build_text_prompt
from app.video_utils import validate_video_duration
from app.queue_worker import (
    JobConflictError,
    JobForbiddenError,
    JobNotFoundError,
    QueueFullError,
    QueueWorker,
)
from app.schemas import (
    AdminClearResponse,
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    MeResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

settings: AppSettings
store: JobStore
worker: QueueWorker
inference: Optional[InferenceService] = None


def get_settings() -> AppSettings:
    return settings


def get_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> AuthContext:
    return require_auth(settings, x_api_key)


async def _load_model_background() -> None:
    """Load the heavy GPU engine after the HTTP server is accepting traffic."""
    global inference
    try:
        if not torch_available():
            raise RuntimeError("PyTorch/CUDA required unless SKIP_MODEL_LOAD=1")
        engine_cfg = build_engine_config(settings)
        # Blocking CUDA init runs in a worker thread so the event loop stays alive.
        svc = await asyncio.to_thread(InferenceService, engine_cfg, 0)
        inference = svc
        worker.inference = svc
        logger.info("FoleyOmniEngine attached to queue worker")
    except Exception:
        logger.exception("Failed to load FoleyOmniEngine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, store, worker, inference

    settings = load_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    store = JobStore(settings.jobs_dir)
    worker = QueueWorker(
        store=store,
        results_dir=settings.results_dir,
        max_queue_size=settings.max_queue_size,
        inference=None,
    )

    await worker.start()

    load_task = None
    if settings.skip_model_load:
        logger.warning("SKIP_MODEL_LOAD=1 — API runs without GPU engine (UI/auth/queue only)")
    else:
        load_task = asyncio.create_task(_load_model_background(), name="load-foley-engine")

    logger.info("Foley-Omni webapp HTTP ready (model may still be loading)")
    yield

    if load_task is not None and not load_task.done():
        load_task.cancel()
    await worker.stop()
    logger.info("Foley-Omni webapp shut down")


def torch_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


app = FastAPI(
    title="Soundstage Preview",
    description="Private video-to-soundtrack preview API (single GPU, queued).",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    model_loaded = inference is not None
    return HealthResponse(
        status="ok" if (model_loaded or settings.skip_model_load) else "starting",
        model_loaded=model_loaded,
        gpu_busy=worker.gpu_busy if worker else False,
        queue_depth=worker.queue_depth if worker else 0,
        max_queue_size=settings.max_queue_size,
    )


@app.get("/api/me", response_model=MeResponse)
async def me(auth: AuthContext = Depends(get_auth)):
    return MeResponse(role=auth.role.value)


@app.post("/api/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    video: UploadFile = File(...),
    words: str = Form(default=""),
    audio_caption: str = Form(default=""),
    music: str = Form(default=""),
    seed: Optional[int] = Form(default=None),
    sample_steps: Optional[int] = Form(default=None),
    auth: AuthContext = Depends(get_auth),
):
    if inference is None:
        if settings.skip_model_load:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference disabled (SKIP_MODEL_LOAD=1)",
            )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading onto the GPU. Retry in a minute.",
        )

    try:
        text_prompt = build_text_prompt(words=words, audio_caption=audio_caption, music=music)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    seed_val = settings.default_seed if seed is None else int(seed)
    steps_val = settings.default_sample_steps if sample_steps is None else int(sample_steps)
    if steps_val < 1 or steps_val > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="sample_steps must be between 1 and 100")

    # Reserve a job id path before create so upload lands in the right folder.
    # We create the job after saving the file.
    upload_tmp_dir = settings.uploads_dir / "_incoming"
    upload_tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Unsupported video type: {suffix}")

    # Create job record first to get id, then move upload into place.
    # Temporary placeholder path updated after save.
    job = store.create(
        owner_token_hash=auth.token_hash,
        video_path="",  # filled below
        text_prompt=text_prompt,
        seed=seed_val,
        sample_steps=steps_val,
        words=words or "",
        audio_caption=audio_caption or "",
        music=music or "",
    )

    job_upload_dir = settings.uploads_dir / job.job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_upload_dir / f"input{suffix}"

    try:
        with video_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        duration = validate_video_duration(video_path, settings.max_video_duration_sec)
        store.update(job.job_id, video_path=str(video_path), duration_sec=duration)
        job = store.get(job.job_id)
        assert job is not None

        position = await worker.enqueue(job)
    except QueueFullError as exc:
        store.update(job.job_id, status="failed", error=str(exc), progress="Rejected")
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except ValueError as exc:
        store.update(job.job_id, status="failed", error=str(exc), progress="Rejected")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        store.update(job.job_id, status="failed", error=str(exc), progress="Rejected")
        logger.exception("Failed to accept job")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to accept job") from exc

    return JobCreateResponse(job_id=job.job_id, status="queued", position=position)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, auth: AuthContext = Depends(get_auth)):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job not found")
    if auth.role != Role.ADMIN and job.owner_token_hash != auth.token_hash:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed to view this job")

    position = None
    if job.status == "queued":
        position = worker.position_of(job_id)

    result_ready = bool(job.result_path and Path(job.result_path).is_file() and job.status == "done")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,  # type: ignore[arg-type]
        position=position,
        progress=job.progress,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        seed=job.seed,
        sample_steps=job.sample_steps,
        duration_sec=job.duration_sec,
        result_ready=result_ready,
    )


@app.delete("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str, auth: AuthContext = Depends(get_auth)):
    try:
        # Admin may cancel any queued job; guests only their own.
        job = store.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if auth.role == Role.ADMIN:
            updated = await worker.cancel_queued(job_id, job.owner_token_hash)
        else:
            updated = await worker.cancel_queued(job_id, auth.token_hash)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job not found")
    except JobForbiddenError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed to cancel this job")
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return JobStatusResponse(
        job_id=updated.job_id,
        status=updated.status,  # type: ignore[arg-type]
        position=None,
        progress=updated.progress,
        error=updated.error,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
        seed=updated.seed,
        sample_steps=updated.sample_steps,
        duration_sec=updated.duration_sec,
        result_ready=False,
    )


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str, auth: AuthContext = Depends(get_auth)):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job not found")
    if auth.role != Role.ADMIN and job.owner_token_hash != auth.token_hash:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed to download this result")
    if job.status != "done" or not job.result_path or not Path(job.result_path).is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Result is not ready")
    return FileResponse(
        path=job.result_path,
        media_type="video/mp4",
        filename=f"soundstage-{job_id}.mp4",
    )


@app.post("/api/admin/queue/clear", response_model=AdminClearResponse)
async def clear_queue(auth: AuthContext = Depends(get_auth)):
    require_admin(auth)
    cancelled = await worker.clear_queued()
    return AdminClearResponse(cancelled_count=len(cancelled), cancelled_job_ids=cancelled)


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
