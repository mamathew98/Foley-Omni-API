"""Single-consumer GPU job queue."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Optional

from app.jobs import Job, JobStore
from app.inference_service import InferenceService

logger = logging.getLogger(__name__)


class QueueWorker:
    """
    Serializes GPU inference so only one job runs at a time.

    Cancel removes a job from the pending deque before it starts.
    Admin clear cancels all queued jobs; the running job is left alone.
    """

    def __init__(
        self,
        store: JobStore,
        results_dir: Path,
        max_queue_size: int,
        inference: Optional[InferenceService] = None,
    ):
        self.store = store
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue_size = max_queue_size
        self.inference = inference

        self._pending: deque[str] = deque()
        self._condition = asyncio.Condition()
        self._running_job_id: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    @property
    def gpu_busy(self) -> bool:
        return self._running_job_id is not None

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    def position_of(self, job_id: str) -> Optional[int]:
        try:
            return list(self._pending).index(job_id) + 1
        except ValueError:
            if self._running_job_id == job_id:
                return 0
            return None

    async def start(self) -> None:
        if self._task is None:
            self._stopped = False
            self._task = asyncio.create_task(self._loop(), name="gpu-queue-worker")

    async def stop(self) -> None:
        self._stopped = True
        async with self._condition:
            self._condition.notify_all()
        if self._task is not None:
            await self._task
            self._task = None

    async def enqueue(self, job: Job) -> int:
        async with self._condition:
            if len(self._pending) + (1 if self._running_job_id else 0) >= self.max_queue_size:
                raise QueueFullError(
                    f"Queue is full (max {self.max_queue_size}). Try again later."
                )
            self._pending.append(job.job_id)
            position = len(self._pending)
            self._condition.notify()
            return position

    async def cancel_queued(self, job_id: str, owner_token_hash: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.owner_token_hash != owner_token_hash:
            raise JobForbiddenError(job_id)
        if job.status == "running":
            raise JobConflictError("Cannot cancel a running job")
        if job.status != "queued":
            raise JobConflictError(f"Job is already {job.status}")

        async with self._condition:
            if job_id in self._pending:
                self._pending = deque(jid for jid in self._pending if jid != job_id)
            updated = self.store.update(
                job_id,
                status="cancelled",
                progress="Cancelled by user",
                error=None,
            )
            return updated  # type: ignore[return-value]

    async def clear_queued(self) -> list[str]:
        """Admin: cancel all queued jobs; leave running alone."""
        cancelled: list[str] = []
        async with self._condition:
            pending_ids = list(self._pending)
            self._pending.clear()
            for job_id in pending_ids:
                job = self.store.get(job_id)
                if job is None:
                    continue
                if job.status == "queued":
                    self.store.update(
                        job_id,
                        status="cancelled",
                        progress="Cancelled by admin (queue cleared)",
                    )
                    cancelled.append(job_id)
        return cancelled

    async def _loop(self) -> None:
        logger.info("GPU queue worker started")
        while not self._stopped:
            async with self._condition:
                while not self._pending and not self._stopped:
                    await self._condition.wait()
                if self._stopped:
                    break
                job_id = self._pending.popleft()

            job = self.store.get(job_id)
            if job is None or job.status != "queued":
                continue

            self._running_job_id = job_id
            self.store.update(job_id, status="running", progress="Starting inference")
            try:
                await self._run_one(job)
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                self.store.update(
                    job_id,
                    status="failed",
                    progress="Failed",
                    error=str(exc),
                )
            finally:
                self._running_job_id = None
        logger.info("GPU queue worker stopped")

    async def _run_one(self, job: Job) -> None:
        if self.inference is None:
            raise RuntimeError("Inference service is not loaded")

        result_dir = self.results_dir / job.job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "result.mp4"

        def on_progress(msg: str) -> None:
            self.store.update(job.job_id, progress=msg)

        await asyncio.to_thread(
            self.inference.run_job,
            video_path=job.video_path,
            text_prompt=job.text_prompt,
            output_path=result_path,
            seed=job.seed,
            sample_steps=job.sample_steps,
            duration_sec=job.duration_sec,
            progress_cb=on_progress,
        )
        self.store.update(
            job.job_id,
            status="done",
            progress="Done",
            result_path=str(result_path),
            error=None,
        )


class QueueFullError(Exception):
    pass


class JobNotFoundError(Exception):
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class JobForbiddenError(Exception):
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Not allowed to modify job: {job_id}")


class JobConflictError(Exception):
    pass
