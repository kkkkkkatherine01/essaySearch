import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

TERMINAL_STATUSES = frozenset({"done", "failed"})


class JobStatus(str, Enum):
    QUEUED = "queued"
    SEARCHING = "searching"
    AWAITING_SELECTION = "awaiting_selection"
    DOWNLOADING = "downloading"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


@dataclass
class LogEntry:
    ts: float
    message: str


@dataclass
class Job:
    id: str
    query: str
    status: JobStatus = JobStatus.QUEUED
    logs: list[LogEntry] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    downloaded: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    answer: Optional[str] = None
    references: Optional[str] = None
    # None = not checked yet; [] = checked, nothing flagged; non-empty =
    # citations the verifier couldn't back up.
    citation_flags: Optional[list[dict]] = None
    # True = every verification attempt failed — distinct from
    # citation_flags staying None (never ran), so the UI can show "checked,
    # but failed" instead of hiding the box entirely.
    citation_check_failed: bool = False
    cost: Optional[float] = None
    duration: Optional[float] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Set by JobManager after construction; not part of equality/repr. Called
    # whenever the job's state changes so it can be persisted to SQLite.
    _on_change: Optional[Callable[["Job"], None]] = field(default=None, repr=False, compare=False)

    def log(self, message: str) -> None:
        self.logs.append(LogEntry(ts=time.time(), message=message))
        self.updated_at = time.time()
        self._touch()

    def set_status(self, status: JobStatus) -> None:
        self.status = status
        self.updated_at = time.time()
        self._touch()

    def _touch(self) -> None:
        if self._on_change:
            self._on_change(self)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "logs": [{"ts": entry.ts, "message": entry.message} for entry in self.logs[-100:]],
            "candidates": self.candidates,
            "downloaded": self.downloaded,
            "evidence": self.evidence,
            "answer": self.answer,
            "references": self.references,
            "citation_flags": self.citation_flags,
            "citation_check_failed": self.citation_check_failed,
            "cost": self.cost,
            "duration": self.duration,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_summary_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self, max_concurrent: int, store=None):
        self.store = store
        self.jobs: dict[str, Job] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        if self.store is not None:
            for job in self.store.load_all():
                job._on_change = self._persist
                self.jobs[job.id] = job

    def _persist(self, job: Job) -> None:
        if self.store is not None:
            self.store.save(job)

    def create_job(self, query: str) -> Job:
        job = Job(id=str(uuid.uuid4()), query=query)
        job._on_change = self._persist
        self.jobs[job.id] = job
        self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Remove a job's history entry. Does not touch any downloaded PDFs
        in the shared library — those are shared across jobs by design."""
        existed = self.jobs.pop(job_id, None) is not None
        if self.store is not None:
            self.store.delete(job_id)
        return existed

    def list_recent(self, limit: int = 30) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def mark_interrupted_jobs_failed(self) -> None:
        """Call once at startup: any job that was left mid-pipeline when the
        server last stopped will never finish (its background task is gone),
        so surface that instead of leaving it stuck in the history list."""
        for job in self.jobs.values():
            if job.status.value not in TERMINAL_STATUSES:
                job.error = "服务重启，任务已中断"
                job.set_status(JobStatus.FAILED)

    async def run(self, job: Job, pipeline: Callable[[Job], Awaitable[None]]) -> None:
        async with self._semaphore:
            await pipeline(job)
