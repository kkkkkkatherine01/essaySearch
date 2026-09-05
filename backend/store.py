import json
import sqlite3
from pathlib import Path

from .job_manager import Job, JobStatus, LogEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    logs TEXT NOT NULL,
    candidates TEXT NOT NULL,
    downloaded TEXT NOT NULL,
    evidence TEXT NOT NULL,
    answer TEXT,
    refs TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""

# Columns added after the initial schema. Applied with ALTER TABLE so
# existing local jobs.db files (from before these fields existed) still load.
MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN cost REAL",
    "ALTER TABLE jobs ADD COLUMN duration REAL",
    "ALTER TABLE jobs ADD COLUMN total_tokens INTEGER",
    "ALTER TABLE jobs ADD COLUMN citation_flags TEXT",
    "ALTER TABLE jobs ADD COLUMN citation_check_failed INTEGER",
]


class JobStore:
    """Minimal SQLite-backed persistence for Job state, so task history
    survives a server restart. Single-user local tool, so a plain sync
    sqlite3 connection (no ORM, no async driver) is plenty."""

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        for migration in MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    def save(self, job: Job) -> None:
        self._conn.execute(
            """
            INSERT INTO jobs
                (id, query, status, logs, candidates, downloaded, evidence, answer, refs, error,
                 cost, duration, total_tokens, citation_flags, citation_check_failed, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                logs=excluded.logs,
                candidates=excluded.candidates,
                downloaded=excluded.downloaded,
                evidence=excluded.evidence,
                answer=excluded.answer,
                refs=excluded.refs,
                error=excluded.error,
                cost=excluded.cost,
                duration=excluded.duration,
                total_tokens=excluded.total_tokens,
                citation_flags=excluded.citation_flags,
                citation_check_failed=excluded.citation_check_failed,
                updated_at=excluded.updated_at
            """,
            (
                job.id,
                job.query,
                job.status.value,
                json.dumps([{"ts": e.ts, "message": e.message} for e in job.logs]),
                json.dumps(job.candidates),
                json.dumps(job.downloaded),
                json.dumps(job.evidence),
                job.answer,
                job.references,
                job.error,
                job.cost,
                job.duration,
                job.total_tokens,
                json.dumps(job.citation_flags) if job.citation_flags is not None else None,
                int(job.citation_check_failed),
                job.created_at,
                job.updated_at,
            ),
        )
        self._conn.commit()

    def delete(self, job_id: str) -> None:
        self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()

    def load_all(self) -> list[Job]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        job = Job(id=row["id"], query=row["query"], status=JobStatus(row["status"]))
        job.logs = [LogEntry(ts=e["ts"], message=e["message"]) for e in json.loads(row["logs"])]
        job.candidates = json.loads(row["candidates"])
        job.downloaded = json.loads(row["downloaded"])
        job.evidence = json.loads(row["evidence"])
        job.answer = row["answer"]
        job.references = row["refs"]
        job.error = row["error"]
        job.cost = row["cost"] if "cost" in row.keys() else None
        job.duration = row["duration"] if "duration" in row.keys() else None
        job.total_tokens = row["total_tokens"] if "total_tokens" in row.keys() else None
        raw_flags = row["citation_flags"] if "citation_flags" in row.keys() else None
        job.citation_flags = json.loads(raw_flags) if raw_flags else None
        job.citation_check_failed = bool(row["citation_check_failed"]) if "citation_check_failed" in row.keys() and row["citation_check_failed"] is not None else False
        job.created_at = row["created_at"]
        job.updated_at = row["updated_at"]
        return job
