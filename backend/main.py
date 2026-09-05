import asyncio
import functools

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .job_manager import JobManager, JobStatus
from .models import PaperCandidate
from .pipeline import run_generate_stage, run_search_stage
from .store import JobStore

app = FastAPI(title="Paper Search & Review Agent")
manager = JobManager(max_concurrent=config.MAX_CONCURRENT_JOBS, store=JobStore(config.DB_PATH))
manager.mark_interrupted_jobs_failed()


@app.on_event("startup")
async def check_config() -> None:
    if not config.ANTHROPIC_API_KEY:
        print("[WARN] ANTHROPIC_API_KEY 未设置，请在 .env 中配置后重启，否则任务会在生成阶段失败。")


class CreateJobRequest(BaseModel):
    query: str


class SelectPapersRequest(BaseModel):
    paper_ids: list[str]


@app.post("/api/jobs")
async def create_job(req: CreateJobRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "研究问题不能为空")
    job = manager.create_job(query)
    asyncio.create_task(manager.run(job, run_search_stage))
    return {"job_id": job.id}


@app.get("/api/jobs")
async def list_jobs():
    return [job.to_summary_dict() for job in manager.list_recent()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/select")
async def select_papers(job_id: str, req: SelectPapersRequest):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status != JobStatus.AWAITING_SELECTION:
        raise HTTPException(400, f"任务当前状态是 {job.status.value}，不能选择论文")
    if not req.paper_ids:
        raise HTTPException(400, "至少要选一篇论文")

    by_id = {c["id"]: c for c in job.candidates}
    selected = [PaperCandidate(**by_id[pid]) for pid in req.paper_ids if pid in by_id]
    if not selected:
        raise HTTPException(400, "选中的论文 id 无效")

    job.log(f"已选择 {len(selected)} 篇论文，开始下载")
    asyncio.create_task(manager.run(job, functools.partial(run_generate_stage, selected=selected)))
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status not in (JobStatus.DONE, JobStatus.FAILED):
        raise HTTPException(400, "任务还在进行中，不能删除")
    manager.delete(job_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
