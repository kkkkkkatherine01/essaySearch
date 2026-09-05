from . import config
from .download import download_pdf
from .job_manager import Job, JobStatus
from .models import PaperCandidate
from .paperqa_engine import run_paperqa
from .search_agent import run_search_agent
from .security import scan_for_injection
from .verifier import check_citation_density, verify_citations


async def run_search_stage(job: Job) -> None:
    """Phase 1: a search agent (see search_agent.py) finds and scores
    candidate papers, then we stop to let the user pick which ones are
    worth downloading before we spend time and Claude tokens generating a
    review from them."""
    try:
        job.set_status(JobStatus.SEARCHING)
        merged = await run_search_agent(job)

        if not merged:
            raise RuntimeError("没有搜索到任何可下载全文的论文，请尝试更换研究问题的措辞")

        job.candidates = [c.__dict__ for c in merged]
        job.log("已按相关性排序，请选择要用于生成综述的论文")
        job.set_status(JobStatus.AWAITING_SELECTION)
    except Exception as e:
        job.error = str(e)
        job.set_status(JobStatus.FAILED)
        job.log(f"失败: {e}")


async def run_generate_stage(job: Job, selected: list[PaperCandidate]) -> None:
    """Phase 2: download the papers the user picked and run paper-qa's
    agent_query over them to gather evidence and generate a cited answer."""
    try:
        job.set_status(JobStatus.DOWNLOADING)
        downloaded = []
        for c in selected:
            path, was_cached = await download_pdf(c, config.LIBRARY_DIR)
            if path:
                downloaded.append({**c.__dict__, "local_path": str(path)})
                job.log(f"已命中缓存（此前任务下载过）: {c.title}" if was_cached else f"已下载: {c.title}")
            else:
                job.log(f"下载失败，跳过: {c.title}")
        job.downloaded = downloaded

        if not downloaded:
            raise RuntimeError("所有选中论文下载均失败")

        job.set_status(JobStatus.GENERATING)
        job.log("正在收集证据并生成带引用的综述（已在共享库中的论文会跳过重新索引，可能需要几分钟）...")
        result = await run_paperqa(job.query)

        job.answer = result["answer"]
        job.references = result["references"]
        job.evidence = result["evidence"]
        job.cost = result["cost"]
        job.duration = result["duration"]
        job.total_tokens = result["total_tokens"]
        job.log(
            f"完成，用时 {result['duration']:.1f} 秒，"
            f"花费 ${result['cost']:.4f}，共 {result['total_tokens']} tokens"
        )

        for e in result["evidence"]:
            hits = scan_for_injection(e.get("context", ""))
            if hits:
                job.log(f"[Security] ⚠ 证据片段中检测到疑似注入内容（{hits[0]}）: 来源 {e.get('source')}")

        # Structural, zero-cost check — not merged into citation_flags (a
        # paragraph with no citation isn't necessarily wrong, e.g. a closing
        # summary), so it's just logged as a secondary signal.
        density_issues = check_citation_density(result["answer"])
        if density_issues:
            job.log(f"[确定性检查] {len(density_issues)} 个段落完全没有引用标记")

        job.log("正在核查综述引用是否有幻觉/证据不支撑的情况...")
        try:
            flags = await verify_citations(result["answer"], result["evidence"])
            job.citation_flags = flags
            job.log(f"引用自查完成：发现 {len(flags)} 处可疑引用" if flags else "引用自查完成：未发现可疑引用")
        except Exception as e:
            # citation_flags stays None; citation_check_failed lets the UI
            # distinguish "checked, found nothing" from "never checked" from
            # "checked, but every attempt failed". Doesn't fail the job —
            # the review itself already generated fine.
            job.citation_check_failed = True
            job.log(f"引用自查失败，跳过（不影响已生成的综述）: {e}")

        job.set_status(JobStatus.DONE)
    except Exception as e:
        job.error = str(e)
        job.set_status(JobStatus.FAILED)
        job.log(f"失败: {e}")
