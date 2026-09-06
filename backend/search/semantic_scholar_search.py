import asyncio
import re
import time

import httpx

from ..models import PaperCandidate

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

# Semantic Scholar's documented limit is 1 request/second, cumulative across
# all endpoints and API keys — process-wide, not per-job. Since multiple jobs
# can run concurrently (config.MAX_CONCURRENT_JOBS), a per-call check isn't
# enough; this lock + timestamp serializes every S2 call across the whole
# process to stay under that ceiling regardless of how many jobs overlap.
_MIN_INTERVAL_SECONDS = 1.1
_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0


async def _wait_for_rate_limit() -> None:
    global _last_request_at
    async with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_INTERVAL_SECONDS:
            await asyncio.sleep(_MIN_INTERVAL_SECONDS - elapsed)
        _last_request_at = time.monotonic()


async def search_semantic_scholar(
    query: str, max_results: int = 10, api_key: str | None = None
) -> list[PaperCandidate]:
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,externalIds,openAccessPdf,url",
    }
    headers = {"x-api-key": api_key} if api_key else {}

    await _wait_for_rate_limit()
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(S2_API_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results: list[PaperCandidate] = []
    for item in data.get("data", []):
        candidate = _candidate_from_s2(item)
        if candidate:
            results.append(candidate)
    return results


def _candidate_from_s2(item: dict) -> PaperCandidate | None:
    if not item or not item.get("paperId"):
        return None
    pdf = item.get("openAccessPdf") or {}
    return PaperCandidate(
        source="semantic_scholar",
        id=item["paperId"],
        title=(item.get("title") or "").strip(),
        authors=[a.get("name", "") for a in (item.get("authors") or [])],
        abstract=(item.get("abstract") or "").strip().replace("\n", " "),
        year=item.get("year"),
        pdf_url=pdf.get("url"),
        landing_url=item.get("url"),
    )


_NEIGHBOR_FIELDS = ["title", "abstract", "year", "authors", "externalIds", "openAccessPdf", "url"]


async def get_citation_neighbors(
    source: str, paper_id: str, max_results: int = 10, api_key: str | None = None
) -> list[PaperCandidate]:
    """Fetch a seed paper's references and citing papers from Semantic
    Scholar's citation graph ("snowballing") and return them as candidates.

    `source`/`paper_id` identify a paper already known to us (from an earlier
    search_arxiv/search_semantic_scholar/expand_citations result) — S2
    accepts an `ARXIV:`-prefixed external id directly as the paper_id path
    param, so an arXiv-sourced seed needs no separate id-resolution call.
    S2 doesn't recognize the version suffix arxiv's own ids carry (e.g.
    "2511.13780v1" 404s; "2511.13780" doesn't) — strip it first."""
    if source == "semantic_scholar":
        s2_id = paper_id
    else:
        s2_id = f"ARXIV:{re.sub(r'v\d+$', '', paper_id)}"
    headers = {"x-api-key": api_key} if api_key else {}

    results: list[PaperCandidate] = []
    seen_ids: set[str] = set()
    for endpoint, wrapper_key in (("references", "citedPaper"), ("citations", "citingPaper")):
        params = {
            "fields": ",".join(f"{wrapper_key}.{f}" for f in _NEIGHBOR_FIELDS),
            "limit": max_results,
        }
        await _wait_for_rate_limit()
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{S2_PAPER_BASE_URL}/{s2_id}/{endpoint}", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        for item in data.get("data", []):
            candidate = _candidate_from_s2(item.get(wrapper_key) or {})
            if candidate and candidate.id not in seen_ids:
                seen_ids.add(candidate.id)
                results.append(candidate)
    return results
