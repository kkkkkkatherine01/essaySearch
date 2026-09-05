import asyncio
import time

import httpx

from ..models import PaperCandidate

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

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
        pdf = item.get("openAccessPdf") or {}
        results.append(
            PaperCandidate(
                source="semantic_scholar",
                id=item.get("paperId"),
                title=(item.get("title") or "").strip(),
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                abstract=(item.get("abstract") or "").strip().replace("\n", " "),
                year=item.get("year"),
                pdf_url=pdf.get("url"),
                landing_url=item.get("url"),
            )
        )
    return results
