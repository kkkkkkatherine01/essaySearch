import arxiv

from ..models import PaperCandidate


def search_arxiv(query: str, max_results: int = 10) -> list[PaperCandidate]:
    """Search arXiv for candidate papers. Runs synchronously (the `arxiv`
    package does blocking HTTP under the hood) — call via asyncio.to_thread."""
    client = arxiv.Client(delay_seconds=5.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results: list[PaperCandidate] = []
    for r in client.results(search):
        results.append(
            PaperCandidate(
                source="arxiv",
                id=r.get_short_id(),
                title=r.title.strip(),
                authors=[a.name for a in r.authors],
                abstract=(r.summary or "").strip().replace("\n", " "),
                year=r.published.year if r.published else None,
                pdf_url=r.pdf_url,
                landing_url=r.entry_id,
            )
        )
    return results
