import re

from ..models import PaperCandidate


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def merge_into_pool(pool: dict[str, PaperCandidate], candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Merge newly found candidates into a running pool dict (keyed by
    normalized title), preferring the entry that actually has a downloadable
    PDF. Candidates without a PDF are dropped (nothing to download later).
    Returns just the candidates that were new to the pool this call, so a
    caller can report "N new" without diffing the whole pool."""
    added: list[PaperCandidate] = []
    for c in candidates:
        if not c.pdf_url:
            continue
        key = normalize_title(c.title)
        if not key:
            continue
        existing = pool.get(key)
        if existing is None:
            pool[key] = c
            added.append(c)
        elif not existing.pdf_url and c.pdf_url:
            pool[key] = c
    return added
