import re
from pathlib import Path

import httpx

from .models import PaperCandidate


def _stable_filename(candidate: PaperCandidate) -> str:
    """Filename derived from (source, id) rather than the title, so the same
    paper always lands at the same path across jobs — that stability is what
    lets paper-qa's index recognize it as already-embedded and skip it."""
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", f"{candidate.source}_{candidate.id}").strip("_")
    return f"{safe_id or candidate.id}.pdf"


async def download_pdf(candidate: PaperCandidate, dest_dir: Path) -> tuple[Path | None, bool]:
    """Download one candidate's PDF into the shared library dir.

    Returns (local_path, was_cached). was_cached is True when the file was
    already there from an earlier job (same paper, same stable filename) —
    no network request is made in that case."""
    if not candidate.pdf_url:
        return None, False

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _stable_filename(candidate)

    if dest_path.exists():
        return dest_path, True

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(candidate.pdf_url)
            resp.raise_for_status()
            content = resp.content
            if not content.startswith(b"%PDF"):
                return None, False
            dest_path.write_bytes(content)
            return dest_path, False
    except Exception:
        return None, False
