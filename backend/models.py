from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PaperCandidate:
    source: str
    id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    pdf_url: Optional[str]
    landing_url: Optional[str]
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = None
