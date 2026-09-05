import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
# Cheaper/faster model for scoring candidate relevance — this step just
# ranks a batch of titles/abstracts, it doesn't need the strongest model.
RERANK_MODEL = os.getenv("RERANK_MODEL", "claude-haiku-4-5-20251001")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "st-multi-qa-MiniLM-L6-cos-v1")

# How many candidates get pre-checked by default in the selection UI.
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "6"))
MAX_CANDIDATES_PER_SOURCE = int(os.getenv("MAX_CANDIDATES_PER_SOURCE", "10"))
# How many deduped candidates to show the user to choose from (not all get
# downloaded — only the ones the user selects).
MAX_CANDIDATES_TO_SHOW = int(os.getenv("MAX_CANDIDATES_TO_SHOW", "15"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

# Hard caps on the search-stage agent loop (backend/search_agent.py) so a
# model that keeps deciding "one more search" can't run away with time or
# tokens. Iteration cap is the primary guard (each iteration is one model
# turn, which may itself call multiple tools e.g. both search sources at
# once); the token cap is a secondary net measured from actual API usage.
SEARCH_AGENT_MAX_ITERATIONS = int(os.getenv("SEARCH_AGENT_MAX_ITERATIONS", "5"))
SEARCH_AGENT_MAX_TOKENS = int(os.getenv("SEARCH_AGENT_MAX_TOKENS", "30000"))

PAPERS_ROOT = ROOT_DIR / "papers"
# Shared across jobs (not per-job) so paper-qa's incremental index
# recognizes an already-embedded paper and skips re-parsing/re-embedding it.
LIBRARY_DIR = PAPERS_ROOT / "library"
LIBRARY_INDEX_DIR = LIBRARY_DIR / ".pqa_index"
FRONTEND_DIR = ROOT_DIR / "frontend"
DB_PATH = ROOT_DIR / "jobs.db"

PAPERS_ROOT.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)
