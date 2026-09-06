"""Unit tests for the deterministic (non-LLM) half of verifier.py's citation
checking — pure string comparison, no API calls, no run-to-run noise.

check_citation_keys_exist/check_citation_density are cheap to get subtly
wrong (regex edge cases) and expensive to notice being wrong — a bug here
would only ever surface as "the LLM layer happened to catch it anyway"
until one day it didn't."""

import json

from backend.tests.eval_verifier import FIXTURE_PATH
from backend.tests.rag_eval import compute_context_precision
from backend.verifier import check_citation_density, check_citation_keys_exist, extract_cited_sources


def _load_fixture():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["answer"], data["evidence"]


def test_clean_answer_has_no_key_existence_issues():
    answer, evidence = _load_fixture()
    assert check_citation_keys_exist(answer, evidence) == []


def test_fabricated_citation_is_caught_with_zero_llm_calls():
    answer, evidence = _load_fixture()
    corrupted = answer.replace(
        "with recognition receiving substantially more attention to date (Bao2026 pages 2-3)",
        "with recognition receiving substantially more attention to date (Smith2023 pages 5-6)",
    )
    issues = check_citation_keys_exist(corrupted, evidence)
    assert len(issues) == 1
    assert issues[0]["cited_as"] == "(Smith2023 pages 5-6)"
    assert issues[0]["problem"] == "no_matching_source"
    assert issues[0]["detected_by"] == "deterministic"
    # The extracted "claim" should be the actual sentence the fabricated
    # citation appears in, not a heading+sentence blob merged together.
    assert issues[0]["claim"].startswith("Sign language processing research")
    assert "Question:" not in issues[0]["claim"]


def test_a_not_supported_style_edit_is_invisible_to_the_deterministic_layer():
    """The deterministic layer can only ever flag no_matching_source — a
    claim attached to a citation that's still perfectly real should never
    trip it, even if the claim content itself is now wrong. That's the LLM
    layer's job (see eval_verifier.py)."""
    answer, evidence = _load_fixture()
    corrupted = answer.replace(
        "achieving 99.03% extraction accuracy with a compact, efficient model (Bao2026 pages 8-9)",
        "achieving 99.03% extraction accuracy while running at 45 frames per second on mobile CPUs "
        "(Bao2026 pages 8-9)",
    )
    assert check_citation_keys_exist(corrupted, evidence) == []


def test_citation_density_flags_a_paragraph_with_no_citations_at_all():
    answer = (
        "# Heading\n\n"
        "This paragraph makes a real claim (Smith2020 pages 1-1).\n\n"
        "This paragraph makes a claim with no citation attached at all.\n\n"
        "References\n\n1. ...\n"
    )
    issues = check_citation_density(answer)
    assert len(issues) == 1
    assert "no citation attached" in issues[0]["claim"]
    assert issues[0]["problem"] == "no_citation_in_paragraph"


def test_citation_density_ignores_headings_and_the_question_echo():
    answer = "Question: something\n\n# A Heading\n\nA cited claim (Smith2020 pages 1-1).\n"
    assert check_citation_density(answer) == []


def test_extract_cited_sources_matches_the_fixture_answer():
    answer, _evidence = _load_fixture()
    assert extract_cited_sources(answer) == {
        "Bao2026 pages 2-3",
        "Bao2026 pages 8-9",
        "Boddu2026 pages 3-4",
        "Johnny2025 pages 1-2",
        "Johnny2025 pages 2-3",
    }


def test_context_precision_on_the_real_fixture():
    """Documents the real, already-measured number (7/15 evidence chunks
    actually got cited) rather than a made-up round value — a regression
    here means the extraction logic changed, not that the fixture is wrong."""
    answer, evidence = _load_fixture()
    assert len(evidence) == 15
    assert compute_context_precision(answer, evidence) == 7 / 15


def test_context_precision_is_zero_with_no_evidence():
    assert compute_context_precision("anything (Smith2020 pages 1-1)", []) == 0.0


def test_context_precision_is_one_when_every_chunk_is_cited():
    evidence = [{"source": "Smith2020 pages 1-1"}, {"source": "Jones2021 pages 2-2"}]
    answer = "A claim (Smith2020 pages 1-1). Another claim (Jones2021 pages 2-2)."
    assert compute_context_precision(answer, evidence) == 1.0
