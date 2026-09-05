"""Synthetic-injection eval for verifier.py's citation checking.

A real precision/recall eval for the *search* agent would need painstaking
human-labeled ground truth — "is this paper actually relevant to this
research question" is a judgment call this script can't fabricate. The
verifier is different: we can manufacture our own ground truth by starting
from a real, already-generated review (backend/tests/fixtures/
sample_review.json, exported from an actual completed job) and deliberately
planting specific, known errors into copies of it. Then:

- recall: does verify_citations catch the errors we know we planted?
- false positives: does it also flag sentences we never touched?

This is possible without a human labeler because we ARE the ground truth
here — we know exactly which sentence we rewrote and how.

Not wired into pytest's default collection (it's `eval_*`, not `test_*`):
this reports numbers for you to read, it isn't a pass/fail gate. Run:

    python -m backend.tests.eval_verifier
"""

import asyncio
import json
from pathlib import Path

from backend.verifier import verify_citations

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_review.json"

# Each case rewrites exactly one sentence of the real answer and records
# what a correct verifier run should find. "contains_any" is checked against
# the union of every flagged issue's claim + cited_as + explanation text
# (case-insensitive) — it doesn't require the verifier to quote our exact
# wording back, just that *something* about the planted error surfaced.
CASES = [
    {
        "name": "fabricated_citation",
        "description": "Swap a real citation for one that doesn't exist anywhere in the evidence pool.",
        "find": "with recognition receiving substantially more attention to date (Bao2026 pages 2-3)",
        "replace": "with recognition receiving substantially more attention to date (Smith2023 pages 5-6)",
        "expected_problem": "no_matching_source",
        "contains_any": ["smith2023"],
    },
    {
        "name": "fabricated_detail_csf",
        "description": (
            "Keep the real CSF citation, but bolt on a specific performance claim "
            "(mobile frame rate) that the cited evidence never mentions."
        ),
        "find": "achieving 99.03% extraction accuracy with a compact, efficient model (Bao2026 pages 8-9)",
        "replace": (
            "achieving 99.03% extraction accuracy while running at 45 frames per second on mobile CPUs "
            "(Bao2026 pages 8-9)"
        ),
        "expected_problem": "not_supported",
        "contains_any": ["45", "frames per second", "mobile", "fps"],
    },
    {
        "name": "fabricated_detail_monolingual",
        "description": (
            "Keep the real citation, but attach a specific accuracy number the cited "
            "evidence never states."
        ),
        "find": "Generation systems have remained predominantly monolingual, typically processing only English input (Bao2026 pages 2-3)",
        "replace": (
            "Generation systems have remained predominantly monolingual, achieving only 62% accuracy "
            "on English input (Bao2026 pages 2-3)"
        ),
        "expected_problem": "not_supported",
        "contains_any": ["62%", "62 percent", "accuracy"],
    },
]


def _issue_text(issue: dict) -> str:
    return " ".join(str(issue.get(k, "")) for k in ("claim", "cited_as", "explanation")).lower()


async def _run_case(answer: str, evidence: list[dict], case: dict) -> dict:
    assert case["find"] in answer, f"fixture drifted — '{case['find'][:50]}...' not found in base answer"
    corrupted = answer.replace(case["find"], case["replace"])
    issues = await verify_citations(corrupted, evidence)

    hit = any(
        issue.get("problem") == case["expected_problem"] and any(k in _issue_text(issue) for k in case["contains_any"])
        for issue in issues
    )
    return {"name": case["name"], "caught": hit, "n_issues_total": len(issues), "issues": issues}


async def main():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    answer, evidence = data["answer"], data["evidence"]

    print(f"=== Baseline: verifying the untouched real answer (query: {data['query']!r}) ===")
    baseline_issues = await verify_citations(answer, evidence)
    print(f"{len(baseline_issues)} issue(s) on the untouched answer (these are the false-positive floor):")
    for issue in baseline_issues:
        print(f"  - [{issue.get('problem')}] {issue.get('claim', '')[:80]!r} — {issue.get('explanation', '')}")

    print(f"\n=== Recall: {len(CASES)} deliberately planted errors ===")
    results = await asyncio.gather(*(_run_case(answer, evidence, case) for case in CASES))
    caught = 0
    for r, case in zip(results, CASES):
        status = "CAUGHT" if r["caught"] else "MISSED"
        if r["caught"]:
            caught += 1
        print(f"  [{status}] {r['name']}: {case['description']}")
        print(f"           ({r['n_issues_total']} total issue(s) flagged in this corrupted copy)")

    recall = caught / len(CASES)
    print(f"\nrecall: {caught}/{len(CASES)} = {recall:.0%}")
    print(f"false-positive floor (untouched answer): {len(baseline_issues)} issue(s)")

    # Soft floor, not a strict gate: a single LLM-judge call has real
    # run-to-run variance, so this catches a wholesale regression (someone
    # breaks the prompt and it stops catching anything), not every dip.
    assert recall >= 0.5, f"verifier recall on planted errors dropped to {recall:.0%}"


if __name__ == "__main__":
    asyncio.run(main())
