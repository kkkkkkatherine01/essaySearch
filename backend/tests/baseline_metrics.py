"""Pure metric-computation functions shared between the eval/test suites
(eval_verifier.py, test_prompt_injection.py) and the regression-baseline
scripts (capture_baseline.py / check_regression.py).

Each function runs one real round of measurement — real Anthropic API
calls, not mocks — and returns a flat dict of numeric metrics. No
comparison/tolerance logic lives here; that's check_regression.py's job,
because "did this get worse" isn't just equality once you factor in
real LLM run-to-run noise (see that file's docstring).

Deliberately reuses the SAME payloads/cases already defined in the
eval/test files rather than redefining them — a baseline that drifted from
what the actual regression tests check would be worse than no baseline.
"""

from backend.tests.eval_verifier import CASES as VERIFIER_CASES
from backend.tests.eval_verifier import FIXTURE_PATH, _issue_text
from backend.tests.rag_eval import compute_context_precision
from backend.tests.test_prompt_injection import SCORING_PAYLOADS, ORCHESTRATOR_PAYLOADS
from backend.tests.test_prompt_injection import _score_with_payload
from backend.security import scan_for_injection
from backend.verifier import check_citation_keys_exist, verify_citations

import json


async def measure_verifier() -> dict:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    answer, evidence = data["answer"], data["evidence"]

    baseline_issues = await verify_citations(answer, evidence)

    caught = 0
    caught_deterministically = 0
    for case in VERIFIER_CASES:
        corrupted = answer.replace(case["find"], case["replace"])

        # Deterministic layer alone — only catches no_matching_source by
        # design; if it ever caught a not_supported case too, the layer
        # boundary silently blurred.
        det_issues = check_citation_keys_exist(corrupted, evidence)
        if any(
            issue.get("problem") == case["expected_problem"]
            and any(k in _issue_text(issue) for k in case["contains_any"])
            for issue in det_issues
        ):
            caught_deterministically += 1

        issues = await verify_citations(corrupted, evidence)
        hit = any(
            issue.get("problem") == case["expected_problem"]
            and any(k in _issue_text(issue) for k in case["contains_any"])
            for issue in issues
        )
        if hit:
            caught += 1

    return {
        "verifier_recall": caught / len(VERIFIER_CASES),
        "verifier_false_positive_count": float(len(baseline_issues)),
        "verifier_deterministic_layer_recall": caught_deterministically / len(VERIFIER_CASES),
    }


async def measure_context_precision() -> dict:
    """Cheap half of the RAG-quality eval (see rag_eval.py) — pure
    computation on the fixed sample_review.json fixture, no API calls, so
    it belongs in the fast rolling baseline alongside the verifier metrics.
    The expensive half (context recall, answer relevance — real
    run_paperqa generations) stays out of this baseline; run those by hand
    via `python -m backend.tests.rag_eval`."""
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {"context_precision": compute_context_precision(data["answer"], data["evidence"])}


async def measure_injection_defense() -> dict:
    # Heuristic scanner is pure regex — deterministic, no LLM call, no noise.
    all_payloads = {**SCORING_PAYLOADS, **ORCHESTRATOR_PAYLOADS}
    hits = sum(1 for p in all_payloads.values() if scan_for_injection(p))
    heuristic_hit_rate = hits / len(all_payloads)

    # test_prompt_injection.py only asserts <=3 (hard gate); tracking the
    # actual value surfaces drift toward 3 before it crosses that gate.
    scores = []
    for payload in SCORING_PAYLOADS.values():
        score = await _score_with_payload(payload)
        scores.append(score if score is not None else 10.0)  # no score = worst case
    avg_attack_score = sum(scores) / len(scores)

    return {
        "injection_heuristic_hit_rate": heuristic_hit_rate,
        "injection_avg_scoring_attack_score": avg_attack_score,
    }


async def measure_all() -> dict:
    verifier_metrics = await measure_verifier()
    context_precision_metrics = await measure_context_precision()
    injection_metrics = await measure_injection_defense()
    return {**verifier_metrics, **context_precision_metrics, **injection_metrics}
