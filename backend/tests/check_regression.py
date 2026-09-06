"""Compare a fresh measurement against the stored regression baseline
(backend/tests/baselines/metrics.json) and flag anything that moved further
than run-to-run noise would explain.

Different from test_prompt_injection.py's fixed thresholds: those catch
"is this fundamentally broken", this catches "did this quietly get worse"
— drift a fixed-threshold test won't see until it crosses the gate.

Not wired into CI — run by hand before a prompt/model change you're unsure
about. Exits non-zero on regression.

Run: python -m backend.tests.check_regression
"""

import asyncio
import json
import sys
from pathlib import Path

from backend.tests.baseline_metrics import measure_all

BASELINE_PATH = Path(__file__).parent / "baselines" / "metrics.json"

# direction: which way is "bad". tolerance: how far past the historical
# worst still counts as noise, not a real regression — sized to roughly
# "one test case flipping outcome", not a hair-trigger on float jitter.
# Tuned against noise actually measured here (verifier recall swings
# 67%-100% run to run on 3 cases), not guessed.
METRIC_SPECS = {
    "verifier_recall": {"direction": "higher_is_better", "tolerance": 0.34},
    # Pure code, no LLM call — structurally always exactly 1/3 (only ever
    # catches the one no_matching_source case among 3 planted errors).
    # Tight tolerance: drift here means the comparison logic broke, not
    # LLM noise.
    "verifier_deterministic_layer_recall": {"direction": "higher_is_better", "tolerance": 0.05},
    "verifier_false_positive_count": {"direction": "lower_is_better", "tolerance": 2.0},
    # Pure computation on a fixed fixture (rag_eval.compute_context_precision)
    # — no LLM call, no noise. Drift here means the fixture changed or the
    # citation-extraction regex it depends on broke, not run-to-run variance.
    "context_precision": {"direction": "higher_is_better", "tolerance": 0.05},
    "injection_heuristic_hit_rate": {"direction": "higher_is_better", "tolerance": 0.17},
    "injection_avg_scoring_attack_score": {"direction": "lower_is_better", "tolerance": 2.0},
}


def _is_regression(name: str, new_value: float, baseline_stats: dict) -> bool:
    spec = METRIC_SPECS[name]
    if spec["direction"] == "higher_is_better":
        return new_value < baseline_stats["min"] - spec["tolerance"]
    return new_value > baseline_stats["max"] + spec["tolerance"]


async def main() -> None:
    if not BASELINE_PATH.exists():
        print(f"No baseline found at {BASELINE_PATH}. Run `python -m backend.tests.capture_baseline` first.")
        sys.exit(2)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["metrics"]

    print("running fresh measurement (real API calls)...")
    fresh = await measure_all()

    regressions = []
    print(f"\n{'metric':<38}{'baseline range':<24}{'new value':<12}status")
    for name, new_value in fresh.items():
        stats = baseline.get(name)
        if stats is None:
            print(f"{name:<38}{'(no baseline)':<24}{new_value:<12.3f}SKIPPED")
            continue
        regressed = _is_regression(name, new_value, stats)
        status = "REGRESSION" if regressed else "ok"
        rng = f"[{stats['min']:.3f}, {stats['max']:.3f}]"
        print(f"{name:<38}{rng:<24}{new_value:<12.3f}{status}")
        if regressed:
            regressions.append(name)

    if regressions:
        print(f"\n{len(regressions)} metric(s) regressed beyond noise tolerance: {regressions}")
        sys.exit(1)
    print("\nno regressions detected.")


if __name__ == "__main__":
    asyncio.run(main())
