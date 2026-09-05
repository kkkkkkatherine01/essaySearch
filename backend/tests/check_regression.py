"""Compare a fresh measurement against the stored regression baseline
(backend/tests/baselines/metrics.json) and report anything that moved
further than run-to-run noise would explain.

This is a different kind of check than test_prompt_injection.py's fixed
thresholds. Those catch "is this fundamentally broken" (a hard floor/ceiling
that shouldn't move). This catches "did this get quietly worse compared to
where it used to be" — which matters most right after a prompt edit or a
model version bump, where nothing crashes and every fixed-threshold test
still passes, but the underlying behavior has drifted. A metric creeping
from consistently ~0.3 to consistently ~2.8 on a scale gated at <=3 would
sail through every existing test while being a real, worth-knowing change.

Not a CI gate by default (this project has no CI) — run it by hand before
committing to a prompt/model change you're unsure about, or wire it into a
CI step yourself; it exits non-zero on regression either way.

Run: python -m backend.tests.check_regression
"""

import asyncio
import json
import sys
from pathlib import Path

from backend.tests.baseline_metrics import measure_all

BASELINE_PATH = Path(__file__).parent / "baselines" / "metrics.json"

# direction: which way is "bad" for this metric. tolerance: how far past the
# worst historically-observed value still counts as noise rather than a real
# regression — sized to roughly "one whole test case flipping outcome", not
# a hair-trigger on float jitter. Tuned against noise actually measured in
# this project (verifier recall swings 67%-100% run to run on 3 cases;
# see 问题记录.txt), not guessed.
METRIC_SPECS = {
    "verifier_recall": {"direction": "higher_is_better", "tolerance": 0.34},
    # Pure code, no LLM call — structurally always exactly 1/3 (it can only
    # ever catch the one no_matching_source case among the 3 planted
    # errors, by design). Tight tolerance is deliberate: unlike the other
    # metrics, any drift here means the regex/comparison logic broke, not
    # LLM noise.
    "verifier_deterministic_layer_recall": {"direction": "higher_is_better", "tolerance": 0.05},
    "verifier_false_positive_count": {"direction": "lower_is_better", "tolerance": 2.0},
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
