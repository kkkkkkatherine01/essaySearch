"""Capture (or extend) the regression baseline: run N fresh samples of each
tracked metric and merge them into backend/tests/baselines/metrics.json.

This is a human-in-the-loop step, not something CI runs automatically —
after an intentional change (prompt edit, model upgrade) that you've
reviewed and accept as the new normal, run this to move the baseline
forward. It APPENDS samples (capped at a rolling window, oldest dropped)
rather than overwriting on each run, so the stored baseline reflects real
run-to-run noise instead of a single lucky/unlucky sample.

Run: python -m backend.tests.capture_baseline [--samples N]
"""

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from backend.tests.baseline_metrics import measure_all

BASELINE_PATH = Path(__file__).parent / "baselines" / "metrics.json"
MAX_SAMPLES_PER_METRIC = 10


def _update_stats(existing_samples: list[float], new_value: float) -> dict:
    samples = (existing_samples + [new_value])[-MAX_SAMPLES_PER_METRIC:]
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
    }


async def main(n_samples: int) -> None:
    BASELINE_PATH.parent.mkdir(exist_ok=True)
    baseline = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {"metrics": {}}
    )

    for i in range(n_samples):
        print(f"sample {i + 1}/{n_samples}...")
        result = await measure_all()
        for key, value in result.items():
            existing = baseline["metrics"].get(key, {}).get("samples", [])
            baseline["metrics"][key] = _update_stats(existing, value)
        print(f"  {result}")

    baseline["captured_at"] = datetime.now(timezone.utc).isoformat()
    BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nbaseline updated: {BASELINE_PATH}")
    for key, stats in baseline["metrics"].items():
        print(f"  {key}: mean={stats['mean']:.3f} range=[{stats['min']:.3f}, {stats['max']:.3f}] (n={len(stats['samples'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3, help="fresh samples to add (default: 3)")
    args = parser.parse_args()
    asyncio.run(main(args.samples))
