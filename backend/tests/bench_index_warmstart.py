"""Real cold-vs-warm measurement of PaperQA2's incremental index reuse
(backend/paperqa_engine.py).

Copies a couple of real PDFs already in the shared library into an
isolated temp library/index dir (never touches the project's real shared
index), then runs the same query against the same papers twice:

    run 1 (cold): temp index is empty -> PaperQA2 must parse + enrich +
                  embed both PDFs from scratch before it can answer.
    run 2 (warm): temp index now has both PDFs cached -> PaperQA2's
                  incremental index (get_directory_index) should recognize
                  them by path and skip re-parsing/re-embedding.

Both runs do the same real LLM generation call, so that part's duration is
comparable noise across both — the gap between the two totals isolates
what the warm index actually buys, rather than asserting a number nobody
measured.

Costs real money (PDF parsing/enrichment LLM calls + 2 full generation
calls per trial) and is slow (~1-2 min per trial, longer with more
papers). Not wired into pytest or the regression baseline — run it by hand
when you want a fresh read on the shared index's payoff, e.g. after
changing _build_settings or the index-sync logic in paperqa_engine.py:

    python -m backend.tests.bench_index_warmstart [n_papers] [n_trials]

Defaults to 6 papers, 1 trial. Pass n_trials > 1 to run several independent
cold+warm pairs (same paper set, fresh temp index each trial) and average
out per-request LLM-generation-time noise.
"""

import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from backend import config

DEFAULT_N_PAPERS = 6
DEFAULT_N_TRIALS = 1
QUERY = "What methods are discussed in these papers?"
# Timestamped per run (not overwritten) — each run is an independent real
# measurement worth keeping, not just the latest one.
RESULT_DIR = Path(__file__).parent / "bench_results" / "index_warmstart_runs"


def _pick_smallest_papers(n: int) -> list[Path]:
    pdfs = sorted(config.LIBRARY_DIR.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if len(pdfs) < n:
        raise RuntimeError(f"Need at least {n} PDFs in {config.LIBRARY_DIR} to run this benchmark.")
    return pdfs[:n]


async def timed_run(paperqa_engine, label: str, filenames: list[str]) -> dict:
    started = time.monotonic()
    result = await paperqa_engine.run_paperqa(QUERY, filenames)
    wall = time.monotonic() - started
    print(f"\n=== {label} ===")
    print(f"wall time: {wall:.2f}s (engine-reported generation duration: {result['duration']:.2f}s)")
    print(f"cost: ${result['cost']:.4f}, total_tokens: {result['total_tokens']}")
    if result["unindexed_papers"]:
        print(f"WARNING unindexed_papers: {result['unindexed_papers']}")
    return {
        "label": label,
        "wall_time": wall,
        "generation_duration": result["duration"],
        "cost": result["cost"],
        "total_tokens": result["total_tokens"],
    }


async def run_trial(source_pdfs: list[Path], trial_idx: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="pqa_bench_") as tmp:
        temp_library = Path(tmp) / "library"
        temp_index = temp_library / ".pqa_index_bench"
        temp_library.mkdir()
        temp_index.mkdir()
        for pdf in source_pdfs:
            shutil.copy(pdf, temp_library / pdf.name)

        # Point the engine at the isolated temp dirs instead of the real
        # shared library — _build_settings reads these attributes at call
        # time, so patching them before importing paperqa_engine is enough
        # for it to pick up the override.
        config.LIBRARY_DIR = temp_library
        config.LIBRARY_INDEX_DIR = temp_index
        from backend import paperqa_engine  # noqa: E402  (import after patch)

        filenames = [p.name for p in source_pdfs]
        cold = await timed_run(paperqa_engine, f"trial {trial_idx}: cold (empty index)", filenames)
        warm = await timed_run(paperqa_engine, f"trial {trial_idx}: warm (index built this trial)", filenames)

    reduction = 1 - warm["wall_time"] / cold["wall_time"] if cold["wall_time"] else 0.0
    print(f"trial {trial_idx} reduction: {reduction:.1%}")
    return {"trial": trial_idx, "cold": cold, "warm": warm, "wall_time_reduction_pct": reduction}


async def main():
    n_papers = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_PAPERS
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_TRIALS

    source_pdfs = _pick_smallest_papers(n_papers)
    print(f"Using {len(source_pdfs)} papers across {n_trials} trial(s): {[p.name for p in source_pdfs]}")

    trials = [await run_trial(source_pdfs, i + 1) for i in range(n_trials)]

    reductions = [t["wall_time_reduction_pct"] for t in trials]
    mean_reduction = statistics.mean(reductions)
    print("\n=== summary ===")
    for t in trials:
        print(
            f"trial {t['trial']}: cold={t['cold']['wall_time']:.2f}s, "
            f"warm={t['warm']['wall_time']:.2f}s, reduction={t['wall_time_reduction_pct']:.1%}"
        )
    print(f"\nmean wall-time reduction across {n_trials} trial(s): {mean_reduction:.1%}")
    if n_trials > 1:
        print(f"stdev: {statistics.stdev(reductions):.1%}")
    print(
        f"(this is a {n_papers}-paper sample — magnitude scales with paper count/size; "
        "pass a larger n_papers/n_trials for a more representative number)"
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_n{n_papers}_t{n_trials}.json"
    result_path.write_text(
        json.dumps(
            {
                "n_papers": n_papers,
                "papers": [p.name for p in source_pdfs],
                "trials": trials,
                "mean_wall_time_reduction_pct": mean_reduction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n(raw result written to {result_path})")


if __name__ == "__main__":
    asyncio.run(main())
