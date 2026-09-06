"""Real measurement of prompt-caching's effect on search-agent input cost.

Runs the actual production run_search_agent() (backend/search_agent.py)
against a real query with real Anthropic API calls, capturing each turn's
raw usage (input_tokens, cache_creation_input_tokens,
cache_read_input_tokens), then compares what was actually billed against
the hypothetical cost of sending every one of those tokens fresh (no
caching at all):

    actual cost   = input*P + cache_write*1.25P + cache_read*0.1P
    no-cache cost = (input + cache_write + cache_read) * P

using Claude Sonnet 5 pricing (P = $2.00/MTok input) and Anthropic's
documented cache multipliers (5-minute-TTL write = 1.25x, read = 0.1x).
This is not a narrative estimate — every number comes from a real
response.usage object for this run.

Costs real money (one real multi-turn agent run, ~5 Sonnet calls + 1-2
Haiku scoring calls) and is *not* wired into pytest or the regression
baseline — run it by hand when you want a fresh read on caching's payoff,
e.g. after changing SYSTEM_PROMPT, TOOLS, or the cache-breakpoint logic
in search_agent.py:

    python -m backend.tests.bench_prompt_cache ["your query here"]
"""

import asyncio
import json
import sys
from pathlib import Path

from backend import search_agent as sa
from backend.job_manager import Job

PRICE_INPUT_PER_MTOK = 2.00  # claude-sonnet-5 (see config.ANTHROPIC_MODEL)
CACHE_WRITE_MULT = 1.25  # 5-minute TTL, the default ephemeral cache_control
CACHE_READ_MULT = 0.10

DEFAULT_QUERY = "prompt injection defenses for LLM agents"
RESULT_PATH = Path(__file__).parent / "bench_results" / "prompt_cache.json"


def _instrument(client) -> list[dict]:
    """Wraps client.messages.create to record each call's raw usage
    without changing search_agent.py itself."""
    records: list[dict] = []
    orig_create = client.messages.create

    def wrapped(*args, **kwargs):
        resp = orig_create(*args, **kwargs)
        u = resp.usage
        records.append(
            {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            }
        )
        return resp

    client.messages.create = wrapped
    return records


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    records = _instrument(sa._get_client())

    job = Job(id="bench-cache", query=query)
    candidates = await sa.run_search_agent(job)
    print(f"\n=== search agent finished: {len(candidates)} candidates ===\n")

    print("=== raw per-call usage ===")
    for i, r in enumerate(records):
        print(f"call {i + 1}: {r}")

    total_input = sum(r["input_tokens"] for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    total_write = sum(r["cache_creation_input_tokens"] for r in records)
    total_read = sum(r["cache_read_input_tokens"] for r in records)

    actual_input_cost = (
        total_input * PRICE_INPUT_PER_MTOK
        + total_write * CACHE_WRITE_MULT * PRICE_INPUT_PER_MTOK
        + total_read * CACHE_READ_MULT * PRICE_INPUT_PER_MTOK
    ) / 1_000_000
    hypothetical_input_cost = ((total_input + total_write + total_read) * PRICE_INPUT_PER_MTOK) / 1_000_000
    reduction = 1 - actual_input_cost / hypothetical_input_cost if hypothetical_input_cost else 0.0

    print("\n=== totals across the run ===")
    print(f"fresh input tokens (full price): {total_input}")
    print(f"cache write tokens (1.25x):       {total_write}")
    print(f"cache read tokens (0.1x):         {total_read}")
    print(f"output tokens:                    {total_output}")
    print(f"\nactual input cost (with caching):    ${actual_input_cost:.6f}")
    print(f"hypothetical input cost (no caching): ${hypothetical_input_cost:.6f}")
    print(f"effective input cost reduction:       {reduction:.1%}")

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "query": query,
                "n_calls": len(records),
                "records": records,
                "total_input": total_input,
                "total_cache_write": total_write,
                "total_cache_read": total_read,
                "total_output": total_output,
                "actual_input_cost_usd": actual_input_cost,
                "hypothetical_input_cost_usd": hypothetical_input_cost,
                "reduction_pct": reduction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n(raw result written to {RESULT_PATH})")


if __name__ == "__main__":
    asyncio.run(main())
