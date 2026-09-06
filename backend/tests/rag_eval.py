"""RAG-quality eval: context precision (cheap, deterministic) and context
recall + answer relevance (expensive — runs real generation).

Named after the standard RAG eval trio (see e.g. the RAGAS framework):
- context precision: of the evidence gather_evidence actually retrieved
  (and paid an LLM call to summarize/score — see paperqa's core.py
  _map_fxn_summary), how much of it made it into the final cited answer.
  Low precision is a signal of over-retrieval: paying for evidence-scoring
  calls on chunks that never influence the generated text.
- context recall: of the key facts a good review of the selected papers
  should surface, how many did retrieval actually gather as evidence.
- answer relevance: does the generated review actually address the stated
  research question (not just "is every claim grounded" — that's
  verifier.py's job).

context_precision reuses backend/tests/fixtures/sample_review.json (the
same fixture eval_verifier.py uses) — free, no API calls, and is also
wired into baseline_metrics.py's cheap regression baseline.

context_recall and answer_relevance need real generation runs
(run_paperqa) against known papers already in the shared library, so
they're kept out of the cheap rolling baseline and run as their own
standalone eval (costs real money — a few Claude Sonnet generation runs
plus a handful of cheap Haiku judge calls):

    python -m backend.tests.rag_eval
"""

import asyncio
import json
from pathlib import Path

import anthropic

from backend import config
from backend.paperqa_engine import run_paperqa
from backend.tests.eval_verifier import FIXTURE_PATH
from backend.verifier import extract_cited_sources

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONTEXT_RECALL_CASES_PATH = FIXTURES_DIR / "context_recall_cases.json"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def compute_context_precision(answer: str, evidence: list[dict]) -> float:
    """Fraction of retrieved evidence chunks that actually got cited
    somewhere in the generated answer. Pure computation, no LLM call —
    reuses the same citation-key extraction verifier.py's deterministic
    checks are built on."""
    if not evidence:
        return 0.0
    cited = extract_cited_sources(answer)
    used = sum(1 for e in evidence if e["source"] in cited)
    return used / len(evidence)


RELEVANCE_TOOL = {
    "name": "score_relevance",
    "description": "Score how directly and completely the review addresses the research question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "description": "10 = fully on-topic and covers the question comprehensively, 0 = off-topic.",
            },
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["score", "reason"],
    },
}


async def judge_answer_relevance(query: str, answer: str) -> dict:
    """LLM-judge: does the generated review directly and completely address
    the stated research question? Distinct from faithfulness (verifier.py)
    — a review can be perfectly grounded in real citations while still
    being incomplete or drifting off the actual question asked."""
    client = _get_client()
    msg = await asyncio.to_thread(
        client.messages.create,
        model=config.RERANK_MODEL,
        max_tokens=512,
        tools=[RELEVANCE_TOOL],
        tool_choice={"type": "tool", "name": "score_relevance"},
        messages=[
            {
                "role": "user",
                "content": (
                    "你是评审员。判断下面这篇综述是否直接、完整地回答了研究问题，"
                    "0-10 打分（10=完全切题且覆盖全面，0=文不对题），给一句中文理由。\n\n"
                    f"研究问题：{query}\n\n综述：\n{answer}"
                ),
            }
        ],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return {"score": block.input.get("score"), "reason": block.input.get("reason")}


KEY_POINT_TOOL = {
    "name": "judge_support",
    "description": "Judge whether the evidence pool supports the given key point.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["supported", "reason"],
    },
}


async def judge_key_point_supported(key_point: str, evidence: list[dict]) -> dict:
    """LLM-judge: does ANY chunk in the retrieved evidence pool support this
    hand-labeled key point (synonymous rewording / partial coverage counts)?
    This is context recall's unit of measurement — it's about what
    retrieval surfaced, not what the final answer happened to cite."""
    pool = "\n\n".join(f"[{e['source']}] {e['context']}" for e in evidence)
    client = _get_client()
    msg = await asyncio.to_thread(
        client.messages.create,
        model=config.RERANK_MODEL,
        max_tokens=512,
        tools=[KEY_POINT_TOOL],
        tool_choice={"type": "tool", "name": "judge_support"},
        messages=[
            {
                "role": "user",
                "content": (
                    "下面是一个应该被检索到的关键论点，以及生成综述时实际检索到的证据池。"
                    "判断证据池里是否有任何一条证据支持/涵盖这个论点（同义转述、部分覆盖也算支持）。\n\n"
                    f"关键论点：{key_point}\n\n证据池：\n{pool}"
                ),
            }
        ],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return {"supported": block.input.get("supported"), "reason": block.input.get("reason")}


async def run_recall_case(case: dict) -> dict:
    """One real generation run against known papers, scored on all three
    axes at once — context_recall and answer_relevance need the real
    run_paperqa call anyway, so context_precision rides along for free."""
    result = await run_paperqa(case["query"], case["filenames"])
    key_point_results = await asyncio.gather(
        *(judge_key_point_supported(kp, result["evidence"]) for kp in case["key_points"])
    )
    recall = sum(1 for r in key_point_results if r["supported"]) / len(case["key_points"])
    relevance = await judge_answer_relevance(case["query"], result["answer"])
    precision = compute_context_precision(result["answer"], result["evidence"])
    return {
        "name": case["name"],
        "context_recall": recall,
        "context_precision": precision,
        "answer_relevance": relevance["score"],
        "key_point_results": list(zip(case["key_points"], key_point_results, strict=True)),
        "cost": result["cost"],
    }


async def main():
    print("=== Context precision on the fixed sample_review.json fixture (free, no API calls) ===")
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_precision = compute_context_precision(data["answer"], data["evidence"])
    cited = extract_cited_sources(data["answer"])
    used = sum(1 for e in data["evidence"] if e["source"] in cited)
    print(f"context_precision = {fixture_precision:.0%} ({used}/{len(data['evidence'])} evidence chunks cited)")

    print("\n=== Context recall / relevance (real generation runs, costs real $) ===")
    cases = json.loads(CONTEXT_RECALL_CASES_PATH.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        print(f"running case '{case['name']}'...")
        r = await run_recall_case(case)
        results.append(r)
        print(
            f"  recall={r['context_recall']:.0%} precision={r['context_precision']:.0%} "
            f"relevance={r['answer_relevance']}/10 cost=${r['cost']:.4f}"
        )
        for kp, judged in r["key_point_results"]:
            status = "COVERED" if judged["supported"] else "MISSED"
            print(f"    [{status}] {kp[:70]}")

    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    print(
        f"\navg context_recall={avg_recall:.0%}  avg context_precision={avg_precision:.0%}  "
        f"avg answer_relevance={avg_relevance:.1f}/10  total_cost=${total_cost:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
