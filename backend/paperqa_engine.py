import time

from paperqa import Settings, agent_query
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

from . import config


def _llm_config_without_temperature(model: str) -> dict:
    """Some Claude models (e.g. claude-sonnet-5) reject the `temperature`
    param outright ("deprecated for this model"). paper-qa's own default
    litellm config always includes it, so build our own model_list config
    that omits it instead."""
    return {
        "name": model,
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {"model": model},
            }
        ],
    }


async def run_paperqa(query: str) -> dict:
    """Point paper-qa's agent at the shared paper library (all PDFs any job
    has downloaded, deduped by paper id) and let it gather evidence and
    generate a cited answer. Search/download already happened before this.

    Every job shares the same paper_directory/index_directory so paper-qa's
    own incremental index (paperqa/agents/search.py's filecheck) skips
    papers it already indexed. Tradeoff: agent_query can see the whole
    library, not just this job's selection — acceptable since evidence
    scoring already filters for relevance."""
    model = config.ANTHROPIC_MODEL
    llm_config = _llm_config_without_temperature(model)

    settings = Settings(
        llm=model,
        summary_llm=model,
        embedding=config.EMBEDDING_MODEL,
        llm_config=llm_config,
        summary_llm_config=llm_config,
        agent=AgentSettings(
            agent_llm=model,
            agent_llm_config=llm_config,
            index=IndexSettings(
                paper_directory=str(config.LIBRARY_DIR),
                index_directory=str(config.LIBRARY_INDEX_DIR),
                sync_with_paper_directory=True,
            ),
            rebuild_index=True,
        ),
        parsing=ParsingSettings(
            enrichment_llm=model,
            enrichment_llm_config=llm_config,
            # We already have metadata from our own arXiv/Semantic Scholar
            # search step; skip paper-qa's own Crossref/S2 lookup so we don't
            # additionally burn through their unauthenticated rate limits.
            use_doc_details=False,
            # Text-only: image extraction needs Pillow (not a dependency we
            # otherwise need) and we only care about cited text evidence.
            multimodal=False,
        ),
    )

    started_at = time.monotonic()
    response = await agent_query(query=query, settings=settings)
    # response.duration is never populated by this paper-qa version (stays
    # 0.0) — time it ourselves.
    duration = time.monotonic() - started_at
    session = response.session

    evidence = [
        {
            "context": c.context,
            "score": c.score,
            "source": c.text.name,
            "citation": c.text.doc.citation,
        }
        for c in sorted(session.contexts, key=lambda c: c.score, reverse=True)
    ]

    total_tokens = sum(sum(counts) for counts in session.token_counts.values())

    return {
        "answer": session.formatted_answer or session.answer,
        "references": session.references,
        "evidence": evidence,
        "cost": session.cost,
        "duration": duration,
        "total_tokens": total_tokens,
    }
