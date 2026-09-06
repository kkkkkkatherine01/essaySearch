import os
import time

from paperqa import Docs, Settings, agent_query
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.search import get_directory_index
from paperqa.settings import (
    AgentSettings,
    AnswerSettings,
    IndexSettings,
    ParsingSettings,
    PromptSettings,
    get_default_pdf_parser,
)

from . import config

# PDF fonts commonly embed these as single glyphs (Alphabetic Presentation
# Forms, U+FB00-FB06) instead of the letter pairs they represent, so
# extracted text comes out as "identiﬁed", "suﬀer", etc. — breaking the
# word apart for anything doing plain word/substring matching downstream.
# A blanket unicodedata.normalize("NFKC", ...) would also fix this, but it
# over-corrects for scientific text: it flattens math notation that
# legitimately depends on distinct codepoints, e.g. ℝ -> "R", superscript
# "x²" -> the wrong "x2", italic/bold math letters -> plain ones. So this
# only translates the small, unambiguous set of ligature characters.
_LIGATURE_MAP = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",  # ﬅ (long s + t)
        "ﬆ": "st",  # ﬆ
    }
)


def _fix_ligatures(text: str) -> str:
    return text.translate(_LIGATURE_MAP)


def _ligature_fixing_pdf_parser(path: str | os.PathLike, **kwargs):
    """Wraps paper-qa's configured PDF parser (resolved the same way it
    resolves its own default, so this stays correct whichever backend —
    pypdf or pymupdf — is installed) and fixes ligature glyphs in the
    extracted per-page text before it's chunked and embedded. Applied once
    at parse time (not at read time), so the fix is baked into whatever
    gets cached in the shared index."""
    parsed = get_default_pdf_parser()(path, **kwargs)
    fixed_content = {}
    for page_num, page_contents in parsed.content.items():
        if isinstance(page_contents, tuple):
            text, media = page_contents
            fixed_content[page_num] = (_fix_ligatures(text), media)
        else:
            fixed_content[page_num] = _fix_ligatures(page_contents)
    parsed.content = fixed_content
    return parsed


class _PreloadedDocsEnvironment(PaperQAEnvironment):
    """paper-qa's environment clears whatever Docs object it's given at the
    start of every run (PaperQAEnvironment._reset_docs -> Docs.clear_docs),
    on the assumption that paper_search will repopulate it for this query —
    harmless when paper_search is in play, but it silently wipes out the
    scoped Docs object we build in _load_scoped_docs before the agent gets
    to look at it. We don't use paper_search (see _SCOPED_TOOL_NAMES), so
    there's nothing to repopulate it with — keep what we were given."""

    async def _reset_docs(self) -> None:
        pass


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


# Every job's agent is restricted to these tools instead of paper-qa's
# default (which also includes paper_search). paper_search runs a full-text
# query over get_directory_index's *entire* shared index — every paper any
# job has ever downloaded — and adds whatever matches into this run's
# evidence, regardless of whether the user selected it for this job; that's
# how a paper from an unrelated past job could end up cited in this review.
# gather_evidence/gen_answer only ever see what's already in the Docs object
# we hand agent_query ourselves (see _load_scoped_docs), so leaving
# paper_search out makes that Docs object the only thing the agent can
# retrieve from. `complete` is required for the agent to terminate; `reset`
# just clears in-memory evidence and is harmless to leave in.
_SCOPED_TOOL_NAMES = {"gather_evidence", "gen_answer", "complete", "reset"}


def _build_settings(num_papers: int) -> Settings:
    model = config.ANTHROPIC_MODEL
    llm_config = _llm_config_without_temperature(model)

    return Settings(
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
            tool_names=_SCOPED_TOOL_NAMES,
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
            parse_pdf=_ligature_fixing_pdf_parser,
        ),
        answer=AnswerSettings(
            # Defaults (evidence_k=10, answer_max_sources=5) are tuned for
            # open-domain QA over a huge corpus, where 5 hits is already
            # generous relative to what's available. Here every paper was
            # already deliberately picked by the user, and both cutoffs
            # apply to the pooled, sorted-by-score list of chunks across
            # *all* selected papers before generation — with more than a
            # handful of papers it's easy for the top-scoring chunks to all
            # come from the 1-2 most textually similar papers, silently
            # excluding the rest from the generated review's citations (they
            # still show up in the evidence panel, which reads
            # session.contexts before this cutoff — they just never get
            # cited in the actual prose). Scale both with the paper count so
            # a multi-paper review has a realistic shot at citing all of
            # them, not just whichever ones scored highest.
            evidence_k=max(10, num_papers * 4),
            answer_max_sources=max(8, num_papers * 2),
            answer_length=(
                "a literature-review-style synthesis (not a short direct answer) of "
                "at least 500-800 words, organized into paragraphs by theme rather "
                "than one paragraph per paper, explicitly comparing/connecting the "
                "selected papers where relevant, and closing with a short paragraph "
                "noting any notable gaps or disagreements across them"
            ),
        ),
        prompts=PromptSettings(
            system=(
                "You are writing a section of a literature review that synthesizes "
                "multiple papers to answer a research question, not answering a "
                "one-off factual question. Compare and connect the selected papers "
                "to each other where relevant — note where they agree, disagree, "
                "extend, or build on each other — rather than treating each as an "
                "isolated source. Your audience is an expert, so be highly "
                "specific. If there are ambiguous terms or acronyms, define them "
                "on first use."
            ),
        ),
    )


async def _load_scoped_docs(settings: Settings, filenames: list[str]) -> tuple[Docs, list[str]]:
    """Build a Docs object containing only this job's selected papers
    (by filename, relative to the shared library dir), reusing
    already-computed chunks + embeddings from the shared directory index
    instead of re-parsing/re-embedding — the same cache paper_search would
    otherwise hit, just scoped to `filenames` instead of the whole shared
    library. `build=True` still syncs/indexes the *entire* library (no
    filter), so this keeps growing the same shared cache every other job
    benefits from; it just doesn't hand the agent a way to search outside
    the slice we assemble here.

    Returns (docs, missing) — missing lists filenames that couldn't be
    loaded (e.g. the PDF failed to parse), for the caller to log without
    failing the whole job over a single bad paper."""
    index = await get_directory_index(settings=settings, build=True)
    combined = Docs()
    missing: list[str] = []
    for filename in filenames:
        cached = await index.get_saved_object(filename)
        if cached is None or not cached.docs:
            missing.append(filename)
            continue
        doc = next(iter(cached.docs.values()))
        await combined.aadd_texts(texts=cached.texts, doc=doc, settings=settings)
    return combined, missing


async def run_paperqa(query: str, filenames: list[str]) -> dict:
    """Point paper-qa's agent at exactly this job's selected papers — not
    the whole shared library, see _load_scoped_docs / _SCOPED_TOOL_NAMES —
    and let it gather evidence and generate a cited answer. Search/download
    already happened before this.

    `filenames` are relative to the shared library dir (config.LIBRARY_DIR),
    matching the stable names download.py writes PDFs under, so a paper any
    past job already indexed is reused via the shared index's cache rather
    than re-embedded."""
    settings = _build_settings(len(filenames))
    docs, missing = await _load_scoped_docs(settings, filenames)
    if not docs.docs:
        raise RuntimeError("选中的论文均未能建立索引（PDF 可能无法解析），请重试或更换论文")

    started_at = time.monotonic()
    response = await agent_query(query=query, settings=settings, docs=docs, env_class=_PreloadedDocsEnvironment)
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
        "unindexed_papers": missing,
    }
