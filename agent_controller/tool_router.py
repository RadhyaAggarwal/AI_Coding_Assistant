"""Lightweight, per-step tool routing: narrows the full tool list to the
tools most plausibly relevant to what's happening right now, since
smaller local models select the wrong tool more often as the tool count
grows. Measured directly against qwen2.5-coder:7b: find_callers succeeded
2/3 times with only itself registered, 0/3 times with 10 tools registered.

Deliberately NOT LLM-based: an extra model call to do the routing itself
would roughly double per-step latency on this CPU-only setup. Instead, a
cheap keyword-overlap heuristic (agent_controller/text_similarity.py)
between the conversation so far and each tool's name + description,
re-applied fresh at every step (not once per whole request) so it
naturally adapts as a multi-tool task progresses — step 1 might narrow
toward file-exploration tools, step 2 (once something's been read)
toward editing tools, etc.

Deliberately biased toward over-inclusion, not under-inclusion: keeping
an irrelevant tool costs a few extra schema tokens; excluding the one
tool actually needed silently breaks the request. So this fails open
(returns every tool) whenever there's no clear keyword signal, and never
narrows below a minimum floor even when scoring is confident.

Note: this alone does not guarantee a multi-tool task actually uses every
tool it needs — routing only controls what's *offered*, not what the
model *chooses*. See agent_controller/request_coverage.py for the
complementary check on whether a compound request's parts were actually
investigated via real tool calls.

Measured live regression this router itself caused: a "there's a bug,
find and fix it" request shares zero vocabulary with edit_file's name or
description (which talk about "create"/"search"/"replace", never
"fix"/"bug"), while "find" and "run the tests" strongly matched three
find_* tools and run_command by name. edit_file scored at/near zero and
was cut below the top-min_keep line every single step of an 8-step run —
the model never once saw edit_file as an option and had no way to apply
the fix it correctly diagnosed. Keyword scoring can't be made to
recognize every way a task might imply "you'll need to edit a file"
without the same whack-a-mole vocabulary-list problem already rejected
for request_coverage.py. Instead, `always_keep_names` lets a caller
protect tools by an existing structural signal instead of guessing at
wording: ToolRegistry.confirmation_required_names() marks exactly the
side-effecting tools (edit_file, run_command) whose silent exclusion is
catastrophic, versus the read-only majority where an occasional miss is
a shrug, not a broken task.
"""
import math

from agent_controller.text_similarity import fuzzy_overlap_count, tokenize
from model_interface.base import ModelInterface

_NAME_MATCH_WEIGHT = 5
_DESCRIPTION_MATCH_WEIGHT = 1
_DEFAULT_MIN_KEEP = 4
# Real tool descriptions are prose, and this project's tools all deal in
# "file"/"directory"/"project" vocabulary, so an absolute "score > 0"
# threshold keeps nearly everything (every tool picks up some incidental
# overlap). Keeping only tools scoring within this fraction of the top
# score discriminates a clear leader from background noise, while still
# keeping multiple tools if several score comparably (a real multi-tool
# need, not noise).
_RELATIVE_KEEP_THRESHOLD = 0.4


def _score(query_words: set[str], schema: dict) -> int:
    function = schema["function"]
    name_words = tokenize(function["name"].replace("_", " "))
    description_words = tokenize(function.get("description", ""))
    return (
        fuzzy_overlap_count(query_words, name_words) * _NAME_MATCH_WEIGHT
        + fuzzy_overlap_count(query_words, description_words) * _DESCRIPTION_MATCH_WEIGHT
    )


def route_tools(
    query_text: str,
    schemas: list[dict],
    min_keep: int = _DEFAULT_MIN_KEEP,
    always_keep_names: frozenset[str] = frozenset(),
) -> list[dict]:
    """Return the schemas most relevant to query_text, or all of them if
    there's no clear signal (fail open) or there aren't more than
    min_keep to begin with (nothing worth narrowing).

    always_keep_names bypasses scoring entirely for the tools it names —
    they survive narrowing regardless of how low they score. Intended for
    tools whose absence would silently break the task rather than just
    cost a slightly worse answer (see module docstring).
    """
    if len(schemas) <= min_keep:
        return schemas

    query_words = tokenize(query_text)
    if not query_words:
        return schemas

    scored = [(schema, _score(query_words, schema)) for schema in schemas]
    max_score = max(score for _, score in scored)
    if max_score == 0:
        return schemas  # no signal at all — fail open, don't guess

    threshold = max_score * _RELATIVE_KEEP_THRESHOLD
    scored.sort(key=lambda pair: pair[1], reverse=True)
    kept = [
        schema
        for schema, score in scored
        if score >= threshold or schema["function"]["name"] in always_keep_names
    ]
    if len(kept) < min_keep:
        kept_names = {schema["function"]["name"] for schema in kept}
        for schema, _ in scored:
            if len(kept) >= min_keep:
                break
            if schema["function"]["name"] not in kept_names:
                kept.append(schema)
                kept_names.add(schema["function"]["name"])
    return kept


# --- Opt-in hybrid (keyword + embedding) routing -----------------------
#
# Everything above this line is route_tools()'s original, already-verified
# keyword-only path -- deliberately untouched by what follows, not
# refactored to share code with it, so enabling the hybrid path below can
# never change behavior for anyone who doesn't opt into it (see
# config.yaml's agent.embedding_aware_routing, default False).
#
# Motivation, confirmed with real data, not assumed: keyword scoring
# alone can score a conceptually-relevant tool at zero just because it's
# worded differently than the request (the edit_file/"there's a bug, fix
# it" case documented above). A real embedding-similarity check against
# the same real tool set correctly ranked edit_file #2 out of 13 tools
# for that exact request, instead of dead last.
#
# First combination attempt (normalize each signal to [0, 1] relative to
# its own max, keep whichever is higher) was tried and rejected after
# live-testing against the real embedding model, not just unit-tested
# with synthetic vectors. Real cosine similarities between this
# project's actual tool descriptions and a real query clustered tightly
# (roughly 0.49-0.62 for one query, 0.52-0.62 for another) -- normalizing
# a band that flat by its own max means even the LEAST similar tool
# lands close to 1.0 relatively (e.g. 0.50/0.62 = 0.81), which clears the
# 0.4-of-max keep threshold easily. Live result: every single tool got
# kept, zero narrowing -- exactly the outcome routing exists to prevent.
#
# Current approach instead combines each tool's RANK under each signal
# (1 = best), not its raw score -- rank is always spread from 1 to N
# regardless of how flat or spiky the underlying scores are, so this
# doesn't inherit the flat-embedding-band problem above. See
# _fractional_ranks() and route_tools_hybrid() for the combination
# itself. Deliberately undamped (no "+k" constant the way textbook
# reciprocal rank fusion uses, typically k=~60): that constant is sized
# for merging results from corpora with hundreds/thousands of candidates
# each, where rank 1 vs. rank 100 should barely matter. With only a
# handful to ~a dozen tools total, a constant that size would flatten
# rank 1 and rank 13 into nearly the same score, reintroducing the exact
# problem being fixed.
#
# Also confirmed with real data: this does NOT fix every routing-adjacent
# problem. Four live tests found semantic_search going unused even when
# it was already being offered under keyword-only routing -- that's a
# model-choice gap, not a routing-exclusion gap, and hybrid routing has
# no effect on it either way.


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _fractional_ranks(scores: dict[str, float]) -> dict[str, float]:
    """1-indexed rank per name, best (highest) score getting rank 1.
    Tied scores share the average of the positions their group
    collectively occupies (e.g. a 3-way tie for 1st-3rd place all get
    rank 2), rather than an arbitrary tie-break by input order -- the
    standard "fractional ranking" approach, so tied tools are never
    silently favored or disfavored by whatever order they happen to be
    registered in.
    """
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        while j < n and ordered[j][1] == ordered[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2  # 1-indexed positions i+1..j, shared
        for name, _ in ordered[i:j]:
            ranks[name] = average_rank
        i = j
    return ranks


class ToolEmbeddingCache:
    """Caches each tool schema's embedding vector for the lifetime of one
    request. Tool schemas don't change between steps, but
    route_tools_hybrid() runs fresh every step (same as route_tools()
    already does) -- without this, every step would re-embed the entire
    tool set from scratch instead of just the one new query.
    """

    def __init__(self, model: ModelInterface):
        self._model = model
        self._tool_vectors: dict[str, list[float]] = {}

    def _tool_text(self, schema: dict) -> str:
        function = schema["function"]
        return f"{function['name']}: {function.get('description', '')}"

    def tool_vector(self, schema: dict) -> list[float] | None:
        """None on a real embedding failure (network, unconfigured
        model, etc.) -- callers treat a missing vector as "no embedding
        signal for this tool" rather than letting the failure propagate,
        so one bad call degrades this tool's score, not the whole step.
        """
        key = self._tool_text(schema)
        if key not in self._tool_vectors:
            try:
                self._tool_vectors[key] = self._model.embed(key)
            except Exception:
                return None
        return self._tool_vectors[key]

    def query_vector(self, query_text: str) -> list[float] | None:
        """Not cached -- the query changes every step, unlike tool
        schemas. None on failure, same fail-open contract as
        tool_vector()."""
        try:
            return self._model.embed(query_text)
        except Exception:
            return None


def route_tools_hybrid(
    query_text: str,
    schemas: list[dict],
    embedding_cache: ToolEmbeddingCache,
    min_keep: int = _DEFAULT_MIN_KEEP,
    always_keep_names: frozenset[str] = frozenset(),
) -> list[dict]:
    """Same narrowing goal as route_tools(), but a tool survives if it
    scores well on EITHER keyword overlap OR embedding similarity to
    query_text, not just keyword overlap alone. Opt-in only -- see the
    module-level note above this function for why route_tools() itself
    is never modified.

    Degrades to exactly route_tools()'s own behavior (by calling it
    directly, not reimplementing it) whenever the embedding side isn't
    usable for this step -- a network hiccup here should never make
    routing worse than today's baseline, only potentially better.
    """
    if len(schemas) <= min_keep:
        return schemas

    query_vector = embedding_cache.query_vector(query_text)
    if query_vector is None:
        return route_tools(query_text, schemas, min_keep, always_keep_names)

    query_words = tokenize(query_text)
    keyword_scores = {
        schema["function"]["name"]: _score(query_words, schema) for schema in schemas
    }
    embedding_scores: dict[str, float] = {}
    for schema in schemas:
        tool_vector = embedding_cache.tool_vector(schema)
        name = schema["function"]["name"]
        embedding_scores[name] = (
            _cosine_similarity(query_vector, tool_vector) if tool_vector is not None else 0.0
        )

    keyword_ranks = _fractional_ranks(keyword_scores)
    embedding_ranks = _fractional_ranks(embedding_scores)
    # Undamped reciprocal rank fusion -- see the module-level note above
    # this function for why no "+k" constant is used here.
    combined = {
        name: 1 / keyword_ranks[name] + 1 / embedding_ranks[name] for name in keyword_ranks
    }

    max_combined = max(combined.values(), default=0.0)
    if max_combined == 0:
        return schemas  # no signal from either side -- fail open, don't guess

    threshold = max_combined * _RELATIVE_KEEP_THRESHOLD
    scored = sorted(schemas, key=lambda schema: combined[schema["function"]["name"]], reverse=True)
    kept = [
        schema
        for schema in scored
        if combined[schema["function"]["name"]] >= threshold
        or schema["function"]["name"] in always_keep_names
    ]
    if len(kept) < min_keep:
        kept_names = {schema["function"]["name"] for schema in kept}
        for schema in scored:
            if len(kept) >= min_keep:
                break
            if schema["function"]["name"] not in kept_names:
                kept.append(schema)
                kept_names.add(schema["function"]["name"])
    return kept
