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
from agent_controller.text_similarity import fuzzy_overlap_count, tokenize

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
