"""Shared lightweight text-matching primitives used by both
tool_router.py (which tool is relevant to what's happening right now)
and request_coverage.py (was a compound request's part actually
investigated). Cheap word-overlap / fuzzy-prefix matching, not real
NLP — deliberately, to avoid needing an extra model call for either
purpose.
"""
import re

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "this", "that", "does", "where", "what", "which", "how",
    "actually", "please", "can", "you", "it", "be", "are", "was", "were",
    "there", "here", "any", "all", "not", "but", "from", "at", "as",
}
MIN_WORD_LEN = 3
# Matching prefix length treated as "close enough" (e.g. "called" and
# "callers" share "call") — crude but avoids missing a match just because
# the phrasing uses a different word form than the tool/request does.
MIN_PREFIX_MATCH = 4


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= MIN_WORD_LEN and w not in STOPWORDS}


def fuzzy_overlap_count(words_a: set[str], words_b: set[str]) -> int:
    count = 0
    for a in words_a:
        for b in words_b:
            prefix_len = min(len(a), len(b), MIN_PREFIX_MATCH)
            if prefix_len >= MIN_PREFIX_MATCH and a[:prefix_len] == b[:prefix_len]:
                count += 1
                break
    return count
