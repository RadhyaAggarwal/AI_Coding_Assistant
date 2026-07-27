"""Semantic (meaning-based) code index: embeds each function/class-sized
chunk of the project so vague, natural-language queries can find relevant
code even when the exact symbol name isn't known -- see
tools/semantic_search.py, the consumer of this. Complements, not
replaces, the exact-match tools (find_symbol/search_code) -- those stay
faster and more precise whenever the exact name or literal text is
already known.

Chunking reuses the ALREADY-EXTRACTED symbol table from repo_index/,
rather than any new per-language parsing: each symbol's chunk runs from
its own start line up to the next symbol's start line in the same file
(sorted by line), capped at _MAX_CHUNK_LINES so one very long function
can't produce one huge, unfocused embedding. This is an approximation,
not true end-line detection -- Symbol only records a start line, the
same established precedent tools/source_snippet.py already relies on for
showing source at a location. In the overwhelming majority of real code
a symbol's definition does run until the next one starts, and a nested
symbol (e.g. a class's own method) naturally produces its own separate,
usually more useful chunk instead of one undifferentiated blob for the
whole class -- a deliberate, disclosed tradeoff, not an oversight.

Caching mirrors repo_index/cache.py's exact shape and granularity (mtime
+size per FILE, not per individual chunk) -- an unchanged file's chunks
are all still fresh; a changed file's chunks are all recomputed together,
the same simplicity tradeoff the AST cache already makes. Each cached
vector's norm is precomputed once here (at embedding time) rather than
recomputed on every search call -- empirically verified this alone gives
roughly a 3x speedup on a generously-sized project (5,000 chunks), with
no new dependency.

A chunk from _chunk_boundaries() can still be too large for the
embedding model's own context window regardless of _MAX_CHUNK_LINES --
line count isn't a reliable proxy for character/token count. Any chunk
over _MAX_CHUNK_CHARS is split further into sequential, line-aligned
sub-chunks (see _split_by_char_limit()) rather than truncated, so no
content is silently discarded; a single line that alone exceeds the
limit (minified/generated content, typically) is skipped rather than
embedded as a corrupted, arbitrarily-cut fragment.
"""
import json
import math
from pathlib import Path
from typing import Callable

from model_interface.base import ModelInterface
from repo_index.indexer import RepoIndex
from repo_index.models import Symbol

_CACHE_FILENAME = "semantic_index_cache.json"
_CACHE_VERSION = 1
_MAX_CHUNK_LINES = 100

# Independent of _MAX_CHUNK_LINES above -- that caps chunk *focus* (one
# function-sized unit, not a whole class), but line count alone doesn't
# bound how many characters an embedding call actually receives (100
# lines of dense code, or an uncapped module-header chunk, can still be
# too large for the embedding model's context window -- confirmed live
# against a real Ollama instance: a 214-line, 13,351-character module
# header was rejected with "input (3053 tokens) is too large to process"
# against a 2048-token context). 4000 chars was chosen from that same
# real failure: 13,351 chars <-> 3053 tokens is ~4.4 chars/token for this
# kind of prose-heavy content, so 4000 chars lands around 900-1000
# tokens -- well under even a bare-minimum 2048-token context, with
# margin for denser tokenization elsewhere.
_MAX_CHUNK_CHARS = 4000


def _cache_path_for(root: Path) -> Path:
    return root / ".agent_state" / _CACHE_FILENAME


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        return {}  # missing/mismatched version -- treat as empty, force a full rebuild
    return data.get("entries", {})


def _save_cache(path: Path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": _CACHE_VERSION, "entries": entries}), encoding="utf-8"
    )


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _chunk_boundaries(symbols: list[Symbol], total_lines: int) -> list[tuple[str, int, int]]:
    """(name, start_line, end_line) for every chunk in one file, 1-indexed
    inclusive, given that file's already-extracted symbols and its total
    line count. Never returns an empty list for a non-empty file -- a
    file with no indexed symbols at all (e.g. an unsupported language, or
    one that's just data/config) still gets one whole-file chunk, capped
    the same way any other chunk is.
    """
    ordered = sorted(symbols, key=lambda s: s.line)
    boundaries: list[tuple[str, int, int]] = []
    if not ordered:
        return [("(whole file)", 1, min(total_lines, _MAX_CHUNK_LINES))] if total_lines else []

    if ordered[0].line > 1:
        boundaries.append(("(module header)", 1, ordered[0].line - 1))
    for i, sym in enumerate(ordered):
        next_line = ordered[i + 1].line if i + 1 < len(ordered) else total_lines + 1
        end = min(next_line - 1, sym.line + _MAX_CHUNK_LINES - 1, total_lines)
        if end >= sym.line:
            boundaries.append((sym.name, sym.line, end))
    return boundaries


def _split_by_char_limit(lines: list[str], start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split a 1-indexed inclusive line range into sub-ranges whose joined
    text stays under max_chars, breaking only on line boundaries so every
    piece is still valid, readable source -- never a mid-line cut. A
    single line longer than max_chars by itself still becomes its own
    one-line sub-range (there's no earlier line to split before); the
    caller decides what to do with that rare case.
    """
    sub_ranges: list[tuple[int, int]] = []
    sub_start = start
    current_len = 0
    for line_num in range(start, end + 1):
        line_len = len(lines[line_num - 1]) + 1  # +1 for the joining newline
        if line_num > sub_start and current_len + line_len > max_chars:
            sub_ranges.append((sub_start, line_num - 1))
            sub_start = line_num
            current_len = 0
        current_len += line_len
    sub_ranges.append((sub_start, end))
    return sub_ranges


class Chunk:
    def __init__(self, file: str, name: str, start_line: int, end_line: int, text: str):
        self.file = file
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.text = text


class SemanticIndex:
    """Builds/maintains per-chunk embeddings for a project, and answers
    similarity searches against them. Construction is potentially slow
    the first time, or after real edits (embeds every not-yet-cached
    chunk) -- on_progress, if given, is called with a human-readable
    message right before any real embedding work happens, so a human
    watching the console sees real evidence of what's happening instead
    of a silent pause (the same spirit as Tool.progress_message(), just
    reaching one level deeper for the one case that actually needs it).
    """

    def __init__(
        self,
        root: str | Path,
        model: ModelInterface,
        on_progress: Callable[[str], None] | None = None,
    ):
        self._root = Path(root).resolve()
        self._model = model
        self._on_progress = on_progress or (lambda message: None)
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._norms: list[float] = []
        self._build()

    def _build(self) -> None:
        repo_index = RepoIndex(self._root)
        cache_file = _cache_path_for(self._root)
        cached_entries = _load_cache(cache_file)
        fresh_entries: dict[str, dict] = {}
        pending: list[tuple[Chunk, dict]] = []  # (chunk, its not-yet-filled cache entry)

        for relative_path, file_index in repo_index.files.items():
            absolute = self._root / relative_path
            try:
                stat = absolute.stat()
                lines = absolute.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            cached_file = cached_entries.get(relative_path)
            file_is_fresh = (
                cached_file is not None
                and cached_file.get("mtime") == stat.st_mtime
                and cached_file.get("size") == stat.st_size
            )
            cached_chunks_by_key = (
                {(c["name"], c["start_line"]): c for c in cached_file["chunks"]}
                if file_is_fresh
                else {}
            )

            file_chunk_entries: list[dict] = []
            for name, start, end in _chunk_boundaries(file_index.symbols, len(lines)):
                text = "\n".join(lines[start - 1 : end])
                if len(text) <= _MAX_CHUNK_CHARS:
                    pieces = [(name, start, end, text)]
                else:
                    sub_ranges = _split_by_char_limit(lines, start, end, _MAX_CHUNK_CHARS)
                    total = len(sub_ranges)
                    pieces = []
                    for i, (sub_start, sub_end) in enumerate(sub_ranges, start=1):
                        sub_text = "\n".join(lines[sub_start - 1 : sub_end])
                        if len(sub_text) > _MAX_CHUNK_CHARS:
                            # A single line too long to embed on its own
                            # (minified/generated content, typically) --
                            # skip it rather than truncate, so nothing is
                            # ever embedded as a corrupted, arbitrarily-cut
                            # fragment. Gated on file_is_fresh so this
                            # doesn't reprint on every unchanged rebuild --
                            # skipped pieces are never cached, so without
                            # this gate the same warning would fire again
                            # every time even though nothing new happened.
                            if not file_is_fresh:
                                self._on_progress(
                                    f"Skipping {relative_path}:{sub_start} for semantic "
                                    "search -- a single line is too long to embed "
                                    "(likely minified/generated content)."
                                )
                            continue
                        pieces.append((f"{name} (part {i}/{total})", sub_start, sub_end, sub_text))

                for piece_name, piece_start, piece_end, piece_text in pieces:
                    chunk = Chunk(relative_path, piece_name, piece_start, piece_end, piece_text)
                    cached = cached_chunks_by_key.get((piece_name, piece_start))
                    if cached is not None:
                        entry = dict(cached)
                        self._chunks.append(chunk)
                        self._vectors.append(entry["vector"])
                        self._norms.append(entry["norm"])
                    else:
                        entry = {"name": piece_name, "start_line": piece_start, "end_line": piece_end, "vector": None, "norm": None}
                        pending.append((chunk, entry))
                    file_chunk_entries.append(entry)

            fresh_entries[relative_path] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "chunks": file_chunk_entries,
            }

        if pending:
            self._on_progress(
                f"Embedding {len(pending)} new/changed chunk"
                f"{'s' if len(pending) != 1 else ''} (first use on a "
                "project, or after real edits, can take a moment)..."
            )
        for chunk, entry in pending:
            vector = self._model.embed(chunk.text)
            norm = _vector_norm(vector)
            entry["vector"] = vector
            entry["norm"] = norm
            self._chunks.append(chunk)
            self._vectors.append(vector)
            self._norms.append(norm)

        _save_cache(cache_file, fresh_entries)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Real chunks ranked by cosine similarity to query, most similar
        first. Empty if the project has no indexed chunks at all."""
        if not self._chunks:
            return []
        query_vector = self._model.embed(query)
        query_norm = _vector_norm(query_vector)
        if query_norm == 0:
            return []

        scored = []
        for chunk, vector, norm in zip(self._chunks, self._vectors, self._norms):
            if norm == 0:
                continue
            dot = sum(x * y for x, y in zip(query_vector, vector))
            scored.append((chunk, dot / (query_norm * norm)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
