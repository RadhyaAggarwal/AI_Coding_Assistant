"""Suggests real files in the project that might be what a tool call
meant when a referenced path doesn't exist -- live-observed gap: across
one test, a model guessed roughly 15 different wrong template paths in
a row (templates/notes.html, templates/note_list.html, core/templates/
index.html, templates/core/note_list.html -- so close, just missing a
leading "core/" -- and more) without ever finding the one real file
that existed, core/templates/core/note_list.html, and without ever
using list_directory on a directory it already knew existed to
discover it. Grounding a "not found" error with the real files that do
exist gives the same real-fact-instead-of-abstract-rejection signal
this project has already found works better than a bare refusal
elsewhere (diff previews, syntax_check's specific parse errors, cached
call results).
"""
from difflib import get_close_matches
from pathlib import Path

from repo_index.scanner import SKIP_DIR_NAMES, iter_project_files

_MAX_SUGGESTIONS = 5
_SIMILARITY_CUTOFF = 0.3


def suggest_similar_paths(project_root: Path, requested_path: str) -> list[str]:
    """Real files in the project that might be what 'requested_path'
    actually meant -- same extension, ranked by filename similarity to
    what was actually asked for. Returns relative path strings (forward
    slashes, matching how paths are given to tools throughout this
    project), or an empty list if the path has no extension or nothing
    with that extension exists anywhere in the project.
    """
    requested = Path(requested_path)
    suffix = requested.suffix
    if not suffix:
        return []

    candidates = [p for p in iter_project_files(project_root) if p.suffix == suffix]
    if not candidates:
        return []

    relative = [p.relative_to(project_root) for p in candidates]
    by_name: dict[str, Path] = {}
    for p in relative:
        by_name.setdefault(p.name, p)

    close_names = get_close_matches(
        requested.name, by_name.keys(), n=_MAX_SUGGESTIONS, cutoff=_SIMILARITY_CUTOFF
    )
    chosen = [by_name[name] for name in close_names] if close_names else relative[:_MAX_SUGGESTIONS]

    return [str(p).replace("\\", "/") for p in chosen]


def suggest_similar_directories(project_root: Path, requested_path: str) -> list[str]:
    """Same idea as suggest_similar_paths, but for a directory path that
    doesn't exist -- ranked by directory-name similarity rather than
    filename+extension, since directories don't have one.
    """
    requested_name = Path(requested_path).name
    if not requested_name:
        return []

    candidates = {
        p.relative_to(project_root)
        for p in project_root.rglob("*")
        if p.is_dir() and not any(part in SKIP_DIR_NAMES for part in p.parts)
    }
    if not candidates:
        return []

    by_name: dict[str, Path] = {}
    for p in candidates:
        by_name.setdefault(p.name, p)

    close_names = get_close_matches(
        requested_name, by_name.keys(), n=_MAX_SUGGESTIONS, cutoff=_SIMILARITY_CUTOFF
    )
    chosen = (
        [by_name[name] for name in close_names]
        if close_names
        else sorted(candidates)[:_MAX_SUGGESTIONS]
    )

    return [str(p).replace("\\", "/") for p in chosen]
