"""Output-path safety guardrails for the MMU v2 HATS pipeline.

The shared Flatiron mount at ``/mnt/ceph/users/polymathic/`` contains datasets
that took months of compute to produce, AND the read-only raw-data mirror
under ``external_data/`` that every build script reads from. Writing to the
wrong path on that mount could destroy months of work or corrupt the raw
inputs every future build depends on.

This module centralizes the validation logic so every code path that accepts
a user-configurable output directory (Snakefile, CLI scripts, tests) can
reuse the same rules. It's the single source of truth for "which output
paths are allowed under /mnt/ceph/users/polymathic/".
"""

from __future__ import annotations

import os


# The Flatiron shared research mount.
POLYMATHIC_ROOT = "/mnt/ceph/users/polymathic"

# The only subdirectories of POLYMATHIC_ROOT that our pipeline is permitted to
# write into. Everything else under POLYMATHIC_ROOT is off-limits — especially
# ``external_data/`` (read-only raw data) and any neighbor project that's not
# part of MMU v2 HATS work.
ALLOWED_HATS_ROOT_PREFIXES: tuple[str, ...] = (
    f"{POLYMATHIC_ROOT}/MultimodalUniverse_v2_hats",
    f"{POLYMATHIC_ROOT}/MultimodalUniverse_v2_hats_cosmos",
)

# Explicit denylist substrings that ABSOLUTELY must not appear in any output
# path. Adds a second layer of defense on top of the allowlist: even if a new
# allowlist entry is added in the future, these will still reject writes that
# touch critical infrastructure.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "external_data",  # read-only raw data mirror — writing here corrupts every future build
)


class UnsafeOutputPathError(ValueError):
    """Raised when the requested output directory is not on the safe list.

    This is a loud, explicit error type so grep / stack traces make it obvious
    that the safety check (not a random misconfiguration) blocked the run.
    """


def _path_components(path: str) -> list[str]:
    """Split a normalized absolute path into its component segments.

    Strips leading/trailing empty strings from an absolute path so that
    ``/a/b/c`` yields ``["a", "b", "c"]`` rather than ``["", "a", "b", "c"]``.
    """
    return [p for p in path.split(os.sep) if p]


def _is_under(path: str, prefix: str) -> bool:
    """True if ``path`` is ``prefix`` or lives beneath it, using path-segment
    comparison rather than string prefix matching.

    This catches the substring false-positive: ``/a/b_extra`` is NOT under
    ``/a/b`` even though ``"/a/b_extra".startswith("/a/b")`` is True.
    """
    path_parts = _path_components(os.path.normpath(path))
    prefix_parts = _path_components(os.path.normpath(prefix))
    return (
        len(path_parts) >= len(prefix_parts)
        and path_parts[: len(prefix_parts)] == prefix_parts
    )


def validate_hats_root(hats_root: str) -> None:
    """Raise ``UnsafeOutputPathError`` if ``hats_root`` is not safe to write into.

    Rules:

    1. ``hats_root`` must be a non-empty string.
    2. Symlinks along ``hats_root`` are resolved via ``os.path.realpath`` so
       a symlink pointing at ``external_data/`` can't bypass the check.
    3. If the resolved path starts with ``/mnt/ceph/users/polymathic/`` it must
       also match one of :data:`ALLOWED_HATS_ROOT_PREFIXES`. Matching is done
       by path-segment comparison so ``/mmu_v2_hats_extra`` does NOT match
       ``/mmu_v2_hats`` (no substring false positives).
    4. The resolved path must not have any component equal to a
       :data:`FORBIDDEN_SUBSTRINGS` entry (e.g. an ``external_data`` segment
       anywhere in the path).
    5. Paths outside ``/mnt/ceph/users/polymathic/`` are allowed without
       further checks (user home dirs, local scratch, test fixtures, etc.).
    """
    if not hats_root or not isinstance(hats_root, str):
        raise UnsafeOutputPathError(
            f"hats_root is empty or not a string: {hats_root!r}. "
            "Refusing to run without a known-good output directory."
        )

    # Resolve symlinks against the real filesystem. For paths that don't yet
    # exist on disk (e.g. a brand-new output dir), realpath still resolves
    # any existing parent components and leaves the leaf alone — good enough
    # to catch the symlink-to-external_data attack.
    resolved = os.path.realpath(os.path.normpath(hats_root)).rstrip("/")
    components = _path_components(resolved)

    # Rule 4: forbidden substrings as path COMPONENTS (not substrings).
    # Also check the raw input so ``/foo/../polymathic/external_data`` is caught
    # even if realpath doesn't fully resolve it.
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in components or forbidden in _path_components(
            os.path.normpath(hats_root)
        ):
            raise UnsafeOutputPathError(
                f"Refusing to use hats_root={hats_root!r} "
                f"(resolved to {resolved!r}): path has a {forbidden!r} "
                "component, which is on the forbidden list. "
                "This check exists specifically to prevent overwriting the "
                "read-only raw data mirror at "
                f"{POLYMATHIC_ROOT}/external_data/."
            )

    # Rule 3: anything under POLYMATHIC_ROOT must match an allowed prefix
    # by path-segment comparison (not string startswith).
    if _is_under(resolved, POLYMATHIC_ROOT):
        is_allowed = any(
            _is_under(resolved, prefix) for prefix in ALLOWED_HATS_ROOT_PREFIXES
        )
        if not is_allowed:
            allowed = "\n  - ".join(ALLOWED_HATS_ROOT_PREFIXES)
            raise UnsafeOutputPathError(
                f"Refusing to use hats_root={hats_root!r} "
                f"(resolved to {resolved!r}). "
                f"Writes under {POLYMATHIC_ROOT}/ are only allowed under:\n"
                f"  - {allowed}\n"
                f"If you really want to write somewhere else on this mount, "
                f"update ALLOWED_HATS_ROOT_PREFIXES in mmu/safety.py explicitly."
            )
    # Non-polymathic paths are fine — the user asked for that location knowing
    # it's on their local disk or a personal scratch area.
