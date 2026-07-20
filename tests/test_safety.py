"""Tests for mmu.safety output-path guardrails.

These tests are load-bearing: they're the ONLY thing standing between a
misconfigured Snakefile and a catastrophic overwrite of the shared
``/mnt/ceph/users/polymathic/`` mount on the cluster. If you break one of
these tests, DO NOT WEAKEN THE TEST — fix the safety module instead.
"""

import pytest

from mmu.safety import (
    ALLOWED_HATS_ROOT_PREFIXES,
    FORBIDDEN_SUBSTRINGS,
    POLYMATHIC_ROOT,
    UnsafeOutputPathError,
    validate_hats_root,
)


class TestAllowedPaths:
    """Paths the pipeline SHOULD accept without complaint."""

    def test_production_root(self):
        validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats")

    def test_cosmos_test_root(self):
        validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_cosmos")

    def test_production_subdir(self):
        validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/sdss")

    def test_cosmos_subdir(self):
        validate_hats_root(
            "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_cosmos/allwise/allwise/allwise"
        )

    def test_trailing_slash(self):
        validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/")

    def test_double_trailing_slashes(self):
        validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats//")

    def test_local_home_dir(self):
        validate_hats_root("/Users/alice/mmu_local")

    def test_tmp_dir(self):
        validate_hats_root("/tmp/test_output")

    def test_relative_path(self):
        validate_hats_root("./test_output/hats")

    def test_other_ceph_mount_unrelated_to_polymathic(self):
        validate_hats_root("/mnt/ceph/users/someoneelse/mmu")


class TestForbiddenPaths:
    """Paths the pipeline MUST reject loudly."""

    def test_bare_polymathic_root(self):
        """The bare polymathic root is the nightmare scenario."""
        with pytest.raises(UnsafeOutputPathError, match="only allowed under"):
            validate_hats_root("/mnt/ceph/users/polymathic")

    def test_bare_polymathic_root_trailing_slash(self):
        with pytest.raises(UnsafeOutputPathError):
            validate_hats_root("/mnt/ceph/users/polymathic/")

    def test_external_data_root(self):
        """The READ-ONLY raw data mirror — writing here corrupts every build."""
        with pytest.raises(UnsafeOutputPathError, match="external_data"):
            validate_hats_root("/mnt/ceph/users/polymathic/external_data")

    def test_external_data_subdir(self):
        with pytest.raises(UnsafeOutputPathError, match="external_data"):
            validate_hats_root("/mnt/ceph/users/polymathic/external_data/astro/SDSS")

    def test_external_data_deep_subdir(self):
        with pytest.raises(UnsafeOutputPathError, match="external_data"):
            validate_hats_root(
                "/mnt/ceph/users/polymathic/external_data/astro/DESI_DR1/coadd-main-dark-0"
            )

    def test_forbidden_substring_outside_polymathic(self):
        """The denylist should fire even if the path isn't under polymathic."""
        with pytest.raises(UnsafeOutputPathError, match="external_data"):
            validate_hats_root("/some/random/place/external_data/mirror")

    def test_unknown_polymathic_sibling(self):
        """A neighbor project under polymathic that isn't on the allowlist."""
        with pytest.raises(UnsafeOutputPathError, match="only allowed under"):
            validate_hats_root("/mnt/ceph/users/polymathic/some_other_project")

    def test_prefix_lookalike(self):
        """Sneaky path that looks like a real prefix but isn't inside it.

        ``MultimodalUniverse_v2_hats_evil`` shares a prefix with the allowed
        ``MultimodalUniverse_v2_hats`` but is a different directory. The
        validator must not treat ``prefix + "_evil"`` as being under ``prefix``.
        """
        with pytest.raises(UnsafeOutputPathError, match="only allowed under"):
            validate_hats_root("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_evil")

    def test_prefix_lookalike_cosmos(self):
        with pytest.raises(UnsafeOutputPathError, match="only allowed under"):
            validate_hats_root(
                "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_cosmos_stolen"
            )

    def test_empty_string(self):
        with pytest.raises(UnsafeOutputPathError, match="empty or not a string"):
            validate_hats_root("")

    def test_none(self):
        with pytest.raises(UnsafeOutputPathError):
            validate_hats_root(None)  # type: ignore[arg-type]

    def test_non_string(self):
        with pytest.raises(UnsafeOutputPathError):
            validate_hats_root(12345)  # type: ignore[arg-type]

    def test_pathlib_object_rejected(self):
        """Pathlib objects aren't strings; require explicit str() conversion
        at the caller so the caller owns any path normalization."""
        from pathlib import PosixPath
        with pytest.raises(UnsafeOutputPathError):
            validate_hats_root(PosixPath("/tmp/foo"))  # type: ignore[arg-type]

    def test_path_traversal_into_external_data(self):
        """Even a relative-traversal path that resolves to external_data
        should be rejected. ``os.path.normpath`` eats the ``..``."""
        with pytest.raises(UnsafeOutputPathError, match="external_data"):
            validate_hats_root(
                "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/../external_data"
            )

    def test_path_traversal_out_of_allowlist(self):
        """A crafted traversal that escapes the allowed prefix should be caught."""
        with pytest.raises(UnsafeOutputPathError, match="only allowed under"):
            validate_hats_root(
                "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/../some_other_project"
            )


class TestAllowlistInvariants:
    """Meta-tests on the allowlist itself, catching changes that weaken safety."""

    def test_no_bare_polymathic_in_allowlist(self):
        """Even an internal typo shouldn't allow the bare polymathic root."""
        assert POLYMATHIC_ROOT not in ALLOWED_HATS_ROOT_PREFIXES
        assert f"{POLYMATHIC_ROOT}/" not in ALLOWED_HATS_ROOT_PREFIXES

    def test_no_external_data_in_allowlist(self):
        for prefix in ALLOWED_HATS_ROOT_PREFIXES:
            assert "external_data" not in prefix, (
                f"allowlist entry {prefix!r} mentions external_data — "
                "that's exactly what the safety rules exist to prevent"
            )

    def test_external_data_in_forbidden(self):
        """Regression guard: external_data must always be on the denylist."""
        assert "external_data" in FORBIDDEN_SUBSTRINGS

    def test_all_allowlist_entries_start_with_polymathic(self):
        for prefix in ALLOWED_HATS_ROOT_PREFIXES:
            assert prefix.startswith(POLYMATHIC_ROOT), (
                f"allowlist entry {prefix!r} is not under {POLYMATHIC_ROOT} "
                "— this check exists because the allowlist's purpose is to "
                "restrict writes under polymathic, not to allow arbitrary paths"
            )

    def test_allowlist_entries_no_trailing_slash(self):
        """Trailing slashes on allowlist entries would break the prefix
        matching logic (``startswith(prefix + "/")`` would produce
        ``foo//bar``). Keep them clean."""
        for prefix in ALLOWED_HATS_ROOT_PREFIXES:
            assert not prefix.endswith("/"), f"{prefix!r} has trailing slash"
