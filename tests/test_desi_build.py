"""Fast unit tests for scripts/desi/build_parent_sample_hats.py.

These tests exercise the pure-Python logic (selection_fn, catalog grouping,
LSF estimation, filepath helpers, schema assembly) without actually running
desispec.io.read_spectra, which requires real DESI coadd FITS files that
aren't trivial to fabricate. The cluster validation step exercises the
full pipeline end-to-end.
"""

import importlib.util
import os
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "desi", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_desi_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _make_catalog(n: int, seed: int = 0) -> Table:
    """Build a synthetic zall-pix-iron-like Table covering all the columns
    the pipeline actually reads."""
    rng = np.random.default_rng(seed)
    survey = np.array(["main"] * n, dtype="U10")
    program = np.array(["dark"] * n, dtype="U10")
    objtype = np.array(["TGT"] * n, dtype="U5")
    healpix = np.arange(n, dtype=np.int64) + 1000
    target_id = np.arange(n, dtype=np.int64) + 10_000
    ra = rng.uniform(150.0, 151.0, n)
    dec = rng.uniform(30.0, 31.0, n)
    cols = {
        "SURVEY": survey,
        "PROGRAM": program,
        "OBJTYPE": objtype,
        "MAIN_PRIMARY": np.ones(n, dtype=bool),
        "COADD_FIBERSTATUS": np.zeros(n, dtype=np.int32),
        "HEALPIX": healpix,
        "TARGETID": target_id,
        "TARGET_RA": ra,
        "TARGET_DEC": dec,
        "ZWARN": np.zeros(n, dtype=np.int64),
    }
    for f in build.FLOAT_FEATURES:
        cols[f] = rng.uniform(0, 10, n).astype(np.float32)
    return Table(cols)


class TestConeCutIntegration:
    """Sanity check that the DESI cone cut (applied via mmu.cone in main())
    trims the catalog correctly. We test the underlying filter directly
    against DESI's TARGET_RA/TARGET_DEC columns."""

    def test_cone_filter_against_target_ra_dec(self):
        from mmu.cone import apply_cone_filter

        cat = _make_catalog(10)
        cat["TARGET_RA"] = np.array(
            [150.0, 150.1, 149.9, 150.3, 200.0, 0.0, 150.0, 150.0, 150.0, 150.0]
        )
        cat["TARGET_DEC"] = np.array(
            [2.0, 2.1, 1.9, 2.3, 30.0, 0.0, 2.0, -30.0, 45.0, 89.0]
        )
        mask = apply_cone_filter(
            np.asarray(cat["TARGET_RA"]), np.asarray(cat["TARGET_DEC"]),
            ra_center=150.0, dec_center=2.0, radius=0.5,
        )
        # Indices 0,1,2,3,6 are within 0.5 deg; rest are not.
        assert mask.tolist() == [
            True, True, True, True, False, False, True, False, False, False
        ]


class TestSelectionFn:
    def test_all_pass_baseline(self):
        cat = _make_catalog(10)
        assert build.selection_fn(cat).all()

    def test_rejects_non_main_survey(self):
        cat = _make_catalog(5)
        cat["SURVEY"][2] = "sv3"
        mask = build.selection_fn(cat)
        assert mask.sum() == 4
        assert not mask[2]

    def test_rejects_non_primary(self):
        cat = _make_catalog(5)
        cat["MAIN_PRIMARY"][1] = False
        assert build.selection_fn(cat).sum() == 4

    def test_rejects_non_tgt(self):
        cat = _make_catalog(5)
        cat["OBJTYPE"][0] = "SKY"
        assert build.selection_fn(cat).sum() == 4

    def test_rejects_bad_fiberstatus(self):
        cat = _make_catalog(5)
        cat["COADD_FIBERSTATUS"][3] = 16777216
        assert build.selection_fn(cat).sum() == 4

    def test_rejects_bad_handles(self):
        """One of the BAD_HANDLES should be filtered out."""
        cat = _make_catalog(5)
        cat["PROGRAM"][0] = "dark"
        cat["HEALPIX"][0] = 26535  # in BAD_HANDLES["dark"]
        mask = build.selection_fn(cat)
        assert not mask[0]

    def test_bad_handles_program_specific(self):
        """A healpix that's bad for 'dark' should NOT be filtered if program is 'bright'."""
        cat = _make_catalog(5)
        cat["PROGRAM"][0] = "bright"
        cat["HEALPIX"][0] = 26535  # bad for dark, not bright
        mask = build.selection_fn(cat)
        assert mask[0]

    def test_handles_byte_strings(self):
        """FITS string columns are sometimes byte arrays."""
        cat = _make_catalog(3)
        cat["SURVEY"] = np.array([b"main", b"main", b"sv3"], dtype="S10")
        mask = build.selection_fn(cat)
        assert mask.sum() == 2


class TestFindMatchingIndices:
    def test_identity(self):
        a = np.array([1, 2, 3, 4, 5])
        b = a.copy()
        np.testing.assert_array_equal(build.find_matching_indices(a, b), np.arange(5))

    def test_reorders_correctly(self):
        a = np.array([10, 20, 30, 40])
        b = np.array([30, 10, 40, 20])
        idx = build.find_matching_indices(a, b)
        np.testing.assert_array_equal(b[idx], a)


class TestGauss:
    def test_peak_at_center(self):
        x = np.linspace(-5, 5, 101)
        y = build._gauss(x, a=1.0, x0=0.0, sigma=1.0)
        assert y.argmax() == 50
        assert np.isclose(y.max(), 1.0)


class TestEstimateLsfSigma:
    def test_gaussian_recovery(self):
        """Feed a known Gaussian resolution profile and confirm recovery."""
        n_fibers, ndiag, n_wave = 3, 11, 100
        x = np.arange(ndiag, dtype=np.float32)
        true_sigma = 1.3
        profile = np.exp(-((x - 5.0) ** 2) / (2 * true_sigma**2)).astype(np.float32)
        res = np.broadcast_to(
            profile[None, :, None], (n_fibers, ndiag, n_wave)
        ).copy()
        est = build.estimate_lsf_sigma(res)
        assert np.isclose(est, true_sigma, atol=0.05)


class TestCoaddFilePath:
    def test_flat_layout(self, tmp_path):
        p = build.coadd_file_path(str(tmp_path), "main", "dark", 1234)
        assert p.endswith("coadd-main-dark-1234.fits")
        assert os.path.dirname(p) == str(tmp_path)


class TestGroupCatalog:
    def test_groups_by_file(self):
        cat = _make_catalog(6)
        # Two groups of three: (main, dark, 100) x3 and (main, dark, 200) x3.
        cat["HEALPIX"] = np.array([100, 200, 100, 200, 100, 200])
        groups = build._group_catalog(cat)
        assert len(groups) == 2
        sizes = sorted(len(g[3]) for g in groups)
        assert sizes == [3, 3]

    def test_preserves_target_ids(self):
        cat = _make_catalog(4)
        cat["HEALPIX"] = np.array([100, 100, 100, 100])
        groups = build._group_catalog(cat)
        assert len(groups) == 1
        _, _, _, target_ids, row_idx = groups[0]
        assert len(target_ids) == 4
        np.testing.assert_array_equal(target_ids, cat["TARGETID"][row_idx])


class TestBuildTable:
    """Exercises build_table with a mocked process_coadd to bypass the
    real desispec.io dependency."""

    def _mock_process_coadd(self, filename, target_ids):
        n = len(target_ids)
        n_wave = 30
        rng = np.random.default_rng(hash(filename) & 0xFFFF)
        return {
            "TARGETID": np.asarray(target_ids),
            "flux": rng.normal(0, 1, (n, n_wave)).astype(np.float32),
            "ivar": rng.uniform(0.1, 1, (n, n_wave)).astype(np.float32),
            "lambda": np.tile(np.linspace(3600, 9800, n_wave, dtype=np.float32), (n, 1)),
            "lsf_sigma": np.full((n, n_wave), 1.2, dtype=np.float32),
            "mask": np.zeros((n, n_wave), dtype=bool),
        }

    def test_builds_full_table(self, tmp_path):
        cat = _make_catalog(4)
        cat["HEALPIX"] = np.array([100, 100, 200, 200])
        # Need the coadd files to "exist" — create empty sentinel files.
        for hp in (100, 200):
            open(tmp_path / f"coadd-main-dark-{hp}.fits", "w").close()

        with patch.object(build, "process_coadd", side_effect=self._mock_process_coadd):
            table = build.build_table(cat, str(tmp_path))

        assert table.num_rows == 4
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "spectrum"} <= names
        # All FLOAT_FEATURES and BOOL_FEATURES should be present.
        assert set(build.FLOAT_FEATURES) <= names
        assert set(build.BOOL_FEATURES) <= names

    def test_spectrum_struct_schema(self, tmp_path):
        cat = _make_catalog(2)
        cat["HEALPIX"] = np.array([500, 500])
        open(tmp_path / "coadd-main-dark-500.fits", "w").close()
        with patch.object(build, "process_coadd", side_effect=self._mock_process_coadd):
            table = build.build_table(cat, str(tmp_path))
        spec_type = table.schema.field("spectrum").type
        assert pa.types.is_struct(spec_type)
        fields = {f.name: f.type for f in spec_type}
        assert set(fields.keys()) == {"flux", "ivar", "lsf_sigma", "lambda", "mask"}
        assert fields["flux"] == pa.list_(pa.float32())
        assert fields["ivar"] == pa.list_(pa.float32())
        assert fields["lsf_sigma"] == pa.list_(pa.float32())
        assert fields["lambda"] == pa.list_(pa.float32())
        assert fields["mask"] == pa.list_(pa.bool_())

    def test_first_row_spectrum_contents(self, tmp_path):
        cat = _make_catalog(2)
        cat["HEALPIX"] = np.array([500, 500])
        open(tmp_path / "coadd-main-dark-500.fits", "w").close()
        with patch.object(build, "process_coadd", side_effect=self._mock_process_coadd):
            table = build.build_table(cat, str(tmp_path))
        first = table.column("spectrum")[0].as_py()
        assert set(first.keys()) == {"flux", "ivar", "lsf_sigma", "lambda", "mask"}
        assert len(first["flux"]) == 30
        assert len(first["lambda"]) == 30
        # lsf_sigma is broadcast from a single scalar so all entries equal.
        assert all(v == first["lsf_sigma"][0] for v in first["lsf_sigma"])

    def test_scalar_column_ordering_matches_targets(self, tmp_path):
        """Scalar columns should be joined on TARGETID such that row i of
        the output corresponds to row i of the target_ids used for spectra."""
        cat = _make_catalog(4)
        cat["HEALPIX"] = np.array([100, 200, 100, 200])
        # Set Z to distinct values so we can check ordering.
        cat["Z"] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for hp in (100, 200):
            open(tmp_path / f"coadd-main-dark-{hp}.fits", "w").close()

        with patch.object(build, "process_coadd", side_effect=self._mock_process_coadd):
            table = build.build_table(cat, str(tmp_path))

        # The spectrum order should match the catalog subset (after _group_catalog's
        # lexsort), so the Z values match the TARGETIDs in the output object_id.
        object_ids = [int(x) for x in table.column("object_id").to_pylist()]
        z_values = table.column("Z").to_pylist()
        expected = dict(zip(cat["TARGETID"].tolist(), cat["Z"].tolist()))
        for oid, z in zip(object_ids, z_values):
            assert z == expected[oid]
