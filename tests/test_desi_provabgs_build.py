"""Fast unit tests for scripts/desi_provabgs/build_parent_sample_hats.py.

These do not touch the cluster — they use a synthetic HDF5 file that mimics
the PROVABGS schema so the selection cuts, Arrow conversion, and MCMC array
handling can be exercised without real DESI data or the provabgs model.
"""

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "desi_provabgs", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_provabgs_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()

N_MCMC = 100
N_THETA = 13


def _make_fake_provabgs_table(n: int = 10, n_valid: int = None) -> Table:
    """Build a synthetic astropy Table matching the PROVABGS HDF5 schema.

    ``n_valid`` objects pass the PROVABGS_LOGMSTAR_BF > 0 and MAG_G/R/Z > 0
    selection; the rest are invalid (LOGMSTAR_BF <= 0).
    """
    if n_valid is None:
        n_valid = n
    rng = np.random.default_rng(7)

    logmstar = np.where(
        np.arange(n) < n_valid,
        rng.uniform(8, 12, n),
        -1.0,
    ).astype(np.float32)

    data = Table({
        "TARGETID": np.arange(10_000, 10_000 + n, dtype=np.int64),
        "RA": rng.uniform(0, 360, n).astype(np.float64),
        "DEC": rng.uniform(-90, 90, n).astype(np.float64),
        "PROVABGS_LOGMSTAR_BF": logmstar,
        "PROVABGS_MCMC": rng.normal(0, 1, (n, N_MCMC, N_THETA)).astype(np.float32),
        "PROVABGS_THETA_BF": rng.normal(0, 1, (n, N_THETA)).astype(np.float32),
        "Z_HP": rng.uniform(0.1, 0.4, n).astype(np.float32),
        "ZERR": rng.uniform(0, 0.01, n).astype(np.float32),
        "TSNR2_BGS": rng.uniform(10, 100, n).astype(np.float32),
        "MAG_G": rng.uniform(17, 22, n).astype(np.float32),
        "MAG_R": rng.uniform(16, 21, n).astype(np.float32),
        "MAG_Z": rng.uniform(15, 20, n).astype(np.float32),
        "MAG_W1": rng.uniform(14, 19, n).astype(np.float32),
        "FIBMAG_R": rng.uniform(16, 21, n).astype(np.float32),
        "HPIX_64": rng.integers(0, 49152, n).astype(np.float32),
        "PROVABGS_Z_MAX": rng.uniform(0.4, 0.6, n).astype(np.float32),
        "SCHLEGEL_COLOR": rng.uniform(0, 1, n).astype(np.float32),
        "PROVABGS_W_ZFAIL": rng.uniform(0, 1, n).astype(np.float32),
        "PROVABGS_W_FIBASSIGN": rng.uniform(0, 1, n).astype(np.float32),
        "IS_BGS_BRIGHT": rng.integers(0, 2, n).astype(bool),
        "IS_BGS_FAINT": rng.integers(0, 2, n).astype(bool),
    })
    return data


def _write_fake_provabgs_hdf5(path: str, n: int = 10, n_valid: int = None) -> str:
    """Write a synthetic PROVABGS HDF5 file matching astropy Table.read format."""
    data = _make_fake_provabgs_table(n=n, n_valid=n_valid)
    data.write(path, overwrite=True)
    return path


@pytest.fixture
def fake_raw_root(tmp_path):
    raw_root = tmp_path / "DESI_PROVABGS"
    raw_root.mkdir()
    _write_fake_provabgs_hdf5(str(raw_root / build.FILENAME), n=12, n_valid=10)
    return str(raw_root)


class TestSelectionMask:
    def test_valid_rows_pass(self):
        data = _make_fake_provabgs_table(n=5, n_valid=3)
        mask = build.selection_mask(data)
        assert mask[:3].all()

    def test_invalid_rows_fail(self):
        data = _make_fake_provabgs_table(n=5, n_valid=3)
        mask = build.selection_mask(data)
        # rows 3,4 have logmstar = -1
        assert not mask[3]
        assert not mask[4]

    def test_all_valid(self):
        data = _make_fake_provabgs_table(n=8, n_valid=8)
        mask = build.selection_mask(data)
        assert mask.all()


class TestMcmcToArrow:
    def test_shape_and_type(self):
        rng = np.random.default_rng(0)
        mcmc = rng.normal(0, 1, (5, N_MCMC, N_THETA)).astype(np.float32)
        arr = build._mcmc_to_arrow(mcmc)
        assert len(arr) == 5
        assert pa.types.is_list(arr.type)
        assert pa.types.is_list(arr.type.value_type)
        assert arr.type.value_type.value_type == pa.float32()

    def test_values_preserved(self):
        rng = np.random.default_rng(1)
        mcmc = rng.normal(0, 1, (2, N_MCMC, N_THETA)).astype(np.float32)
        arr = build._mcmc_to_arrow(mcmc)
        # Check the first sample's first chain step
        row0 = arr[0].as_py()
        assert len(row0) == N_MCMC
        assert len(row0[0]) == N_THETA
        np.testing.assert_allclose(row0[0], mcmc[0, 0].tolist(), rtol=1e-5)


class TestBuildArrowTable:
    def test_required_columns(self):
        data = _make_fake_provabgs_table(n=5, n_valid=5)
        table = build.build_arrow_table(data, include_best_fit=False)
        names = table.schema.names
        for col in ["ra", "dec", "object_id", "PROVABGS_MCMC", "PROVABGS_THETA_BF",
                    "PROVABGS_LOGMSTAR_BF"]:
            assert col in names, f"Missing column: {col}"
        for feat in build.FLOAT_FEATURES:
            assert feat in names, f"Missing float feature: {feat}"
        for feat in build.BOOL_FEATURES:
            assert feat in names, f"Missing bool feature: {feat}"

    def test_best_fit_columns_excluded_when_skipped(self):
        data = _make_fake_provabgs_table(n=5, n_valid=5)
        table = build.build_arrow_table(data, include_best_fit=False)
        for feat in build.BEST_FIT_FLOAT_FEATURES:
            assert feat not in table.schema.names

    def test_best_fit_columns_included(self):
        data = _make_fake_provabgs_table(n=5, n_valid=5)
        # Add synthetic best-fit columns as would be produced by compute_best_fit_properties
        data["Z_MW"] = np.ones(5, dtype=np.float32) * 0.02
        data["TAGE_MW"] = np.ones(5, dtype=np.float32) * 8.0
        data["AVG_SFR"] = np.ones(5, dtype=np.float32) * 0.5
        table = build.build_arrow_table(data, include_best_fit=True)
        for feat in build.BEST_FIT_FLOAT_FEATURES:
            assert feat in table.schema.names, f"Missing: {feat}"

    def test_object_id_string_type(self):
        data = _make_fake_provabgs_table(n=3, n_valid=3)
        table = build.build_arrow_table(data, include_best_fit=False)
        assert table.schema.field("object_id").type == pa.string()

    def test_ra_dec_float64(self):
        data = _make_fake_provabgs_table(n=3, n_valid=3)
        table = build.build_arrow_table(data, include_best_fit=False)
        assert table.schema.field("ra").type == pa.float64()
        assert table.schema.field("dec").type == pa.float64()

    def test_mcmc_nested_list_type(self):
        data = _make_fake_provabgs_table(n=3, n_valid=3)
        table = build.build_arrow_table(data, include_best_fit=False)
        mcmc_field = table.schema.field("PROVABGS_MCMC")
        assert pa.types.is_list(mcmc_field.type)
        assert pa.types.is_list(mcmc_field.type.value_type)
        assert mcmc_field.type.value_type.value_type == pa.float32()

    def test_theta_bf_list_type(self):
        data = _make_fake_provabgs_table(n=3, n_valid=3)
        table = build.build_arrow_table(data, include_best_fit=False)
        theta_field = table.schema.field("PROVABGS_THETA_BF")
        assert pa.types.is_list(theta_field.type)
        assert theta_field.type.value_type == pa.float32()

    def test_row_count(self):
        n = 7
        data = _make_fake_provabgs_table(n=n, n_valid=n)
        table = build.build_arrow_table(data, include_best_fit=False)
        assert table.num_rows == n


class TestLoadData:
    def test_loads_hdf5(self, fake_raw_root):
        data = build.load_data(fake_raw_root)
        assert len(data) == 12
        assert "TARGETID" in data.colnames
        assert "PROVABGS_MCMC" in data.colnames

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PROVABGS"):
            build.load_data(str(tmp_path / "no_such_dir"))


class TestMainEntryPoint:
    def test_returns_1_with_missing_file(self, tmp_path):
        ret = build.main([
            "--raw-root", str(tmp_path / "nonexistent"),
            "--output-root", str(tmp_path / "out"),
            "--skip-best-fit",
        ])
        assert ret == 1

    def test_returns_0_with_fake_data(self, fake_raw_root, tmp_path):
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--skip-best-fit",
            "--pixel-threshold", "64",
        ])
        assert ret == 0

    def test_cone_cuts_to_zero_returns_1(self, fake_raw_root, tmp_path):
        # Use a cone that's essentially a point at the south pole — no objects there
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--skip-best-fit",
            "--ra-center", "0.0",
            "--dec-center", "-89.99",
            "--radius", "0.001",
        ])
        assert ret == 1

    def test_max_files_arg_accepted(self, fake_raw_root, tmp_path):
        # --max-files is a no-op for this single-file dataset but must be accepted
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--skip-best-fit",
            "--max-files", "1",
            "--pixel-threshold", "64",
        ])
        assert ret == 0

    @pytest.mark.slow
    def test_full_build_produces_catalog_dir(self, fake_raw_root, tmp_path):
        out = tmp_path / "out"
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(out),
            "--skip-best-fit",
            "--pixel-threshold", "64",
        ])
        assert ret == 0
        catalog_dir = out / "desi_provabgs" / "desi_provabgs"
        assert catalog_dir.exists()
