"""Fast unit tests for scripts/tess/build_parent_sample_hats.py."""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "tess", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_tess_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_lightcurve(path: str, tic: int, sector: int, n_cadences: int = 100) -> None:
    """Write a tiny SPOC FFI lightcurve FITS file with the columns we read."""
    rng = np.random.default_rng(tic)
    cols = [
        fits.Column(name="TIME", format="D",
                    array=np.linspace(2825.0, 2853.0, n_cadences) + rng.normal(0, 0.001, n_cadences)),
        fits.Column(name="SAP_FLUX", format="E",
                    array=(2300.0 + rng.normal(0, 5, n_cadences)).astype(np.float32)),
        fits.Column(name="SAP_FLUX_ERR", format="E",
                    array=rng.uniform(1, 10, n_cadences).astype(np.float32)),
        fits.Column(name="QUALITY", format="J",
                    array=np.zeros(n_cadences, dtype=np.int32)),
    ]
    lc = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    lc.header["RA_OBJ"] = 180.0 + tic * 0.001
    lc.header["DEC_OBJ"] = 20.0 + tic * 0.001
    fits.HDUList([fits.PrimaryHDU(), lc]).writeto(path, overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "tess"
    root.mkdir()
    for tic, sector in [(1668875, 64), (1668887, 64)]:
        fname = f"hlsp_tess-spoc_tess_phot_{tic:016d}-s{sector:04d}_tess_v1_lc.fits"
        _write_fake_lightcurve(str(root / fname), tic, sector)
    return str(root)


class TestParseFilename:
    def test_extracts_tic_and_sector(self):
        out = build.parse_filename("hlsp_tess-spoc_tess_phot_0000000001668875-s0064_tess_v1_lc.fits")
        assert out == (1668875, 64)

    def test_returns_none_on_unmatched(self):
        assert build.parse_filename("not_a_tess_file.fits") is None


class TestFindRawFiles:
    def test_finds_files(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 2

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadLightcurve:
    def test_returns_dict(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        row = build.read_lightcurve(files[0])
        assert row is not None
        assert "tic_id" in row
        assert "sector" in row
        assert row["sector"] == 64
        assert "time" in row and "flux" in row

    def test_drops_nan_cadences(self, tmp_path):
        path = tmp_path / "hlsp_tess-spoc_tess_phot_0000000000000001-s0001_tess_v1_lc.fits"
        n = 10
        cols = [
            fits.Column(name="TIME", format="D",
                        array=np.array([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0])),
            fits.Column(name="SAP_FLUX", format="E", array=np.ones(n, dtype=np.float32)),
            fits.Column(name="SAP_FLUX_ERR", format="E", array=np.ones(n, dtype=np.float32)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(n, dtype=np.int32)),
        ]
        lc = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
        lc.header["RA_OBJ"] = 0.0
        lc.header["DEC_OBJ"] = 0.0
        fits.HDUList([fits.PrimaryHDU(), lc]).writeto(path)
        row = build.read_lightcurve(str(path))
        assert len(row["time"]) == 7  # 3 NaNs dropped


class TestBuildTable:
    def test_schema(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        rows = [build.read_lightcurve(p) for p in files]
        table = build.build_table([r for r in rows if r is not None])
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "object_id" in table.schema.names
        assert "lightcurve" in table.schema.names
        assert table.num_rows == 2

    def test_lightcurve_struct_fields(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        rows = [build.read_lightcurve(p) for p in files]
        table = build.build_table([r for r in rows if r is not None])
        spec = table.schema.field("lightcurve").type
        names = [spec.field(i).name for i in range(spec.num_fields)]
        assert set(names) == {"time", "flux", "flux_err", "quality"}


class TestVariableLengthLightcurves:
    """Regression: lightcurves of different lengths must round-trip without padding."""

    def _make_row(self, tic, n):
        return {
            "tic_id": tic,
            "sector": 1,
            "ra": float(tic),
            "dec": 0.0,
            "time": np.linspace(0, 10, n, dtype=np.float64),
            "flux": np.ones(n, dtype=np.float32),
            "flux_err": np.ones(n, dtype=np.float32) * 0.01,
            "quality": np.zeros(n, dtype=np.int32),
        }

    def test_different_lengths_preserved(self):
        rows = [self._make_row(1, 10), self._make_row(2, 50), self._make_row(3, 7)]
        table = build.build_table(rows)
        lcs = table.column("lightcurve")
        assert len(lcs[0].as_py()["flux"]) == 10
        assert len(lcs[1].as_py()["flux"]) == 50
        assert len(lcs[2].as_py()["flux"]) == 7

    def test_lightcurve_field_is_list_type(self):
        rows = [self._make_row(1, 10), self._make_row(2, 5)]
        table = build.build_table(rows)
        spec = table.schema.field("lightcurve").type
        flux_field = spec.field("flux")
        assert pa.types.is_list(flux_field.type), (
            "flux should be a variable-length pa.list_, not a fixed-width array"
        )

    def test_no_padding_artifacts(self):
        # 3-row lightcurve should NOT have any extra zeros appended.
        row = self._make_row(99, 3)
        row["flux"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        table = build.build_table([row])
        flux = table.column("lightcurve")[0].as_py()["flux"]
        assert flux == [1.0, 2.0, 3.0]
