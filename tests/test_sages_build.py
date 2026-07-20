"""Fast unit tests for scripts/sages/build_parent_sample_hats.py."""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "sages", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_sages_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_dr1_uv(path: str, n_rows: int = 8) -> None:
    rng = np.random.default_rng(0)
    cols = [
        # SAGES IDs are strings like 'SAGE000206.1+000720', not integers.
        fits.Column(name="SAGE_ID", format="20A",
                    array=np.array([f"SAGE{i:06d}.1+000000" for i in range(n_rows)])),
        fits.Column(name="RA", format="D", array=rng.uniform(0, 360, n_rows)),
        fits.Column(name="DEC", format="D", array=rng.uniform(-90, 90, n_rows)),
        # Two rows below the magnitude floor + one with bad flag => 5 should pass cuts.
        fits.Column(name="MAG_U", format="E", array=np.array([15.0, 14.5, -999.0, 16.2, 13.1, -999.0, 14.0, 15.5], dtype=np.float32)),
        fits.Column(name="MAG_V", format="E", array=np.array([14.5, 14.0, 13.5, 15.7, 12.8, 13.0, 13.6, 14.9], dtype=np.float32)),
        fits.Column(name="FLAG_U", format="J", array=np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)),
        fits.Column(name="FLAG_V", format="J", array=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)),
        fits.Column(name="ERR_U", format="E", array=rng.uniform(0.01, 0.1, n_rows).astype(np.float32)),
        fits.Column(name="ERR_V", format="E", array=rng.uniform(0.01, 0.1, n_rows).astype(np.float32)),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "sages"
    root.mkdir()
    _write_fake_dr1_uv(str(root / "dr1-uv.fits"))
    return str(root)


class TestSelectionFn:
    def test_drops_invalid_mags_and_flags(self):
        t = Table({
            "MAG_U": [15.0, -999.0, 14.0, 13.0],
            "MAG_V": [14.0, 14.5, -999.0, 12.8],
            "FLAG_U": [0, 0, 0, 1],
            "FLAG_V": [0, 0, 0, 0],
        })
        mask = build.selection_fn(t)
        assert mask.tolist() == [True, False, False, False]


class TestReadTable:
    def test_basic_columns(self, fake_raw_root):
        table = build.read_table(fake_raw_root)
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "object_id" in table.schema.names
        assert "MAG_U" in table.schema.names
        assert "MAG_V" in table.schema.names

    def test_object_id_string(self, fake_raw_root):
        table = build.read_table(fake_raw_root)
        assert table.schema.field("object_id").type == pa.string()
        # Should retain the SAGES designation format, not be re-formatted as int.
        assert table.column("object_id")[0].as_py().startswith("SAGE")

    def test_cuts_applied(self, fake_raw_root):
        # Of the 8 fake rows, 2 have MAG_U=-999 and 1 has FLAG_U=1, so 5 survive.
        table = build.read_table(fake_raw_root)
        assert table.num_rows == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build.read_table(str(tmp_path))
