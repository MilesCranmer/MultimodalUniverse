"""Fast unit tests for scripts/galex/build_parent_sample_hats.py."""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "galex", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_galex_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_shard(path: str, n_rows: int = 6) -> None:
    rng = np.random.default_rng(0)
    cols = [
        # Note: GUVCat columns are upper-case in the file; the script lowercases them.
        fits.Column(name="objid", format="K", array=np.arange(2_000_000_000, 2_000_000_000 + n_rows)),
        fits.Column(name="ra", format="D", array=rng.uniform(0, 360, n_rows)),
        fits.Column(name="dec", format="D", array=rng.uniform(-90, 90, n_rows)),
        fits.Column(name="fuv_mag", format="E", array=rng.uniform(15, 22, n_rows).astype(np.float32)),
        fits.Column(name="nuv_mag", format="E", array=rng.uniform(15, 22, n_rows).astype(np.float32)),
        fits.Column(name="band", format="J", array=rng.integers(1, 4, n_rows).astype(np.int32)),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "galex"
    root.mkdir()
    _write_fake_shard(str(root / "GUVCat_AIS_FOV055_glat00_00N__05_00N.fits.gz"), n_rows=6)
    _write_fake_shard(str(root / "GUVCat_AIS_FOV055_glat05_00N__10_00N.fits.gz"), n_rows=4)
    return str(root)


class TestFindRawFiles:
    def test_finds_all_shards(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 2

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadShard:
    def test_basic_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "object_id" in table.schema.names
        assert "fuv_mag" in table.schema.names
        assert "nuv_mag" in table.schema.names
        assert table.num_rows == 6

    def test_object_id_string(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert table.schema.field("object_id").type == pa.string()
        assert table.column("object_id")[0].as_py() == "2000000000"

    def test_lowercased_columns(self, fake_raw_root):
        # GUVCat columns are upper-case in real files; we lowercase them.
        # In our fake fixture they're already lowercase, but the test still
        # confirms the script doesn't break on already-lowercase input.
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        for name in table.schema.names:
            assert name == name.lower(), f"{name} not lowercased"
