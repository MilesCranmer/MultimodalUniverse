"""Fast unit tests for scripts/ssl_legacysurvey/build_parent_sample_hats.py."""

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "ssl_legacysurvey", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_ssl_legacysurvey_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_chunk(path: str, n_rows: int = 8) -> None:
    """Write a tiny images_npix152_*.h5 chunk matching the real Stein-et-al layout."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("inds", data=np.arange(n_rows, dtype=np.int64))
        f.create_dataset("ra", data=rng.uniform(0, 360, n_rows).astype(np.float64))
        f.create_dataset("dec", data=rng.uniform(-90, 90, n_rows).astype(np.float64))
        f.create_dataset("ebv", data=rng.uniform(0, 0.1, n_rows).astype(np.float32))
        f.create_dataset("z_spec", data=rng.uniform(0, 1, n_rows).astype(np.float32))
        f.create_dataset("flux", data=rng.uniform(0, 100, (n_rows, 3)).astype(np.float32))
        f.create_dataset("fiberflux", data=rng.uniform(0, 100, (n_rows, 3)).astype(np.float32))
        f.create_dataset("psfdepth", data=rng.uniform(20, 25, (n_rows, 3)).astype(np.float32))
        f.create_dataset("psfsize", data=rng.uniform(1, 2, (n_rows, 3)).astype(np.float32))
        # (n_rows, 3, 152, 152) is the real shape but we use 16x16 for speed
        f.create_dataset(
            "images",
            data=rng.normal(0, 1, (n_rows, 3, build.IMAGE_SIZE, build.IMAGE_SIZE)).astype(np.float32),
        )


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "ssl"
    root.mkdir()
    _write_fake_chunk(str(root / "images_npix152_000000000_001000000.h5"), n_rows=5)
    _write_fake_chunk(str(root / "images_npix152_001000000_002000000.h5"), n_rows=3)
    return str(root)


class TestFindRawFiles:
    def test_finds_all(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root)) == 2

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadChunk:
    def test_basic_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        names = table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "image" in names
        assert "ebv" in names
        assert "z_spec" in names
        # per-band flux unrolled
        assert "flux_g" in names
        assert "flux_r" in names
        assert "flux_z" in names
        assert "fiberflux_g" in names
        assert "psfdepth_z" in names

    def test_row_count(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        assert table.num_rows == 5

    def test_object_id_is_string(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        assert table.schema.field("object_id").type == pa.string()

    def test_image_is_struct_of_parallel_lists(self, fake_raw_root):
        """Image column must be a struct<band, flux, psf_fwhm, scale>
        matching Mike's v1 SSL LegacySurvey transformer schema, with plain
        nested lists (no extension type) to avoid the nested_pandas crashes
        in hats-import's finishing stage."""
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        image_type = table.schema.field("image").type
        assert pa.types.is_struct(image_type)
        fields = {f.name: f.type for f in image_type}
        assert set(fields.keys()) == {"band", "flux", "psf_fwhm", "scale"}
        assert fields["band"] == pa.list_(pa.string())
        assert fields["flux"] == pa.large_list(pa.large_list(pa.large_list(pa.float32())))
        assert fields["psf_fwhm"] == pa.list_(pa.float32())
        assert fields["scale"] == pa.list_(pa.float32())

    def test_image_roundtrip_shape(self, fake_raw_root):
        """Reading image[0].flux back via as_py() and wrapping in np.asarray
        should give (N_BANDS, IMAGE_SIZE, IMAGE_SIZE)."""
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        first = table.column("image")[0].as_py()
        assert first["band"] == build.BANDS
        assert len(first["psf_fwhm"]) == build.N_BANDS
        assert len(first["scale"]) == build.N_BANDS
        arr = np.asarray(first["flux"], dtype=np.float32)
        assert arr.shape == (build.N_BANDS, build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_max_rows_per_file(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0], max_rows=2)
        assert table.num_rows == 2
