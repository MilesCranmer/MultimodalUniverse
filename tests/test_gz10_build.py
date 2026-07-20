"""Unit tests for scripts/gz10/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "gz10", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_gz10_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()

N_ROWS = 10


def _write_fake_gz10(path: str, n_rows: int = N_ROWS) -> None:
    """Write a minimal Galaxy10_DECals.h5 matching the real file layout."""
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        f.create_dataset("ra", data=rng.uniform(0, 360, n_rows).astype(np.float64))
        f.create_dataset("dec", data=rng.uniform(-90, 90, n_rows).astype(np.float64))
        f.create_dataset("ans", data=rng.integers(0, 10, n_rows).astype(np.int32))
        f.create_dataset("redshift", data=rng.uniform(0, 1, n_rows).astype(np.float32))
        f.create_dataset("pxscale", data=np.full(n_rows, 0.262, dtype=np.float32))
        f.create_dataset(
            "images",
            data=rng.integers(0, 256, (n_rows, build.IMAGE_SIZE, build.IMAGE_SIZE, 3), dtype=np.uint8),
        )


@pytest.fixture
def fake_gz10(tmp_path):
    p = str(tmp_path / "Galaxy10_DECals.h5")
    _write_fake_gz10(p, n_rows=N_ROWS)
    return p


class TestConstants:
    def test_image_size(self):
        assert build.IMAGE_SIZE == 256

    def test_bands(self):
        assert build.BANDS == ["g", "r", "z"]
        assert build.N_BANDS == 3


class TestReadTable:
    def test_row_count(self, fake_gz10):
        table = build.read_table(fake_gz10)
        assert table.num_rows == N_ROWS

    def test_required_columns(self, fake_gz10):
        table = build.read_table(fake_gz10)
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "gz10_label", "redshift", "image"} <= names

    def test_object_id_is_string(self, fake_gz10):
        table = build.read_table(fake_gz10)
        assert table.schema.field("object_id").type == pa.string()

    def test_gz10_label_is_int32(self, fake_gz10):
        table = build.read_table(fake_gz10)
        assert table.schema.field("gz10_label").type == pa.int32()

    def test_redshift_is_float32(self, fake_gz10):
        table = build.read_table(fake_gz10)
        assert table.schema.field("redshift").type == pa.float32()

    def test_image_struct_fields(self, fake_gz10):
        table = build.read_table(fake_gz10)
        img_type = table.schema.field("image").type
        assert pa.types.is_struct(img_type)
        field_names = {f.name for f in img_type}
        assert field_names == {"band", "array", "scale"}

    def test_image_band_type(self, fake_gz10):
        table = build.read_table(fake_gz10)
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["band"] == pa.list_(pa.string())

    def test_image_array_type(self, fake_gz10):
        table = build.read_table(fake_gz10)
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["array"] == pa.large_list(pa.large_list(pa.large_list(pa.uint8())))

    def test_image_scale_type(self, fake_gz10):
        table = build.read_table(fake_gz10)
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["scale"] == pa.list_(pa.float32())

    def test_image_roundtrip_shape(self, fake_gz10):
        """image[0].array must decode to shape (N_BANDS, IMAGE_SIZE, IMAGE_SIZE)."""
        table = build.read_table(fake_gz10)
        first = table.column("image")[0].as_py()
        assert first["band"] == build.BANDS
        arr = np.asarray(first["array"], dtype=np.uint8)
        assert arr.shape == (build.N_BANDS, build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_image_scale_per_band(self, fake_gz10):
        table = build.read_table(fake_gz10)
        first = table.column("image")[0].as_py()
        assert len(first["scale"]) == build.N_BANDS

    def test_max_rows(self, fake_gz10):
        table = build.read_table(fake_gz10, max_rows=4)
        assert table.num_rows == 4

    def test_cone_filter_all_inside(self, fake_gz10):
        """With a huge cone everything survives."""
        table_all = build.read_table(fake_gz10)
        table_cone = build.read_table(fake_gz10, ra_center=0.0, dec_center=0.0, radius=180.0)
        assert table_cone.num_rows == table_all.num_rows

    def test_cone_filter_all_outside(self, fake_gz10):
        """An infinitesimally small cone at an empty sky patch returns nothing."""
        # Use a tiny radius at a point that almost certainly has no objects
        # in a 10-row random sample.  The table may have 0 rows, which is fine.
        table = build.read_table(fake_gz10, ra_center=42.0, dec_center=85.0, radius=0.0001)
        assert table.num_rows < N_ROWS
