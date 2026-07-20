"""Unit tests for scripts/btsbot/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "btsbot", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_btsbot_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()

N_ROWS = 8


def _make_fake_meta(n_rows: int, split: str, rng: np.random.Generator) -> pd.DataFrame:
    """Return a fake CSV DataFrame matching the BTSbot metadata layout."""
    df = pd.DataFrame({
        "candid": rng.integers(1_000_000, 9_999_999, n_rows),
        "objectId": [f"ZTF{i:09d}" for i in range(n_rows)],
        "ra": rng.uniform(0, 360, n_rows),
        "dec": rng.uniform(-30, 30, n_rows),
        "fid": rng.choice([1, 2], n_rows),
        "label": rng.integers(0, 2, n_rows),
        "programid": np.ones(n_rows, dtype=int),
        "field": rng.integers(200, 900, n_rows),
        "nneg": rng.integers(0, 5, n_rows),
        "nbad": rng.integers(0, 2, n_rows),
        "ndethist": rng.integers(1, 20, n_rows),
        "ncovhist": rng.integers(1, 50, n_rows),
        "nmtchps": rng.integers(0, 5, n_rows),
        "nnotdet": rng.integers(0, 5, n_rows),
        "N": rng.integers(1, 100, n_rows),
        "isdiffpos": rng.choice([True, False], n_rows),
        "is_SN": rng.choice([True, False], n_rows),
        "near_threshold": rng.choice([True, False], n_rows),
        "is_rise": rng.choice([True, False], n_rows),
        "source_set": [split] * n_rows,
        # A subset of float features
        "jd": rng.uniform(2_458_000, 2_459_000, n_rows),
        "magpsf": rng.uniform(17, 21, n_rows),
        "sigmapsf": rng.uniform(0.01, 0.1, n_rows),
        "drb": rng.uniform(0, 1, n_rows),
        "acai_h": rng.uniform(0, 1, n_rows),
        "age": rng.uniform(0, 100, n_rows),
    })
    return df


def _write_fake_btsbot(raw_root: str, n_rows: int = N_ROWS) -> None:
    """Write minimal fake BTSbot .npy + .csv files."""
    rng = np.random.default_rng(7)
    os.makedirs(raw_root, exist_ok=True)
    for idx, split in enumerate(build._SPLITS):
        meta = _make_fake_meta(n_rows, split, rng)
        meta.to_csv(os.path.join(raw_root, build._META_FILES[idx]), index=False)
        images = rng.uniform(0, 1, (n_rows, build.IMAGE_SIZE, build.IMAGE_SIZE, 3)).astype(np.float32)
        np.save(os.path.join(raw_root, build._IMG_FILES[idx]), images)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = str(tmp_path / "btsbot")
    _write_fake_btsbot(root, n_rows=N_ROWS)
    return root


class TestConstants:
    def test_image_size(self):
        assert build.IMAGE_SIZE == 63

    def test_pixel_scale(self):
        assert build.PIXEL_SCALE == 1.01

    def test_views(self):
        assert build.VIEWS == ["science", "reference", "difference"]

    def test_splits(self):
        assert build._SPLITS == ["train", "val", "test"]


class TestReadSplit:
    def test_row_count(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        assert table.num_rows == N_ROWS

    def test_required_columns(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "split", "image"} <= names

    def test_split_label(self, fake_raw_root):
        for i, split_name in enumerate(build._SPLITS):
            table = build.read_split(fake_raw_root, i)
            assert table.column("split")[0].as_py() == split_name

    def test_object_id_is_int64(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        assert table.schema.field("object_id").type == pa.int64()

    def test_image_struct_fields(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        img_type = table.schema.field("image").type
        assert pa.types.is_struct(img_type)
        field_names = {f.name for f in img_type}
        assert field_names == {"band", "view", "array", "scale"}

    def test_image_array_type(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["array"] == pa.large_list(pa.large_list(pa.large_list(pa.float32())))

    def test_image_band_is_single_entry(self, fake_raw_root):
        """Each ZTF alert has a single band (g or r) in the band list."""
        table = build.read_split(fake_raw_root, 0)
        first = table.column("image")[0].as_py()
        assert len(first["band"]) == 1
        assert first["band"][0] in ("g", "r")

    def test_image_view_list(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        first = table.column("image")[0].as_py()
        assert first["view"] == build.VIEWS

    def test_image_array_shape(self, fake_raw_root):
        """array should decode to (3, 63, 63) — 3 views × H × W."""
        table = build.read_split(fake_raw_root, 0)
        first = table.column("image")[0].as_py()
        arr = np.asarray(first["array"], dtype=np.float32)
        assert arr.shape == (3, build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_image_scale_length(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        first = table.column("image")[0].as_py()
        assert len(first["scale"]) == 3
        assert all(abs(s - build.PIXEL_SCALE) < 1e-5 for s in first["scale"])

    def test_float_features_present(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0)
        names = set(table.schema.names)
        for f in ["jd", "magpsf", "drb"]:
            assert f in names

    def test_max_rows(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0, max_rows=3)
        assert table.num_rows == 3

    def test_cone_all_inside(self, fake_raw_root):
        table_all = build.read_split(fake_raw_root, 0)
        table_cone = build.read_split(fake_raw_root, 0, ra_center=0.0, dec_center=0.0, radius=180.0)
        assert table_cone.num_rows == table_all.num_rows

    def test_cone_all_outside(self, fake_raw_root):
        table = build.read_split(fake_raw_root, 0, ra_center=42.0, dec_center=85.0, radius=0.0001)
        assert table is None or table.num_rows < N_ROWS


class TestReadAllSplits:
    def test_returns_three_tables(self, fake_raw_root):
        tables = build.read_all_splits(fake_raw_root)
        assert len(tables) == 3

    def test_total_rows(self, fake_raw_root):
        tables = build.read_all_splits(fake_raw_root)
        total = sum(t.num_rows for t in tables)
        assert total == N_ROWS * 3
