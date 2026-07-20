"""Fast unit tests for scripts/gaia/build_parent_sample_hats.py (full DR3).

This script reads only GaiaSource_*.hdf5 shards (no XP join) and emits a
HATS catalog with ~1.8B rows at full scale. Tests fabricate tiny synthetic
shards and exercise the per-shard helpers. No write_hats call.
"""

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "gaia", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_gaia_full_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_source(path: str, source_ids: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> None:
    n = len(source_ids)
    rng = np.random.default_rng(int(source_ids[0]) if n else 0)
    with h5py.File(path, "w") as f:
        f.create_dataset("source_id", data=source_ids.astype(np.int64))
        f.create_dataset("ra", data=ra.astype(np.float64))
        f.create_dataset("dec", data=dec.astype(np.float64))
        for col in (
            build.PHOTOMETRY_FEATURES
            + [c for c in build.ASTROMETRY_FEATURES if c not in ("ra", "dec")]
            + build.RV_FEATURES
            + build.GSPPHOT_FEATURES
            + build.FLAG_FEATURES
            + build.CORRECTION_FEATURES
        ):
            if col in f:
                continue
            f.create_dataset(col, data=rng.normal(size=n).astype(np.float32))


def test_find_source_shards(tmp_path):
    p1 = tmp_path / "GaiaSource_000000-003111.hdf5"
    p2 = tmp_path / "GaiaSource_003112-005263.hdf5"
    other = tmp_path / "AstrophysicalParameters_000000-003111.hdf5"
    for p in (p1, p2, other):
        p.touch()
    found = build.find_source_shards(str(tmp_path))
    assert len(found) == 2
    assert all("GaiaSource_" in f for f in found)


def test_read_source_columns_missing_col_nan(tmp_path):
    path = tmp_path / "GaiaSource_000000-000010.hdf5"
    _write_fake_source(path, np.arange(5, dtype=np.int64), np.linspace(150, 151, 5), np.linspace(2, 3, 5))
    # Add a column that does NOT exist and verify it's NaN-filled.
    cols = build._read_source_columns(str(path), ["source_id", "ra", "dec", "bogus_col"])
    assert cols["source_id"].shape == (5,)
    assert np.isnan(cols["bogus_col"]).all()


def test_process_shard_no_cone(tmp_path):
    path = tmp_path / "GaiaSource_000000-000010.hdf5"
    _write_fake_source(path, np.arange(10, dtype=np.int64),
                       np.linspace(0, 360, 10), np.linspace(-90, 90, 10))
    table = build.process_shard(str(path))
    assert isinstance(table, pa.Table)
    assert table.num_rows == 10
    assert set(table.schema.names) >= {
        "object_id", "ra", "dec", "photometry", "astrometry",
        "radial_velocity", "gspphot", "flags", "corrections",
    }


def test_process_shard_cone_keeps_only_inside(tmp_path):
    path = tmp_path / "GaiaSource_000000-000010.hdf5"
    # 10 sources, half near COSMOS (150, 2), half at the south pole.
    ra = np.array([150.0, 150.1, 150.2, 149.9, 150.05,
                   0.0, 10.0, 50.0, 100.0, 200.0])
    dec = np.array([2.0, 2.1, 1.9, 2.05, 2.02,
                    -89.0, -88.0, -85.0, -80.0, -70.0])
    _write_fake_source(path, np.arange(10, dtype=np.int64), ra, dec)
    table = build.process_shard(str(path), ra_center=150.0, dec_center=2.0, radius=0.5)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 5
    # All surviving rows should be near COSMOS.
    assert all(abs(float(r) - 150.0) < 0.5 for r in table.column("ra").to_pylist())


def test_process_shard_cone_empty_returns_none(tmp_path):
    path = tmp_path / "GaiaSource_000000-000010.hdf5"
    _write_fake_source(path, np.arange(5, dtype=np.int64),
                       np.full(5, 0.0), np.full(5, -89.0))
    table = build.process_shard(str(path), ra_center=150.0, dec_center=2.0, radius=0.5)
    assert table is None


def test_build_table_struct_fields(tmp_path):
    path = tmp_path / "GaiaSource_000000-000010.hdf5"
    _write_fake_source(path, np.arange(3, dtype=np.int64),
                       np.array([150.0, 150.1, 150.2]), np.array([2.0, 2.05, 2.1]))
    cols = build._read_source_columns(str(path), build.ALL_SOURCE_COLUMNS)
    table = build.build_table(cols)
    for struct_name, features in [
        ("photometry", build.PHOTOMETRY_FEATURES),
        ("astrometry", build.ASTROMETRY_FEATURES),
        ("radial_velocity", build.RV_FEATURES),
        ("gspphot", build.GSPPHOT_FEATURES),
        ("flags", build.FLAG_FEATURES),
        ("corrections", build.CORRECTION_FEATURES),
    ]:
        struct_type = table.schema.field(struct_name).type
        inner = {struct_type.field(i).name for i in range(struct_type.num_fields)}
        assert inner == set(features), f"{struct_name}: {inner} != {set(features)}"
