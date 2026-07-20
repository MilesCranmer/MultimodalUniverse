"""Tests for scripts/allwise/build_parent_sample_hats.py.

These tests build a tiny synthetic raw-AllWISE-shaped parquet shard, then run
the script's helpers against it. They DO NOT call ``write_hats`` (which is slow
because of hats-import). The end-to-end build is exercised separately as a
slow integration test on real cluster data.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_build_module():
    """Load scripts/allwise/build_parent_sample_hats.py under a unique name."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "allwise", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_allwise_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _make_fake_allwise_shard(path: str, n_rows: int = 50) -> str:
    """Write a tiny parquet file with the columns we care about + some extras."""
    rng = np.random.default_rng(42)
    table = pa.table({
        "designation": pa.array([f"J{i:08d}+000000" for i in range(n_rows)], type=pa.string()),
        "ra": pa.array(rng.uniform(0, 360, n_rows), type=pa.float64()),
        "dec": pa.array(rng.uniform(-90, 90, n_rows), type=pa.float64()),
        "cntr": pa.array(np.arange(1_000_000_000, 1_000_000_000 + n_rows, dtype=np.int64)),
        "w1mpro": pa.array(rng.uniform(8, 18, n_rows), type=pa.float32()),
        "w1snr": pa.array(rng.uniform(0, 100, n_rows), type=pa.float32()),
    })
    pq.write_table(table, path)
    return path


@pytest.fixture
def fake_raw_root(tmp_path):
    raw_root = tmp_path / "allwise"
    shard_dir = raw_root / "healpix_k0=0" / "healpix_k5=0"
    shard_dir.mkdir(parents=True)
    _make_fake_allwise_shard(str(shard_dir / "part0.snappy.parquet"), n_rows=20)

    second_dir = raw_root / "healpix_k0=0" / "healpix_k5=1"
    second_dir.mkdir(parents=True)
    _make_fake_allwise_shard(str(second_dir / "part0.snappy.parquet"), n_rows=15)
    return str(raw_root)


class TestHealpixPrefilter:
    """find_raw_files should restrict returned shards to the set of k5 pixels
    touching the cone, using healpy.query_disc."""

    def _make_dataset(self, tmp_path, k5_pixels):
        """Create empty parquet files at healpix_k5={pix} for each given pixel."""
        raw_root = tmp_path / "allwise"
        for pix in k5_pixels:
            # k0 is nside=1 parent of each k5 pixel. For the test we just pick one.
            d = raw_root / "healpix_k0=6" / f"healpix_k5={pix}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "part0.snappy.parquet").write_bytes(b"")
        return str(raw_root)

    def test_cone_filters_to_cosmos_pixels(self, tmp_path):
        # Create pixel dirs including the two that COSMOS (150, 2) should hit
        # plus some decoys.
        raw_root = self._make_dataset(tmp_path, [6811, 6814, 0, 100, 9999])
        files = build.find_raw_files(
            raw_root, ra_center=150.0, dec_center=2.0, radius=0.5
        )
        pixels = sorted(
            int(build.ALLWISE_PATH_K5_RE.search(f).group(1)) for f in files
        )
        assert pixels == [6811, 6814]

    def test_no_cone_returns_all(self, tmp_path):
        raw_root = self._make_dataset(tmp_path, [100, 200, 6811])
        files = build.find_raw_files(raw_root)
        assert len(files) == 3


class TestConeCutMain:
    """End-to-end sanity check for the cone-cut plumbing in main()."""

    def test_cone_filter_trims_rows(self, tmp_path):
        # Build a tiny raw shard where 3 of 10 objects fall inside the cone.
        raw_root = tmp_path / "allwise"
        shard_dir = raw_root / "healpix_k0=0" / "healpix_k5=0"
        shard_dir.mkdir(parents=True)
        ra = np.array([150.0, 150.05, 149.95, 200.0, 0.0, 30.0, 60.0, 90.0, 120.0, 180.0])
        dec = np.array([2.0, 2.05, 1.95, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cntr = np.arange(1_000, 1_010, dtype=np.int64)
        tbl = pa.table({
            "ra": pa.array(ra),
            "dec": pa.array(dec),
            "cntr": pa.array(cntr),
            "w1mpro": pa.array(np.zeros(10, dtype=np.float32)),
        })
        shard_path = shard_dir / "part0.snappy.parquet"
        pq.write_table(tbl, str(shard_path))

        # Read via read_shard (which the main loop calls) and then apply
        # the same cone filter the main loop applies.
        from mmu.cone import apply_cone_filter
        table = build.read_shard(str(shard_path))
        mask = apply_cone_filter(
            table.column("ra").to_numpy(),
            table.column("dec").to_numpy(),
            ra_center=150.0, dec_center=2.0, radius=0.5,
        )
        filtered = table.filter(pa.array(mask))
        assert filtered.num_rows == 3
        # The surviving ra values should all be within 0.5° of 150°.
        assert all(abs(x - 150.0) <= 0.5 for x in filtered.column("ra").to_pylist())


class TestFindRawFiles:
    def test_finds_all_shards(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 2
        assert all(f.endswith(".parquet") for f in files)

    def test_max_files(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root, max_files=1)
        assert len(files) == 1

    def test_missing_root(self, tmp_path):
        files = build.find_raw_files(str(tmp_path / "does_not_exist"))
        assert files == []


class TestReadShard:
    def test_basic_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "object_id" in table.schema.names
        assert table.num_rows == 20

    def test_object_id_derived_from_cntr(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        obj_ids = table.column("object_id").to_pylist()
        assert obj_ids[0] == "1000000000"
        assert table.schema.field("object_id").type == pa.string()

    def test_other_columns_preserved(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert "w1mpro" in table.schema.names
        assert "designation" in table.schema.names

    def test_missing_radec_raises(self, tmp_path):
        bad = tmp_path / "bad.parquet"
        pq.write_table(pa.table({"x": pa.array([1, 2, 3])}), bad)
        with pytest.raises(ValueError, match="ra/dec"):
            build.read_shard(str(bad))

    def test_missing_object_id_and_cntr_raises(self, tmp_path):
        bad = tmp_path / "bad.parquet"
        pq.write_table(
            pa.table({
                "ra": pa.array([1.0, 2.0]),
                "dec": pa.array([3.0, 4.0]),
            }),
            bad,
        )
        with pytest.raises(ValueError, match="object_id.*cntr"):
            build.read_shard(str(bad))

    def test_existing_object_id_passthrough(self, tmp_path):
        path = tmp_path / "with_oid.parquet"
        pq.write_table(
            pa.table({
                "ra": pa.array([1.0, 2.0]),
                "dec": pa.array([3.0, 4.0]),
                "object_id": pa.array(["a", "b"], type=pa.string()),
            }),
            path,
        )
        table = build.read_shard(str(path))
        assert table.column("object_id").to_pylist() == ["a", "b"]
