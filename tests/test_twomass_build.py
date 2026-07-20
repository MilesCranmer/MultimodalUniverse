"""Fast unit tests for scripts/twomass/build_parent_sample_hats.py.

Builds a tiny synthetic gzipped pipe-delimited file with the columns the script
expects, then exercises the parsing and post-processing helpers. No real 2MASS
data and no `write_hats` call.
"""

import gzip
import importlib.util
import os

import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "twomass", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_twomass_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_shard(path: str, n_rows: int = 5) -> None:
    """Write a tiny gzipped pipe-delimited shard.

    Real 2MASS PSC files have N columns (60) without a separate `dec` field —
    `dec` is a v1 alias for `decl` added at parse time. The fake shard mirrors
    the real layout: it does NOT include the `dec` slot.
    """
    cols = [c for c in build.COLUMN_MAPPING if c != "dec"]
    n_cols = len(cols)
    idx = {name: cols.index(name) for name in ("ra", "decl", "pts_key", "j_m", "h_m", "k_m")}

    rows: list[str] = []
    for i in range(n_rows):
        fields = ["\\N"] * n_cols
        fields[idx["ra"]] = f"{180.0 + i * 0.1}"
        fields[idx["decl"]] = f"{20.0 + i * 0.1}"
        fields[idx["pts_key"]] = f"{1_000_000_000 + i}"
        fields[idx["j_m"]] = f"{14.5 + i * 0.1}"
        fields[idx["h_m"]] = f"{13.5 + i * 0.1}"
        fields[idx["k_m"]] = f"{12.5 + i * 0.1}"
        rows.append("|".join(fields))

    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "twomass_psc"
    root.mkdir()
    _write_fake_shard(str(root / "psc_aaa.gz"), n_rows=5)
    _write_fake_shard(str(root / "psc_aab.gz"), n_rows=3)
    return str(root)


class TestFindRawFiles:
    def test_finds_all_shards(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 2
        assert all(f.endswith(".gz") for f in files)

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadShard:
    def test_columns_present(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names  # aliased from decl
        assert "decl" in table.schema.names  # original kept too
        assert "object_id" in table.schema.names
        assert table.num_rows == 5

    def test_dec_aliased_from_decl(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        assert table.column("dec").to_pylist() == table.column("decl").to_pylist()

    def test_object_id_is_string_from_pts_key(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        ids = table.column("object_id").to_pylist()
        assert ids[0] == "1000000000"
        assert table.schema.field("object_id").type == pa.string()

    def test_photometry_preserved(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_shard(files[0])
        j = table.column("j_m").to_pylist()
        assert j[0] == pytest.approx(14.5)
