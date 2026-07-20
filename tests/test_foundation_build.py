"""Unit tests for mmu.sn_ia_snana (shared SN-Ia SNANA logic) via the
foundation wrapper. The four other SN-Ia datasets (snls, ps1_sne_ia,
des_y3_sne_ia, swift_sne_ia) all call the same :func:`build_main`, so
exhaustive per-dataset tests aren't needed — a smoke test per wrapper
is enough and lives in tests/test_sn_ia_wrappers.py.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from mmu import sn_ia_snana as bps


SNANA_SAMPLE = """SURVEY: FOUNDATION
SNTYPE: 1
SNID: TEST01
RA: 150.1000000
DECL: 2.0500000
MWEBV: 0.03 +- 0.002
FILTERS: griz
REDSHIFT_HELIO: 0.05 +- 0.0001
REDSHIFT_CMB: 0.051 +- 0.0001
REDSHIFT_FINAL: 0.052 +- 0.0001
HOSTGAL_LOGMASS: 10.0 +- 0.1

NOBS: 4
NVAR: 5
VARLIST: MJD FLT FIELD FLUXCAL FLUXCALERR
OBS: 57000.0 g NULL 100.0 5.0
OBS: 57001.0 g NULL 110.0 5.5
OBS: 57000.0 r NULL 200.0 7.0
OBS: 57002.0 i NULL 300.0 8.0
"""


@pytest.fixture
def sample_dir():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "Foundation_DR1_TEST01.txt")
        with open(path, "w") as f:
            f.write(SNANA_SAMPLE)
        path2 = os.path.join(td, "Foundation_DR1_TEST02.txt")
        with open(path2, "w") as f:
            f.write(SNANA_SAMPLE.replace("TEST01", "TEST02")
                    .replace("150.1000000", "10.0")
                    .replace("2.0500000", "-30.0"))
        yield td


def test_read_snana_file_basic(sample_dir):
    path = os.path.join(sample_dir, "Foundation_DR1_TEST01.txt")
    r = bps.read_snana_file(path)
    assert r is not None
    assert r["object_id"] == "TEST01"
    assert r["obj_type"] == "Ia"
    assert r["ra"] == pytest.approx(150.1)
    assert r["dec"] == pytest.approx(2.05)
    # REDSHIFT_FINAL has priority
    assert r["redshift"] == pytest.approx(0.052)
    assert r["host_log_mass"] == pytest.approx(10.0)
    assert len(r["MJD"]) == 4
    assert set(r["FLT"]) == {"g", "r", "i"}


def test_read_snana_file_with_prefix(sample_dir):
    path = os.path.join(sample_dir, "Foundation_DR1_TEST01.txt")
    r = bps.read_snana_file(path, object_id_prefix="PS1_")
    assert r["object_id"] == "PS1_TEST01"


def test_stack_pads_to_max_length(sample_dir):
    path = os.path.join(sample_dir, "Foundation_DR1_TEST01.txt")
    r = bps.read_snana_file(path)
    all_bands = np.array(["g", "i", "r", "z"])
    stacked = bps.stack_per_object([r], all_bands)
    s = stacked[0]
    # max_length = 2 (g-band has the most obs), n_bands = 4 → flattened = 8
    assert len(s["band"]) == 8
    assert list(s["band"]) == ["g", "g", "i", "i", "r", "r", "z", "z"]
    g_mask = s["band"] == "g"
    assert sorted(s["time"][g_mask].tolist()) == [57000.0, 57001.0]
    z_mask = s["band"] == "z"
    assert (s["time"][z_mask] == -99.0).all()
    assert (s["flux"][z_mask] == 0.0).all()
    i_mask = s["band"] == "i"
    assert 57002.0 in s["time"][i_mask].tolist()
    assert -99.0 in s["time"][i_mask].tolist()


def test_build_table_schema(sample_dir):
    rows = [
        bps.read_snana_file(os.path.join(sample_dir, f))
        for f in sorted(os.listdir(sample_dir))
    ]
    all_bands = np.array(sorted(set().union(*[set(r["FLT"]) for r in rows])))
    stacked = bps.stack_per_object(rows, all_bands)
    tbl = bps.build_table(stacked)
    assert tbl.num_rows == 2
    names = tbl.schema.names
    for col in ("ra", "dec", "object_id", "obj_type", "redshift",
                "host_log_mass", "lightcurve"):
        assert col in names
    lc_type = tbl.schema.field("lightcurve").type
    lc_fields = {lc_type.field(i).name for i in range(lc_type.num_fields)}
    assert lc_fields == {"band", "time", "flux", "flux_err"}


def test_build_main_writes_hats(sample_dir):
    with tempfile.TemporaryDirectory() as out:
        rc = bps.build_main(
            catalog_name="foundation",
            raw_subdir=sample_dir,
            filename_glob="Foundation_DR1_*.txt",
            argv=["--output-root", out],
        )
        assert rc == 0
        catalog_dir = os.path.join(out, "foundation", "foundation")
        assert os.path.isfile(os.path.join(catalog_dir, "hats.properties"))
        ds_dir = os.path.join(catalog_dir, "dataset")
        parts = []
        for root, _, names in os.walk(ds_dir):
            for n in names:
                if n.endswith(".parquet") and not n.startswith("_"):
                    parts.append(os.path.join(root, n))
        assert parts
        tbl = pq.read_table(parts[0])
        assert "lightcurve" in tbl.schema.names
        assert tbl.num_rows >= 1


def test_build_main_cone_cut(sample_dir):
    with tempfile.TemporaryDirectory() as out:
        rc = bps.build_main(
            catalog_name="foundation",
            raw_subdir=sample_dir,
            filename_glob="Foundation_DR1_*.txt",
            argv=[
                "--output-root", out,
                "--ra-center", "150.0",
                "--dec-center", "2.0",
                "--radius", "0.5",
            ],
        )
        assert rc == 0
        catalog_dir = os.path.join(out, "foundation", "foundation")
        ds_dir = os.path.join(catalog_dir, "dataset")
        parts = []
        for root, _, names in os.walk(ds_dir):
            for n in names:
                if n.endswith(".parquet") and not n.startswith("_"):
                    parts.append(os.path.join(root, n))
        tbl = pq.read_table(parts[0])
        ids = tbl.column("object_id").to_pylist()
        assert "TEST01" in ids
        assert "TEST02" not in ids
