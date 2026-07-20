"""Fast unit tests for scripts/gaia_xp/build_parent_sample_hats.py.

These tests fabricate tiny synthetic GaiaSource + XpContinuousMeanSpectrum
HDF5 shards that match the real Gaia Archive bulk schema, then run the
script's helpers against them. They don't exercise write_hats (that's
covered by cluster integration tests).
"""

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "gaia_xp", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_gaia_xp_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_gaia_source(path: str, source_ids: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> None:
    """Write a tiny GaiaSource_*.hdf5 matching the real Gaia Archive bulk schema."""
    n = len(source_ids)
    rng = np.random.default_rng(int(source_ids[0]) if n else 0)
    with h5py.File(path, "w") as f:
        f.create_dataset("source_id", data=source_ids.astype(np.int64))
        f.create_dataset("ra", data=ra.astype(np.float64))
        f.create_dataset("dec", data=dec.astype(np.float64))
        # All photometry/astrometry/rv/gspphot/flags/corrections fields.
        # rv_template_teff lives in both RV and corrections in v1's schema;
        # dedupe when writing so we don't try to create the same dataset twice.
        all_fields = (
            build.PHOTOMETRY_FEATURES + build.ASTROMETRY_FEATURES
            + build.RV_FEATURES + build.GSPPHOT_FEATURES
            + build.FLAG_FEATURES + build.CORRECTION_FEATURES
        )
        for c in dict.fromkeys(all_fields):
            if c in ("ra", "dec"):
                continue  # already set above
            f.create_dataset(c, data=rng.normal(0, 1, n).astype(np.float32))


def _write_fake_xp(
    path: str,
    source_ids: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> None:
    """Write a tiny XpContinuousMeanSpectrum_*.hdf5 with 55-coefficient arrays.

    The ``ra``/``dec`` arrays must match the GaiaSource ra/dec for the same
    source_ids. In the real Gaia Archive bulk files, XP contains the same
    astrometric coordinates as GaiaSource for its subset of sources.
    """
    n = len(source_ids)
    rng = np.random.default_rng(int(source_ids[0]) if n else 0)
    with h5py.File(path, "w") as f:
        f.create_dataset("source_id", data=source_ids.astype(np.int64))
        f.create_dataset("ra", data=ra.astype(np.float64))
        f.create_dataset("dec", data=dec.astype(np.float64))
        f.create_dataset("bp_coefficients", data=rng.normal(0, 1, (n, 55)).astype(np.float32))
        f.create_dataset("rp_coefficients", data=rng.normal(0, 1, (n, 55)).astype(np.float32))
        f.create_dataset("bp_coefficient_errors", data=rng.uniform(0.01, 0.1, (n, 55)).astype(np.float32))
        f.create_dataset("rp_coefficient_errors", data=rng.uniform(0.01, 0.1, (n, 55)).astype(np.float32))


@pytest.fixture
def fake_gaia_shards(tmp_path):
    """Create two matched (source, xp) shard pairs where XP is a subset of source."""
    raw = tmp_path / "Gaia"
    raw.mkdir()

    # Shard 1: 10 source_ids, 4 of which have XP spectra.
    # Put 3 sources inside the COSMOS cone (ra=150, dec=2).
    src_ids_1 = np.arange(1000, 1010, dtype=np.int64)
    ra_1 = np.array([150.0, 150.1, 150.2, 180.0, 200.0, 50.0, 25.0, 300.0, 150.05, 30.0])
    dec_1 = np.array([2.0, 2.05, 1.95, 30.0, -20.0, 45.0, 60.0, -45.0, 2.1, 15.0])
    _write_fake_gaia_source(
        str(raw / "GaiaSource_000000-003111.hdf5"),
        src_ids_1, ra_1, dec_1,
    )
    # XP has source_ids for index 0, 2, 5, 8 (4 sources; 3 in cone)
    xp_idx_1 = [0, 2, 5, 8]
    xp_ids_1 = src_ids_1[xp_idx_1]
    xp_ra_1 = ra_1[xp_idx_1]
    xp_dec_1 = dec_1[xp_idx_1]
    _write_fake_xp(
        str(raw / "XpContinuousMeanSpectrum_000000-003111.hdf5"),
        xp_ids_1, xp_ra_1, xp_dec_1,
    )

    # Shard 2: 5 source_ids, 2 of which have XP. None in the cone.
    src_ids_2 = np.arange(2000, 2005, dtype=np.int64)
    ra_2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    dec_2 = np.array([-60.0, 70.0, 80.0, -80.0, 65.0])
    _write_fake_gaia_source(
        str(raw / "GaiaSource_003112-005263.hdf5"),
        src_ids_2, ra_2, dec_2,
    )
    xp_idx_2 = [1, 3]
    xp_ids_2 = src_ids_2[xp_idx_2]
    xp_ra_2 = ra_2[xp_idx_2]
    xp_dec_2 = dec_2[xp_idx_2]
    _write_fake_xp(
        str(raw / "XpContinuousMeanSpectrum_003112-005263.hdf5"),
        xp_ids_2, xp_ra_2, xp_dec_2,
    )

    return str(raw)


class TestFindShardPairs:
    def test_pairs_by_suffix(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        assert len(pairs) == 2
        # Each pair should have matching suffix.
        for src, xp in pairs:
            assert build.PART_RE.search(src).group(0) == build.PART_RE.search(xp).group(0)

    def test_ignores_unpaired(self, tmp_path):
        raw = tmp_path / "Gaia"
        raw.mkdir()
        # Only a GaiaSource file, no matching XP.
        _write_fake_gaia_source(
            str(raw / "GaiaSource_000000-001000.hdf5"),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        pairs = build.find_shard_pairs(str(raw))
        assert pairs == []

    def test_empty_root(self, tmp_path):
        assert build.find_shard_pairs(str(tmp_path)) == []


class TestProcessShard:
    def test_joins_xp_to_source(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        source_path, xp_path = pairs[0]
        table = build.process_shard(source_path, xp_path)
        # Shard 1 has 4 XP sources; all 4 should match a GaiaSource row.
        assert table.num_rows == 4

    def test_schema_has_all_struct_fields(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        table = build.process_shard(*pairs[0])
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "spectral_coefficients",
                "photometry", "astrometry", "radial_velocity",
                "gspphot", "flags", "corrections"} <= names

    def test_spectral_coefficients_struct(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        table = build.process_shard(*pairs[0])
        spec_type = table.schema.field("spectral_coefficients").type
        assert pa.types.is_struct(spec_type)
        fields = {f.name: f.type for f in spec_type}
        assert fields["coeff"] == pa.list_(pa.float32())
        assert fields["coeff_error"] == pa.list_(pa.float32())
        first = table.column("spectral_coefficients")[0].as_py()
        assert len(first["coeff"]) == 110
        assert len(first["coeff_error"]) == 110

    def test_photometry_struct_fields(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        table = build.process_shard(*pairs[0])
        phot_type = table.schema.field("photometry").type
        assert pa.types.is_struct(phot_type)
        field_names = {f.name for f in phot_type}
        assert field_names == set(build.PHOTOMETRY_FEATURES)

    def test_cone_cut_trims(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        # Shard 1: XP sources at indices 0, 2, 5, 8. ra/dec at these:
        #   0: (150.0, 2.0)      → in cone
        #   2: (150.2, 1.95)     → in cone (~0.21°)
        #   5: (50.0, 45.0)      → out
        #   8: (150.05, 2.1)     → in cone
        # So cone cut leaves 3 of 4.
        table = build.process_shard(
            *pairs[0],
            ra_center=150.0, dec_center=2.0, radius=0.5,
        )
        assert table.num_rows == 3

    def test_cone_cut_no_survivors(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        # Shard 2 has no sources in the cone.
        table = build.process_shard(
            *pairs[1],
            ra_center=150.0, dec_center=2.0, radius=0.5,
        )
        assert table is None

    def test_object_id_is_int64_source_id(self, fake_gaia_shards):
        pairs = build.find_shard_pairs(fake_gaia_shards)
        table = build.process_shard(*pairs[0])
        assert table.schema.field("object_id").type == pa.int64()
        obj_ids = set(table.column("object_id").to_pylist())
        # Shard 1 XP source_ids: 1000, 1002, 1005, 1008
        assert obj_ids == {1000, 1002, 1005, 1008}
