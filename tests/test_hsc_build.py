"""Unit tests for scripts/hsc/build_parent_sample_hats.py.

Covers the pure-Python helpers (path construction, mask cleaning, PSF FWHM
formula, struct/table assembly) plus an end-to-end build_table run on
synthetic records that match the dict shape produced by ``process_patch``.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "_hsc_build",
        os.path.join(
            os.path.dirname(__file__), "..", "scripts", "hsc",
            "build_parent_sample_hats.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()


class TestPathConstruction:
    def test_patch_dir_low(self):
        # patch=305 → 3,5
        assert build._patch_dir(305) == "3,5"

    def test_patch_dir_zero(self):
        assert build._patch_dir(0) == "0,0"

    def test_patch_dir_high(self):
        # patch=905 → 9,5
        assert build._patch_dir(905) == "9,5"

    def test_image_path(self):
        p = build.patch_image_path("/data", "G", 9813, 305)
        assert p.endswith("HSC-G/9813/3,5/calexp-HSC-G-9813-3,5.fits")


class TestMaskClean:
    def test_clears_bad_bits(self):
        # Bit 0 (BAD), bit 1 (SAT), bit 8 (NO_DATA) → should become False
        # Other bits → True
        data = np.array([
            0b00000000,  # all clean → True
            0b00000001,  # BAD → False
            0b00000010,  # SAT → False
            0b00000100,  # other → True
            0b00010000,  # other → True
            0b100000000,  # NO_DATA (bit 8) → False
            0b00000011,  # BAD + SAT → False
        ], dtype=np.int32)
        clean = build._clean_mask(data)
        assert clean.dtype == bool
        assert clean.tolist() == [True, False, False, True, True, False, False]


class TestPsfFwhm:
    def test_round_circular_psf(self):
        # mxx = myy = 1, mxy = 0 → det = 1 → FWHM = 2.355 * 1**(1/4) = 2.355
        obj = {f"{b.lower()}_sdssshape_psf_shape{c}": v
               for b in build.BANDS
               for c, v in (("11", 1.0), ("22", 1.0), ("12", 0.0))}
        out = build._psf_fwhm_arcsec(obj)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.full(5, 2.355), atol=1e-5)

    def test_negative_det_zeroed(self):
        # mxy^2 > mxx*myy → det < 0 → 0 instead of NaN/imag
        obj = {f"{b.lower()}_sdssshape_psf_shape{c}": v
               for b in build.BANDS
               for c, v in (("11", 0.5), ("22", 0.5), ("12", 1.0))}
        out = build._psf_fwhm_arcsec(obj)
        assert np.all(out == 0.0)

    def test_nan_zeroed(self):
        # NaN inputs propagate to NaN det → nan_to_num → 0
        obj = {f"{b.lower()}_sdssshape_psf_shape{c}": v
               for b in build.BANDS
               for c, v in (("11", float("nan")), ("22", 1.0), ("12", 0.0))}
        out = build._psf_fwhm_arcsec(obj)
        assert np.all(out == 0.0)


def _fake_record(object_id="123", n_bands=None):
    """Synthetic record matching the per-row dict produced by process_patch."""
    n_bands = n_bands or build.N_BANDS
    rec = {
        "object_id": object_id,
        "ra": 150.0,
        "dec": 2.0,
        "image_flux": np.ones((n_bands, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
        "image_ivar": np.ones((n_bands, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
        "image_mask": np.ones((n_bands, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=bool),
        "image_psf_fwhm": np.full(n_bands, 0.7, dtype=np.float32),
        "image_scale": np.full(n_bands, build.PIXEL_SCALE, dtype=np.float32),
    }
    for f in build.FLOAT_FEATURES:
        rec[f] = 1.0
    return rec


class TestBuildTable:
    def test_top_level_columns(self):
        table = build.build_table([_fake_record(), _fake_record(object_id="456")])
        assert table.num_rows == 2
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "image"} <= names
        # All 65 scalar features present.
        for f in build.FLOAT_FEATURES:
            assert f in names
        assert len(build.FLOAT_FEATURES) == 65

    def test_image_struct_subfields(self):
        table = build.build_table([_fake_record()])
        img = table.schema.field("image").type
        assert pa.types.is_struct(img)
        names = {f.name for f in img}
        assert names == {"band", "flux", "ivar", "mask", "psf_fwhm", "scale"}

    def test_image_shapes(self):
        table = build.build_table([_fake_record()])
        row = table.column("image")[0].as_py()
        assert row["band"] == list(build.BANDS)
        # flux is list-of-list-of-list with shape (5, 160, 160)
        assert len(row["flux"]) == build.N_BANDS
        assert len(row["flux"][0]) == build.IMAGE_SIZE
        assert len(row["flux"][0][0]) == build.IMAGE_SIZE
        # mask is bool
        assert isinstance(row["mask"][0][0][0], bool)
        # per-band scalars
        assert len(row["psf_fwhm"]) == build.N_BANDS
        np.testing.assert_allclose(row["scale"], [build.PIXEL_SCALE] * build.N_BANDS, atol=1e-6)

    def test_scalar_dtypes(self):
        table = build.build_table([_fake_record()])
        for f in build.FLOAT_FEATURES[:5]:
            assert table.schema.field(f).type == pa.float32()


class TestConstants:
    def test_image_size(self):
        assert build.IMAGE_SIZE == 160

    def test_pixel_scale(self):
        assert build.PIXEL_SCALE == 0.168

    def test_bands(self):
        assert build.BANDS == ["G", "R", "I", "Z", "Y"]
        assert build.N_BANDS == 5
