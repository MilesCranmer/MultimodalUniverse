"""Fast unit tests for scripts/legacysurvey/build_parent_sample_hats.py.

Most tests target the pure-Python helpers (sweep filename parsing, cone
bbox intersection, selection function, schema assembly). The full image
extraction path (Cutout2D + WCS + JPEG decoding) is exercised by the
cluster integration test, not here, because faking a brick FITS+WCS+JPEG
properly is more code than the helpers themselves.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "legacysurvey", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_ls_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestParseSweepBbox:
    def test_north_positive(self):
        assert build.parse_sweep_bbox("sweep-150p000-155p005.fits") == (150.0, 155.0, 0.0, 5.0)

    def test_south_negative(self):
        assert build.parse_sweep_bbox("sweep-150m005-155p000.fits") == (150.0, 155.0, -5.0, 0.0)

    def test_full_negative(self):
        assert build.parse_sweep_bbox("sweep-150m015-155m010.fits") == (150.0, 155.0, -15.0, -10.0)

    def test_path_with_dirs(self):
        assert build.parse_sweep_bbox(
            "/some/dir/dr10/south/sweep/10.1/sweep-000m005-005p000.fits"
        ) == (0.0, 5.0, -5.0, 0.0)

    def test_invalid_returns_none(self):
        assert build.parse_sweep_bbox("not_a_sweep.fits") is None
        assert build.parse_sweep_bbox("sweep-15m5-15p5.fits") is None


class TestBboxIntersectsCone:
    def test_center_inside(self):
        assert build._bbox_intersects_cone((150, 155, 0, 5), 150.5, 2.5, 0.5)

    def test_far_below(self):
        assert not build._bbox_intersects_cone((150, 155, 0, 5), 150.5, -10, 0.5)

    def test_far_above(self):
        assert not build._bbox_intersects_cone((150, 155, 0, 5), 150.5, 20, 0.5)

    def test_far_left(self):
        assert not build._bbox_intersects_cone((150, 155, 0, 5), 100, 2, 0.5)

    def test_far_right(self):
        assert not build._bbox_intersects_cone((150, 155, 0, 5), 200, 2, 0.5)

    def test_just_outside_padded_in(self):
        # Cone center at (149.7, 2) with radius 0.5 should still intersect
        # the (150,155,0,5) bbox because the cone extends past 150.0 in RA.
        assert build._bbox_intersects_cone((150, 155, 0, 5), 149.7, 2.5, 0.5)


class TestSelectObservations:
    def _make_catalog(self, n=5):
        return Table({
            "FLUX_Z":               np.array([100, 100, 100, 0.001, 100], dtype=np.float64),
            "MW_TRANSMISSION_Z":    np.ones(n, dtype=np.float64),
            "NOBS_G":               np.ones(n, dtype=np.int32),
            "NOBS_R":               np.ones(n, dtype=np.int32),
            "NOBS_I":               np.ones(n, dtype=np.int32),
            "NOBS_Z":               np.array([1, 1, 0, 1, 1], dtype=np.int32),  # row 2 = no Z
            "TYPE":                 np.array(["EXP", "PSF", "EXP", "EXP", "EXP"]),  # row 1 = PSF
            "MASKBITS":             np.array([0, 0, 0, 0, 1 << 0], dtype=np.int64),  # row 4 = bad bit
        })

    def test_passes_baseline(self):
        cat = self._make_catalog()
        # Row 0 should pass (zmag = 22.5 - 2.5*log10(100) = 17.5 < 21).
        # Rows 1-4 should fail one cut each.
        mask = build.select_observations(cat)
        assert mask.tolist() == [True, False, False, False, False]

    def test_zmag_cut(self):
        n = 3
        cat = Table({
            "FLUX_Z":            np.array([100, 0.1, 0.001], dtype=np.float64),
            "MW_TRANSMISSION_Z": np.ones(n, dtype=np.float64),
            "NOBS_G":            np.ones(n, dtype=np.int32),
            "NOBS_R":            np.ones(n, dtype=np.int32),
            "NOBS_I":            np.ones(n, dtype=np.int32),
            "NOBS_Z":            np.ones(n, dtype=np.int32),
            "TYPE":              np.array(["EXP"] * n),
            "MASKBITS":          np.zeros(n, dtype=np.int64),
        })
        mask = build.select_observations(cat)
        assert mask[0] and not mask[1] and not mask[2]


class TestFindSweepFiles:
    def test_cone_filter_to_cosmos(self, tmp_path):
        sweep_dir = tmp_path / "dr10" / "south" / "sweep" / "10.1"
        sweep_dir.mkdir(parents=True)
        for name in [
            "sweep-150p000-155p005.fits",  # COSMOS overlap
            "sweep-145p000-150p005.fits",  # adjacent in RA (overlaps)
            "sweep-000m005-005p000.fits",  # far away
            "sweep-200p000-205p005.fits",  # far away
        ]:
            (sweep_dir / name).write_bytes(b"")
        files = build.find_sweep_files(
            str(tmp_path), ra_center=150.0, dec_center=2.0, radius=0.5,
        )
        names = sorted(os.path.basename(f) for f in files)
        # The two RA-150 sweeps should pass (one center-overlap, one edge-pad).
        assert "sweep-150p000-155p005.fits" in names
        assert "sweep-145p000-150p005.fits" in names
        assert "sweep-000m005-005p000.fits" not in names
        assert "sweep-200p000-205p005.fits" not in names

    def test_no_cone_returns_all(self, tmp_path):
        sweep_dir = tmp_path / "dr10" / "south" / "sweep" / "10.1"
        sweep_dir.mkdir(parents=True)
        for name in ["sweep-000p000-005p005.fits", "sweep-150p000-155p005.fits"]:
            (sweep_dir / name).write_bytes(b"")
        assert len(build.find_sweep_files(str(tmp_path))) == 2


class TestNearbyCatalogPadding:
    def test_returns_padded_when_empty(self):
        # _build_nearby_catalog falls back to all-zero padded record if there
        # are no objects in the cutout. We bypass the WCS-heavy path here by
        # constructing a fake "no nearby" call directly.
        from astropy.nddata import Cutout2D
        from astropy.wcs import WCS
        # Build a tiny synthetic 200x200 image with a dummy WCS.
        data = np.zeros((200, 200), dtype=np.float32)
        w = WCS(naxis=2)
        w.wcs.crpix = [100, 100]
        w.wcs.cdelt = [-0.262 / 3600, 0.262 / 3600]
        w.wcs.crval = [150.0, 2.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        cutout = Cutout2D(data, position=(100, 100), size=(160, 160), wcs=w)
        # Catalog has rows but FAR from the cutout center, so none survive.
        cat = Table({
            "RA": np.array([0.0, 90.0]),
            "DEC": np.array([-30.0, 30.0]),
            "FLUX_G": np.array([1.0, 2.0], dtype=np.float32),
            "FLUX_R": np.array([1.0, 2.0], dtype=np.float32),
            "FLUX_I": np.array([1.0, 2.0], dtype=np.float32),
            "FLUX_Z": np.array([1.0, 2.0], dtype=np.float32),
            "TYPE": np.array(["EXP", "EXP"]),
            "SHAPE_R": np.array([1.0, 1.0], dtype=np.float32),
            "SHAPE_E1": np.array([0.0, 0.0], dtype=np.float32),
            "SHAPE_E2": np.array([0.0, 0.0], dtype=np.float32),
        })
        nearby = build._build_nearby_catalog(cat, cutout, n_objects=5)
        for key in build.CATALOG_FEATURES:
            assert nearby[key] == [0.0] * 5


class TestObjectMaskPainting:
    def test_paints_non_padding_rows(self):
        # 4 nearby objects with various positions, only first two non-padding.
        nearby = {key: [0.0] * build.NEARBY_CATALOG_N for key in build.CATALOG_FEATURES}
        nearby["X"][0] = 80
        nearby["Y"][0] = 80
        nearby["TYPE"][0] = float(build.OBJECT_TYPE_COLOR["EXP"])  # 3
        nearby["SHAPE_R"][0] = 5.0  # arcsec
        nearby["X"][1] = 40
        nearby["Y"][1] = 40
        nearby["TYPE"][1] = float(build.OBJECT_TYPE_COLOR["DEV"])  # 4
        nearby["SHAPE_R"][1] = 3.0
        mask = build._build_object_mask(nearby, (160, 160))
        # Some EXP=3 pixels exist
        assert (mask == 3).any()
        assert (mask == 4).any()
        # No padding (TYPE=0 rows) painted as 0... well, the BG is 0 too, but
        # let's just make sure there are no spurious values from padding.
        assert set(np.unique(mask).tolist()) <= {0, 3, 4}


class TestBuildTableSchema:
    def _fake_record(self, ra=150.0, dec=2.0, oid="abc-1"):
        return {
            "ra": ra,
            "dec": dec,
            "object_id": oid,
            "image_band": list(build.BANDS),
            "image_flux": np.zeros((4, 160, 160), dtype=np.float32),
            "image_ivar": np.ones((4, 160, 160), dtype=np.float32),
            "image_mask": np.zeros((4, 160, 160), dtype=bool),
            "image_psf_fwhm": np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),
            "image_scale": np.array([0.262, 0.262, 0.262, 0.262], dtype=np.float32),
            "rgb": np.zeros((3, 160, 160), dtype=np.uint8),
            "blobmodel": np.zeros((3, 160, 160), dtype=np.uint8),
            "object_mask": np.zeros((160, 160), dtype=np.uint8),
            "nearby_catalog": {key: [0.0] * build.NEARBY_CATALOG_N for key in build.CATALOG_FEATURES},
            "scalars": {f: 0.0 for f in build.FLOAT_FEATURES},
        }

    def test_table_top_level_columns(self):
        table = build.build_table([self._fake_record(), self._fake_record(oid="abc-2")])
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "image", "rgb", "blobmodel",
                "object_mask", "catalog"} <= names
        for f in build.FLOAT_FEATURES:
            assert f in names

    def test_image_struct_subfields(self):
        table = build.build_table([self._fake_record()])
        image_type = table.schema.field("image").type
        assert pa.types.is_struct(image_type)
        sub = {f.name: f.type for f in image_type}
        assert set(sub.keys()) == {"band", "flux", "ivar", "mask", "psf_fwhm", "scale"}
        assert sub["flux"] == pa.list_(pa.list_(pa.list_(pa.float32())))
        assert sub["ivar"] == pa.list_(pa.list_(pa.list_(pa.float32())))
        assert sub["mask"] == pa.list_(pa.list_(pa.list_(pa.bool_())))
        assert sub["band"] == pa.list_(pa.string())
        assert sub["psf_fwhm"] == pa.list_(pa.float32())
        assert sub["scale"] == pa.list_(pa.float32())

    def test_jpeg_columns_uint8_nested(self):
        table = build.build_table([self._fake_record()])
        for c in ("rgb", "blobmodel"):
            t = table.schema.field(c).type
            assert t == pa.list_(pa.list_(pa.list_(pa.uint8())))
        assert table.schema.field("object_mask").type == pa.list_(pa.list_(pa.uint8()))

    def test_image_roundtrip_shape(self):
        table = build.build_table([self._fake_record()])
        first = table.column("image")[0].as_py()
        flux = np.asarray(first["flux"], dtype=np.float32)
        assert flux.shape == (4, 160, 160)
        assert first["band"] == build.BANDS

    def test_catalog_struct_lists_have_n_objects(self):
        table = build.build_table([self._fake_record()])
        cat = table.column("catalog")[0].as_py()
        for key in build.CATALOG_FEATURES:
            assert key in cat
            assert len(cat[key]) == build.NEARBY_CATALOG_N
