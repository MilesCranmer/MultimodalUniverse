"""Unit tests for scripts/jwst/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "jwst", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_jwst_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()


def _make_wcs_header(size: int = 128):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = size
    hdr["NAXIS2"] = size
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = size / 2
    hdr["CRPIX2"] = size / 2
    hdr["CRVAL1"] = 150.0
    hdr["CRVAL2"] = 2.0
    hdr["CDELT1"] = -0.000011111
    hdr["CDELT2"] = 0.000011111
    return hdr


def _write_image(path: str, value: float) -> None:
    fits.PrimaryHDU(np.full((128, 128), value, dtype=np.float32), header=_make_wcs_header()).writeto(path, overwrite=True)


def _write_catalog(path: str) -> None:
    cat = Table({
        "id": np.array([1, 2], dtype=np.int64),
        "ra": np.array([150.0, 150.0001], dtype=np.float64),
        "dec": np.array([2.0, 2.0001], dtype=np.float64),
        "mag_auto": np.array([26.0, 26.5], dtype=np.float32),
        "flux_radius": np.array([1.0, 1.1], dtype=np.float32),
        "flux_auto": np.array([2.0, 2.1], dtype=np.float32),
        "fluxerr_auto": np.array([0.1, 0.1], dtype=np.float32),
        "cxx_image": np.array([0.2, 0.2], dtype=np.float32),
        "cyy_image": np.array([0.3, 0.3], dtype=np.float32),
        "cxy_image": np.array([0.0, 0.0], dtype=np.float32),
        "f090w_flux_aper_0": np.array([1.0, 1.0], dtype=np.float32),
        "f115w_flux_aper_0": np.array([1.0, 1.0], dtype=np.float32),
        "f150w_flux_aper_0": np.array([1.0, 1.0], dtype=np.float32),
        "f200w_flux_aper_0": np.array([1.0, 1.0], dtype=np.float32),
    })
    cat.write(path, format="fits", overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "JWST"
    mosaic = root / "ceers-full-grizli-v7.0"
    mosaic.mkdir(parents=True)
    _write_catalog(str(mosaic / "ceers-full-grizli-v7.0-fix_phot_apcorr.fits"))
    for filt, val in [("f090w", 1.0), ("f115w", 2.0), ("f150w", 3.0), ("f200w", 4.0)]:
        _write_image(str(mosaic / f"ceers-full-grizli-v7.0-{filt}-clear_drc_sci.fits.gz"), val)
        _write_image(str(mosaic / f"ceers-full-grizli-v7.0-{filt}-clear_drc_wht_full.fits.gz"), 10.0)
    return str(root)


class TestFindMosaics:
    def test_find_mosaic_dirs(self, fake_raw_root):
        mosaics = build.find_mosaic_dirs(fake_raw_root)
        assert len(mosaics) == 1
        assert mosaics[0][0] == "ceers-full-grizli-v7.0"


class TestSelection:
    def test_selection_function(self, fake_raw_root):
        path = os.path.join(fake_raw_root, "ceers-full-grizli-v7.0", "ceers-full-grizli-v7.0-fix_phot_apcorr.fits")
        catalog = Table.read(path)
        mask = build.selection_function(catalog, mag_cut=27.0)
        assert mask.all()


class TestProcessMosaic:
    def test_process_mosaic(self, fake_raw_root, tmp_path):
        rows = build.process_mosaic(
            "ceers-full-grizli-v7.0",
            os.path.join(fake_raw_root, "ceers-full-grizli-v7.0"),
            pixel_threshold=32,
            scratch_dir=str(tmp_path / "scratch"),
            ra_center=None,
            dec_center=None,
            radius=None,
        )
        assert rows == 2
        files = list((tmp_path / "scratch").glob("*.parquet"))
        assert len(files) == 1


class TestMain:
    def test_main_fake_data(self, fake_raw_root, tmp_path):
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--scratch-dir", str(tmp_path / "scratch"),
            "--max-files", "1",
            "--pixel-threshold", "32",
        ])
        assert ret == 0
        hats = list((tmp_path / "out").rglob("hats.properties"))
        assert hats

    def test_missing_raw(self, tmp_path):
        ret = build.main([
            "--raw-root", str(tmp_path / "missing"),
            "--output-root", str(tmp_path / "out"),
            "--scratch-dir", str(tmp_path / "scratch"),
        ])
        assert ret == 1
