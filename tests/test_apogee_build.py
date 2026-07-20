"""Unit tests for scripts/apogee/build_parent_sample_hats.py."""

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
        os.path.dirname(__file__), "..", "scripts", "apogee", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_apogee_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()


def _write_visit_file(path: str, value: float = 1.0) -> None:
    n = 8575
    flux = np.full((1, n), value, dtype=np.float32)
    sigma = np.full((1, n), 2.0, dtype=np.float32)
    mask = np.zeros((1, n), dtype=np.int16)
    hdus = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(flux),
        fits.ImageHDU(sigma),
        fits.ImageHDU(mask),
    ])
    hdus.writeto(path, overwrite=True)


def _write_continuum_file(path: str, value: float = 3.0) -> None:
    n = 8575
    primary = fits.PrimaryHDU()
    flux = fits.ImageHDU(np.full(n, value, dtype=np.float32))
    sigma = fits.ImageHDU(np.full(n, 4.0, dtype=np.float32))
    fits.HDUList([primary, flux, sigma]).writeto(path, overwrite=True)


def _write_allstar(path: str, n_rows: int = 3) -> None:
    table = Table({
        "APOGEE_ID": [f"2M{i:06d}" for i in range(n_rows)],
        "FIELD": ["fieldA"] * n_rows,
        "TELESCOPE": ["apo25m", "lco25m", "apo25m"][:n_rows],
        "FILE": [f"apStar-{i}.fits" for i in range(n_rows)],
        "SNR": np.array([100.0, 120.0, 10.0][:n_rows], dtype=np.float32),
        "RA": np.array([10.0, 20.0, 30.0][:n_rows], dtype=np.float64),
        "DEC": np.array([0.0, 1.0, 2.0][:n_rows], dtype=np.float64),
        "TEFF": np.array([4500.0, 4600.0, 4700.0][:n_rows], dtype=np.float32),
        "LOGG": np.array([2.0, 2.1, 2.2][:n_rows], dtype=np.float32),
        "M_H": np.array([-0.2, -0.1, 0.0][:n_rows], dtype=np.float32),
        "ALPHA_M": np.array([0.1, 0.2, 0.3][:n_rows], dtype=np.float32),
        "TEFF_ERR": np.array([50.0, 50.0, 50.0][:n_rows], dtype=np.float32),
        "LOGG_ERR": np.array([0.1, 0.1, 0.1][:n_rows], dtype=np.float32),
        "M_H_ERR": np.array([0.05, 0.05, 0.05][:n_rows], dtype=np.float32),
        "ALPHA_M_ERR": np.array([0.02, 0.02, 0.02][:n_rows], dtype=np.float32),
        "VHELIO_AVG": np.array([10.0, 20.0, 30.0][:n_rows], dtype=np.float32),
    })
    table.write(path, format="fits", overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    raw_root = tmp_path / "apogee"
    allstar_dir = raw_root / "spectro" / "aspcap" / "dr17" / "synspec_rev1"
    allstar_dir.mkdir(parents=True)
    _write_allstar(str(allstar_dir / "allStar-dr17-synspec_rev1.fits"))

    for telescope in ["apo25m", "lco25m"]:
        visit_dir = raw_root / "spectro" / "redux" / "dr17" / "stars" / telescope / "fieldA"
        cont_dir = raw_root / "spectro" / "aspcap" / "dr17" / "synspec_rev1" / telescope / "fieldA"
        visit_dir.mkdir(parents=True, exist_ok=True)
        cont_dir.mkdir(parents=True, exist_ok=True)

    _write_visit_file(str(raw_root / "spectro" / "redux" / "dr17" / "stars" / "apo25m" / "fieldA" / "apStar-0.fits"), 1.0)
    _write_continuum_file(str(raw_root / "spectro" / "aspcap" / "dr17" / "synspec_rev1" / "apo25m" / "fieldA" / "aspcapStar-dr17-2M000000.fits"), 3.0)
    _write_visit_file(str(raw_root / "spectro" / "redux" / "dr17" / "stars" / "lco25m" / "fieldA" / "apStar-1.fits"), 2.0)
    _write_continuum_file(str(raw_root / "spectro" / "aspcap" / "dr17" / "synspec_rev1" / "lco25m" / "fieldA" / "aspcapStar-dr17-2M000001.fits"), 4.0)
    return str(raw_root)


class TestCatalogLoading:
    def test_load_catalog(self, fake_raw_root):
        catalog = build.load_catalog(fake_raw_root, cache_root=os.path.join(fake_raw_root, "_cache"))
        assert len(catalog) == 3
        assert "APOGEE_ID" in catalog.colnames

    def test_selection_mask(self, fake_raw_root):
        catalog = build.load_catalog(fake_raw_root, cache_root=os.path.join(fake_raw_root, "_cache"))
        mask = build.selection_mask(catalog, fake_raw_root, cache_root=os.path.join(fake_raw_root, "_cache"))
        assert mask.tolist() == [True, True, False]


class TestSpectrumReading:
    def test_read_visit_and_continuum_shapes(self, fake_raw_root):
        visit = build._visit_path(fake_raw_root, "fieldA", "apo25m", "apStar-0.fits")
        cont = build._continuum_path(fake_raw_root, "fieldA", "apo25m", "2M000000")
        out = build._read_visit_and_continuum(visit, cont)
        assert out["flux"].shape == build.LAM_CROPPED.shape
        assert out["ivar"].shape == build.LAM_CROPPED.shape
        assert out["mask"].shape == build.LAM_CROPPED.shape

    def test_row_to_record(self, fake_raw_root):
        row = {
            "APOGEE_ID": "2M000000",
            "FIELD": "fieldA",
            "TELESCOPE": "apo25m",
            "FILE": "apStar-0.fits",
            "RA": 10.0,
            "DEC": 0.0,
            "TEFF": 4500.0,
            "LOGG": 2.0,
            "M_H": -0.2,
            "ALPHA_M": 0.1,
            "TEFF_ERR": 50.0,
            "LOGG_ERR": 0.1,
            "M_H_ERR": 0.05,
            "ALPHA_M_ERR": 0.02,
            "VHELIO_AVG": 10.0,
        }
        record = build._row_to_record((0, row, fake_raw_root))
        assert record is not None
        assert record["object_id"] == "2M000000"
        assert record["restframe"] is True


class TestArrowTable:
    def test_build_arrow_table(self, fake_raw_root):
        row = {
            "APOGEE_ID": "2M000000",
            "FIELD": "fieldA",
            "TELESCOPE": "apo25m",
            "FILE": "apStar-0.fits",
            "RA": 10.0,
            "DEC": 0.0,
            "TEFF": 4500.0,
            "LOGG": 2.0,
            "M_H": -0.2,
            "ALPHA_M": 0.1,
            "TEFF_ERR": 50.0,
            "LOGG_ERR": 0.1,
            "M_H_ERR": 0.05,
            "ALPHA_M_ERR": 0.02,
            "VHELIO_AVG": 10.0,
        }
        record = build._row_to_record((0, row, fake_raw_root))
        table = build.build_arrow_table([record])
        assert table.schema.field("object_id").type == pa.string()
        assert table.schema.field("ra").type == pa.float64()
        assert table.schema.field("dec").type == pa.float64()
        spectrum_type = table.schema.field("spectrum").type
        assert pa.types.is_struct(spectrum_type)
        names = {field.name for field in spectrum_type}
        assert names == {
            "flux",
            "ivar",
            "lsf_sigma",
            "lambda",
            "mask",
            "pseudo_continuum_flux",
            "pseudo_continuum_ivar",
        }


class TestMain:
    def test_missing_catalog(self, tmp_path):
        ret = build.main([
            "--raw-root", str(tmp_path / "missing"),
            "--output-root", str(tmp_path / "out"),
            "--cache-root", str(tmp_path / "cache"),
        ])
        assert ret in (0, 1)

    def test_main_fake_data(self, fake_raw_root, tmp_path):
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--scratch-dir", str(tmp_path / "scratch"),
            "--cache-root", str(tmp_path / "cache"),
            "--max-files", "2",
            "--num-processes", "1",
            "--batch-size", "1",
            "--pixel-threshold", "64",
        ])
        assert ret == 0
