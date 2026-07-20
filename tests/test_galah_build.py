"""Unit tests for scripts/galah/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os
import tarfile

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "galah", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_galah_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()


def _write_galah_spectrum(path: str, value: float) -> None:
    n = 12
    primary = fits.PrimaryHDU(np.full(n, value, dtype=np.float32))
    primary.header["CRVAL1"] = 5000.0
    primary.header["CDELT1"] = 0.1
    primary.header["NAXIS1"] = n
    primary.header["CRPIX1"] = 1
    primary.header["UTMJD"] = 58000.0

    sigma = fits.ImageHDU(np.full(n, 2.0, dtype=np.float32))
    sigma.header["CRVAL1"] = 5000.0
    sigma.header["CDELT1"] = 0.1
    sigma.header["NAXIS1"] = n
    sigma.header["CRPIX1"] = 1

    filler2 = fits.ImageHDU(np.zeros(n, dtype=np.float32))
    filler3 = fits.ImageHDU(np.zeros(n, dtype=np.float32))
    norm = fits.ImageHDU(np.full(n, value + 1.0, dtype=np.float32))
    norm.header["CRVAL1"] = 5000.0
    norm.header["CDELT1"] = 0.1
    norm.header["NAXIS1"] = n
    norm.header["CRPIX1"] = 1
    fits.HDUList([primary, sigma, filler2, filler3, norm]).writeto(path, overwrite=True)


def _make_catalog_table() -> Table:
    data = {
        "sobject_id": np.array([1311160005010021], dtype=np.int64),
        "ra_dr2": np.array([10.0], dtype=np.float64),
        "dec_dr2": np.array([0.5], dtype=np.float64),
        "rv_galah": np.array([25.0], dtype=np.float32),
        "e_rv_galah": np.array([0.2], dtype=np.float32),
    }
    for name in build._FLOAT_FEATURES:
        if name in data:
            continue
        data[name] = np.array([1.0], dtype=np.float32)
    for name in build._INT_FEATURES:
        data[name] = np.array([1], dtype=np.int32)
    return Table(data)


def _make_vac_table() -> Table:
    data = {
        "sobject_id": np.array([1311160005010021], dtype=np.int64),
    }
    for src_name in build.VAC_KEY_MAP.values():
        data[src_name] = np.array([1.0], dtype=np.float32)
    return Table(data)


@pytest.fixture
def fake_raw_root(tmp_path):
    raw_root = tmp_path / "galah" / "dr3"
    raw_root.mkdir(parents=True)

    _make_catalog_table().write(raw_root / "GALAH_DR3_main_allspec_v2.fits", format="fits", overwrite=True)
    _make_vac_table().write(raw_root / "GALAH_DR3_VAC_ages_v2.fits", format="fits", overwrite=True)
    (raw_root / "resolution_maps").mkdir()
    for i in range(1, 5):
        fits.PrimaryHDU(np.ones((2, 12), dtype=np.float32)).writeto(
            raw_root / "resolution_maps" / f"ccd{i}_piv.fits",
            overwrite=True,
        )

    spec_dir = tmp_path / "specsrc" / "galah" / "dr3" / "spectra" / "hermes"
    spec_dir.mkdir(parents=True)
    for suffix, value in zip([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0]):
        _write_galah_spectrum(str(spec_dir / f"1311160005010021{suffix}.fits"), value)

    tar_path = raw_root / "fake_spectra.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in spec_dir.iterdir():
            tar.add(path, arcname=f"galah/dr3/spectra/hermes/{path.name}")

    return str(raw_root)


class TestPrepareRows:
    def test_prepare_rows(self, fake_raw_root):
        rows = build.prepare_rows(fake_raw_root, nside=16, max_files=None, ra_center=None, dec_center=None, radius=None)
        assert len(rows) == 1
        assert rows[0]["catalog"]["sobject_id"] == 1311160005010021


class TestProcessObject:
    def test_process_object(self, fake_raw_root):
        rows = build.prepare_rows(fake_raw_root, nside=16, max_files=None, ra_center=None, dec_center=None, radius=None)
        record = build._process_object(rows[0])
        assert record is not None
        assert record["object_id"] == "1311160005010021"
        assert "flux" in record
        assert "filter_indices" in record


class TestArrowTable:
    def test_build_arrow_table(self, fake_raw_root):
        rows = build.prepare_rows(fake_raw_root, nside=16, max_files=None, ra_center=None, dec_center=None, radius=None)
        record = build._process_object(rows[0])
        table = build.build_arrow_table([record])
        assert table.schema.field("object_id").type == pa.string()
        assert table.schema.field("ra").type == pa.float64()
        assert table.schema.field("dec").type == pa.float64()
        assert pa.types.is_struct(table.schema.field("spectrum").type)
        assert pa.types.is_struct(table.schema.field("filter_indices").type)


class TestMain:
    def test_main_fake_data(self, fake_raw_root, tmp_path):
        ret = build.main([
            "--raw-root", fake_raw_root,
            "--output-root", str(tmp_path / "out"),
            "--scratch-dir", str(tmp_path / "scratch"),
            "--max-files", "1",
            "--num-processes", "1",
            "--batch-size", "1",
            "--pixel-threshold", "8",
        ])
        assert ret == 0
