"""Tests for the direct MaNGA HATS builder."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "manga", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_manga_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_catalogs(raw_root: Path, plateifu: str = "8485-1901") -> tuple[Path, Path]:
    drpall = Table(
        {
            "plateifu": np.array([plateifu, "9999-0001"]),
            "ifura": np.array([150.0, 151.0], dtype=np.float64),
            "ifudec": np.array([2.0, 3.0], dtype=np.float64),
            "nsa_z": np.array([0.05, 0.07], dtype=np.float32),
        }
    )
    dapall = Table(
        {
            "PLATEIFU": np.array([plateifu, "9999-0001"]),
            "DAPDONE": np.array([True, False]),
        }
    )

    drpall_path = raw_root / "drpall-v3_1_1.fits"
    dapall_path = raw_root / "dapall-v3_1_1-3.1.0.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(drpall, name="MANGA")]).writeto(
        drpall_path, overwrite=True
    )
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(dapall, name=build.DAPTYPE)]
    ).writeto(dapall_path, overwrite=True)
    return drpall_path, dapall_path


def _write_cube(raw_root: Path, plateifu: str = "8485-1901") -> Path:
    plate, _ = plateifu.split("-")
    cube_dir = raw_root / "dr17" / "manga" / "spectro" / "redux" / "v3_1_1" / plate / "stack"
    cube_dir.mkdir(parents=True, exist_ok=True)
    cube_path = cube_dir / f"manga-{plateifu}-LOGCUBE.fits.gz"

    nwave, ny, nx = 4, 2, 3
    base = np.arange(nwave * ny * nx, dtype=np.float32).reshape(nwave, ny, nx)
    mask = (100 + np.arange(nwave * ny * nx, dtype=np.int64)).reshape(nwave, ny, nx)
    image = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)

    flux_hdu = fits.ImageHDU(base.copy(), name="FLUX")
    flux_hdu.header["BUNIT"] = "1e-17 erg/s/cm^2/Ang/spaxel"
    flux_hdu.header["CUNIT3"] = "Angstrom"

    hdus = [fits.PrimaryHDU(), flux_hdu]
    hdus.append(fits.ImageHDU((base + 100).astype(np.float32), name="IVAR"))
    hdus.append(fits.ImageHDU(mask, name="MASK"))
    hdus.append(fits.ImageHDU((base + 200).astype(np.float32), name="LSFPOST"))
    hdus.append(fits.ImageHDU(np.linspace(3500, 3503, nwave, dtype=np.float32), name="WAVE"))

    for idx, band in enumerate(build.BANDS):
        hdus.append(fits.ImageHDU((image + idx).astype(np.float32), name=f"{band.upper()}IMG"))
    for idx, band in enumerate(build.BANDS):
        hdus.append(fits.ImageHDU((image + 10 + idx).astype(np.float32), name=f"{band.upper()}PSF"))

    fits.HDUList(hdus).writeto(cube_path, overwrite=True)
    return cube_path


def _write_maps(raw_root: Path, plateifu: str = "8485-1901") -> Path:
    plate, ifu = plateifu.split("-")
    maps_dir = (
        raw_root
        / "dr17"
        / "manga"
        / "spectro"
        / "analysis"
        / "v3_1_1"
        / "3.1.0"
        / build.DAPTYPE
        / plate
        / ifu
    )
    maps_dir.mkdir(parents=True, exist_ok=True)
    maps_path = maps_dir / f"manga-{plateifu}-MAPS-{build.DAPTYPE}.fits.gz"

    ny, nx = 2, 3
    skycoo = np.array(
        [
            [[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]],
            [[-0.5, -0.25, 0.0], [0.25, 0.5, 0.75]],
        ],
        dtype=np.float32,
    )
    ellcoo = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24]],
        ],
        dtype=np.float32,
    )

    spx_skycoo = fits.ImageHDU(skycoo, name="SPX_SKYCOO")
    spx_skycoo.header["BUNIT"] = "arcsec"
    spx_skycoo.header["C01"] = "skycoo_x"
    spx_skycoo.header["C02"] = "skycoo_y"

    spx_ellcoo = fits.ImageHDU(ellcoo, name="SPX_ELLCOO")
    spx_ellcoo.header["C01"] = "ellcoo_r"
    spx_ellcoo.header["C02"] = "ellcoo_rre"
    spx_ellcoo.header["C03"] = "ellcoo_rkpc"
    spx_ellcoo.header["C04"] = "ellcoo_theta"
    spx_ellcoo.header["U01"] = "arcsec"
    spx_ellcoo.header["U02"] = "r/re"
    spx_ellcoo.header["U03"] = "kpc"
    spx_ellcoo.header["U04"] = "deg"

    binid = fits.ImageHDU(np.arange(ny * nx, dtype=np.float32).reshape(ny, nx), name="BINID")
    binid.header["BUNIT"] = "bin"

    emline = np.arange(2 * ny * nx, dtype=np.float32).reshape(2, ny, nx)
    emline_hdu = fits.ImageHDU(emline, name="EMLINE_GFLUX")
    emline_hdu.header["ERRDATA"] = "EMLINE_GFLUX_IVAR"
    emline_hdu.header["QUALDATA"] = "EMLINE_GFLUX_MASK"
    emline_hdu.header["BUNIT"] = "1e-17"
    emline_hdu.header["C01"] = "Ha 6564"
    emline_hdu.header["C02"] = "Hb 4862"
    emline_hdu.header["U01"] = "1e-17"
    emline_hdu.header["U02"] = "1e-17"

    emline_ivar = fits.ImageHDU((emline + 100).astype(np.float32), name="EMLINE_GFLUX_IVAR")
    emline_mask = fits.ImageHDU((emline + 200).astype(np.float32), name="EMLINE_GFLUX_MASK")

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            spx_skycoo,
            spx_ellcoo,
            binid,
            emline_hdu,
            emline_ivar,
            emline_mask,
        ]
    ).writeto(maps_path, overwrite=True)
    return maps_path


def _make_raw_fixture(tmp_path: Path, plateifu: str = "8485-1901") -> Path:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_catalogs(raw_root, plateifu=plateifu)
    _write_cube(raw_root, plateifu=plateifu)
    _write_maps(raw_root, plateifu=plateifu)
    return raw_root


class TestCatalogLoading:
    def test_load_catalog_filters_dapdone(self, tmp_path):
        raw_root = _make_raw_fixture(tmp_path)
        catalog = build.load_catalog(str(raw_root))
        assert len(catalog) == 1
        assert catalog["plateifu"][0] == "8485-1901"


class TestProcessCube:
    def test_process_cube_preserves_native_shape(self, tmp_path):
        raw_root = _make_raw_fixture(tmp_path)
        catalog = build.load_catalog(str(raw_root))
        record = build.process_cube(catalog[0], str(raw_root))

        assert record["object_id"] == "8485-1901"
        assert record["spatial_shape_y"] == 2
        assert record["spatial_shape_x"] == 3
        assert record["spaxels"]["flux"].shape == (2, 3, 4)
        assert record["spaxels"]["lambda"].shape == (4,)
        assert record["images"]["flux"].shape == (4, 2, 3)
        assert len(record["maps"]) == 9  # SPX_SKYCOO(2) + SPX_ELLCOO(4) + BINID + EMLINE_GFLUX(2)
        assert record["maps"][0]["label"] == "spx_skycoo_skycoo_x"
        assert record["maps"][-1]["label"] == "emline_gflux_hb_4862"
        assert record["spaxels"]["ellcoo_r_units"] == "arcsec"
        assert record["spaxels"]["ellcoo_theta_units"] == "deg"


class TestBuildTable:
    def test_schema_contains_native_shape_and_nested_payloads(self, tmp_path):
        raw_root = _make_raw_fixture(tmp_path)
        catalog = build.load_catalog(str(raw_root))
        record = build.process_cube(catalog[0], str(raw_root))
        table = build.build_table([record])

        names = set(table.schema.names)
        assert {
            "ra",
            "dec",
            "object_id",
            "healpix",
            "z",
            "spaxel_size",
            "spaxel_size_units",
            "spatial_shape_y",
            "spatial_shape_x",
            "spaxels",
            "images",
            "maps",
        } <= names

        spaxels_type = table.schema.field("spaxels").type
        assert pa.types.is_struct(spaxels_type)
        assert spaxels_type["flux"].type == pa.list_(pa.list_(pa.list_(pa.float32())))
        assert spaxels_type["lambda_units"].type == pa.string()

        row = table.column("spaxels")[0].as_py()
        assert len(row["flux"]) == 2
        assert len(row["flux"][0]) == 3
        assert len(row["flux"][0][0]) == 4

        images = table.column("images")[0].as_py()
        assert images["filter"] == build.BANDS
        assert len(images["flux"]) == 4
        assert len(images["flux"][0]) == 2
        assert len(images["flux"][0][0]) == 3


class TestEndToEndHatsWrite:
    def test_main_writes_hats_catalog(self, tmp_path):
        raw_root = _make_raw_fixture(tmp_path)
        output_root = tmp_path / "out"
        work_dir = tmp_path / "work"

        try:
            rc = build.main(
                [
                    "--raw-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--work-dir",
                    str(work_dir),
                    "--plateifu",
                    "8485-1901",
                    "--rows-per-shard",
                    "1",
                    "--debug",
                ]
            )
        except RuntimeError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local Dask scheduler sockets are blocked in this sandbox")
            raise
        assert rc == 0

        hats_root = output_root / "manga" / "manga"
        dataset = ds.dataset(str(hats_root / "dataset"), format="parquet")
        table = dataset.to_table(
            columns=["object_id", "ra", "dec", "spatial_shape_y", "spatial_shape_x", "images"]
        )

        assert table.num_rows == 1
        assert table.column("object_id")[0].as_py() == "8485-1901"
        assert table.column("ra")[0].as_py() == 150.0
        assert table.column("dec")[0].as_py() == 2.0
        assert table.column("spatial_shape_y")[0].as_py() == 2
        assert table.column("spatial_shape_x")[0].as_py() == 3

        images = table.column("images")[0].as_py()
        assert images["filter"] == build.BANDS
        assert len(images["flux"]) == 4
        assert len(images["flux"][0]) == 2
        assert len(images["flux"][0][0]) == 3
