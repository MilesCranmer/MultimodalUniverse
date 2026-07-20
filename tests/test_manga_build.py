"""Fast unit tests for scripts/manga/build_parent_sample_hats.py.

The full process_cube path needs real LOGCUBE/MAPS FITS files which are
hard to fabricate, so we test the pure-Python helpers (path construction,
padding, struct assembly) and exercise build_table with hand-rolled fake
records that match the expected dict shape.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
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


NY, NX, NWAVE = 32, 27, 10  # tiny native shape for tests


class TestPathConstruction:
    def test_cube_path(self):
        p = build.cube_path("/data", "8485-1901")
        assert p.endswith("dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz")

    def test_maps_path(self):
        p = build.maps_path("/data", "8485-1901")
        assert "spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/8485/1901" in p
        assert p.endswith("manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz")


def _fake_record(plateifu="8485-1901", n_maps=3, ny=NY, nx=NX, nwave=NWAVE):
    """Build a synthetic record matching the per-row dict produced by
    ``process_cube`` at native (ny, nx) shape — no 96x96 padding."""
    yy, xx = np.indices((ny, nx))
    spaxels = {
        "flux":   np.ones((ny, nx, nwave), dtype=np.float32),
        "ivar":   np.ones((ny, nx, nwave), dtype=np.float32),
        "mask":   np.zeros((ny, nx, nwave), dtype=np.int64),
        "lsf":    np.ones((ny, nx, nwave), dtype=np.float32),
        "lambda": np.linspace(3500, 9000, nwave, dtype=np.float32),
        "x": xx.astype(np.int8),
        "y": yy.astype(np.int8),
        "spaxel_idx": np.arange(ny * nx, dtype=np.int16).reshape(ny, nx),
        "flux_units":   "1e-17 erg/s/cm^2/Å",
        "lambda_units": "Angstrom",
        "skycoo_x":     np.zeros((ny, nx), dtype=np.float32),
        "skycoo_y":     np.zeros((ny, nx), dtype=np.float32),
        "ellcoo_r":     np.zeros((ny, nx), dtype=np.float32),
        "ellcoo_rre":   np.zeros((ny, nx), dtype=np.float32),
        "ellcoo_rkpc":  np.zeros((ny, nx), dtype=np.float32),
        "ellcoo_theta": np.zeros((ny, nx), dtype=np.float32),
        "skycoo_units":       "arcsec",
        "ellcoo_r_units":     "arcsec",
        "ellcoo_rre_units":   "",
        "ellcoo_rkpc_units":  "kpc",
        "ellcoo_theta_units": "degree",
    }
    images = {
        "filter": list(build.BANDS),
        "flux": np.ones((build.N_BANDS, ny, nx), dtype=np.float32),
        "flux_units": ["nanomaggies/pixel"] * build.N_BANDS,
        "psf": np.ones((build.N_BANDS, ny, nx), dtype=np.float32),
        "psf_units": ["nanomaggies/pixel"] * build.N_BANDS,
        "scale": [build.SPAXEL_SIZE_ARCSEC] * build.N_BANDS,
        "scale_units": ["arcsec"] * build.N_BANDS,
    }
    maps = []
    for i in range(n_maps):
        maps.append({
            "group": f"group_{i}",
            "label": f"label_{i}",
            "flux": np.ones((ny, nx), dtype=np.float32),
            "ivar": np.ones((ny, nx), dtype=np.float32),
            "mask": np.zeros((ny, nx), dtype=np.float32),
            "array_units": "unit",
        })
    return {
        "plateifu": plateifu,
        "spatial_shape_y": ny,
        "spatial_shape_x": nx,
        "spaxels": spaxels,
        "images": images,
        "maps": maps,
    }


def _fake_catalog(plateifus=("8485-1901",)):
    return Table({
        "plateifu": np.array(list(plateifus)),
        "ifura": np.full(len(plateifus), 150.0, dtype=np.float64),
        "ifudec": np.full(len(plateifus), 2.0, dtype=np.float64),
        "nsa_z": np.full(len(plateifus), 0.05, dtype=np.float32),
    })


class TestBuildTable:
    def test_top_level_columns(self):
        rec = _fake_record()
        cat = _fake_catalog(["8485-1901"])
        table = build.build_table([rec], cat)
        assert table.num_rows == 1
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "z", "spaxel_size",
                "spaxel_size_units", "spaxels", "images", "maps"} <= names

    def test_spaxels_struct_subfields(self):
        table = build.build_table([_fake_record()], _fake_catalog(["8485-1901"]))
        spx = table.schema.field("spaxels").type
        assert pa.types.is_struct(spx)
        names = {f.name for f in spx}
        assert {"flux", "ivar", "mask", "lsf", "lambda", "x", "y",
                "spaxel_idx", "flux_units", "lambda_units",
                "skycoo_x", "skycoo_y", "ellcoo_r", "ellcoo_rre",
                "ellcoo_rkpc", "ellcoo_theta"} <= names

    def test_images_struct_subfields(self):
        table = build.build_table([_fake_record()], _fake_catalog(["8485-1901"]))
        ims = table.schema.field("images").type
        assert pa.types.is_struct(ims)
        names = {f.name for f in ims}
        assert names == {"filter", "flux", "flux_units", "psf", "psf_units", "scale", "scale_units"}

    def test_maps_struct_subfields(self):
        table = build.build_table([_fake_record(n_maps=5)], _fake_catalog(["8485-1901"]))
        mps = table.schema.field("maps").type
        assert pa.types.is_struct(mps)
        names = {f.name for f in mps}
        assert names == {"group", "label", "flux", "ivar", "mask", "array_units"}

    def test_first_row_shapes(self):
        rec = _fake_record(n_maps=3)
        table = build.build_table([rec], _fake_catalog(["8485-1901"]))
        # Top-level row exposes the IFU's native shape.
        assert table.column("spatial_shape_y")[0].as_py() == NY
        assert table.column("spatial_shape_x")[0].as_py() == NX

        spx = table.column("spaxels")[0].as_py()
        # x/y/spaxel_idx are 2D (ny, nx) at native shape.
        assert len(spx["x"]) == NY
        assert len(spx["x"][0]) == NX
        # flux is 3D (ny, nx, nwave), spectral axis last.
        assert len(spx["flux"]) == NY
        assert len(spx["flux"][0]) == NX
        assert len(spx["flux"][0][0]) == NWAVE
        # lambda is shared per row (1D).
        assert len(spx["lambda"]) == NWAVE
        # Unit fields are scalar strings, not per-spaxel lists.
        assert spx["flux_units"] == "1e-17 erg/s/cm^2/Å"
        assert spx["lambda_units"] == "Angstrom"

        ims = table.column("images")[0].as_py()
        assert ims["filter"] == build.BANDS
        flux = np.asarray(ims["flux"])
        assert flux.shape == (build.N_BANDS, NY, NX)

        mps = table.column("maps")[0].as_py()
        assert len(mps["group"]) == 3
        assert mps["group"] == ["group_0", "group_1", "group_2"]
        flux_maps = np.asarray(mps["flux"])
        assert flux_maps.shape == (3, NY, NX)

    def test_drops_records_not_in_catalog(self):
        rec = _fake_record(plateifu="9999-9999")  # not in catalog
        with pytest.raises(RuntimeError, match="No records joined"):
            build.build_table([rec], _fake_catalog(["8485-1901"]))
