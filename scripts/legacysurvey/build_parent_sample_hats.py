"""Convert raw DECaLS DR10 South into a HATS catalog.

Matches v1 MMU's ``scripts/legacysurvey/build_parent_sample.py`` exactly,
with HATS output instead of HDF5:

    1. Read the sweep catalogs (``dr10/south/sweep/10.1/sweep-*.fits``).
       Each filename encodes its RA/Dec bounding box, so when a cone cut is
       active we can skip every sweep that doesn't intersect the cone.
    2. Apply v1 ``select_observations``: z-mag < 21, NOBS_{G,R,I,Z} > 0,
       TYPE != "PSF", clean MASKBITS bits.
    3. (If a cone cut is active) trim the catalog to the cone.
    4. For each surviving object, group by BRICKNAME, open the brick coadd
       FITS files (image-{g,r,i,z}, invvar-{g,r,i,z}, maskbits) and the
       blobmodel/RGB JPEGs, and extract a 160×160 cutout per object via WCS.
    5. For each cutout, also build:
         - the brightest-N nearby-object catalog within the cutout
         - the elliptical object mask painted from those nearby objects
         - the cleaned binary mask from MASKBITS
    6. Bundle each per-object record into a row of a single PyArrow table
       matching v1's HuggingFace `Features(...)` declaration:

           image: struct<
               band:     list<string>,
               flux:     list<list<list<float32>>>,   # (4, 160, 160)
               ivar:     list<list<list<float32>>>,
               mask:     list<list<list<bool>>>,
               psf_fwhm: list<float32>,
               scale:    list<float32>,
           >
           blobmodel:   list<list<list<uint8>>>     # (3, 160, 160) JPEG cutout
           rgb:         list<list<list<uint8>>>     # (3, 160, 160) JPEG cutout
           object_mask: list<list<uint8>>           # (160, 160) painted mask
           catalog: struct<
               FLUX_G/R/I/Z, TYPE, SHAPE_R/E1/E2, X, Y: list<float32>
           >
           ra, dec, object_id (= BRICKNAME-OBJID), + 12 photometric scalars

       Plain nested lists (no extension types anywhere) — see
       ``project_image_storage`` memory for why nested_pandas crashes
       when HF ``datasets`` extension types live inside list/struct.
    7. Write the HATS catalog via ``mmu.hats_import.write_hats``.

Cluster layout::

    /mnt/ceph/users/polymathic/external_data/astro/legacysurvey/dr10/south/
        sweep/10.1/sweep-{ra1}{p|m}{dec1}-{ra2}{p|m}{dec2}.fits
        coadd/{brick3}/{brickname}/legacysurvey-{brickname}-image-{g,r,i,z}.fits.fz
                                  /legacysurvey-{brickname}-invvar-{g,r,i,z}.fits.fz
                                  /legacysurvey-{brickname}-maskbits.fits.fz
                                  /legacysurvey-{brickname}-image.jpg
                                  /legacysurvey-{brickname}-blobmodel.jpg

The sweep filename format is ``sweep-RRRsDDD-RRRsDDD.fits`` where ``RRR`` is
RA in degrees zero-padded to 3 digits, ``s`` is ``p`` (positive) or ``m``
(negative), and ``DDD`` is ``|Dec|`` in degrees zero-padded to 3 digits.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import skimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from PIL import Image, ImageOps

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats_from_parquet_dir


CATALOG_NAME = "legacysurvey"

ARCSEC_PER_PIXEL = 0.262
IMAGE_SIZE = 160
N_BANDS = 4
BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]
PSF_KEYS = ["PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_I", "PSFSIZE_Z"]
NEARBY_CATALOG_N = 20  # cap on # of nearby-object rows kept per cutout

# v1 maskbit cleanup, both for the catalog selection and the per-pixel mask.
SELECTION_MASKBITS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 14, 15]

# Top-level scalar columns from the sweep catalog (v1 HF schema).
FLOAT_FEATURES = [
    "EBV",
    "FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z",
    "FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4",
    "SHAPE_R", "SHAPE_E1", "SHAPE_E2",
]

# Per-cutout nearby-object catalog (v1 HF schema, in the order v1 stores them).
CATALOG_FEATURES = [
    "FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z",
    "TYPE",
    "SHAPE_R", "SHAPE_E1", "SHAPE_E2",
    "X", "Y",
]

# v1 OBJECT_TYPE_COLOR maps morphological TYPE strings to small integers; we
# use the same mapping so the painted-object-mask values are comparable.
OBJECT_TYPE_COLOR = {
    name: i for i, name in enumerate(["PSF", "REX", "EXP", "DEV", "SER", "DUP"], start=1)
}

# Sweep filename: e.g. sweep-150p000-155p005.fits → ra (150,155), dec (+0,+5).
SWEEP_RE = re.compile(
    r"sweep-(\d{3})([pm])(\d{3})-(\d{3})([pm])(\d{3})\.fits$"
)


def parse_sweep_bbox(filename: str) -> tuple[float, float, float, float] | None:
    """Return ``(ra_min, ra_max, dec_min, dec_max)`` parsed from a sweep filename.

    Returns ``None`` if the filename doesn't match the standard pattern.
    """
    m = SWEEP_RE.search(os.path.basename(filename))
    if not m:
        return None
    ra_min = float(m.group(1))
    dec_min = float(m.group(3)) * (1 if m.group(2) == "p" else -1)
    ra_max = float(m.group(4))
    dec_max = float(m.group(6)) * (1 if m.group(5) == "p" else -1)
    return ra_min, ra_max, dec_min, dec_max


def _bbox_intersects_cone(
    bbox: tuple[float, float, float, float],
    ra_center: float,
    dec_center: float,
    radius: float,
) -> bool:
    """Return True if a (ra_min, ra_max, dec_min, dec_max) sweep bbox could
    contain any source within ``radius`` degrees of the cone center."""
    ra_min, ra_max, dec_min, dec_max = bbox
    # Pad by radius/cos(dec) in RA to be conservative near the box edges.
    cos_d = max(np.cos(np.deg2rad(max(abs(dec_min), abs(dec_max)))), 1e-3)
    if dec_center < dec_min - radius:
        return False
    if dec_center > dec_max + radius:
        return False
    if ra_center < ra_min - radius / cos_d:
        return False
    if ra_center > ra_max + radius / cos_d:
        return False
    return True


def find_sweep_files(
    raw_root: str,
    max_files: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> list[str]:
    """Glob ``dr10/south/sweep/10.1/sweep-*.fits`` and (optionally) skip any
    file whose RA/Dec bounding box doesn't intersect a cone cut."""
    sweep_dir = os.path.join(raw_root, "dr10", "south", "sweep", "10.1")
    files = sorted(glob.glob(os.path.join(sweep_dir, "sweep-*.fits")))
    cone_active = (
        ra_center is not None and dec_center is not None and radius is not None
    )
    if cone_active:
        keep: list[str] = []
        for f in files:
            bbox = parse_sweep_bbox(f)
            if bbox is None:
                continue
            if _bbox_intersects_cone(bbox, ra_center, dec_center, radius):
                keep.append(f)
        files = keep
    if max_files is not None:
        files = files[:max_files]
    return files


def select_observations(catalog: Table, zmag_cut: float = 21.0) -> np.ndarray:
    """v1 selection: zmag < 21, all 4 bands observed, non-PSF, clean maskbits."""
    flux_z = np.asarray(catalog["FLUX_Z"], dtype=np.float64)
    mw_z = np.asarray(catalog["MW_TRANSMISSION_Z"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        zmag = 22.5 - 2.5 * np.log10(flux_z / mw_z)
    mask_mag = zmag < zmag_cut

    nobs = np.array([np.asarray(catalog[f"NOBS_{b}"]) for b in ("G", "R", "I", "Z")]).T
    mask_obs = ~np.any(nobs == 0, axis=1)

    type_col = np.array([
        s.decode().strip() if isinstance(s, (bytes, np.bytes_)) else str(s).strip()
        for s in catalog["TYPE"]
    ])
    mask_type = type_col != "PSF"

    mb = np.asarray(catalog["MASKBITS"], dtype=np.int64)
    mask_clean = np.ones(len(catalog), dtype=bool)
    for bit in SELECTION_MASKBITS:
        mask_clean &= (mb & (1 << bit)) == 0

    return mask_mag & mask_obs & mask_type & mask_clean


def read_sweep(
    path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> Table | None:
    """Read one sweep FITS, apply v1 selection, then optionally cone cut.

    Returns ``None`` if no rows survive.
    """
    catalog = Table.read(path)
    catalog = catalog[select_observations(catalog)]
    if len(catalog) == 0:
        return None
    if ra_center is not None and dec_center is not None and radius is not None:
        mask = apply_cone_filter(
            np.asarray(catalog["RA"], dtype=np.float64),
            np.asarray(catalog["DEC"], dtype=np.float64),
            ra_center, dec_center, radius,
        )
        catalog = catalog[mask]
    if len(catalog) == 0:
        return None
    return catalog


def _clean_maskbits(data: np.ndarray) -> np.ndarray:
    """v1 mask cleanup: 0 → masked-bad, 1 → good. Same bit set as selection."""
    out = np.ones_like(data, dtype=bool)
    for bit in SELECTION_MASKBITS:
        out &= (data & (1 << bit)) == 0
    return out


def _load_brick_images(brick_path: str, brick_name: str) -> dict | None:
    """Load all per-band image/invvar HDUs and the maskbits HDU for one brick.

    Returns None if any required file is missing.
    """
    images: dict[str, fits.ImageHDU] = {}
    for kind in ("image", "invvar"):
        for b in ("g", "r", "i", "z"):
            fname = os.path.join(
                brick_path, f"legacysurvey-{brick_name}-{kind}-{b}.fits.fz"
            )
            if not os.path.exists(fname):
                return None
            with fits.open(fname) as hdul:
                images[f"{kind}-{b}"] = hdul[1].copy()
    mb_path = os.path.join(brick_path, f"legacysurvey-{brick_name}-maskbits.fits.fz")
    if not os.path.exists(mb_path):
        return None
    with fits.open(mb_path) as hdul:
        mb_hdu = hdul[1].copy()
    mb_hdu.data = _clean_maskbits(mb_hdu.data).astype(mb_hdu.data.dtype)
    images["maskbits"] = mb_hdu
    return images


def _load_brick_jpeg(path: str) -> np.ndarray | None:
    """Load a brick-level JPEG (rgb or blobmodel) and return as (C, H, W) uint8."""
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    img = ImageOps.flip(img)
    arr = np.asarray(img, dtype=np.uint8)
    return np.moveaxis(arr, -1, 0)  # (H, W, C) → (C, H, W)


def _build_nearby_catalog(
    brick_catalog: Table,
    cutout: Cutout2D,
    n_objects: int = NEARBY_CATALOG_N,
) -> dict[str, list[float]]:
    """Build a CATALOG_FEATURES dict of the brightest nearby objects in the
    cutout, padded to ``n_objects`` rows. Mirrors v1 ``CatalogSelector``.
    """
    catalog_coords = SkyCoord(
        ra=brick_catalog["RA"], dec=brick_catalog["DEC"], unit="deg",
    )
    cutout_world_center = cutout.wcs.wcs_pix2world(*cutout.input_position_cutout, 1)
    cutout_center = SkyCoord(
        ra=float(cutout_world_center[0]),
        dec=float(cutout_world_center[1]),
        unit="deg",
    )

    # First filter by sky separation roughly equal to the cutout half-diagonal.
    separations = cutout_center.separation(catalog_coords).degree
    pixel_separations = separations * 3600.0 / ARCSEC_PER_PIXEL
    radius_pix = np.sqrt(2.0) * np.linalg.norm(cutout.shape) / 2
    close_mask = pixel_separations < radius_pix
    if not close_mask.any():
        # Fall back to empty record padded with zeros.
        return {key: [0.0] * n_objects for key in CATALOG_FEATURES}
    close_catalog = brick_catalog[close_mask]
    close_coords = catalog_coords[close_mask]

    # Then keep only those whose pixel coordinates fall inside the cutout.
    pixel_i, pixel_j = cutout.wcs.world_to_array_index(close_coords)
    bbox = cutout.bbox_cutout
    in_cutout = (
        (pixel_j >= bbox[1][0]) & (pixel_j < bbox[1][1])
        & (pixel_i >= bbox[0][0]) & (pixel_i < bbox[0][1])
    )
    in_cutout_catalog = close_catalog[in_cutout]
    in_cutout_coords = close_coords[in_cutout]
    if len(in_cutout_catalog) == 0:
        return {key: [0.0] * n_objects for key in CATALOG_FEATURES}

    # Sort brightest first by FLUX_I (descending — largest flux = brightest).
    flux_i = np.asarray(in_cutout_catalog["FLUX_I"], dtype=np.float64)
    order = np.argsort(-flux_i)
    in_cutout_catalog = in_cutout_catalog[order][:n_objects]
    in_cutout_coords = in_cutout_coords[order][:n_objects]

    out: dict[str, list[float]] = {key: [] for key in CATALOG_FEATURES}
    for obj, coord in zip(in_cutout_catalog, in_cutout_coords):
        x_pix_i, y_pix_i = cutout.wcs.world_to_array_index(coord)
        for key in CATALOG_FEATURES:
            if key == "TYPE":
                t = obj["TYPE"]
                if isinstance(t, (bytes, np.bytes_)):
                    t = t.decode()
                out[key].append(float(OBJECT_TYPE_COLOR.get(str(t).strip(), 0)))
            elif key == "X":
                out[key].append(float(y_pix_i))
            elif key == "Y":
                out[key].append(float(x_pix_i))
            else:
                out[key].append(float(obj[key]))

    # Pad to n_objects with zeros.
    for key in CATALOG_FEATURES:
        while len(out[key]) < n_objects:
            out[key].append(0.0)
    return out


def _build_object_mask(
    nearby_catalog: dict[str, list[float]],
    cutout_shape: tuple[int, int],
) -> np.ndarray:
    """Paint elliptical apertures of the nearby objects into a uint8 mask.

    Pixel value = OBJECT_TYPE_COLOR[type]. Mirrors v1 ``get_object_mask``.
    """
    mask = np.zeros(cutout_shape, dtype=np.uint8)
    for i in range(NEARBY_CATALOG_N):
        x = nearby_catalog["X"][i]
        y = nearby_catalog["Y"][i]
        type_color = nearby_catalog["TYPE"][i]
        if type_color == 0:
            continue  # padding row
        radius = nearby_catalog["SHAPE_R"][i] / ARCSEC_PER_PIXEL
        e1 = nearby_catalog["SHAPE_E1"][i]
        e2 = nearby_catalog["SHAPE_E2"][i]
        e_mag = np.sqrt(e1 * e1 + e2 * e2)
        if e_mag > 0.999:
            continue  # degenerate ellipse
        angle = 0.5 * np.arctan2(e2, e1)
        q = (1 - e_mag) / (1 + e_mag)
        height = max(2 * radius, 1)
        width = max(2 * radius * q, 1)
        rr, cc = skimage.draw.ellipse(
            int(x), int(y), width, height, shape=cutout_shape, rotation=angle,
        )
        mask[cc, rr] = int(type_color)
    return mask


def process_brick(
    brick_catalog: Table,
    raw_root: str,
) -> list[dict] | None:
    """Open all brick files for one brick group of the catalog and emit one
    record per surviving object."""
    brick_name = (
        brick_catalog["BRICKNAME"][0].decode().strip()
        if isinstance(brick_catalog["BRICKNAME"][0], (bytes, np.bytes_))
        else str(brick_catalog["BRICKNAME"][0]).strip()
    )
    brick_group = brick_name[:3]
    brick_path = os.path.join(raw_root, "dr10", "south", "coadd", brick_group, brick_name)
    if not os.path.isdir(brick_path):
        return None

    images = _load_brick_images(brick_path, brick_name)
    if images is None:
        return None

    rgb = _load_brick_jpeg(
        os.path.join(brick_path, f"legacysurvey-{brick_name}-image.jpg")
    )
    blob = _load_brick_jpeg(
        os.path.join(brick_path, f"legacysurvey-{brick_name}-blobmodel.jpg")
    )
    if rgb is None or blob is None:
        return None

    wcs_g = WCS(images["image-g"].header)

    out_records: list[dict] = []
    for obj in brick_catalog:
        ra = float(obj["RA"])
        dec = float(obj["DEC"])
        x, y = wcs_g.all_world2pix(ra, dec, 1)
        position = (x, y)
        size = (IMAGE_SIZE, IMAGE_SIZE)

        try:
            cutout_g = Cutout2D(images["image-g"].data, position, size, wcs=wcs_g)
        except Exception:
            continue
        if cutout_g.data.shape != size:
            continue

        flux_cube = []
        for b in ("g", "r", "i", "z"):
            try:
                c = Cutout2D(images[f"image-{b}"].data, position, size, wcs=wcs_g)
            except Exception:
                c = None
            if c is None or c.data.shape != size:
                flux_cube = None
                break
            flux_cube.append(c.data)
        if flux_cube is None:
            continue

        ivar_cube = []
        for b in ("g", "r", "i", "z"):
            try:
                c = Cutout2D(images[f"invvar-{b}"].data, position, size, wcs=wcs_g)
            except Exception:
                c = None
            if c is None or c.data.shape != size:
                ivar_cube = None
                break
            ivar_cube.append(c.data)
        if ivar_cube is None:
            continue

        try:
            mb_cutout = Cutout2D(images["maskbits"].data, position, size, wcs=wcs_g).data
        except Exception:
            continue
        if mb_cutout.shape != size:
            continue

        # Per-band stack into (4, 160, 160). The maskbits coadd is single-band,
        # but v1 stores it as a 2D bool with no per-band dimension. We replicate
        # to 4 bands so the schema mirrors flux/ivar shape exactly.
        mask_cube = np.broadcast_to(mb_cutout.astype(bool), (N_BANDS, *size)).copy()

        # Three-channel JPEG cutouts (blobmodel + RGB), each (3, 160, 160).
        rgb_cutout_channels = []
        blob_cutout_channels = []
        for ch in range(rgb.shape[0]):
            try:
                rgb_cutout_channels.append(
                    Cutout2D(rgb[ch], position, size, wcs=wcs_g).data
                )
            except Exception:
                rgb_cutout_channels = None
                break
        if rgb_cutout_channels is None or len(rgb_cutout_channels) == 0:
            continue
        for ch in range(blob.shape[0]):
            try:
                blob_cutout_channels.append(
                    Cutout2D(blob[ch], position, size, wcs=wcs_g).data
                )
            except Exception:
                blob_cutout_channels = None
                break
        if blob_cutout_channels is None or len(blob_cutout_channels) == 0:
            continue
        rgb_cutout = np.stack(rgb_cutout_channels).astype(np.uint8)
        blob_cutout = np.stack(blob_cutout_channels).astype(np.uint8)
        if rgb_cutout.shape[1:] != size or blob_cutout.shape[1:] != size:
            continue

        # Nearby-object catalog + painted object mask.
        nearby = _build_nearby_catalog(brick_catalog, cutout_g)
        object_mask = _build_object_mask(nearby, size)

        psf_fwhm = np.array([float(obj[k]) for k in PSF_KEYS], dtype=np.float32)
        scale = np.array([ARCSEC_PER_PIXEL] * N_BANDS, dtype=np.float32)

        try:
            obj_id = obj["OBJID"]
            if isinstance(obj_id, (bytes, np.bytes_)):
                obj_id = obj_id.decode()
            object_id = f"{brick_name}-{int(obj_id)}"
        except Exception:
            continue

        out_records.append({
            "ra": ra,
            "dec": dec,
            "object_id": object_id,
            "image_band": list(BANDS),
            "image_flux": np.stack(flux_cube).astype(np.float32),
            "image_ivar": np.stack(ivar_cube).astype(np.float32),
            "image_mask": mask_cube,
            "image_psf_fwhm": psf_fwhm,
            "image_scale": scale,
            "rgb": rgb_cutout,
            "blobmodel": blob_cutout,
            "object_mask": object_mask,
            "nearby_catalog": nearby,
            "scalars": {f: float(obj[f]) for f in FLOAT_FEATURES},
        })

    return out_records


def _as_array(a: pa.Array | pa.ChunkedArray) -> pa.Array:
    """pa.array(...) on a large nested Python list can return a ChunkedArray,
    which pa.StructArray.from_arrays rejects with 'Expected Array, got
    ChunkedArray'. Force it to a contiguous Array via combine_chunks().
    """
    if isinstance(a, pa.ChunkedArray):
        return a.combine_chunks()
    return a


def _build_image_struct(records: list[dict]) -> pa.StructArray:
    band_arr = pa.array([r["image_band"] for r in records], type=pa.list_(pa.string()))

    def nest_4d(arrs: list[np.ndarray]) -> list:
        return [
            [[list(row) for row in band_img] for band_img in cube]
            for cube in arrs
        ]

    flux_arr = pa.array(
        nest_4d([r["image_flux"] for r in records]),
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    ivar_arr = pa.array(
        nest_4d([r["image_ivar"] for r in records]),
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    mask_arr = pa.array(
        [
            [[list(row) for row in band_img] for band_img in r["image_mask"]]
            for r in records
        ],
        type=pa.large_list(pa.large_list(pa.large_list(pa.bool_()))),
    )
    psf_arr = pa.array(
        [list(r["image_psf_fwhm"]) for r in records], type=pa.list_(pa.float32())
    )
    scale_arr = pa.array(
        [list(r["image_scale"]) for r in records], type=pa.list_(pa.float32())
    )
    return pa.StructArray.from_arrays(
        [_as_array(a) for a in
         (band_arr, flux_arr, ivar_arr, mask_arr, psf_arr, scale_arr)],
        names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"],
    )


def _build_jpeg_column(records: list[dict], key: str) -> pa.Array:
    """Build a list<list<list<uint8>>> column for a JPEG cutout (3, H, W)."""
    nested = [
        [[list(row) for row in channel] for channel in r[key]]
        for r in records
    ]
    return pa.array(nested, type=pa.large_list(pa.large_list(pa.large_list(pa.uint8()))))


def _build_object_mask_column(records: list[dict]) -> pa.Array:
    nested = [[list(row) for row in r["object_mask"]] for r in records]
    return pa.array(nested, type=pa.large_list(pa.large_list(pa.uint8())))


def _build_nearby_catalog_struct(records: list[dict]) -> pa.StructArray:
    arrays = []
    names = []
    for key in CATALOG_FEATURES:
        arrays.append(
            _as_array(
                pa.array(
                    [r["nearby_catalog"][key] for r in records],
                    type=pa.list_(pa.float32()),
                )
            )
        )
        names.append(key)
    return pa.StructArray.from_arrays(arrays, names=names)


def build_table(records: list[dict]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "image": _build_image_struct(records),
        "rgb": _build_jpeg_column(records, "rgb"),
        "blobmodel": _build_jpeg_column(records, "blobmodel"),
        "object_mask": _build_object_mask_column(records),
        "catalog": _build_nearby_catalog_struct(records),
    }
    for f in FLOAT_FEATURES:
        columns[f] = pa.array(
            [r["scalars"][f] for r in records], type=pa.float32()
        )
    return pa.table(columns)


def _process_sweep_to_parquet(args: tuple) -> tuple[str, int, int, str | None]:
    """Pool worker: process ONE sweep file into ONE parquet shard under ``scratch``.

    Input tuple: ``(sweep_path, raw_root, scratch, ra_center, dec_center, radius)``
    Returns ``(sweep_basename, n_catalog_rows, n_cutouts_written, err)`` where
    ``err`` is ``None`` on success or a stringified error on failure.

    ALL exceptions are caught and converted to string error messages so the
    pool worker always returns a picklable value. Some exception types raised
    deep in astropy/cfitsio are dynamically generated and CANNOT be pickled;
    letting them propagate crashes the whole Pool, which is how an earlier
    run of this job died on one corrupted brick FITS file.

    Must be module-level (not nested) so multiprocessing.Pool can pickle it.
    """
    sweep_path, raw_root, scratch, ra_center, dec_center, radius = args
    basename = os.path.basename(sweep_path)
    out_path = os.path.join(scratch, f"part-{basename}.parquet")
    if os.path.exists(out_path):
        n_rows = pq.read_metadata(out_path).num_rows
        return basename, 0, n_rows, "already done"
    try:
        cat = read_sweep(
            sweep_path,
            ra_center=ra_center,
            dec_center=dec_center,
            radius=radius,
        )
        if cat is None:
            return basename, 0, 0, None

        brickname_col = np.array([
            s.decode().strip() if isinstance(s, (bytes, np.bytes_)) else str(s).strip()
            for s in cat["BRICKNAME"]
        ])
        unique_bricks = np.unique(brickname_col)
        # One parquet per sweep (not per-256-cutouts). With Pool(8), peak
        # memory is ~8 sweeps × 80 GB/sweep ≈ 640 GB, well within the
        # 900 GB slurm allocation. This produces ~1436 parquet files for
        # the full survey instead of the 231k files the per-256 approach
        # generated (ceph hates small files).
        sweep_records: list[dict] = []
        brick_errors: list[str] = []
        n_cat = len(cat)

        n_bricks = len(unique_bricks)
        for bi, brick in enumerate(unique_bricks):
            brick_cat = cat[brickname_col == brick]
            try:
                recs = process_brick(brick_cat, raw_root)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                brick_errors.append(f"{brick}: {type(exc).__name__}: {exc}")
                continue
            if recs:
                sweep_records.extend(recs)
            if (bi + 1) % 50 == 0 or bi == n_bricks - 1:
                sys.stderr.write(
                    f"  {basename}: brick {bi+1}/{n_bricks}, "
                    f"{len(sweep_records)} cutouts so far\n"
                )
                sys.stderr.flush()
            del recs

        del cat, brickname_col, unique_bricks

        err_summary = (
            f"{len(brick_errors)} brick error(s): " + "; ".join(brick_errors[:3])
            + (" ..." if len(brick_errors) > 3 else "")
        ) if brick_errors else None

        if not sweep_records:
            return basename, n_cat, 0, err_summary

        table = build_table(sweep_records)
        del sweep_records
        # Atomic write: write to .tmp then rename so a crash mid-write
        # doesn't leave a corrupt parquet that skip-if-exists treats as done.
        tmp_path = out_path + ".tmp"
        pq.write_table(table, tmp_path)
        os.rename(tmp_path, out_path)
        n_cutouts = table.num_rows
        del table
        return basename, n_cat, n_cutouts, err_summary
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        return basename, 0, 0, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of sweep files to process.")
    parser.add_argument("--pixel-threshold", type=int, default=100_000)
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Pool size. Default 1 — big sweeps peak at 588 GB in "
                             "build_table (196k cutouts × 1 MB × 3x). Pool(2)+ OOMs.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory for per-sweep parquet "
                             "shards. REQUIRED when running sharded — all shard "
                             "jobs write here and the ingest job reads from here. "
                             "When unset (single-job mode), defaults to a "
                             "pid-namespaced dir that's wiped on exit.")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="This shard's stride offset (0..num-shards-1). Sweeps "
                             "are assigned via sweeps[shard_idx::num_shards].")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (matches the number of "
                             "independent sbatch jobs in the sharded launcher).")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-sweep parquet shards; do NOT run "
                             "write_hats_from_parquet_dir. Used by shard jobs.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Skip the sweep-processing phase; run ONLY "
                             "write_hats_from_parquet_dir against --scratch-dir. "
                             "Used by the dependent ingest job.")
    parser.add_argument("--ingest-workers", type=int, default=96,
                        help="Dask workers for the ingest step (used with "
                             "--only-ingest and the single-job fallback).")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    if args.skip_ingest and args.only_ingest:
        print("ERROR: --skip-ingest and --only-ingest are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print(f"ERROR: shard-idx {args.shard_idx} outside [0, {args.num_shards})",
              file=sys.stderr)
        return 2

    # Single-job mode: auto-generate a pid-namespaced scratch dir and clean
    # it up on exit. Sharded mode: caller MUST provide --scratch-dir (shared
    # across all shard jobs); we never delete a caller-provided scratch dir.
    sharded_mode = args.num_shards > 1 or args.skip_ingest or args.only_ingest
    if sharded_mode and args.scratch_dir is None:
        print("ERROR: --scratch-dir is required in sharded mode "
              "(--num-shards>1 / --skip-ingest / --only-ingest)",
              file=sys.stderr)
        return 2

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)

    try:
        # ----------------------------------------------------------------- #
        # Build phase: write per-sweep parquet shards to the shared scratch.
        # ----------------------------------------------------------------- #
        if not args.only_ingest:
            t0 = time.time()
            sweeps = find_sweep_files(
                args.raw_root,
                max_files=args.max_files,
                ra_center=args.ra_center,
                dec_center=args.dec_center,
                radius=args.radius,
            )
            if not sweeps:
                print(f"ERROR: no sweep files match under {args.raw_root}",
                      file=sys.stderr)
                return 1
            print(f"[shard {args.shard_idx}/{args.num_shards}] "
                  f"Found {len(sweeps)} total sweep file(s)", flush=True)

            if args.num_shards > 1:
                sweeps = sweeps[args.shard_idx::args.num_shards]
                print(f"[shard {args.shard_idx}] my stride: {len(sweeps)} sweep(s)",
                      flush=True)

            print(f"[shard {args.shard_idx}] streaming per-sweep parquet to {scratch}",
                  flush=True)

            work = [
                (sw, args.raw_root, scratch, args.ra_center, args.dec_center, args.radius)
                for sw in sweeps
            ]
            total = 0
            n_written = 0
            n_total = len(work)
            sweep_start = time.time()

            def _report(i: int, result: tuple) -> None:
                nonlocal sweep_start
                basename, n_cat, n_cut, err = result
                elapsed = time.time() - sweep_start
                sweep_start = time.time()
                # `err` is now a WARNING summary if the sweep had some bad
                # bricks but also some good ones (n_cut > 0). The sweep is
                # NOT skipped in that case; we still write its good-brick
                # parquet. Only skip logging when n_cut == 0 AND err is set.
                if err and n_cut == 0:
                    print(f"[shard {args.shard_idx}] [{i}/{n_total}] {basename}: "
                          f"SKIPPED {err} ({elapsed:.0f}s)", file=sys.stderr, flush=True)
                    return
                if err:
                    print(f"[shard {args.shard_idx}] [{i}/{n_total}] {basename}: "
                          f"{n_cat} cat rows → {n_cut} cutouts ({elapsed:.0f}s) "
                          f"(WARN: {err})", file=sys.stderr, flush=True)
                    return
                if n_cut == 0:
                    return
                rate = n_cut / elapsed if elapsed > 0 else 0
                print(f"[shard {args.shard_idx}] [{i}/{n_total}] {basename}: "
                      f"{n_cat} cat rows → {n_cut} cutouts in {elapsed:.0f}s "
                      f"({rate:.1f} cutouts/s)", flush=True)

            if args.num_processes > 1 and n_total > 1:
                with Pool(args.num_processes) as pool:
                    for i, result in enumerate(
                        pool.imap_unordered(_process_sweep_to_parquet, work), 1
                    ):
                        _report(i, result)
                        _, _, n_cut, _ = result
                        if n_cut > 0:
                            n_written += 1
                            total += n_cut
            else:
                for i, item in enumerate(work, 1):
                    result = _process_sweep_to_parquet(item)
                    _report(i, result)
                    _, _, n_cut, _ = result
                    if n_cut > 0:
                        n_written += 1
                        total += n_cut

            dt = time.time() - t0
            print(
                f"[shard {args.shard_idx}] BUILD DONE: {n_written} sweeps with "
                f"cutouts, {total} total cutouts in {dt:.1f}s "
                f"({n_written / dt:.2f} sweeps/s)",
                flush=True,
            )

        # ----------------------------------------------------------------- #
        # Ingest phase: read the shared scratch and write the final HATS.
        # ----------------------------------------------------------------- #
        if args.skip_ingest:
            return 0

        print(f"Ingesting scratch dir {scratch} → HATS catalog "
              f"(dask workers={args.ingest_workers})", flush=True)
        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
            n_workers=args.ingest_workers,
            debug=False,
        )
        print(f"Done: {catalog_dir}", flush=True)
    finally:
        # Only wipe auto-generated single-job scratch dirs. Never touch a
        # caller-provided shared scratch (the other shard jobs / the ingest
        # job may still need it).
        if args.scratch_dir is None and not sharded_mode:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
