"""Convert HSC SSP PDR3 Deep/UltraDeep into a HATS catalog.

Reads the v1-cached parent catalog (already produced from HSC archive SQL
query — we don't re-run the query) plus the per-band calexp FITS images on
the polymathic mirror, and writes a HATS catalog matching v1's
``Sequence(struct<image>)`` schema as a struct-of-parallel-lists with plain
pyarrow types.

Cluster layout::

    /mnt/ceph/users/polymathic/MultimodalUniverse/hsc/pdr3_dud_22.5.fits
        # 477,104 rows, columns: object_id, ra, dec, tract, patch,
        # a_{g,r,i,z,y}, {g,r,i,z,y}_extendedness_value, _variance_value,
        # _cmodel_mag, _cmodel_magerr, _cmodel_flux, _cmodel_fluxerr,
        # _sdssshape_psf_shape{11,22,12}, _sdssshape_shape{11,22,12}

    /mnt/ceph/users/polymathic/external_data/astro/hsc/pdr3_dud/
        HSC-{G,R,I,Z,Y}/{tract}/{patch_x},{patch_y}/calexp-HSC-{band}-{tract}-{patch_x},{patch_y}.fits

Per (tract, patch) pool worker:
    1. Open the 5 calexp FITS files (one per band).
    2. WCS-cutout each catalog object to (160, 160) per band.
    3. Convert mask bitmask to boolean (clears bad-pixel/saturated/no-data bits).
    4. Compute PSF FWHM in arcsec from sdssshape_psf moments
       (``2.355 * (mxx*myy - mxy^2)^(1/4)``) per band.
    5. Emit one row per object with the image struct + all scalar features.

Schema::

    image: struct<
        band:     list<string>,                       # 5 entries per row
        flux:     list<list<list<float32>>>,          # (5, 160, 160)
        ivar:     list<list<list<float32>>>,          # (5, 160, 160)
        mask:     list<list<list<bool>>>,             # (5, 160, 160)
        psf_fwhm: list<float32>,                      # 5 floats
        scale:    list<float32>,                      # 5 floats (constant 0.168)
    >
    + 65 scalar float32 columns (the v1 _FLOAT_FEATURES + the columns v1
      commented out — per the "no normalization" rule we keep them all)
    + object_id (string), ra (float64), dec (float64)

The mask bit clean (bits 0/1/8 = bad/saturated/no-data) matches v1
``build_parent_sample.py``. The cmodel_mag/cmodel_flux columns are kept
even though v1 commented out the _flux versions, because the catalog has
them and we don't drop survey-native columns.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats_from_parquet_dir


CATALOG_NAME = "hsc"

IMAGE_SIZE = 160
PIXEL_SCALE = 0.168  # arcsec / pixel
BANDS = ["G", "R", "I", "Z", "Y"]
N_BANDS = len(BANDS)
BAND_FILTERS = [f"HSC-{b}" for b in BANDS]
# Mask bits we clear to produce a "clean pixel" boolean mask. v1 uses these
# three: 0 = BAD, 1 = SAT (saturated), 8 = NO_DATA. A cleared pixel reads
# True in the resulting mask, matching v1.
MASK_BITS_TO_CLEAR = (0, 1, 8)

# All 65 per-object scalar columns from the v1 catalog. Includes the
# *_variance_value, *_cmodel_flux, *_cmodel_fluxerr columns that v1's
# `_FLOAT_FEATURES` list commented out — per "no normalization" we keep
# every column the survey gave us.
FLOAT_FEATURES = [
    "a_g", "a_r", "a_i", "a_z", "a_y",
    "g_extendedness_value", "r_extendedness_value", "i_extendedness_value",
    "z_extendedness_value", "y_extendedness_value",
    "g_variance_value", "r_variance_value", "i_variance_value",
    "z_variance_value", "y_variance_value",
    "g_cmodel_mag", "g_cmodel_magerr",
    "r_cmodel_mag", "r_cmodel_magerr",
    "i_cmodel_mag", "i_cmodel_magerr",
    "z_cmodel_mag", "z_cmodel_magerr",
    "y_cmodel_mag", "y_cmodel_magerr",
    "g_cmodel_flux", "g_cmodel_fluxerr",
    "r_cmodel_flux", "r_cmodel_fluxerr",
    "i_cmodel_flux", "i_cmodel_fluxerr",
    "z_cmodel_flux", "z_cmodel_fluxerr",
    "y_cmodel_flux", "y_cmodel_fluxerr",
    "g_sdssshape_psf_shape11", "g_sdssshape_psf_shape22", "g_sdssshape_psf_shape12",
    "r_sdssshape_psf_shape11", "r_sdssshape_psf_shape22", "r_sdssshape_psf_shape12",
    "i_sdssshape_psf_shape11", "i_sdssshape_psf_shape22", "i_sdssshape_psf_shape12",
    "z_sdssshape_psf_shape11", "z_sdssshape_psf_shape22", "z_sdssshape_psf_shape12",
    "y_sdssshape_psf_shape11", "y_sdssshape_psf_shape22", "y_sdssshape_psf_shape12",
    "g_sdssshape_shape11", "g_sdssshape_shape22", "g_sdssshape_shape12",
    "r_sdssshape_shape11", "r_sdssshape_shape22", "r_sdssshape_shape12",
    "i_sdssshape_shape11", "i_sdssshape_shape22", "i_sdssshape_shape12",
    "z_sdssshape_shape11", "z_sdssshape_shape22", "z_sdssshape_shape12",
    "y_sdssshape_shape11", "y_sdssshape_shape22", "y_sdssshape_shape12",
]


def _b2s(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def _to_native(arr: np.ndarray) -> np.ndarray:
    """Make the buffer native byte order so PyArrow accepts it."""
    if arr.dtype.byteorder == ">":
        return arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return arr


def _patch_dir(patch: int) -> str:
    """HSC patches are encoded as ``patch_x*100 + patch_y`` integers in the
    catalog. The on-disk dir is ``"patch_x,patch_y"``."""
    return f"{patch // 100},{patch % 100}"


def patch_image_path(raw_root: str, band: str, tract: int, patch: int) -> str:
    """``calexp-HSC-{band}-{tract}-{patch_x},{patch_y}.fits`` under the
    per-band per-tract patch dir."""
    patch_str = _patch_dir(patch)
    return os.path.join(
        raw_root,
        f"HSC-{band}",
        str(tract),
        patch_str,
        f"calexp-HSC-{band}-{tract}-{patch_str}.fits",
    )


def load_catalog(catalog_path: str) -> Table:
    """Read the v1 SQL-query result FITS table."""
    return Table.read(catalog_path)


def _clean_mask(data: np.ndarray) -> np.ndarray:
    """Clear v1's bad-pixel/saturated/no-data bits and return a bool array.

    True = clean pixel (no flagged bits set), matching v1.
    """
    clean = np.ones_like(data, dtype=bool)
    for bit in MASK_BITS_TO_CLEAR:
        clean &= (data & (1 << bit)) == 0
    return clean


def _open_patch_images(raw_root: str, tract: int, patch: int) -> dict | None:
    """Open the 5-band calexp FITS files for one patch and return a dict
    keyed by band → ``{"image": HDU, "var": HDU, "mask": HDU}`` with the
    mask already converted to a boolean clean-pixel array.

    Returns ``None`` if any band's FITS file is missing or unreadable.
    """
    images: dict[str, dict] = {}
    for band in BANDS:
        path = patch_image_path(raw_root, band, tract, patch)
        if not os.path.exists(path):
            return None
        with fits.open(path) as hdul:
            image_hdu = hdul[1].copy()
            mask_hdu = hdul[2].copy()
            var_hdu = hdul[3].copy()
        mask_hdu.data = _clean_mask(mask_hdu.data).astype(mask_hdu.data.dtype)
        images[band] = {"image": image_hdu, "var": var_hdu, "mask": mask_hdu}
    return images


def _make_cutout(hdu: fits.ImageHDU, ra: float, dec: float) -> np.ndarray | None:
    """160×160 WCS cutout around (ra, dec). Returns ``None`` if the cutout
    falls partially outside the patch."""
    wcs = WCS(hdu.header)
    x, y = wcs.all_world2pix(ra, dec, 1)
    try:
        cut = Cutout2D(hdu.data, (x, y), (IMAGE_SIZE, IMAGE_SIZE), wcs=wcs)
    except Exception:  # noqa: BLE001
        return None
    if cut.data.shape != (IMAGE_SIZE, IMAGE_SIZE):
        return None
    return _to_native(np.asarray(cut.data))


def _psf_fwhm_arcsec(obj) -> np.ndarray:
    """v1's per-band PSF FWHM from sdssshape_psf moments::

        FWHM = 2.355 * (mxx * myy - mxy^2) ** (1/4)

    Returns a (5,) float32 array, NaN values replaced with 0.
    """
    out = np.zeros(N_BANDS, dtype=np.float32)
    for i, band in enumerate(BANDS):
        b = band.lower()
        mxx = float(obj[f"{b}_sdssshape_psf_shape11"])
        myy = float(obj[f"{b}_sdssshape_psf_shape22"])
        mxy = float(obj[f"{b}_sdssshape_psf_shape12"])
        det = mxx * myy - mxy * mxy
        out[i] = 2.355 * (det ** 0.25) if det > 0 else 0.0
    return np.nan_to_num(out, nan=0.0)


def process_patch(
    patch_catalog: Table,
    raw_root: str,
) -> list[dict]:
    """Read one (tract, patch) group's calexp FITS, cutout each catalog
    object, return a list of per-object record dicts. Returns ``[]`` on
    missing image files."""
    tract = int(patch_catalog["tract"][0])
    patch = int(patch_catalog["patch"][0])

    images = _open_patch_images(raw_root, tract, patch)
    if images is None:
        return []

    out: list[dict] = []
    for obj in patch_catalog:
        ra = float(obj["ra"])
        dec = float(obj["dec"])

        flux_cube = []
        ivar_cube = []
        mask_cube = []
        ok = True
        for band in BANDS:
            image_cut = _make_cutout(images[band]["image"], ra, dec)
            var_cut = _make_cutout(images[band]["var"], ra, dec)
            mask_cut = _make_cutout(images[band]["mask"], ra, dec)
            if image_cut is None or var_cut is None or mask_cut is None:
                ok = False
                break
            flux_cube.append(image_cut.astype(np.float32))
            # ivar = 1/var, but variance can be zero or NaN at masked pixels.
            with np.errstate(divide="ignore", invalid="ignore"):
                ivar = np.where(var_cut > 0, 1.0 / var_cut, 0.0).astype(np.float32)
            ivar_cube.append(np.nan_to_num(ivar, nan=0.0))
            mask_cube.append(mask_cut.astype(bool))
        if not ok:
            continue

        record = {
            "object_id": str(int(obj["object_id"])),
            "ra": ra,
            "dec": dec,
            "image_flux": np.stack(flux_cube),       # (5, 160, 160)
            "image_ivar": np.stack(ivar_cube),
            "image_mask": np.stack(mask_cube),
            "image_psf_fwhm": _psf_fwhm_arcsec(obj),
            "image_scale": np.full(N_BANDS, PIXEL_SCALE, dtype=np.float32),
        }
        for f in FLOAT_FEATURES:
            record[f] = float(obj[f]) if obj[f] is not np.ma.masked else 0.0
        out.append(record)
    return out


def _ndarray_to_nested_array(array: np.ndarray, value_type: pa.DataType) -> pa.Array:
    """Convert one ndarray (any rank) into a nested PyArrow list array.

    Same helper as the manga build — recursive ``pa.ListArray.from_arrays``
    around a flat values array, with offsets computed from the input shape.
    """
    arr = np.ascontiguousarray(array)
    values = pa.array(arr.reshape(-1), type=value_type)
    nested: pa.Array = values
    for dim in reversed(arr.shape):
        offsets = np.arange(0, len(nested) + 1, dim, dtype=np.int32)
        nested = pa.ListArray.from_arrays(offsets, nested)
    return nested


def _nested_column(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.Array:
    """Build a column where row i is the nested-list version of arrays[i].
    Per-row arrays don't need to share shape; they're concatenated as
    independent nested trees."""
    return pa.concat_arrays(
        [_ndarray_to_nested_array(a, value_type) for a in arrays]
    )


def _build_image_struct(records: list[dict]) -> pa.StructArray:
    """``image`` struct-of-parallel-lists: 5 bands per row, fixed (160, 160)
    cutout shape. flux/ivar are float32, mask is bool, psf_fwhm/scale are
    per-band float32 lists, band is a string list."""
    return pa.StructArray.from_arrays(
        [
            pa.array([list(BANDS) for _ in records], type=pa.list_(pa.string())),
            _nested_column([r["image_flux"] for r in records], pa.float32()),
            _nested_column([r["image_ivar"] for r in records], pa.float32()),
            _nested_column([r["image_mask"] for r in records], pa.bool_()),
            _nested_column([r["image_psf_fwhm"] for r in records], pa.float32()),
            _nested_column([r["image_scale"] for r in records], pa.float32()),
        ],
        names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"],
    )


def build_table(records: list[dict]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "image": _build_image_struct(records),
    }
    for f in FLOAT_FEATURES:
        columns[f] = pa.array([r[f] for r in records], type=pa.float32())
    return pa.table(columns)


def _process_patch_to_parquet(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: read one (tract, patch) group, write a parquet shard.

    Input tuple: ``(tract, patch, patch_catalog, raw_root, scratch)``.
    Returns ``(label, n_objects_written, err)``.
    """
    tract, patch, patch_catalog, raw_root, scratch = args
    label = f"{tract}-{_patch_dir(patch)}"
    out_path = os.path.join(scratch, f"part-{label}.parquet")
    if os.path.exists(out_path):
        return label, 0, None
    try:
        records = process_patch(patch_catalog, raw_root)
        if not records:
            return label, 0, None
        table = build_table(records)
        tmp_path = out_path + ".tmp"
        pq.write_table(table, tmp_path)
        os.rename(tmp_path, out_path)
        n = table.num_rows
        del records, table
        return label, n, None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        return label, 0, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Per-band calexp FITS root, containing HSC-{G,R,I,Z,Y}/<tract>/<x,y>/.",
    )
    parser.add_argument(
        "--catalog-path",
        default="/mnt/ceph/users/polymathic/MultimodalUniverse/hsc/pdr3_dud_22.5.fits",
        help="v1-cached SQL-query parent catalog FITS file (object_id + ra/dec + 65 scalars).",
    )
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of (tract,patch) groups to process.")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--num-processes", type=int, default=32,
                        help="Pool worker count. Each holds 5 calexp FITS HDUs "
                             "(~150 MB) plus per-object cutouts; 32 × ~200 MB ≈ 6 GB.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory. REQUIRED in sharded mode.")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-patch parquet shards.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Skip build phase; run only write_hats_from_parquet_dir.")
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args(argv)

    if args.skip_ingest and args.only_ingest:
        print("ERROR: --skip-ingest and --only-ingest are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print(f"ERROR: shard-idx {args.shard_idx} outside [0, {args.num_shards})",
              file=sys.stderr)
        return 2

    sharded_mode = args.num_shards > 1 or args.skip_ingest or args.only_ingest
    if sharded_mode and args.scratch_dir is None:
        print("ERROR: --scratch-dir is required in sharded mode", file=sys.stderr)
        return 2

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)

    try:
        # ----------------------------------------------------------------- #
        # Build phase.
        # ----------------------------------------------------------------- #
        if not args.only_ingest:
            print(f"[shard {args.shard_idx}/{args.num_shards}] "
                  f"Loading catalog from {args.catalog_path}", flush=True)
            catalog = load_catalog(args.catalog_path)
            print(f"  {len(catalog)} catalog rows", flush=True)

            if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
                cone_mask = apply_cone_filter(
                    np.asarray(catalog["ra"], dtype=np.float64),
                    np.asarray(catalog["dec"], dtype=np.float64),
                    args.ra_center, args.dec_center, args.radius,
                )
                catalog = catalog[cone_mask]
                print(f"  {len(catalog)} rows after cone cut", flush=True)
                if len(catalog) == 0:
                    print("  no HSC objects in cone", file=sys.stderr)
                    return 1

            # Group catalog by (tract, patch) — each pool worker handles one
            # patch's worth of catalog rows.
            catalog.sort(["tract", "patch"])
            tract_arr = np.asarray(catalog["tract"])
            patch_arr = np.asarray(catalog["patch"])
            tp = np.column_stack([tract_arr, patch_arr])
            _, group_starts = np.unique(tp, axis=0, return_index=True)
            group_starts = np.sort(group_starts)
            group_ends = np.concatenate([group_starts[1:], [len(catalog)]])
            groups = [
                (int(tract_arr[s]), int(patch_arr[s]), catalog[s:e])
                for s, e in zip(group_starts, group_ends)
            ]
            print(f"  {len(groups)} unique (tract, patch) groups", flush=True)

            if args.max_files is not None:
                groups = groups[:args.max_files]
                print(f"  capped to {len(groups)} groups via --max-files",
                      flush=True)

            if args.num_shards > 1:
                groups = groups[args.shard_idx::args.num_shards]
                print(f"[shard {args.shard_idx}] my stride: {len(groups)} groups",
                      flush=True)

            print(f"[shard {args.shard_idx}] streaming per-patch parquet to {scratch}",
                  flush=True)

            work = [
                (tract, patch, patch_cat, args.raw_root, scratch)
                for tract, patch, patch_cat in groups
            ]
            n_total = len(work)
            n_written = 0
            total = 0

            if args.num_processes > 1 and n_total > 1:
                with Pool(args.num_processes) as pool:
                    for i, (label, n, err) in enumerate(
                        pool.imap_unordered(_process_patch_to_parquet, work), 1
                    ):
                        if err:
                            print(f"[shard {args.shard_idx}] [{i}/{n_total}] "
                                  f"{label}: SKIPPED {err}",
                                  file=sys.stderr, flush=True)
                            continue
                        if n > 0:
                            n_written += 1
                            total += n
                        if (i % 50 == 0) or (i == n_total):
                            print(f"[shard {args.shard_idx}] [{i}/{n_total}] "
                                  f"{label}: {n} objects (cum {total})",
                                  flush=True)
            else:
                for i, item in enumerate(work, 1):
                    label, n, err = _process_patch_to_parquet(item)
                    if err:
                        print(f"[shard {args.shard_idx}] [{i}/{n_total}] "
                              f"{label}: SKIPPED {err}",
                              file=sys.stderr, flush=True)
                        continue
                    if n > 0:
                        n_written += 1
                        total += n

            print(f"[shard {args.shard_idx}] BUILD DONE: {n_written} patches, "
                  f"{total} total objects", flush=True)

        # ----------------------------------------------------------------- #
        # Ingest phase.
        # ----------------------------------------------------------------- #
        if args.skip_ingest:
            return 0

        print(f"Ingesting {scratch} → HATS catalog (workers={args.ingest_workers})",
              flush=True)
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
        if args.scratch_dir is None and not sharded_mode:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
