"""Convert COSMOS-Web JWST mosaics into a HATS catalog.

JWST filters only (as specified in make_stamps.py):
    F115W, F150W, F277W, F444W  — NIRCam, 30 mas/pix
    F770W                        — MIRI,   60 mas/pix

Selection: 0 < MAG_MODEL_F277W < 27

Data layout on the Candide n17/n03 filesystem:
    Catalog: /n03data/huertas/COSMOS-Web/cats/
             COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits
    NIRCam:  /n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/
             mosaic_nircam_{filt}_COSMOS-Web_30mas_{tile}_v0_8_sci.fits
             mosaic_nircam_{filt}_COSMOS-Web_30mas_{tile}_v0_8_wht.fits  [optional]
    MIRI:    /n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7/
             mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_sci.fits
             mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_wht.fits    [optional]

Processing: one Pool worker per COSMOS-Web tile (A1–A10, B1–B10).
Each worker opens the 5 filter mosaics for its tile (memory-mapped), cuts
IMAGE_SIZE×IMAGE_SIZE stamps for every catalog object in that tile, and
writes a parquet shard to scratch_dir.

Schema::

    image: struct<
        band:     list<string>,                    # 5 entries per row
        flux:     list<list<list<float32>>>,       # (5, 96, 96)
        ivar:     list<list<list<float32>>>,       # (5, 96, 96); synthetic if no wht
        mask:     list<list<list<bool>>>,          # True = valid pixel
        psf_fwhm: list<float32>,                   # arcsec per band
        scale:    list<float32>,                   # arcsec/pix per band
    >
    + ra (float64), dec (float64)
    + object_id (string, from ID_SE++)
    + FLOAT_FEATURES scalar columns (float32)
"""

from __future__ import annotations

import argparse
import glob
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
from mmu.hats_configs import MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, to_native_endian, write_hats_from_parquet_dir


CATALOG_NAME = "cosmos"

# Default data paths on the Flatiron n17/n03 servers (same as make_stamps.py)
DEFAULT_CATALOG_PATH = (
    "/n03data/huertas/COSMOS-Web/cats/"
    "COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits"
)
DEFAULT_NIRCAM_ROOT = "/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8"
DEFAULT_MIRI_ROOT   = "/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7"

IMAGE_SIZE = 96  # pixels, fixed for all bands

# JWST filters to include (F770W is MIRI; the rest are NIRCam).
FILTERS = ["F115W", "F150W", "F277W", "F444W", "F770W"]
MIRI_FILTERS = {"F770W"}

# Native pixel scale (arcsec/pixel) per filter.
PIXEL_SCALE = {
    "F115W": 0.03,
    "F150W": 0.03,
    "F277W": 0.03,
    "F444W": 0.03,
    "F770W": 0.06,
}

# Empirical PSF FWHM (arcsec). NIRCam values from DJA/COSMOS-Web release notes;
# MIRI F770W from Rigby et al. 2023.
EMPIRICAL_PSF_FWHM = {
    "F115W": 0.040,
    "F150W": 0.050,
    "F277W": 0.092,
    "F444W": 0.145,
    "F770W": 0.269,
}

# Magnitude cut applied to the detection filter.
MAG_CUT_FILTER = "MAG_MODEL_F277W"
MAG_CUT        = 27.0

# All 20 COSMOS-Web tiles from make_stamps.py
ALL_TILES = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)]

# Objects per parquet chunk written during tile processing.  Keeping this
# small limits peak RAM: the intermediate nested-Python-list representation
# of one chunk is ~2×CHUNK_SIZE×5×96×96×24 bytes ≈ 1.1 GB at 500 objects.
CHUNK_SIZE = 500

# Catalog scalar columns written to HATS. Subset the caller knows exist in
# the COSMOSWeb v3.1.0 catalog. Missing columns are silently skipped.
FLOAT_FEATURES = [
    "RADIUS",
    "AXRATIO",
    "MAG_MODEL_F115W",
    "MAG_MODEL_F150W",
    "MAG_MODEL_F277W",
    "MAG_MODEL_F444W",
    "MAG_MODEL_F770W",
]


# ── File-path helpers ────────────────────────────────────────────────────────


def _sci_path(tile: str, filt: str, nircam_root: str, miri_root: str) -> str:
    filt_l = filt.lower()
    if filt in MIRI_FILTERS:
        return os.path.join(
            miri_root,
            f"mosaic_miri_{filt_l}_COSMOS-Web_60mas_{tile}_v0_7_sci.fits",
        )
    return os.path.join(
        nircam_root,
        f"mosaic_nircam_{filt_l}_COSMOS-Web_30mas_{tile}_v0_8_sci.fits",
    )


def _wht_path(tile: str, filt: str, nircam_root: str, miri_root: str) -> str:
    return _sci_path(tile, filt, nircam_root, miri_root).replace("_sci.fits", "_wht.fits")


# ── Low-level FITS helpers ────────────────────────────────────────────────────


def _read_image(path: str, is_miri: bool) -> tuple[np.ndarray, fits.Header]:
    """Read science data from a FITS file, returning (float32 array, header).

    MIRI files store data in extension 1; NIRCam in extension 0.
    Falls back to extension 0 if extension 1 is absent.
    Uses memory-mapping so the OS loads only the pages touched by cutouts.
    """
    with fits.open(path, memmap=True) as hdul:
        ext = 1 if (is_miri and len(hdul) > 1 and hdul[1].data is not None) else 0
        data   = to_native_endian(np.asarray(hdul[ext].data, dtype=np.float32))
        header = hdul[ext].header.copy()
    return data, header


def _make_cutout(
    data: np.ndarray,
    wcs: WCS,
    ra: float,
    dec: float,
) -> np.ndarray:
    """Extract an IMAGE_SIZE×IMAGE_SIZE stamp centred on (ra, dec).

    Uses ``mode="partial"`` so edge objects get zero-padded instead of
    raising an exception, matching the JWST builder's behaviour.
    """
    x, y = wcs.all_world2pix(ra, dec, 0)
    cut = Cutout2D(
        data,
        (x, y),
        (IMAGE_SIZE, IMAGE_SIZE),
        wcs=wcs,
        mode="partial",
        fill_value=0.0,
    )
    return np.nan_to_num(cut.data, nan=0.0).astype(np.float32)


# ── Per-tile processing ───────────────────────────────────────────────────────


def _load_tile_images(
    tile: str,
    nircam_root: str,
    miri_root: str,
) -> dict | None:
    """Open all 5 filter images for one tile.

    Returns a dict keyed by filter name with entries::

        {"sci": np.ndarray, "ivar": np.ndarray, "wcs": WCS,
         "scale": float32, "psf_fwhm": float32}

    Returns None if any required science file is missing.
    """
    images: dict[str, dict] = {}
    for filt in FILTERS:
        sci_file = _sci_path(tile, filt, nircam_root, miri_root)
        if not os.path.exists(sci_file):
            print(f"  [{tile}] missing {filt}: {sci_file}", file=sys.stderr)
            return None

        is_miri = filt in MIRI_FILTERS
        sci, header = _read_image(sci_file, is_miri)
        wcs = WCS(header)

        wht_file = _wht_path(tile, filt, nircam_root, miri_root)
        if os.path.exists(wht_file):
            ivar, _ = _read_image(wht_file, is_miri)
        else:
            # Synthetic ivar: 1 where data is non-zero/finite, 0 otherwise.
            ivar = np.where(np.isfinite(sci) & (sci != 0), 1.0, 0.0).astype(np.float32)

        images[filt] = {
            "sci":      sci,
            "ivar":     ivar,
            "wcs":      wcs,
            "scale":    np.float32(PIXEL_SCALE[filt]),
            "psf_fwhm": np.float32(EMPIRICAL_PSF_FWHM[filt]),
        }
    return images


def _decode_bytes(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def _flush_chunk(
    chunk: list[dict],
    tile: str,
    chunk_idx: int,
    scratch_dir: str,
    float_features: list[str],
) -> int:
    """Write one chunk of records to a parquet file and return the row count."""
    path = os.path.join(scratch_dir, f"cosmos_{tile}_{chunk_idx:04d}.parquet")
    table = _build_table(chunk, float_features)
    tmp = path + ".tmp"
    pq.write_table(table, tmp)
    os.rename(tmp, path)
    n = table.num_rows
    del table
    return n


def _process_tile(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: build all object records for one COSMOS-Web tile.

    Records are flushed to parquet every CHUNK_SIZE objects so the
    intermediate nested-Python-list representation in _build_table never
    exceeds ~1 GB per worker regardless of tile size.

    A ``cosmos_{tile}.done`` sentinel is written on success; its presence
    on the next run causes the tile to be skipped and its rows re-counted
    from the existing chunk files.

    Input tuple::
        (tile, tile_rows, nircam_root, miri_root, scratch_dir,
         float_features, ra_center, dec_center, radius)

    Returns ``(tile, n_rows_written, error_string_or_None)``.
    """
    (tile, tile_rows, nircam_root, miri_root, scratch_dir,
     float_features, ra_center, dec_center, radius) = args

    done_marker = os.path.join(scratch_dir, f"cosmos_{tile}.done")
    if os.path.exists(done_marker):
        existing = sorted(glob.glob(os.path.join(scratch_dir, f"cosmos_{tile}_*.parquet")))
        n = sum(pq.read_metadata(p).num_rows for p in existing)
        return tile, n, "already done"

    # Remove any partial chunks from a previous failed run so we don't
    # accumulate duplicate rows on retry.
    for stale in glob.glob(os.path.join(scratch_dir, f"cosmos_{tile}_*.parquet")):
        os.remove(stale)

    try:
        images = _load_tile_images(tile, nircam_root, miri_root)
        if images is None:
            return tile, 0, "missing image files"

        chunk:      list[dict] = []
        chunk_idx  = 0
        total      = 0
        n_cat      = len(tile_rows)

        for ri in range(n_cat):
            row = tile_rows[ri]
            try:
                ra  = float(row["RA_MODEL"])
                dec = float(row["DEC_MODEL"])
            except (KeyError, TypeError):
                continue

            flux_stack: list[np.ndarray] = []
            ivar_stack: list[np.ndarray] = []
            mask_stack: list[np.ndarray] = []
            bands:      list[str]       = []
            psf_vals:   list[float]     = []
            scale_vals: list[float]     = []

            ok = True
            for filt in FILTERS:
                img = images[filt]
                try:
                    flux_cut = _make_cutout(img["sci"],  img["wcs"], ra, dec)
                    ivar_cut = _make_cutout(img["ivar"], img["wcs"], ra, dec)
                except Exception:
                    ok = False
                    break
                flux_stack.append(flux_cut)
                ivar_stack.append(ivar_cut)
                mask_stack.append((ivar_cut > 0))
                bands.append(filt)
                psf_vals.append(float(img["psf_fwhm"]))
                scale_vals.append(float(img["scale"]))

            if not ok or not flux_stack:
                continue

            try:
                obj_id_raw = row["ID_SE++"]
                obj_id = str(int(obj_id_raw))
            except (KeyError, TypeError, ValueError):
                obj_id = f"{tile}_{ri}"

            record: dict = {
                "object_id":    obj_id,
                "ra":           np.float64(ra),
                "dec":          np.float64(dec),
                "image_band":   bands,
                "image_flux":   np.stack(flux_stack).astype(np.float32),
                "image_ivar":   np.stack(ivar_stack).astype(np.float32),
                "image_mask":   np.stack(mask_stack).astype(bool),
                "image_psf_fwhm": np.array(psf_vals,   dtype=np.float32),
                "image_scale":    np.array(scale_vals,  dtype=np.float32),
            }
            for feat in float_features:
                try:
                    v = row[feat]
                    record[feat] = np.float32(v) if v is not np.ma.masked else np.float32(0.0)
                except (KeyError, TypeError):
                    record[feat] = np.float32(0.0)
            chunk.append(record)

            if len(chunk) >= CHUNK_SIZE:
                total += _flush_chunk(chunk, tile, chunk_idx, scratch_dir, float_features)
                chunk_idx += 1
                chunk = []

            if (ri + 1) % 2000 == 0 or ri == n_cat - 1:
                print(f"  [{tile}] {ri+1}/{n_cat} → {total + len(chunk)} cutouts",
                      flush=True)

        # Flush the final partial chunk.
        if chunk:
            total += _flush_chunk(chunk, tile, chunk_idx, scratch_dir, float_features)

        if total == 0:
            return tile, 0, "no cutouts produced"

        # Mark tile as fully done so reruns skip it.
        open(done_marker, "w").close()
        return tile, total, None

    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        return tile, 0, f"{type(exc).__name__}: {exc}"


# ── PyArrow table assembly ────────────────────────────────────────────────────


def _as_array(a: pa.Array | pa.ChunkedArray) -> pa.Array:
    return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a


def _build_image_struct(records: list[dict]) -> pa.StructArray:
    band_arr = pa.array(
        [r["image_band"] for r in records], type=pa.list_(pa.string())
    )
    flux_arr = pa.array(
        [[[list(row) for row in band] for band in r["image_flux"]] for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    ivar_arr = pa.array(
        [[[list(row) for row in band] for band in r["image_ivar"]] for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    mask_arr = pa.array(
        [[[list(map(bool, row)) for row in band] for band in r["image_mask"]]
         for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.bool_()))),
    )
    psf_arr = pa.array(
        [r["image_psf_fwhm"].tolist() for r in records], type=pa.list_(pa.float32())
    )
    scale_arr = pa.array(
        [r["image_scale"].tolist() for r in records], type=pa.list_(pa.float32())
    )
    return pa.StructArray.from_arrays(
        [_as_array(a) for a in [band_arr, flux_arr, ivar_arr, mask_arr, psf_arr, scale_arr]],
        names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"],
    )


def _build_table(records: list[dict], float_features: list[str]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "ra":        pa.array(
            np.array([r["ra"] for r in records], dtype=np.float64), type=pa.float64()
        ),
        "dec":       pa.array(
            np.array([r["dec"] for r in records], dtype=np.float64), type=pa.float64()
        ),
        "image": _build_image_struct(records),
    }
    for feat in float_features:
        columns[feat] = pa.array(
            np.array([r.get(feat, 0.0) for r in records], dtype=np.float32),
            type=pa.float32(),
        )
    return pa.table(columns)


# ── Catalog loading ───────────────────────────────────────────────────────────


def load_catalog(catalog_path: str) -> Table:
    """Read the COSMOSWeb FITS catalog, keeping only scalar columns."""
    cat = Table.read(catalog_path, format="fits")
    scalar_cols = [c for c in cat.colnames if len(cat[c].shape) <= 1]
    return cat[scalar_cols]


def filter_catalog(cat: Table, float_features: list[str]) -> tuple[Table, list[str]]:
    """Apply MAG_MODEL_F277W < 27 selection and resolve available float features."""
    mag_col = MAG_CUT_FILTER
    if mag_col not in cat.colnames:
        raise ValueError(f"Selection column '{mag_col}' not found in catalog. "
                         f"Available: {cat.colnames[:20]} ...")

    mag = np.asarray(cat[mag_col], dtype=np.float64)
    mask = (mag > 0) & (mag < MAG_CUT)
    cat = cat[mask]
    print(f"  {np.sum(mask)} objects pass {mag_col} < {MAG_CUT}", flush=True)

    # Keep only float features that actually exist in the catalog.
    available = [f for f in float_features if f in cat.colnames]
    missing   = [f for f in float_features if f not in cat.colnames]
    if missing:
        print(f"  WARNING: float features not found in catalog, skipping: {missing}",
              file=sys.stderr)
    return cat, available


def group_by_tile(cat: Table) -> dict[str, Table]:
    """Return a dict mapping tile name → subset of catalog rows for that tile."""
    tile_col = np.array([_decode_bytes(v) for v in cat["TILE"]])
    groups: dict[str, Table] = {}
    for tile in np.unique(tile_col):
        groups[tile] = cat[tile_col == tile]
    return groups


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-path",
        default=DEFAULT_CATALOG_PATH,
        help="Path to the COSMOSWeb master FITS catalog.",
    )
    parser.add_argument(
        "--nircam-root",
        default=DEFAULT_NIRCAM_ROOT,
        help="Directory containing NIRCam v0.8 mosaic science FITS files.",
    )
    parser.add_argument(
        "--miri-root",
        default=DEFAULT_MIRI_ROOT,
        help="Directory containing MIRI Full v0.7 mosaic science FITS files.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Directory for per-tile parquet shards. Auto-created if unset.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap on number of tiles to process (useful for smoke-tests).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Pool size. Each worker loads 5 FITS mosaics for one tile. "
             "Memory per worker ≈ 2–10 GB depending on tile size.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ingest-workers",  type=int, default=8)
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Parent directory for hats-import's intermediate files. "
             "Defaults to the system /tmp, which may be too small on "
             "cluster nodes. Set to a path on a large shared filesystem "
             "(e.g. the same parent as --scratch-dir).",
    )
    parser.add_argument("--ra-center",  type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius",     type=float, default=None,
                        help="Cone radius in degrees.")
    args = parser.parse_args(argv)

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    auto_scratch = args.scratch_dir is None

    try:
        # ── Load & filter catalog ──────────────────────────────────────────
        print(f"Loading catalog from {args.catalog_path}", flush=True)
        catalog = load_catalog(args.catalog_path)
        print(f"  {len(catalog)} total catalog rows", flush=True)

        catalog, float_features = filter_catalog(catalog, FLOAT_FEATURES)

        if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
            cone_mask = apply_cone_filter(
                np.asarray(catalog["RA_MODEL"], dtype=np.float64),
                np.asarray(catalog["DEC_MODEL"], dtype=np.float64),
                args.ra_center, args.dec_center, args.radius,
            )
            catalog = catalog[cone_mask]
            print(f"  {len(catalog)} rows after cone cut", flush=True)
            if len(catalog) == 0:
                print("ERROR: no COSMOS objects in cone", file=sys.stderr)
                return 1

        # ── Group by tile ─────────────────────────────────────────────────
        tile_groups = group_by_tile(catalog)
        tiles = sorted(tile_groups.keys())
        print(f"  {len(tiles)} tiles with objects: {tiles}", flush=True)

        if args.max_files is not None:
            tiles = tiles[:args.max_files]
            print(f"  capped to {len(tiles)} tile(s) via --max-files", flush=True)

        work = [
            (
                tile,
                tile_groups[tile],
                args.nircam_root,
                args.miri_root,
                scratch,
                float_features,
                args.ra_center,
                args.dec_center,
                args.radius,
            )
            for tile in tiles
        ]

        # ── Build phase: one parquet shard per tile ───────────────────────
        print(f"Processing {len(work)} tile(s) with Pool({args.num_processes}) "
              f"→ {scratch}", flush=True)

        n_total   = len(work)
        n_written = 0
        total     = 0

        if args.num_processes > 1 and n_total > 1:
            with Pool(args.num_processes) as pool:
                for i, (tile, n, err) in enumerate(
                    pool.imap_unordered(_process_tile, work), 1
                ):
                    if err and err != "already done":
                        print(f"  [{i}/{n_total}] {tile}: SKIPPED — {err}",
                              file=sys.stderr, flush=True)
                        continue
                    if n > 0:
                        n_written += 1
                        total     += n
                    print(f"  [{i}/{n_total}] {tile}: {n} objects written "
                          f"(cum {total})", flush=True)
        else:
            for i, item in enumerate(work, 1):
                tile, n, err = _process_tile(item)
                if err and err != "already done":
                    print(f"  [{i}/{n_total}] {tile}: SKIPPED — {err}",
                          file=sys.stderr, flush=True)
                    continue
                if n > 0:
                    n_written += 1
                    total     += n
                print(f"  [{i}/{n_total}] {tile}: {n} objects", flush=True)

        print(f"BUILD DONE: {n_written} tiles, {total} total objects", flush=True)

        if total == 0:
            print("ERROR: no objects were written", file=sys.stderr)
            return 1

        # ── Ingest phase: parquet → HATS ──────────────────────────────────
        print(f"Ingesting {scratch} → HATS (workers={args.ingest_workers})", flush=True)
        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
            n_workers=args.ingest_workers,
            debug=False,
            tmp_dir=args.tmp_dir,
        )
        print(f"Done: {catalog_dir}", flush=True)

    finally:
        if auto_scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
