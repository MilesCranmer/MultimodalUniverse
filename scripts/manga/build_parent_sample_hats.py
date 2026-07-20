"""Convert raw SDSS-IV MaNGA into a HATS catalog.

Reads the same DR17 inputs as v1 MMU's ``scripts/manga/build_parent_sample.py``
but stores cubes / maps at their **native** spatial shape rather than v1's
psycho 96×96 zero-padding. Each row carries ``spatial_shape_y`` and
``spatial_shape_x`` so a reader can reconstruct the per-IFU geometry without
trusting a hard-coded constant. The spectral axis is moved to the **last**
position so per-spaxel spectra are contiguous in memory:

    spaxels.flux  : list<list<list<float32>>>   (ny, nx, nwave)
    spaxels.ivar  : list<list<list<float32>>>   (ny, nx, nwave)
    spaxels.mask  : list<list<list<int64>>>     (ny, nx, nwave)
    spaxels.lsf   : list<list<list<float32>>>   (ny, nx, nwave)
    spaxels.lambda: list<float32>               (nwave,)   shared per object
    spaxels.x, y  : list<list<int8>>            (ny, nx)
    spaxels.spaxel_idx          : list<list<int16>>   (ny, nx)
    spaxels.skycoo_x, skycoo_y  : list<list<float32>> (ny, nx)
    spaxels.ellcoo_{r,rre,rkpc,theta} : list<list<float32>> (ny, nx)
    spaxels.{flux,lambda,skycoo,ellcoo_*}_units : string  (scalar per row)

    images.flux/psf : list<list<list<float32>>>  (n_bands, ny, nx)
    images.{filter,flux_units,psf_units,scale,scale_units} : list<...>

    maps.flux/ivar/mask : list<list<list<float32>>>  (n_maps, ny, nx)
    maps.{group,label,array_units} : list<string>

    + ra, dec, object_id (=plateifu), z, spaxel_size, spaxel_size_units,
      spatial_shape_y, spatial_shape_x  (top-level scalars)

Pipeline:

    1. Read drpall + dapall, inner-join on plateifu, keep only DAPDONE rows.
    2. (Optional) cone-cut by ifura/ifudec.
    3. For each surviving plate-ifu, open LOGCUBE + DAP MAPS and assemble a
       per-row dict at the IFU's native spatial shape.
    4. Write per-plate-ifu parquet shards in the build phase, then run
       ``write_hats_from_parquet_dir`` for the ingest phase. Build/ingest
       split + multiprocessing pool match the other sharded datasets.
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
from astropy.table import Table, join

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats_from_parquet_dir


CATALOG_NAME = "manga"

N_BANDS = 4
BANDS = ["g", "r", "i", "z"]
SPAXEL_SIZE_ARCSEC = 0.5
DAPTYPE = "HYB10-MILESHC-MASTARSSP"


def _b2s(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def load_catalog(raw_root: str) -> Table:
    """Read drpall + dapall and inner-join on plate-ifu, keeping DAPDONE rows."""
    drpall = Table.read(os.path.join(raw_root, "drpall-v3_1_1.fits"), hdu="MANGA")
    dapall = Table.read(
        os.path.join(raw_root, "dapall-v3_1_1-3.1.0.fits"),
        hdu=DAPTYPE,
    )
    catalog = join(
        drpall, dapall,
        keys_left="plateifu", keys_right="PLATEIFU", join_type="inner",
    )
    catalog = catalog[np.asarray(catalog["DAPDONE"]).astype(bool)]
    return catalog


def cube_path(raw_root: str, plateifu: str) -> str:
    plate, _ = plateifu.split("-")
    return os.path.join(
        raw_root, "dr17", "manga", "spectro", "redux", "v3_1_1",
        plate, "stack", f"manga-{plateifu}-LOGCUBE.fits.gz",
    )


def maps_path(raw_root: str, plateifu: str) -> str:
    plate, ifu = plateifu.split("-")
    return os.path.join(
        raw_root, "dr17", "manga", "spectro", "analysis", "v3_1_1", "3.1.0",
        DAPTYPE, plate, ifu,
        f"manga-{plateifu}-MAPS-{DAPTYPE}.fits.gz",
    )


def _to_native(arr: np.ndarray) -> np.ndarray:
    """Make the buffer little-endian native so PyArrow accepts it."""
    if arr.dtype.byteorder == ">":
        return arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return arr


def _spectral_last(arr: np.ndarray, dtype) -> np.ndarray:
    """FITS LOGCUBE arrays come in as ``(nwave, ny, nx)``; move the spectral
    axis to the last position so per-spaxel spectra are contiguous."""
    arr = _to_native(np.asarray(arr, dtype=dtype))
    return np.moveaxis(arr, 0, -1)


def process_cube(
    plateifu: str,
    cube_file: str,
    map_file: str,
) -> dict | None:
    """Read one MaNGA plate-ifu (LOGCUBE + DAP MAPS) into a per-row dict.

    Returns ``None`` if either input file is missing.
    """
    if not os.path.exists(cube_file):
        return None
    if not os.path.exists(map_file):
        return None

    with fits.open(cube_file) as cube:
        flux = _spectral_last(cube["FLUX"].data, np.float32)        # (ny, nx, nwave)
        ivar = _spectral_last(cube["IVAR"].data, np.float32)
        mask = _spectral_last(cube["MASK"].data, np.int64)
        lsf  = _spectral_last(cube["LSFPOST"].data, np.float32)
        wave = _to_native(np.asarray(cube["WAVE"].data, dtype=np.float32))  # (nwave,)
        flux_units = _b2s(cube["FLUX"].header.get("BUNIT", ""))
        lambda_units = _b2s(cube["FLUX"].header.get("CUNIT3", ""))

        ny, nx, nwave = flux.shape

        # Per-spaxel index grids at native shape (no padding).
        yy, xx = np.indices((ny, nx))
        x_arr = xx.astype(np.int8)
        y_arr = yy.astype(np.int8)
        spaxel_idx = np.arange(ny * nx, dtype=np.int16).reshape(ny, nx)

        # Reconstructed griz images and PSFs at native shape (n_bands, ny, nx).
        img_stack = np.stack([
            _to_native(np.asarray(cube[f"{b.upper()}IMG"].data, dtype=np.float32))
            for b in BANDS
        ])
        psf_stack = np.stack([
            _to_native(np.asarray(cube[f"{b.upper()}PSF"].data, dtype=np.float32))
            for b in BANDS
        ])

    # DAP MAPS file: spaxel coordinate grids + analysis maps.
    with fits.open(map_file) as mapf:
        skycoo = _to_native(np.asarray(mapf["SPX_SKYCOO"].data, dtype=np.float32))  # (2, ny, nx)
        skycoo_units = _b2s(mapf["SPX_SKYCOO"].header.get("BUNIT", ""))

        ellcoo = _to_native(np.asarray(mapf["SPX_ELLCOO"].data, dtype=np.float32))  # (4, ny, nx)
        ellcoo_units = [
            _b2s(mapf["SPX_ELLCOO"].header.get(f"U{i}", ""))
            for i in range(1, 5)
        ]

        # Walk every non-PRIMARY, non-IVAR/MASK extension as a "map".
        maps_data: list[dict] = []
        for ext in mapf:
            if ext.name == "PRIMARY":
                continue
            if ext.name.endswith(("IVAR", "MASK")):
                continue
            arr = ext.data
            if arr is None:
                continue
            arr = _to_native(np.asarray(arr, dtype=np.float32))
            errdata = ext.header.get("ERRDATA")
            qualdata = ext.header.get("QUALDATA")
            if errdata and errdata in mapf:
                err = _to_native(np.asarray(mapf[errdata].data, dtype=np.float32))
            else:
                err = np.zeros_like(arr)
            if qualdata and qualdata in mapf:
                qual = _to_native(np.asarray(mapf[qualdata].data, dtype=np.float32))
            else:
                qual = np.full_like(arr, 1073741824.0)
            unit = _b2s(ext.header.get("BUNIT", ""))
            base_name = ext.name.lower()
            if arr.ndim == 3:
                # Multi-channel: emit one map per channel.
                for ch in range(arr.shape[0]):
                    chan_label = (
                        ext.header.get(f"C{ch + 1:02}", ext.header.get(f"C{ch + 1}", ""))
                    ).replace("-", "_").strip().replace(". ", "").replace(" ", "_")
                    chan_unit = (
                        ext.header.get(f"U{ch + 1:02}", ext.header.get(f"U{ch + 1}", ""))
                    ) or unit
                    maps_data.append({
                        "group": base_name,
                        "label": f"{base_name}_{chan_label.lower()}",
                        "flux": arr[ch],
                        "ivar": err[ch] if err.ndim == 3 else err,
                        "mask": qual[ch] if qual.ndim == 3 else qual,
                        "array_units": chan_unit,
                    })
            else:
                maps_data.append({
                    "group": base_name,
                    "label": base_name,
                    "flux": arr,
                    "ivar": err,
                    "mask": qual,
                    "array_units": unit,
                })

    return {
        "plateifu": plateifu,
        "spatial_shape_y": int(ny),
        "spatial_shape_x": int(nx),
        "spaxels": {
            "flux": flux, "ivar": ivar, "mask": mask, "lsf": lsf,
            "lambda": wave,
            "x": x_arr, "y": y_arr, "spaxel_idx": spaxel_idx,
            "flux_units": flux_units,
            "lambda_units": lambda_units,
            "skycoo_x": skycoo[0],
            "skycoo_y": skycoo[1],
            "ellcoo_r": ellcoo[0],
            "ellcoo_rre": ellcoo[1],
            "ellcoo_rkpc": ellcoo[2],
            "ellcoo_theta": ellcoo[3],
            "skycoo_units": skycoo_units,
            "ellcoo_r_units": ellcoo_units[0],
            "ellcoo_rre_units": ellcoo_units[1],
            "ellcoo_rkpc_units": ellcoo_units[2],
            "ellcoo_theta_units": ellcoo_units[3],
        },
        "images": {
            "filter": list(BANDS),
            "flux": img_stack,                  # (n_bands, ny, nx)
            "flux_units": ["nanomaggies/pixel"] * N_BANDS,
            "psf": psf_stack,
            "psf_units": ["nanomaggies/pixel"] * N_BANDS,
            "scale": [SPAXEL_SIZE_ARCSEC] * N_BANDS,
            "scale_units": ["arcsec"] * N_BANDS,
        },
        "maps": maps_data,
    }


def _ndarray_to_nested_array(array: np.ndarray, value_type: pa.DataType) -> pa.Array:
    """Convert one ndarray (any rank) into a nested PyArrow list array.

    The result has ``ndim`` levels of nesting matching the input shape.
    Used per-row; the caller stitches per-row arrays with ``pa.concat_arrays``.
    """
    arr = np.ascontiguousarray(array)
    values = pa.array(arr.reshape(-1), type=value_type)
    nested: pa.Array = values
    for dim in reversed(arr.shape):
        offsets = np.arange(0, len(nested) + 1, dim, dtype=np.int64)
        nested = pa.LargeListArray.from_arrays(offsets, nested)
    return nested


def _nested_column(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.Array:
    """Build a column where row i is the nested-list version of ``arrays[i]``.

    Each row's ndarray can have a different shape — they don't need to share
    any dimension, since each row is encoded into its own list-of-lists tree
    and we just concatenate the resulting per-row arrays.
    """
    return pa.concat_arrays(
        [_ndarray_to_nested_array(a, value_type) for a in arrays]
    )


def _build_spaxels_struct(records: list[dict]) -> pa.StructArray:
    """Build the spaxels struct column at native (ny, nx) shape per row.

    Float/int cube fields become ``list<list<list<...>>>`` (ny, nx, nwave) or
    ``list<list<...>>`` (ny, nx); ``lambda`` is shared per row as ``list<f32>``;
    unit fields are scalar strings (one value per row, not one per spaxel).
    """
    spx = [r["spaxels"] for r in records]
    return pa.StructArray.from_arrays(
        [
            _nested_column([s["flux"]   for s in spx], pa.float32()),  # (ny,nx,nwave)
            _nested_column([s["ivar"]   for s in spx], pa.float32()),
            _nested_column([s["mask"]   for s in spx], pa.int64()),
            _nested_column([s["lsf"]    for s in spx], pa.float32()),
            _nested_column([s["lambda"] for s in spx], pa.float32()),  # (nwave,)
            _nested_column([s["x"]            for s in spx], pa.int8()),    # (ny,nx)
            _nested_column([s["y"]            for s in spx], pa.int8()),
            _nested_column([s["spaxel_idx"]   for s in spx], pa.int16()),
            pa.array([s["flux_units"]    for s in spx], type=pa.string()),
            pa.array([s["lambda_units"]  for s in spx], type=pa.string()),
            _nested_column([s["skycoo_x"]    for s in spx], pa.float32()),
            _nested_column([s["skycoo_y"]    for s in spx], pa.float32()),
            _nested_column([s["ellcoo_r"]    for s in spx], pa.float32()),
            _nested_column([s["ellcoo_rre"]  for s in spx], pa.float32()),
            _nested_column([s["ellcoo_rkpc"] for s in spx], pa.float32()),
            _nested_column([s["ellcoo_theta"] for s in spx], pa.float32()),
            pa.array([s["skycoo_units"]       for s in spx], type=pa.string()),
            pa.array([s["ellcoo_r_units"]     for s in spx], type=pa.string()),
            pa.array([s["ellcoo_rre_units"]   for s in spx], type=pa.string()),
            pa.array([s["ellcoo_rkpc_units"]  for s in spx], type=pa.string()),
            pa.array([s["ellcoo_theta_units"] for s in spx], type=pa.string()),
        ],
        names=[
            "flux", "ivar", "mask", "lsf", "lambda",
            "x", "y", "spaxel_idx",
            "flux_units", "lambda_units",
            "skycoo_x", "skycoo_y",
            "ellcoo_r", "ellcoo_rre", "ellcoo_rkpc", "ellcoo_theta",
            "skycoo_units",
            "ellcoo_r_units", "ellcoo_rre_units",
            "ellcoo_rkpc_units", "ellcoo_theta_units",
        ],
    )


def _build_images_struct(records: list[dict]) -> pa.StructArray:
    """Images struct at native (n_bands, ny, nx) shape per row."""
    return pa.StructArray.from_arrays(
        [
            pa.array([r["images"]["filter"] for r in records], type=pa.list_(pa.string())),
            _nested_column([r["images"]["flux"] for r in records], pa.float32()),
            pa.array([r["images"]["flux_units"] for r in records], type=pa.list_(pa.string())),
            _nested_column([r["images"]["psf"] for r in records], pa.float32()),
            pa.array([r["images"]["psf_units"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["images"]["scale"] for r in records], type=pa.list_(pa.float32())),
            pa.array([r["images"]["scale_units"] for r in records], type=pa.list_(pa.string())),
        ],
        names=["filter", "flux", "flux_units", "psf", "psf_units", "scale", "scale_units"],
    )


def _maps_cube(record: dict, key: str, dtype) -> np.ndarray:
    """Stack a row's per-map 2D arrays into a single (n_maps, ny, nx) ndarray
    so it serializes as a single nested list-of-list-of-list field."""
    maps = record["maps"]
    if not maps:
        return np.zeros(
            (0, record["spatial_shape_y"], record["spatial_shape_x"]),
            dtype=dtype,
        )
    return np.stack([np.asarray(m[key], dtype=dtype) for m in maps], axis=0)


def _build_maps_struct(records: list[dict]) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array(
                [[m["group"] for m in r["maps"]] for r in records],
                type=pa.list_(pa.string()),
            ),
            pa.array(
                [[m["label"] for m in r["maps"]] for r in records],
                type=pa.list_(pa.string()),
            ),
            _nested_column([_maps_cube(r, "flux", np.float32) for r in records], pa.float32()),
            _nested_column([_maps_cube(r, "ivar", np.float32) for r in records], pa.float32()),
            _nested_column([_maps_cube(r, "mask", np.float32) for r in records], pa.float32()),
            pa.array(
                [[m["array_units"] for m in r["maps"]] for r in records],
                type=pa.list_(pa.string()),
            ),
        ],
        names=["group", "label", "flux", "ivar", "mask", "array_units"],
    )


def build_table(records: list[dict], catalog: Table) -> pa.Table:
    """Stitch the per-cube records back to the joined catalog and build the
    PyArrow table at native shape."""
    cat_lookup = {}
    plateifu_col = np.array([_b2s(p) for p in catalog["plateifu"]])
    for i, p in enumerate(plateifu_col):
        cat_lookup[p] = i

    keep_records: list[dict] = []
    keep_indices: list[int] = []
    for r in records:
        i = cat_lookup.get(_b2s(r["plateifu"]))
        if i is None:
            continue
        keep_records.append(r)
        keep_indices.append(i)
    if not keep_records:
        raise RuntimeError("No records joined to catalog rows")

    catalog_subset = catalog[np.array(keep_indices)]
    ra = np.asarray(catalog_subset["ifura"], dtype=np.float64)
    dec = np.asarray(catalog_subset["ifudec"], dtype=np.float64)
    z = np.asarray(catalog_subset["nsa_z"], dtype=np.float32)
    plateifu_arr = np.array([_b2s(p) for p in catalog_subset["plateifu"]])

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra),
        "dec": pa.array(dec),
        "object_id": pa.array(plateifu_arr, type=pa.string()),
        "z": pa.array(z, type=pa.float32()),
        "spaxel_size": pa.array(
            np.full(len(keep_records), SPAXEL_SIZE_ARCSEC, dtype=np.float32),
            type=pa.float32(),
        ),
        "spaxel_size_units": pa.array(
            ["arcsec"] * len(keep_records), type=pa.string()
        ),
        "spatial_shape_y": pa.array(
            [r["spatial_shape_y"] for r in keep_records], type=pa.int16(),
        ),
        "spatial_shape_x": pa.array(
            [r["spatial_shape_x"] for r in keep_records], type=pa.int16(),
        ),
        "spaxels": _build_spaxels_struct(keep_records),
        "images":  _build_images_struct(keep_records),
        "maps":    _build_maps_struct(keep_records),
    }
    return pa.table(columns)


def _process_cube_to_parquet(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: read one MaNGA cube+MAPS pair, build a 1-row pa.Table,
    and write it as ``part-<plateifu>.parquet`` under ``scratch``.

    Input tuple: ``(plateifu, cube_file, map_file, row_dict, scratch)``
    where ``row_dict`` is a dict of the catalog row fields build_table needs.

    Returns ``(plateifu, n_maps_written, err)``. ``err`` is ``None`` on
    success, a string on failure. All exceptions are converted to string
    errors (not raised) so one bad cube doesn't kill the pool.

    Must be module-level (not nested) for multiprocessing.Pool to pickle it.
    """
    plateifu, cube_file, map_file, row_dict, scratch = args
    try:
        rec = process_cube(plateifu, cube_file, map_file)
        if rec is None:
            return plateifu, 0, "missing cube or map file"
        # Reconstruct a single-row astropy Table from the dict for build_table.
        row_table = Table({k: [v] for k, v in row_dict.items()})
        table = build_table([rec], row_table)
        out_path = os.path.join(scratch, f"part-{plateifu}.parquet")
        tmp_path = out_path + ".tmp"
        pq.write_table(table, tmp_path)
        os.rename(tmp_path, out_path)
        n_maps = len(rec["maps"])
        del rec, table, row_table
        return plateifu, n_maps, None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        return plateifu, 0, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of plate-ifus to process.")
    parser.add_argument("--pixel-threshold", type=int, default=100_000,
                        help="Max rows per HATS partition. Default 100k — manga has "
                             "only ~10k objects so this produces ~1 partition, which "
                             "avoids the massive reduce overhead of 8192.")
    parser.add_argument("--num-processes", type=int, default=32,
                        help="Per-shard multiprocessing.Pool size. Default 32 "
                             "(not 96) because each worker holds a full IFU cube ~500 MB.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory for per-plate-ifu parquet "
                             "files. REQUIRED in sharded mode (--num-shards>1 / "
                             "--skip-ingest / --only-ingest). Single-job mode auto-"
                             "generates a pid-namespaced dir that's wiped on exit.")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="This shard's stride offset (0..num-shards-1).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-plate-ifu parquet shards.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Skip build phase; run only write_hats_from_parquet_dir.")
    parser.add_argument("--ingest-workers", type=int, default=96,
                        help="Dask workers for the ingest step.")
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

    sharded_mode = args.num_shards > 1 or args.skip_ingest or args.only_ingest
    if sharded_mode and args.scratch_dir is None:
        print("ERROR: --scratch-dir is required in sharded mode", file=sys.stderr)
        return 2

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)

    try:
        # ----------------------------------------------------------------- #
        # Build phase: write per-plate-ifu parquet shards.
        # ----------------------------------------------------------------- #
        if not args.only_ingest:
            print(f"[shard {args.shard_idx}/{args.num_shards}] "
                  f"Loading drpall + dapall from {args.raw_root}", flush=True)
            catalog = load_catalog(args.raw_root)
            print(f"  {len(catalog)} plate-ifus after DAPDONE join", flush=True)

            if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
                mask = apply_cone_filter(
                    np.asarray(catalog["ifura"], dtype=np.float64),
                    np.asarray(catalog["ifudec"], dtype=np.float64),
                    args.ra_center, args.dec_center, args.radius,
                )
                catalog = catalog[mask]
                print(f"  {len(catalog)} plate-ifus after cone cut", flush=True)
                if len(catalog) == 0:
                    print("  no MaNGA targets in cone — nothing to write", file=sys.stderr)
                    return 1

            if args.max_files is not None:
                catalog = catalog[:args.max_files]
                print(f"  capped to {len(catalog)} plate-ifus via --max-files", flush=True)

            if args.num_shards > 1:
                # Stride-slice the catalog for this shard.
                keep_idx = np.arange(args.shard_idx, len(catalog), args.num_shards)
                catalog = catalog[keep_idx]
                print(f"[shard {args.shard_idx}] my stride: {len(catalog)} plate-ifus",
                      flush=True)

            print(f"[shard {args.shard_idx}] streaming per-plate-ifu parquet to {scratch}",
                  flush=True)

            n_written = 0
            work = []
            for row in catalog:
                plateifu = _b2s(row["plateifu"])
                cube_file = cube_path(args.raw_root, plateifu)
                map_file = maps_path(args.raw_root, plateifu)
                row_dict = {col: row[col] for col in catalog.colnames}
                work.append((plateifu, cube_file, map_file, row_dict, scratch))

            if args.num_processes > 1 and len(work) > 1:
                with Pool(args.num_processes) as pool:
                    for i, (plateifu, n_maps, err) in enumerate(
                        pool.imap_unordered(_process_cube_to_parquet, work), 1
                    ):
                        if err:
                            print(f"[shard {args.shard_idx}] [{i}/{len(work)}] "
                                  f"{plateifu}: SKIPPED {err}", file=sys.stderr, flush=True)
                            continue
                        n_written += 1
                        if n_written % 20 == 0 or n_written == len(work):
                            print(f"[shard {args.shard_idx}] [{i}/{len(work)}] "
                                  f"{plateifu}: {n_maps} maps", flush=True)
            else:
                for i, item in enumerate(work, 1):
                    plateifu, n_maps, err = _process_cube_to_parquet(item)
                    if err:
                        print(f"[shard {args.shard_idx}] [{i}/{len(work)}] "
                              f"{plateifu}: SKIPPED {err}", file=sys.stderr, flush=True)
                        continue
                    n_written += 1

            print(f"[shard {args.shard_idx}] BUILD DONE: {n_written} plate-ifus",
                  flush=True)

        # ----------------------------------------------------------------- #
        # Ingest phase.
        # ----------------------------------------------------------------- #
        if args.skip_ingest:
            return 0

        import glob as _glob
        n_parquets = len(_glob.glob(os.path.join(scratch, "*.parquet")))
        print(f"=== MANGA GATHER CONFIG ===", flush=True)
        print(f"  scratch: {scratch} ({n_parquets} parquets)", flush=True)
        print(f"  output: {args.output_root}", flush=True)
        print(f"  pixel_threshold: {args.pixel_threshold}", flush=True)
        print(f"  ingest_workers: {args.ingest_workers}", flush=True)
        print(f"  mem alloc: {os.environ.get('SLURM_MEM_PER_NODE', 'unknown')} MB", flush=True)
        print(f"  time limit: {os.environ.get('SLURM_TIMELIMIT', 'unknown')}", flush=True)
        print(f"===========================", flush=True)

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
