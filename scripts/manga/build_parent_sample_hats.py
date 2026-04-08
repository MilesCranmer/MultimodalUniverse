"""Build a MaNGA HATS catalog directly from DR17 FITS products."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, join
from cdshealpix import lonlat_to_healpix
from dask.distributed import Client

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats_from_parquet_dir


CATALOG_NAME = "manga"
DAPTYPE = "HYB10-MILESHC-MASTARSSP"
SPAXEL_SIZE_ARCSEC = 0.5
HEALPIX_DEPTH = 4
MAP_MASK_FILL_VALUE = 1073741824.0
BANDS = ["g", "r", "i", "z"]


def _b2s(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def _to_native(array: np.ndarray, dtype=None) -> np.ndarray:
    """Return ``array`` in native byte order."""
    arr = np.asarray(array, dtype=dtype)
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _normalize_channel_name(value: str) -> str:
    return value.replace("-", "_").strip().replace(". ", "").replace(" ", "_")


def _header_indexed_value(header, prefix: str, index: int, default="") -> str:
    """Read ``U1``/``U01``-style FITS header cards."""
    return header.get(f"{prefix}{index:02}", header.get(f"{prefix}{index}", default))


def _require_spatial_shape(name: str, array: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if array.shape[-2:] != expected_shape:
        raise ValueError(
            f"{name} has spatial shape {array.shape[-2:]}, expected {expected_shape}"
        )


def _move_spectral_axis_last(array: np.ndarray, dtype) -> np.ndarray:
    """Move a FITS spectral axis from front to back."""
    array = _to_native(array, dtype=dtype)
    return np.moveaxis(array, 0, -1)


def load_catalog(
    raw_root: str,
    *,
    drpall_path: str | None = None,
    dapall_path: str | None = None,
) -> Table:
    """Load the joined DRP/DAP catalog."""
    drpall_file = drpall_path or os.path.join(raw_root, "drpall-v3_1_1.fits")
    dapall_file = dapall_path or os.path.join(raw_root, "dapall-v3_1_1-3.1.0.fits")

    drpall = Table.read(drpall_file, hdu="MANGA")
    dapall = Table.read(dapall_file, hdu=DAPTYPE)
    catalog = join(
        drpall,
        dapall,
        keys_left="plateifu",
        keys_right="PLATEIFU",
        join_type="inner",
    )
    return catalog[np.asarray(catalog["DAPDONE"]).astype(bool)]


def cube_path(raw_root: str, plateifu: str) -> str:
    plate, _ = plateifu.split("-")
    return os.path.join(
        raw_root,
        "dr17",
        "manga",
        "spectro",
        "redux",
        "v3_1_1",
        plate,
        "stack",
        f"manga-{plateifu}-LOGCUBE.fits.gz",
    )


def maps_path(raw_root: str, plateifu: str) -> str:
    plate, ifu = plateifu.split("-")
    return os.path.join(
        raw_root,
        "dr17",
        "manga",
        "spectro",
        "analysis",
        "v3_1_1",
        "3.1.0",
        DAPTYPE,
        plate,
        ifu,
        f"manga-{plateifu}-MAPS-{DAPTYPE}.fits.gz",
    )


def _read_optional_map_data(
    mapf: fits.HDUList,
    ext_name: str | None,
    shape: tuple[int, ...],
    *,
    fill_value: float,
) -> np.ndarray:
    if ext_name and ext_name in mapf:
        return _to_native(np.asarray(mapf[ext_name].data), dtype=np.float32)
    return np.full(shape, fill_value, dtype=np.float32)


def process_cube(summary_row, raw_root: str) -> dict | None:
    """Read one plate-IFU into a row."""
    plateifu = _b2s(summary_row["plateifu"])
    cube_file = cube_path(raw_root, plateifu)
    map_file = maps_path(raw_root, plateifu)

    if not os.path.exists(cube_file) or not os.path.exists(map_file):
        return None

    with fits.open(cube_file) as cube:
        flux = _move_spectral_axis_last(cube["FLUX"].data, np.float32)
        ivar = _move_spectral_axis_last(cube["IVAR"].data, np.float32)
        mask = _move_spectral_axis_last(cube["MASK"].data, np.int64)
        lsf = _move_spectral_axis_last(cube["LSFPOST"].data, np.float32)
        wave = _to_native(np.asarray(cube["WAVE"].data), dtype=np.float32)

        ny, nx, nwave = flux.shape
        if nwave != len(wave):
            raise ValueError(f"{plateifu}: FLUX spectral axis {nwave} != WAVE length {len(wave)}")

        flux_units = _b2s(cube["FLUX"].header.get("BUNIT", ""))
        lambda_units = _b2s(cube["FLUX"].header.get("CUNIT3", ""))

        yy, xx = np.indices((ny, nx))
        x_arr = xx.astype(np.int8)
        y_arr = yy.astype(np.int8)
        spaxel_idx = np.arange(ny * nx, dtype=np.int16).reshape(ny, nx)

        image_flux = np.stack(
            [_to_native(np.asarray(cube[f"{band.upper()}IMG"].data), dtype=np.float32) for band in BANDS],
            axis=0,
        )
        image_psf = np.stack(
            [_to_native(np.asarray(cube[f"{band.upper()}PSF"].data), dtype=np.float32) for band in BANDS],
            axis=0,
        )
        _require_spatial_shape(f"{plateifu} image_flux", image_flux, (ny, nx))
        _require_spatial_shape(f"{plateifu} image_psf", image_psf, (ny, nx))

    with fits.open(map_file) as mapf:
        skycoo = _to_native(np.asarray(mapf["SPX_SKYCOO"].data), dtype=np.float32)
        ellcoo = _to_native(np.asarray(mapf["SPX_ELLCOO"].data), dtype=np.float32)
        _require_spatial_shape(f"{plateifu} SPX_SKYCOO", skycoo, (ny, nx))
        _require_spatial_shape(f"{plateifu} SPX_ELLCOO", ellcoo, (ny, nx))

        skycoo_units = _b2s(mapf["SPX_SKYCOO"].header.get("BUNIT", ""))
        ellcoo_units = [
            _b2s(_header_indexed_value(mapf["SPX_ELLCOO"].header, "U", i, ""))
            for i in range(1, 5)
        ]

        maps_data: list[dict] = []
        for ext in mapf:
            if ext.name == "PRIMARY" or ext.name.endswith(("IVAR", "MASK")):
                continue
            if ext.data is None:
                continue

            array = _to_native(np.asarray(ext.data), dtype=np.float32)
            _require_spatial_shape(f"{plateifu} {ext.name}", array, (ny, nx))

            err = _read_optional_map_data(
                mapf,
                ext.header.get("ERRDATA"),
                array.shape,
                fill_value=0.0,
            )
            qual = _read_optional_map_data(
                mapf,
                ext.header.get("QUALDATA"),
                array.shape,
                fill_value=MAP_MASK_FILL_VALUE,
            )
            _require_spatial_shape(f"{plateifu} {ext.name} ERRDATA", err, (ny, nx))
            _require_spatial_shape(f"{plateifu} {ext.name} QUALDATA", qual, (ny, nx))

            unit = _b2s(ext.header.get("BUNIT", ""))
            base_name = ext.name.lower()
            if array.ndim == 3:
                for ch in range(array.shape[0]):
                    chan_label = _normalize_channel_name(
                        _b2s(_header_indexed_value(ext.header, "C", ch + 1, ""))
                    )
                    chan_unit = _b2s(
                        _header_indexed_value(ext.header, "U", ch + 1, "") or unit
                    )
                    maps_data.append(
                        {
                            "group": base_name,
                            "label": f"{base_name}_{chan_label.lower()}",
                            "flux": array[ch],
                            "ivar": err[ch] if err.ndim == 3 else err,
                            "mask": qual[ch] if qual.ndim == 3 else qual,
                            "array_units": chan_unit,
                        }
                    )
            else:
                maps_data.append(
                    {
                        "group": base_name,
                        "label": base_name,
                        "flux": array,
                        "ivar": err,
                        "mask": qual,
                        "array_units": unit,
                    }
                )

    healpix = lonlat_to_healpix(
        float(summary_row["ifura"]) * u.deg,
        float(summary_row["ifudec"]) * u.deg,
        HEALPIX_DEPTH,
    ).item()

    return {
        "object_id": plateifu,
        "ra": float(summary_row["ifura"]),
        "dec": float(summary_row["ifudec"]),
        "healpix": healpix,
        "z": float(summary_row["nsa_z"]),
        "spaxel_size": float(SPAXEL_SIZE_ARCSEC),
        "spaxel_size_units": "arcsec",
        "spatial_shape_y": int(ny),
        "spatial_shape_x": int(nx),
        "spaxels": {
            "flux": flux,
            "ivar": ivar,
            "mask": mask,
            "lsf": lsf,
            "lambda": wave,
            "x": x_arr,
            "y": y_arr,
            "spaxel_idx": spaxel_idx,
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
            "flux": image_flux,
            "flux_units": ["nanomaggies/pixel"] * len(BANDS),
            "psf": image_psf,
            "psf_units": ["nanomaggies/pixel"] * len(BANDS),
            "scale": [SPAXEL_SIZE_ARCSEC] * len(BANDS),
            "scale_units": ["arcsec"] * len(BANDS),
        },
        "maps": maps_data,
    }


def _ndarray_to_nested_array(array: np.ndarray, value_type: pa.DataType | None = None) -> pa.Array:
    """Convert an ndarray to nested Arrow lists without ``tolist()``."""
    arr = np.asarray(array)
    if value_type is None:
        value_type = pa.from_numpy_dtype(arr.dtype)

    values = pa.array(arr.reshape(-1), type=value_type)
    nested = values
    for dim in reversed(arr.shape):
        offsets = np.arange(0, len(nested) + 1, dim, dtype=np.int32)
        nested = pa.ListArray.from_arrays(offsets, nested)
    return nested


def _ndarray_column(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.Array:
    return pa.concat_arrays([_ndarray_to_nested_array(arr, value_type) for arr in arrays])


def _build_spaxels_struct(records: list[dict]) -> pa.StructArray:
    spx = [record["spaxels"] for record in records]
    return pa.StructArray.from_arrays(
        [
            _ndarray_column([row["flux"] for row in spx], pa.float32()),
            _ndarray_column([row["ivar"] for row in spx], pa.float32()),
            _ndarray_column([row["mask"] for row in spx], pa.int64()),
            _ndarray_column([row["lsf"] for row in spx], pa.float32()),
            _ndarray_column([row["lambda"] for row in spx], pa.float32()),
            _ndarray_column([row["x"] for row in spx], pa.int8()),
            _ndarray_column([row["y"] for row in spx], pa.int8()),
            _ndarray_column([row["spaxel_idx"] for row in spx], pa.int16()),
            pa.array([row["flux_units"] for row in spx], type=pa.string()),
            pa.array([row["lambda_units"] for row in spx], type=pa.string()),
            _ndarray_column([row["skycoo_x"] for row in spx], pa.float32()),
            _ndarray_column([row["skycoo_y"] for row in spx], pa.float32()),
            _ndarray_column([row["ellcoo_r"] for row in spx], pa.float32()),
            _ndarray_column([row["ellcoo_rre"] for row in spx], pa.float32()),
            _ndarray_column([row["ellcoo_rkpc"] for row in spx], pa.float32()),
            _ndarray_column([row["ellcoo_theta"] for row in spx], pa.float32()),
            pa.array([row["skycoo_units"] for row in spx], type=pa.string()),
            pa.array([row["ellcoo_r_units"] for row in spx], type=pa.string()),
            pa.array([row["ellcoo_rre_units"] for row in spx], type=pa.string()),
            pa.array([row["ellcoo_rkpc_units"] for row in spx], type=pa.string()),
            pa.array([row["ellcoo_theta_units"] for row in spx], type=pa.string()),
        ],
        names=[
            "flux",
            "ivar",
            "mask",
            "lsf",
            "lambda",
            "x",
            "y",
            "spaxel_idx",
            "flux_units",
            "lambda_units",
            "skycoo_x",
            "skycoo_y",
            "ellcoo_r",
            "ellcoo_rre",
            "ellcoo_rkpc",
            "ellcoo_theta",
            "skycoo_units",
            "ellcoo_r_units",
            "ellcoo_rre_units",
            "ellcoo_rkpc_units",
            "ellcoo_theta_units",
        ],
    )


def _build_images_struct(records: list[dict]) -> pa.StructArray:
    images = [record["images"] for record in records]
    return pa.StructArray.from_arrays(
        [
            pa.array([row["filter"] for row in images], type=pa.list_(pa.string())),
            _ndarray_column([row["flux"] for row in images], pa.float32()),
            pa.array([row["flux_units"] for row in images], type=pa.list_(pa.string())),
            _ndarray_column([row["psf"] for row in images], pa.float32()),
            pa.array([row["psf_units"] for row in images], type=pa.list_(pa.string())),
            pa.array([row["scale"] for row in images], type=pa.list_(pa.float32())),
            pa.array([row["scale_units"] for row in images], type=pa.list_(pa.string())),
        ],
        names=["filter", "flux", "flux_units", "psf", "psf_units", "scale", "scale_units"],
    )


def _maps_cube(record: dict, key: str, dtype) -> np.ndarray:
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
            pa.array([[m["group"] for m in record["maps"]] for record in records], type=pa.list_(pa.string())),
            pa.array([[m["label"] for m in record["maps"]] for record in records], type=pa.list_(pa.string())),
            _ndarray_column([_maps_cube(record, "flux", np.float32) for record in records], pa.float32()),
            _ndarray_column([_maps_cube(record, "ivar", np.float32) for record in records], pa.float32()),
            _ndarray_column([_maps_cube(record, "mask", np.float32) for record in records], pa.float32()),
            pa.array([[m["array_units"] for m in record["maps"]] for record in records], type=pa.list_(pa.string())),
        ],
        names=["group", "label", "flux", "ivar", "mask", "array_units"],
    )


def build_table(records: list[dict]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "ra": pa.array([record["ra"] for record in records], type=pa.float64()),
        "dec": pa.array([record["dec"] for record in records], type=pa.float64()),
        "object_id": pa.array([record["object_id"] for record in records], type=pa.string()),
        "healpix": pa.array([record["healpix"] for record in records], type=pa.int64()),
        "z": pa.array([record["z"] for record in records], type=pa.float32()),
        "spaxel_size": pa.array([record["spaxel_size"] for record in records], type=pa.float32()),
        "spaxel_size_units": pa.array([record["spaxel_size_units"] for record in records], type=pa.string()),
        "spatial_shape_y": pa.array([record["spatial_shape_y"] for record in records], type=pa.int16()),
        "spatial_shape_x": pa.array([record["spatial_shape_x"] for record in records], type=pa.int16()),
        "spaxels": _build_spaxels_struct(records),
        "images": _build_images_struct(records),
        "maps": _build_maps_struct(records),
    }
    return pa.table(columns)


def _write_shard(records: list[dict], shard_dir: str, shard_index: int) -> str:
    path = os.path.join(shard_dir, f"{CATALOG_NAME}_shard_{shard_index:05d}.parquet")
    pq.write_table(build_table(records), path, compression="zstd")
    return path


def _load_plateifu_filter(args) -> set[str]:
    selected = {_b2s(v) for v in args.plateifu}
    if args.plateifu_file:
        with open(args.plateifu_file, "r", encoding="utf-8") as handle:
            selected.update(line.strip() for line in handle if line.strip())
    return selected


def _prepare_work_dir(work_dir: str | None) -> tuple[str, bool]:
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        return work_dir, False
    return tempfile.mkdtemp(prefix="manga_hats_shards_"), True


def _make_hats_client(debug: bool, workers: int) -> Client:
    if debug:
        return Client(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        )
    return Client(
        n_workers=workers,
        threads_per_worker=1,
        dashboard_address=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--drpall-path", default=None)
    parser.add_argument("--dapall-path", default=None)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--work-dir", default=None, help="Optional directory for intermediate parquet shards.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep an auto-created work dir after success.")
    parser.add_argument("--max-files", type=int, default=None, help="Cap on number of plate-ifus to process.")
    parser.add_argument("--plateifu", action="append", default=[], help="Restrict to a specific plate-IFU. Repeat as needed.")
    parser.add_argument("--plateifu-file", default=None, help="Text file with one plate-IFU per line.")
    parser.add_argument("--rows-per-shard", type=int, default=1, help="How many MaNGA objects to pack into each parquet shard.")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=1, help="Dask workers for the hats-import phase.")
    parser.add_argument("--debug", action="store_true", help="Run hats-import in a single process.")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None, help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    if args.rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be >= 1")

    print(f"Loading drpall + dapall from {args.raw_root}")
    catalog = load_catalog(
        args.raw_root,
        drpall_path=args.drpall_path,
        dapall_path=args.dapall_path,
    )
    print(f"  {len(catalog)} plate-ifus after DAPDONE join")

    selected_plateifus = _load_plateifu_filter(args)
    if selected_plateifus:
        mask = np.array([_b2s(v) in selected_plateifus for v in catalog["plateifu"]], dtype=bool)
        catalog = catalog[mask]
        print(f"  {len(catalog)} plate-ifus after explicit plate-IFU filtering")

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        mask = apply_cone_filter(
            np.asarray(catalog["ifura"], dtype=np.float64),
            np.asarray(catalog["ifudec"], dtype=np.float64),
            args.ra_center,
            args.dec_center,
            args.radius,
        )
        catalog = catalog[mask]
        print(
            f"  {len(catalog)} plate-ifus after cone cut "
            f"(ra={args.ra_center}, dec={args.dec_center}, radius={args.radius})"
        )

    if args.max_files is not None:
        catalog = catalog[:args.max_files]
        print(f"  capped to {len(catalog)} plate-ifus via --max-files")

    if len(catalog) == 0:
        print("ERROR: no MaNGA targets selected", file=sys.stderr)
        return 1

    work_dir, cleanup_work_dir = _prepare_work_dir(args.work_dir)
    parquet_files: list[str] = []
    chunk: list[dict] = []
    shard_index = 0
    processed = 0

    try:
        for i, row in enumerate(catalog, 1):
            plateifu = _b2s(row["plateifu"])
            try:
                record = process_cube(row, args.raw_root)
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                print(f"  [{i}/{len(catalog)}] {plateifu}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            if record is None:
                print(f"  [{i}/{len(catalog)}] {plateifu}: missing cube or map file", file=sys.stderr)
                continue

            chunk.append(record)
            processed += 1
            print(
                f"  [{i}/{len(catalog)}] {plateifu}: "
                f"{record['spatial_shape_y']}x{record['spatial_shape_x']} "
                f"with {len(record['maps'])} maps"
            )

            if len(chunk) >= args.rows_per_shard:
                parquet_files.append(_write_shard(chunk, work_dir, shard_index))
                shard_index += 1
                chunk.clear()

        if chunk:
            parquet_files.append(_write_shard(chunk, work_dir, shard_index))

        if not parquet_files:
            print("ERROR: no plate-ifus successfully processed", file=sys.stderr)
            return 1

        print(f"\nBuilt {len(parquet_files)} parquet shard(s) for {processed} MaNGA targets")
        with _make_hats_client(args.debug, args.workers) as client:
            catalog_dir = write_hats_from_parquet_dir(
                work_dir,
                output_path=args.output_root,
                catalog_name=CATALOG_NAME,
                pixel_threshold=args.pixel_threshold,
                n_workers=args.workers,
                debug=args.debug,
                client=client,
            )
        print(f"Done: {catalog_dir}")
        if cleanup_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 0
    finally:
        if cleanup_work_dir and not args.keep_work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
