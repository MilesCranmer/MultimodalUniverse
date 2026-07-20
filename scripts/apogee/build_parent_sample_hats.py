"""Convert raw APOGEE DR17 spectra into a HATS catalog.

This is a raw-data path, not a v1-HDF5 re-ingest. It reads the local APOGEE
mirror under::

    /mnt/ceph/users/polymathic/external_data/astro/APOGEE/apogee

using:
    - allStar-dr17-synspec_rev1.fits for the global stellar catalog
    - apStar visit files under spectro/redux/dr17/stars/{telescope}/{field}/
    - aspcapStar continuum files under spectro/aspcap/dr17/synspec_rev1/{telescope}/{field}/

The emitted HATS schema matches the existing HF-facing APOGEE contract:
    - object_id (string)
    - ra, dec (float64) for HATS partitioning
    - spectrum struct with flux/ivar/lsf_sigma/lambda/mask/pseudo_continuum_*
    - teff, logg, m_h, alpha_m, teff_err, logg_err, m_h_err, alpha_m_err,
      radial_velocity (float32)
    - restframe (bool)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import warnings
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, to_native_endian, write_hats_from_parquet_dir


CATALOG_NAME = "apogee"

LAM = 10.0 ** np.arange(
    4.179, 4.179 + 8575 * 6.0 * 10.0**-6.0, 6.0 * 10.0**-6.0
).astype(np.float32)

BLUE_START = 246
BLUE_END = 3274
GREEN_START = 3585
GREEN_END = 6080
RED_START = 6344
RED_END = 8335
CROP_INDEX = np.r_[BLUE_START:BLUE_END, GREEN_START:GREEN_END, RED_START:RED_END]
LAM_CROPPED = LAM[CROP_INDEX]

FLOAT_FEATURE_MAP = {
    "teff": "TEFF",
    "logg": "LOGG",
    "m_h": "M_H",
    "alpha_m": "ALPHA_M",
    "teff_err": "TEFF_ERR",
    "logg_err": "LOGG_ERR",
    "m_h_err": "M_H_ERR",
    "alpha_m_err": "ALPHA_M_ERR",
    "radial_velocity": "VHELIO_AVG",
}


def _catalog_path(raw_root: str) -> str:
    direct = os.path.join(
        raw_root,
        "spectro",
        "aspcap",
        "dr17",
        "synspec_rev1",
        "allStar-dr17-synspec_rev1.fits",
    )
    if os.path.exists(direct):
        return direct

    sibling_v2 = os.path.join(
        os.path.dirname(raw_root),
        "apogee_v2",
        "spectro",
        "aspcap",
        "dr17",
        "synspec_rev1",
        "allStar-dr17-synspec_rev1.fits",
    )
    if os.path.exists(sibling_v2):
        return sibling_v2
    return direct


def _default_cache_root() -> str:
    return os.path.expanduser("~/ceph/general_data/mmu_apogee_raw")


def _is_readable_fits_table(path: str) -> bool:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Table.read(path, hdu=1)
        for w in caught:
            if "truncated" in str(w.message).lower():
                return False
        return True
    except Exception:
        return False


def _visit_path(raw_root: str, field: str, telescope: str, filename: str) -> str:
    return os.path.join(
        raw_root,
        "spectro",
        "redux",
        "dr17",
        "stars",
        telescope,
        str(field),
        str(filename),
    )


def _continuum_path(raw_root: str, field: str, telescope: str, apogee_id: str) -> str:
    return os.path.join(
        raw_root,
        "spectro",
        "aspcap",
        "dr17",
        "synspec_rev1",
        telescope,
        str(field),
        f"aspcapStar-dr17-{apogee_id}.fits",
    )


def _masked_to_array(column) -> np.ndarray:
    return np.ma.asarray(column).filled(np.nan)


def _masked_str(column) -> np.ndarray:
    return np.ma.asarray(column).filled("").astype(str)


def _download_file(url: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


def ensure_catalog_path(raw_root: str, cache_root: str) -> str:
    path = _catalog_path(raw_root)
    if os.path.exists(path) and _is_readable_fits_table(path):
        return path
    filename = "allStar-dr17-synspec_rev1.fits"
    url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{filename}"
    cache_path = os.path.join(cache_root, "spectro", "aspcap", "dr17", "synspec_rev1", filename)
    if os.path.exists(cache_path) and not _is_readable_fits_table(cache_path):
        os.remove(cache_path)
    return _download_file(url, cache_path)


def ensure_visit_path(raw_root: str, field: str, telescope: str, filename: str, cache_root: str) -> str:
    path = _visit_path(raw_root, field, telescope, filename)
    if os.path.exists(path):
        return path
    url = f"https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/{telescope}/{field}/{filename}"
    cache_path = os.path.join(cache_root, "spectro", "redux", "dr17", "stars", telescope, str(field), filename)
    return _download_file(url, cache_path)


def ensure_continuum_path(raw_root: str, field: str, telescope: str, apogee_id: str, cache_root: str) -> str:
    filename = f"aspcapStar-dr17-{apogee_id}.fits"
    path = _continuum_path(raw_root, field, telescope, apogee_id)
    if os.path.exists(path):
        return path
    url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{telescope}/{field}/{filename}"
    cache_path = os.path.join(cache_root, "spectro", "aspcap", "dr17", "synspec_rev1", telescope, str(field), filename)
    return _download_file(url, cache_path)


def load_catalog(raw_root: str, cache_root: str | None = None) -> Table:
    cache_root = cache_root or _default_cache_root()
    path = ensure_catalog_path(raw_root, cache_root)
    return Table.read(path, hdu=1)


def selection_mask(catalog: Table, raw_root: str, cache_root: str | None = None) -> np.ndarray:
    cache_root = cache_root or _default_cache_root()
    telescope = _masked_str(catalog["TELESCOPE"])
    filename = _masked_str(catalog["FILE"])
    field = _masked_str(catalog["FIELD"])
    apogee_id = _masked_str(catalog["APOGEE_ID"])
    snr = _masked_to_array(catalog["SNR"]).astype(np.float32)

    mask = np.isin(telescope, ["apo25m", "lco25m"])
    mask &= filename != ""
    mask &= snr > 30

    _, unique_idx = np.unique(apogee_id, return_index=True)
    unique_mask = np.zeros(len(catalog), dtype=bool)
    unique_mask[unique_idx] = True
    mask &= unique_mask

    existing = np.zeros(len(catalog), dtype=bool)
    for i in np.where(mask)[0]:
        try:
            visit = ensure_visit_path(raw_root, field[i], telescope[i], filename[i], cache_root)
            continuum = ensure_continuum_path(raw_root, field[i], telescope[i], apogee_id[i], cache_root)
            existing[i] = os.path.exists(visit) and os.path.exists(continuum)
        except OSError:
            existing[i] = False
    mask &= existing
    return mask


def _read_visit_and_continuum(visit_path: str, continuum_path: str) -> dict[str, np.ndarray]:
    with fits.open(visit_path) as hdus:
        raw_flux = to_native_endian(np.asarray(hdus[1].data[0], dtype=np.float32))
        raw_sigma = to_native_endian(np.asarray(hdus[2].data[0], dtype=np.float32))
        raw_mask = to_native_endian(np.asarray(hdus[3].data[0] > 0, dtype=bool))

    raw_ivar = np.zeros_like(raw_sigma, dtype=np.float32)
    valid_sigma = raw_sigma != 0
    raw_ivar[valid_sigma] = 1.0 / np.square(raw_sigma[valid_sigma])
    raw_ivar[np.isnan(raw_ivar)] = 0.0

    invalid = raw_mask | (raw_ivar < 1e-6) | np.isnan(raw_flux) | np.isnan(raw_ivar)

    with fits.open(continuum_path) as hdus:
        continuum_flux = to_native_endian(np.asarray(hdus[1].data, dtype=np.float32))
        continuum_sigma = to_native_endian(np.asarray(hdus[2].data, dtype=np.float32))

    continuum_ivar = np.zeros_like(continuum_sigma, dtype=np.float32)
    valid_cont = continuum_sigma != 0
    continuum_ivar[valid_cont] = 1.0 / np.square(continuum_sigma[valid_cont])
    continuum_ivar[np.isnan(continuum_ivar)] = 0.0

    lsf_sigma = np.ones_like(raw_flux, dtype=np.float32)
    lsf_sigma[:BLUE_END] *= 0.326
    lsf_sigma[BLUE_END:GREEN_END] *= 0.283
    lsf_sigma[GREEN_END:] *= 0.236

    return {
        "flux": raw_flux[CROP_INDEX],
        "ivar": raw_ivar[CROP_INDEX],
        "lsf_sigma": lsf_sigma[CROP_INDEX],
        "lambda": LAM_CROPPED,
        "mask": invalid[CROP_INDEX],
        "pseudo_continuum_flux": continuum_flux[CROP_INDEX],
        "pseudo_continuum_ivar": continuum_ivar[CROP_INDEX],
    }


def _row_to_record(args: tuple[int, dict[str, str], str] | tuple[int, dict[str, str], str, str]) -> dict | None:
    if len(args) == 3:
        idx, row, raw_root = args
        cache_root = _default_cache_root()
    else:
        idx, row, raw_root, cache_root = args
    try:
        visit_path = ensure_visit_path(raw_root, row["FIELD"], row["TELESCOPE"], row["FILE"], cache_root)
        continuum_path = ensure_continuum_path(raw_root, row["FIELD"], row["TELESCOPE"], row["APOGEE_ID"], cache_root)
        spectrum = _read_visit_and_continuum(visit_path, continuum_path)
    except (OSError, FileNotFoundError, KeyError, IndexError, ValueError) as exc:
        print(
            f"WARNING: skipping APOGEE row {idx} ({row['APOGEE_ID']}): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None

    record = {
        "object_id": row["APOGEE_ID"],
        "ra": np.float64(row["RA"]),
        "dec": np.float64(row["DEC"]),
        "restframe": True,
    }
    record.update(spectrum)
    for out_name, src_name in FLOAT_FEATURE_MAP.items():
        record[out_name] = np.float32(row[src_name])
    return record


def build_arrow_table(records: list[dict]) -> pa.Table:
    spectrum = pa.StructArray.from_arrays(
        arrays=[
            pa.array([r["flux"].tolist() for r in records], type=pa.list_(pa.float32())),
            pa.array([r["ivar"].tolist() for r in records], type=pa.list_(pa.float32())),
            pa.array([r["lsf_sigma"].tolist() for r in records], type=pa.list_(pa.float32())),
            pa.array([r["lambda"].tolist() for r in records], type=pa.list_(pa.float32())),
            pa.array([r["mask"].tolist() for r in records], type=pa.list_(pa.bool_())),
            pa.array(
                [r["pseudo_continuum_flux"].tolist() for r in records],
                type=pa.list_(pa.float32()),
            ),
            pa.array(
                [r["pseudo_continuum_ivar"].tolist() for r in records],
                type=pa.list_(pa.float32()),
            ),
        ],
        names=[
            "flux",
            "ivar",
            "lsf_sigma",
            "lambda",
            "mask",
            "pseudo_continuum_flux",
            "pseudo_continuum_ivar",
        ],
    )

    columns: dict[str, pa.Array] = {
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "ra": pa.array(np.array([r["ra"] for r in records], dtype=np.float64), type=pa.float64()),
        "dec": pa.array(np.array([r["dec"] for r in records], dtype=np.float64), type=pa.float64()),
        "restframe": pa.array([r["restframe"] for r in records], type=pa.bool_()),
        "spectrum": spectrum,
    }
    for out_name in FLOAT_FEATURE_MAP:
        columns[out_name] = pa.array(
            np.array([r[out_name] for r in records], dtype=np.float32),
            type=pa.float32(),
        )
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root of the local APOGEE DR17 mirror.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument("--max-files", type=int, default=None, help="Cap selected stars for smoke tests.")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args(argv)

    cache_root = args.cache_root or _default_cache_root()
    os.makedirs(cache_root, exist_ok=True)

    try:
        catalog = load_catalog(args.raw_root, cache_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mask = selection_mask(catalog, args.raw_root, cache_root)
    catalog = catalog[mask]

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        cone_mask = apply_cone_filter(
            _masked_to_array(catalog["RA"]).astype(np.float64),
            _masked_to_array(catalog["DEC"]).astype(np.float64),
            args.ra_center,
            args.dec_center,
            args.radius,
        )
        catalog = catalog[cone_mask]

    if args.max_files is not None:
        catalog = catalog[:args.max_files]

    if len(catalog) == 0:
        print("ERROR: no APOGEE rows selected", file=sys.stderr)
        return 1

    row_dicts = []
    for row in catalog:
        row_dicts.append({
            "APOGEE_ID": str(row["APOGEE_ID"]),
            "FIELD": str(row["FIELD"]),
            "TELESCOPE": str(row["TELESCOPE"]),
            "FILE": str(row["FILE"]),
            "RA": float(row["RA"]),
            "DEC": float(row["DEC"]),
            **{src_name: float(row[src_name]) for src_name in FLOAT_FEATURE_MAP.values()},
        })

    scratch_dir = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch_dir, exist_ok=True)

    shard_idx = 0
    pending: list[dict] = []
    work = [(i, row_dicts[i], args.raw_root, cache_root) for i in range(len(row_dicts))]

    n_total = len(work)
    n_done = 0

    def _process(record):
        nonlocal pending, shard_idx, n_done
        n_done += 1
        if n_done % 500 == 0 or n_done == n_total:
            sys.stderr.write(f"  [{n_done}/{n_total}] {len(pending)} pending, "
                           f"{shard_idx} shards written\n")
            sys.stderr.flush()
        if record is None:
            return
        pending.append(record)
        if len(pending) >= args.batch_size:
            table = build_arrow_table(pending)
            out = os.path.join(scratch_dir, f"part-{shard_idx:04d}.parquet")
            tmp = out + ".tmp"
            pq.write_table(table, tmp)
            os.rename(tmp, out)
            shard_idx += 1
            pending = []

    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            for record in pool.imap_unordered(_row_to_record, work):
                _process(record)
    else:
        for item in work:
            _process(_row_to_record(item))

    if pending:
        table = build_arrow_table(pending)
        pq.write_table(table, os.path.join(scratch_dir, f"part-{shard_idx:04d}.parquet"))
        shard_idx += 1

    if shard_idx == 0:
        print("ERROR: no APOGEE rows were processed successfully", file=sys.stderr)
        return 1

    catalog_dir = write_hats_from_parquet_dir(
        scratch_dir,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=True,
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
