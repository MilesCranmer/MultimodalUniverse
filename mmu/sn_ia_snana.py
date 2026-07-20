"""Shared HATS build logic for SN-Ia lightcurve datasets stored as SNANA ASCII.

Foundation DR1, SNLS, PS1 SN-Ia, DES Y3 SN-Ia, and Swift SN-Ia all ship as
directories of SNANA-format ASCII files (one file per SN). Their v1 HF
schemas are identical:

    lightcurve : struct<band, flux, flux_err, time>
    redshift, host_log_mass : float32
    object_id, obj_type : string

The only per-dataset variation is the filename glob and the ``object_id``
prefix (e.g. ``PS1_<SNID>`` vs bare ``<SNID>``). This module exposes
:func:`build_main` that :file:`scripts/<dataset>/build_parent_sample_hats.py`
calls with those parameters; the ~5-line wrapper is all that's needed per
dataset.

Output matches v1's flattened HF form: each object's lightcurve has parallel
``band`` / ``time`` / ``flux`` / ``flux_err`` lists of length
``n_bands * max_length_per_object`` (padded with MJD=-99, flux=0 so padding
samples are identifiable on readback).

The SNANA ASCII parser is inline (below) rather than delegated to
``sncosmo.read_snana_ascii`` because sncosmo doesn't have a Linux wheel
on PyPI and builds from source against Python.h, which the Flatiron
compute nodes don't have installed. The format is trivial:
``KEY: value [...]`` metadata lines, then a ``VARLIST: ...`` header,
then ``OBS: <values...>`` rows. Comments start with ``#``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow as pa

from mmu.cone import apply_cone_filter
from mmu.hats_configs import MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


# SNANA header prefixes that are NOT metadata and should not be stuffed into
# the meta dict by the catch-all colon parser. These have structural meaning
# (they drive the obs table) or are free-form comments that can recur and
# whose "value" is not what we want to remember as e.g. COMMENT=<first word>.
_NON_META_PREFIXES = frozenset({
    "COMMENT", "END", "END_PHOTOMETRY", "BEGIN_PHOTOMETRY",
    "NOBS", "NVAR", "VARLIST", "OBS",
})


def _parse_snana_ascii(path: str) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """Parse a SNANA ASCII lightcurve file.

    Returns ``(meta, varlist, obs_rows)`` where:
    - ``meta`` maps header key → first-token value (ignores the rest of the line
      so e.g. ``RA: 150.1 deg`` becomes ``meta["RA"] = "150.1"``).
    - ``varlist`` is the column names of the obs table.
    - ``obs_rows`` is a list of per-obs lists of strings (one row per ``OBS:`` line).

    Defensive behavior:
    - If ``OBS:`` lines appear before ``VARLIST:``, raise ValueError rather
      than silently producing empty rows.
    - Non-metadata prefixes (``COMMENT``, ``END``, ``BEGIN_PHOTOMETRY``, etc.)
      are skipped rather than stuffed into the meta dict.
    - Duplicate metadata keys raise ValueError — SNANA files shouldn't
      repeat real metadata keys, and silently overwriting hides upstream
      data issues.
    """
    meta: dict[str, str] = {}
    varlist: list[str] = []
    obs_rows: list[list[str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OBS:"):
                if not varlist:
                    raise ValueError(
                        f"{path}: OBS: line before VARLIST: header. "
                        "Cannot interpret observation columns."
                    )
                parts = line.split()
                obs_rows.append(parts[1 : 1 + len(varlist)])
                continue
            if line.startswith("VARLIST:"):
                varlist = line.split()[1:]
                continue
            # Recognized structural prefixes we explicitly skip (COMMENT:,
            # END:, BEGIN_PHOTOMETRY:, NOBS:, NVAR:, ...).
            head, sep, _ = line.partition(":")
            if not sep:
                continue
            key = head.strip()
            if key in _NON_META_PREFIXES:
                continue
            # A metadata line like "RA: 150.1 deg" or "REDSHIFT_HELIO: 0.05 +- 0.001".
            # Take the first whitespace-separated token after the colon as the value;
            # SNANA files are inconsistent about trailing units / errors.
            rest = line.partition(":")[2].strip().split()
            if not rest:
                continue
            if key in meta:
                # Repeating a real metadata key is a real upstream problem —
                # fail loud instead of silently letting the second line win.
                raise ValueError(
                    f"{path}: duplicate metadata key {key!r} "
                    f"({meta[key]!r} vs {rest[0]!r})."
                )
            meta[key] = rest[0]
    return meta, varlist, obs_rows


def read_snana_file(path: str, object_id_prefix: str = "") -> dict | None:
    """Parse one SNANA ASCII SN light-curve file.

    Returns ``None`` if RA/Dec/redshift can't be extracted (the ~2% of files
    with missing metadata in some of these surveys).

    ``object_id_prefix`` is prepended to the raw ``SNID`` (e.g. ``"PS1_"``).
    """
    meta, varlist, obs_rows = _parse_snana_ascii(path)

    try:
        ra = float(meta["RA"])
        dec = float(meta["DECL"])
        snid = str(meta["SNID"])
    except KeyError:
        return None

    redshift = None
    for k in ("REDSHIFT_FINAL", "REDSHIFT_CMB", "REDSHIFT_HELIO"):
        if k in meta:
            redshift = float(meta[k])
            break
    if redshift is None:
        return None

    host_log_mass = float("nan")
    for k in ("HOST_LOGMASS", "HOSTGAL_LOGMASS"):
        if k in meta:
            host_log_mass = float(meta[k])
            break

    # Index the obs columns we need. Different surveys label the filter
    # column as FLT or BAND; the others are consistent.
    try:
        flt_col = varlist.index("FLT") if "FLT" in varlist else varlist.index("BAND")
        mjd_col = varlist.index("MJD")
        fluxcal_col = varlist.index("FLUXCAL")
        fluxcalerr_col = varlist.index("FLUXCALERR")
    except ValueError:
        return None

    if not obs_rows:
        return None

    flt = np.array([row[flt_col] for row in obs_rows], dtype=str)
    mjd = np.array([float(row[mjd_col]) for row in obs_rows], dtype=np.float32)
    fluxcal = np.array([float(row[fluxcal_col]) for row in obs_rows], dtype=np.float32)
    fluxcalerr = np.array([float(row[fluxcalerr_col]) for row in obs_rows], dtype=np.float32)

    return {
        "object_id": f"{object_id_prefix}{snid}",
        "obj_type": "Ia",
        "ra": ra,
        "dec": dec,
        "redshift": redshift,
        "host_log_mass": host_log_mass,
        "FLT": flt,
        "MJD": mjd,
        "FLUXCAL": fluxcal,
        "FLUXCALERR": fluxcalerr,
    }


def stack_per_object(rows: list[dict], all_bands: np.ndarray) -> list[dict]:
    """For each object, build per-band padded arrays of length ``max_length``
    (the longest single-band run of observations for that object), then
    flatten into parallel lists of band/time/flux/flux_err.

    This matches v1's two-step behavior: HDF5 stores a ``(n_bands, max_len)``
    2D padded array, and the HF loader flattens it to parallel 1D sequences
    with band labels repeated ``max_len`` times per band. We skip the
    intermediate 2D representation and emit the flattened sequence directly,
    but the final values (including padded sentinels) are identical.
    """
    out = []
    n_bands = len(all_bands)
    for r in rows:
        flt = r["FLT"]
        _, counts = np.unique(flt, return_counts=True)
        max_length = int(counts.max()) if len(counts) else 0

        time_2d = np.zeros((n_bands, max_length), dtype=np.float32)
        flux_2d = np.zeros((n_bands, max_length), dtype=np.float32)
        flux_err_2d = np.zeros((n_bands, max_length), dtype=np.float32)
        time_2d[:] = -99.0

        for j, band in enumerate(all_bands):
            mask = flt == band
            n = int(mask.sum())
            if n == 0:
                continue
            time_2d[j, :n] = r["MJD"][mask]
            flux_2d[j, :n] = r["FLUXCAL"][mask]
            flux_err_2d[j, :n] = r["FLUXCALERR"][mask]

        band_labels = np.repeat(all_bands, max_length)
        out.append(
            {
                "object_id": r["object_id"],
                "obj_type": r["obj_type"],
                "ra": r["ra"],
                "dec": r["dec"],
                "redshift": r["redshift"],
                "host_log_mass": r["host_log_mass"],
                "band": band_labels,
                "time": time_2d.reshape(-1),
                "flux": flux_2d.reshape(-1),
                "flux_err": flux_err_2d.reshape(-1),
            }
        )
    return out


def build_table(rows: list[dict]) -> pa.Table:
    columns = {
        "ra": pa.array([r["ra"] for r in rows], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in rows], type=pa.float64()),
        "object_id": pa.array([r["object_id"] for r in rows], type=pa.string()),
        "obj_type": pa.array([r["obj_type"] for r in rows], type=pa.string()),
        "redshift": pa.array([r["redshift"] for r in rows], type=pa.float32()),
        "host_log_mass": pa.array(
            [r["host_log_mass"] for r in rows], type=pa.float32()
        ),
        "lightcurve": pa.StructArray.from_arrays(
            [
                pa.array([r["band"].tolist() for r in rows], type=pa.list_(pa.string())),
                pa.array([r["time"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["flux"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["flux_err"] for r in rows], type=pa.list_(pa.float32())),
            ],
            names=["band", "time", "flux", "flux_err"],
        ),
    }
    return pa.table(columns)


def build_main(
    *,
    catalog_name: str,
    raw_subdir: str,
    filename_glob: str,
    object_id_prefix: str = "",
    argv: list[str] | None = None,
) -> int:
    """Entry-point shared by every SN-Ia SNANA build script.

    Args:
        catalog_name: HATS catalog name (e.g. ``"foundation"``, ``"ps1_sne_ia"``).
        raw_subdir: Relative subdir under RAW_DATA_ROOT (passed in by caller
            since ``mmu.hats_configs`` already knows it).
        filename_glob: Shell glob that matches one SN per file under the raw dir.
        object_id_prefix: String prepended to SNID in the output ``object_id``.
        argv: Command-line argv (``None`` → sys.argv).
    """
    parser = argparse.ArgumentParser(description=f"Build HATS catalog for {catalog_name}")
    parser.add_argument("--raw-root", default=raw_subdir)
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, catalog_name),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.raw_root, filename_glob)))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(f"ERROR: no {filename_glob} under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} {catalog_name} SN file(s)")

    rows: list[dict] = []
    for i, p in enumerate(files, 1):
        try:
            r = read_snana_file(p, object_id_prefix=object_id_prefix)
        except (ValueError, KeyError, OSError) as exc:
            print(
                f"  [{i}/{len(files)}] {os.path.basename(p)}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        if r is None:
            print(
                f"  [{i}/{len(files)}] {os.path.basename(p)}: missing RA/Dec/redshift",
                file=sys.stderr,
            )
            continue
        rows.append(r)
        print(
            f"  [{i}/{len(files)}] {r['object_id']}: "
            f"{len(r['MJD'])} obs across {len(np.unique(r['FLT']))} bands"
        )

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        ra = np.array([r["ra"] for r in rows], dtype=np.float64)
        dec = np.array([r["dec"] for r in rows], dtype=np.float64)
        mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
        rows = [r for r, keep in zip(rows, mask) if keep]
        print(f"After cone cut: {len(rows)} SNe")

    if not rows:
        print("ERROR: no SNe after filtering", file=sys.stderr)
        return 1

    all_bands = np.array(
        sorted(set().union(*[set(r["FLT"]) for r in rows])), dtype=str
    )
    print(f"All bands: {list(all_bands)}")

    stacked = stack_per_object(rows, all_bands)
    table = build_table(stacked)
    total_samples = sum(len(r["time"]) for r in stacked)
    print(
        f"\nWriting HATS catalog: {table.num_rows} SNe, "
        f"{total_samples} lightcurve samples"
    )
    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=catalog_name,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0
