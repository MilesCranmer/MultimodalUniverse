"""Convert raw DESI PROVABGS HDF5 file into a single HATS catalog.

The cluster mirrors the DESI PROVABGS VAC at:

    /mnt/ceph/users/polymathic/external_data/astro/DESI_PROVABGS/
        BGS_ANY_full.provabgs.sv3.v0.hdf5

This script:
1. Reads the HDF5 file using astropy.table.Table.
2. Filters to objects with valid best-fit (PROVABGS_LOGMSTAR_BF > 0,
   MAG_G/R/Z > 0), matching the v1 selection.
3. Optionally computes Z_MW, TAGE_MW, AVG_SFR via the provabgs NMF model
   (requires ``provabgs`` package; skip with ``--skip-best-fit``).
4. Converts the PROVABGS_MCMC (N, 100, 13) array to
   pa.list_(pa.list_(pa.float32())).
5. Hands the table to ``write_hats`` to produce a HATS catalog under
   ``{output_root}/desi_provabgs/desi_provabgs/``.

Usage:
    python -m scripts.desi_provabgs.build_parent_sample_hats \\
        --raw-root /mnt/ceph/users/polymathic/external_data/astro/DESI_PROVABGS \\
        --output-root /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/desi_provabgs \\
        --max-files 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa


def _as_array(arr):
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr

from astropy.table import Table
from tqdm import tqdm

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


CATALOG_NAME = "desi_provabgs"

FILENAME = "BGS_ANY_full.provabgs.sv3.v0.hdf5"

FLOAT_FEATURES = [
    "Z_HP",
    "ZERR",
    "TSNR2_BGS",
    "MAG_G",
    "MAG_R",
    "MAG_Z",
    "MAG_W1",
    "FIBMAG_R",
    "HPIX_64",
    "PROVABGS_Z_MAX",
    "SCHLEGEL_COLOR",
    "PROVABGS_W_ZFAIL",
    "PROVABGS_W_FIBASSIGN",
]

BOOL_FEATURES = [
    "IS_BGS_BRIGHT",
    "IS_BGS_FAINT",
]

BEST_FIT_FLOAT_FEATURES = ["Z_MW", "TAGE_MW", "AVG_SFR"]


def selection_mask(data: Table) -> np.ndarray:
    """Return boolean mask matching v1 _get_best_fit filter."""
    mask = np.asarray(data["PROVABGS_LOGMSTAR_BF"]) > 0
    mask &= np.asarray(data["MAG_G"]) > 0
    mask &= np.asarray(data["MAG_R"]) > 0
    mask &= np.asarray(data["MAG_Z"]) > 0
    return mask


def compute_best_fit_properties(data: Table) -> Table:
    """Compute Z_MW, TAGE_MW, AVG_SFR using the provabgs NMF model.

    Requires the ``provabgs`` package.
    """
    from provabgs import models as Models  # noqa: PLC0415

    m_nmf = Models.NMF(burst=True, emulator=True)
    thetas = np.asarray(data["PROVABGS_THETA_BF"])[:, :12]
    zreds = np.asarray(data["Z_HP"])

    z_mw = []
    tage_mw = []
    avg_sfr = []

    for i in tqdm(range(len(thetas)), desc="Computing best-fit properties"):
        z_mw.append(m_nmf.Z_MW(thetas[i], zred=zreds[i]))
        tage_mw.append(m_nmf.tage_MW(thetas[i], zred=zreds[i]))
        avg_sfr.append(m_nmf.avgSFR(thetas[i], zred=zreds[i]))

    data["Z_MW"] = np.array(z_mw)
    data["TAGE_MW"] = np.array(tage_mw)
    data["AVG_SFR"] = np.array(avg_sfr)
    return data


def _mcmc_to_arrow(mcmc: np.ndarray) -> pa.Array:
    """Convert PROVABGS_MCMC (N, 100, 13) to pa.list_(pa.list_(pa.float32())).

    Each row becomes a list of 100 lists of 13 floats.
    """
    n = mcmc.shape[0]
    mcmc = mcmc.astype(np.float32)
    inner_type = pa.list_(pa.float32())
    outer_type = pa.list_(inner_type)

    rows = []
    for i in range(n):
        row = mcmc[i]  # shape (100, 13)
        inner = [row[j].tolist() for j in range(row.shape[0])]
        rows.append(inner)
    return pa.array(rows, type=outer_type)


def build_arrow_table(data: Table, include_best_fit: bool = True) -> pa.Table:
    """Convert an astropy Table of PROVABGS rows into a PyArrow table."""
    ra = np.asarray(data["RA"], dtype=np.float64)
    dec = np.asarray(data["DEC"], dtype=np.float64)

    columns: dict[str, pa.Array] = {
        "object_id": pa.array([str(int(t)) for t in np.asarray(data["TARGETID"])], type=pa.string()),
        "ra": pa.array(ra),
        "dec": pa.array(dec),
        "PROVABGS_LOGMSTAR_BF": pa.array(
            np.asarray(data["PROVABGS_LOGMSTAR_BF"]).astype(np.float32)
        ),
        "PROVABGS_MCMC": _mcmc_to_arrow(np.asarray(data["PROVABGS_MCMC"])),
        "PROVABGS_THETA_BF": pa.array(
            [row.astype(np.float32).tolist() for row in np.asarray(data["PROVABGS_THETA_BF"])],
            type=pa.list_(pa.float32()),
        ),
    }

    for feat in FLOAT_FEATURES:
        columns[feat] = pa.array(np.asarray(data[feat]).astype(np.float32))

    for feat in BOOL_FEATURES:
        columns[feat] = pa.array(np.asarray(data[feat]).astype(bool))

    if include_best_fit:
        for feat in BEST_FIT_FLOAT_FEATURES:
            columns[feat] = pa.array(np.asarray(data[feat]).astype(np.float32))

    return pa.table(columns)


def load_data(raw_root: str) -> Table:
    """Read the PROVABGS HDF5 file into an astropy Table."""
    path = os.path.join(raw_root, FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PROVABGS file: {path}")
    return Table.read(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Directory containing BGS_ANY_full.provabgs.sv3.v0.hdf5.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
        help="Where to write the HATS catalog (collection root).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ignored (single-file dataset; kept for CLI consistency).",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8192,
        help="Max rows per HATS partition.",
    )
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    parser.add_argument(
        "--skip-best-fit",
        action="store_true",
        help="Skip Z_MW/TAGE_MW/AVG_SFR computation (requires provabgs package).",
    )
    args = parser.parse_args(argv)

    print(f"Reading PROVABGS data from {args.raw_root}")
    try:
        data = load_data(args.raw_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(data)} rows")

    mask = selection_mask(data)
    data = data[mask]
    print(f"{len(data)} rows after selection cuts")

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        ra = np.asarray(data["RA"])
        dec = np.asarray(data["DEC"])
        cone_mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
        data = data[cone_mask]
        print(f"{len(data)} rows survive cone cut")
        if len(data) == 0:
            print("ERROR: no rows survived the cone cut", file=sys.stderr)
            return 1

    if not args.skip_best_fit:
        print("Computing best-fit properties (Z_MW, TAGE_MW, AVG_SFR)...")
        data = compute_best_fit_properties(data)

    print(f"Building Arrow table from {len(data)} rows")
    table = build_arrow_table(data, include_best_fit=not args.skip_best_fit)

    print(f"Writing HATS catalog to {args.output_root}")
    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
