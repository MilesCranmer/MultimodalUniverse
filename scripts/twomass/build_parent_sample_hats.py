"""Convert raw 2MASS Point Source Catalog into a HATS catalog.

The cluster mirrors the 2MASS PSC at:

    /mnt/ceph/users/polymathic/external_data/astro/twomass/psc/psc_*.gz

Each file is a gzipped pipe-delimited table (~3-5M rows). Columns are
documented at https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_2a.html
and the column types live in ``scripts/twomass/twomass.py``'s ``_mapping``.

Conversion is straightforward: parse each gzipped CSV with PyArrow, normalize
``decl`` -> ``dec``, cast ``pts_key`` -> ``object_id`` (string), and feed all
shards to ``write_hats``.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats_from_parquet_dir


CATALOG_NAME = "twomass"


def _load_mapping() -> dict[str, str]:
    """Import ``_mapping`` from scripts/twomass/twomass.py without polluting sys.modules."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = spec_from_file_location("_twomass_legacy", os.path.join(here, "twomass.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._mapping


COLUMN_MAPPING = _load_mapping()


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "psc_*.gz")))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_shard(path: str) -> pa.Table:
    """Parse one gzipped pipe-delimited 2MASS PSC shard into a PyArrow table."""
    # The legacy _mapping dict has both `decl` (the actual 2MASS column) and
    # `dec` (a v1 alias). The raw files only have `decl`, so we strip `dec`
    # from the read schema and add it as a copy after parsing.
    read_columns = [c for c in COLUMN_MAPPING if c != "dec"]
    read_types = {k: v for k, v in COLUMN_MAPPING.items() if k != "dec"}

    table = pcsv.read_csv(
        path,
        read_options=pcsv.ReadOptions(use_threads=True, column_names=read_columns),
        parse_options=pcsv.ParseOptions(delimiter="|"),
        convert_options=pcsv.ConvertOptions(
            column_types=read_types,
            null_values=["\\N"],
        ),
    )
    # 2MASS uses `decl`; expose it as `dec` for HATS.
    table = table.append_column("dec", table.column("decl"))
    # object_id is the integer pts_key, cast to string.
    pts_key = table.column("pts_key").to_numpy()
    obj = pa.array([str(int(k)) for k in pts_key], type=pa.string())
    table = table.append_column("object_id", obj)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root of the raw 2MASS PSC files (psc_*.gz).",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--scratch-dir", default=None,
                        help="Scratch dir for per-shard parquet files (default: "
                             "ceph MultimodalUniverse_v2_hats_scratch/twomass_<pid>).")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    cone_active = (
        args.ra_center is not None
        and args.dec_center is not None
        and args.radius is not None
    )

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no psc_*.gz under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} 2MASS shard(s)")

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    print(f"Streaming per-shard parquet to {scratch}")

    try:
        total = 0
        n_written = 0
        for i, p in enumerate(files, 1):
            t = read_shard(p)
            if cone_active:
                ra = t.column("ra").to_numpy()
                dec = t.column("dec").to_numpy()
                mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
                t = t.filter(pa.array(mask))
                if t.num_rows == 0:
                    print(f"  [{i}/{len(files)}] {os.path.basename(p)}: 0 rows (cone filtered)")
                    continue
            out_path = os.path.join(scratch, f"part-{os.path.basename(p)}.parquet")
            pq.write_table(t, out_path)
            n_written += 1
            total += t.num_rows
            print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} rows → part-{i:04d}")
            del t

        if n_written == 0:
            print("ERROR: no rows survived the cone cut", file=sys.stderr)
            return 1

        print(f"\nIngesting {n_written} shard(s), {total} rows → HATS catalog")
        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
        )
        print(f"Done: {catalog_dir}")
    finally:
        if args.scratch_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
