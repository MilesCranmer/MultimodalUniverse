"""One-off ingest helper.

Reads all parquets in ``--scratch-dir`` and runs ``write_hats_from_parquet_dir``
to produce a HATS catalog. Used by manual ``sbatch --wrap=...`` gather jobs
that don't go through the Snakefile (e.g. relaunching a partial gather without
restarting the whole snakemake DAG).

Must use the ``if __name__ == "__main__":`` guard because dask's
``LocalCluster`` spawns worker processes via ``multiprocessing.spawn``, which
re-imports the script as ``__main__``. Without the guard, every spawned worker
re-runs ``write_hats_from_parquet_dir`` recursively and the parent loops on the
"Safe importing of main module" RuntimeError.
"""

from __future__ import annotations

import argparse


def main() -> int:
    from mmu.hats_import import write_hats_from_parquet_dir

    p = argparse.ArgumentParser()
    p.add_argument("--scratch-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--catalog-name", required=True)
    p.add_argument("--pixel-threshold", type=int, default=8192)
    p.add_argument("--ingest-workers", type=int, default=8)
    args = p.parse_args()

    print(
        f"Ingesting {args.scratch_dir} -> {args.output_root} ({args.catalog_name})",
        flush=True,
    )
    out = write_hats_from_parquet_dir(
        args.scratch_dir,
        output_path=args.output_root,
        catalog_name=args.catalog_name,
        pixel_threshold=args.pixel_threshold,
        n_workers=args.ingest_workers,
        debug=False,
    )
    print(f"Done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
