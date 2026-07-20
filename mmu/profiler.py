"""Pre-flight profiler for HATS build scripts.

Processes a small sample of rows, runs a mini gather, measures peak RSS
and throughput, checks for common pitfalls, and recommends resource
parameters (workers, walltime, pixel_threshold) for production.

Usage::

    python -m mmu.profiler manga --n-sample 3
    python -m mmu.profiler legacysurvey --n-sample 5
"""

from __future__ import annotations

import argparse
import glob
import importlib
import inspect
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ProfileResult:
    dataset: str
    n_sample: int
    n_output_rows: int
    scatter_peak_rss_mb: float
    scatter_rss_per_row_mb: float
    scatter_rows_per_sec: float
    scatter_elapsed_sec: float
    gather_peak_rss_mb: float | None
    gather_elapsed_sec: float | None
    warnings: list[str] = field(default_factory=list)
    recommended_workers: int = 1
    recommended_scatter_walltime_h: float = 48
    recommended_gather_walltime_h: float = 48
    recommended_pixel_threshold: int = 100_000

    def __str__(self) -> str:
        lines = [
            f"=== Profile: {self.dataset} ({self.n_sample} sample files → {self.n_output_rows} rows) ===",
            f"  Scatter:",
            f"    Peak RSS:        {self.scatter_peak_rss_mb:.0f} MB",
            f"    RSS per row:     {self.scatter_rss_per_row_mb:.0f} MB",
            f"    Throughput:      {self.scatter_rows_per_sec:.2f} rows/sec",
            f"    Elapsed:         {self.scatter_elapsed_sec:.1f}s",
        ]
        if self.gather_peak_rss_mb is not None:
            lines += [
                f"  Gather (mini):",
                f"    Peak RSS:        {self.gather_peak_rss_mb:.0f} MB",
                f"    Elapsed:         {self.gather_elapsed_sec:.1f}s",
            ]
        lines += [
            f"  Recommendations (900 GB node):",
            f"    Workers:         {self.recommended_workers}",
            f"    Scatter wall:    {self.recommended_scatter_walltime_h:.0f}h",
            f"    Gather wall:     {self.recommended_gather_walltime_h:.0f}h",
            f"    pixel_threshold: {self.recommended_pixel_threshold}",
        ]
        if self.warnings:
            lines.append(f"  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


def _measure_rss_mb() -> float:
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rss / 1e6


def _check_schema_warnings(parquet_path: str) -> list[str]:
    """Check a parquet file's schema for common pitfalls."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    warnings = []
    schema = pq.read_schema(parquet_path)

    def _check_type(name: str, t: pa.DataType, depth: int = 0):
        if pa.types.is_list(t) and not pa.types.is_large_list(t):
            inner = t.value_type
            if pa.types.is_list(inner) or pa.types.is_large_list(inner):
                warnings.append(
                    f"Column '{name}' uses pa.list_ for nested arrays. "
                    f"Will overflow on >30k rows. Use pa.large_list instead."
                )
        if pa.types.is_struct(t):
            for i in range(t.num_fields):
                _check_type(f"{name}.{t.field(i).name}", t.field(i).type, depth + 1)
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            _check_type(name, t.value_type, depth + 1)

    for i in range(len(schema)):
        _check_type(schema.field(i).name, schema.field(i).type)

    if "ra" not in schema.names:
        warnings.append("No 'ra' column — hats-import will fail")
    if "dec" not in schema.names:
        warnings.append("No 'dec' column — hats-import will fail")

    return warnings


def profile_build(
    dataset: str,
    n_sample: int = 5,
    node_mem_mb: int = 900_000,
    safety_factor: float = 2.5,
    run_gather: bool = True,
) -> ProfileResult:
    """Profile a dataset's build script on a small sample."""
    import pyarrow.parquet as pq

    from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Known: {list(DATASETS.keys())}")

    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", dataset,
        "build_parent_sample_hats.py",
    )
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"No build script at {script_path}")

    spec = importlib.util.spec_from_file_location(f"_profile_{dataset}", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tmp_output = f"/tmp/mmu_profile_{dataset}_{os.getpid()}"
    tmp_scratch = f"/tmp/mmu_profile_scratch_{dataset}_{os.getpid()}"
    os.makedirs(tmp_output, exist_ok=True)
    os.makedirs(tmp_scratch, exist_ok=True)

    main_src = inspect.getsource(mod.main)

    argv = [
        "--output-root", tmp_output,
        "--max-files", str(n_sample),
    ]
    # Only pass --num-processes if the script accepts it
    if "num-processes" in main_src or "num_processes" in main_src:
        argv += ["--num-processes", "1"]
    has_scratch = "scratch-dir" in main_src or "scratch_dir" in main_src
    if has_scratch:
        argv += ["--scratch-dir", tmp_scratch, "--skip-ingest"]

    # === SCATTER PHASE ===
    rss_before = _measure_rss_mb()
    t0 = time.time()
    try:
        rc = mod.main(argv)
    except SystemExit as e:
        rc = e.code or 0
    scatter_elapsed = time.time() - t0
    scatter_peak_rss = _measure_rss_mb()

    parquets = sorted(glob.glob(os.path.join(tmp_scratch, "**/*.parquet"), recursive=True)) or \
               sorted(glob.glob(os.path.join(tmp_output, "**/*.parquet"), recursive=True))
    n_output_rows = sum(pq.read_metadata(f).num_rows for f in parquets) if parquets else 0

    warnings = []

    # Check: did we produce any output?
    if n_output_rows == 0:
        warnings.append(
            f"ZERO output rows from {n_sample} sample files! "
            f"The build script is silently failing. Do NOT launch at scale."
        )

    # Check: schema pitfalls
    if parquets:
        warnings.extend(_check_schema_warnings(parquets[0]))

    # Check: stale output at production path
    prod_output = os.path.join(MMU_V2_HATS_ROOT, dataset, dataset, dataset)
    if os.path.isdir(prod_output):
        existing = glob.glob(os.path.join(prod_output, "dataset", "**/*.parquet"), recursive=True)
        if existing:
            warnings.append(
                f"Production output dir has {len(existing)} existing parquets. "
                f"Pre-gather cleanup will wipe them, but verify this is intended."
            )

    # === GATHER PHASE (mini) ===
    gather_peak_rss = None
    gather_elapsed = None
    if run_gather and parquets and n_output_rows > 0:
        from mmu.hats_import import write_hats_from_parquet_dir

        gather_input = tmp_scratch if has_scratch else tmp_output
        gather_output = f"/tmp/mmu_profile_gather_{dataset}_{os.getpid()}"
        os.makedirs(gather_output, exist_ok=True)

        rss_before_gather = _measure_rss_mb()
        t1 = time.time()
        try:
            write_hats_from_parquet_dir(
                gather_input,
                output_path=gather_output,
                catalog_name=dataset,
                pixel_threshold=100_000,
                n_workers=1,
                debug=True,
            )
        except Exception as e:
            warnings.append(f"Mini gather FAILED: {type(e).__name__}: {e}")
        gather_elapsed = time.time() - t1
        gather_peak_rss = _measure_rss_mb()
        shutil.rmtree(gather_output, ignore_errors=True)

    # Clean up
    shutil.rmtree(tmp_output, ignore_errors=True)
    shutil.rmtree(tmp_scratch, ignore_errors=True)

    # === RECOMMENDATIONS ===
    rss_delta = max(scatter_peak_rss - rss_before, 50)
    rss_per_row = rss_delta / max(n_output_rows, 1)
    rows_per_sec = n_output_rows / max(scatter_elapsed, 0.01)

    # Scatter OOM prediction: build_table triples memory (records → lists → arrow).
    # With num_processes workers, worst case = all workers in build_table simultaneously.
    # Estimate per-worker peak from sample parquet sizes.
    BUILD_TABLE_MULTIPLIER = 3
    if parquets and n_output_rows > 0:
        avg_parquet_bytes = sum(os.path.getsize(f) for f in parquets) / len(parquets)
        avg_rows_per_parquet = n_output_rows / len(parquets)
        bytes_per_row_on_disk = avg_parquet_bytes / max(avg_rows_per_parquet, 1)
        in_mem_per_row_mb = bytes_per_row_on_disk * 2.0 / 1e6  # parquet → in-memory ~2x

        # Get num_processes from script default
        num_processes = 4  # fallback
        if "num_processes" in main_src or "num-processes" in main_src:
            import re
            m = re.search(r'default\s*=\s*(\d+)',
                          main_src[main_src.find("num-processes"):main_src.find("num-processes")+100]
                          if "num-processes" in main_src else
                          main_src[main_src.find("num_processes"):main_src.find("num_processes")+100])
            if m:
                num_processes = int(m.group(1))

        # Estimate max rows per work unit (sweep/chunk/plate-ifu)
        # From the sample: n_output_rows from n_sample files
        max_rows_per_unit = n_output_rows  # conservative: assume one unit had all rows
        if len(parquets) > 1:
            max_rows_per_unit = max(pq.read_metadata(f).num_rows for f in parquets)

        # Peak per worker during build_table
        peak_per_worker_mb = max_rows_per_unit * in_mem_per_row_mb * BUILD_TABLE_MULTIPLIER
        # All workers hitting build_table simultaneously (worst case)
        peak_total_mb = num_processes * peak_per_worker_mb

        if peak_total_mb > node_mem_mb * 0.9:
            safe_workers = max(1, int(node_mem_mb * 0.8 / max(peak_per_worker_mb, 1)))
            warnings.append(
                f"SCATTER OOM RISK: build_table triples memory. "
                f"With Pool({num_processes}), worst case = {num_processes} × "
                f"{peak_per_worker_mb/1000:.0f} GB/worker = {peak_total_mb/1000:.0f} GB "
                f"(node has {node_mem_mb/1000:.0f} GB). "
                f"Recommend Pool({safe_workers})."
            )
            recommended_workers = safe_workers
        else:
            recommended_workers = num_processes
    else:
        recommended_workers = 1

    # Cap workers
    recommended_workers = max(1, min(32, recommended_workers))

    # Scatter walltime
    recommended_scatter_h = 48.0  # safe default

    # Gather walltime: extrapolate from mini gather
    recommended_gather_h = 48.0
    if gather_elapsed and n_output_rows > 0:
        gather_per_row = gather_elapsed / n_output_rows
        # Estimate total rows (rough: n_output_rows / n_sample * total_files)
        # We don't know total_files here, so just use a generous multiplier
        recommended_gather_h = max(4, gather_per_row * n_output_rows * 1000 / 3600 * safety_factor)
        recommended_gather_h = min(recommended_gather_h, 48)

    # Pixel threshold
    recommended_pixel_threshold = 100_000

    # Gather memory estimation: biggest partition = min(total_estimated_rows, pixel_threshold)
    # Each row in a partition is held in memory during reduce.
    if parquets and n_output_rows > 0:
        avg_parquet_bytes = sum(os.path.getsize(f) for f in parquets) / len(parquets)
        avg_rows_per_parquet = n_output_rows / len(parquets)
        bytes_per_row = avg_parquet_bytes / max(avg_rows_per_parquet, 1)
        # In-memory representation is typically 2-5x larger than parquet (decompression)
        mem_per_row_mb = bytes_per_row * 3.0 / 1e6  # 3x decompression factor
        worst_partition_rows = min(recommended_pixel_threshold, n_output_rows * 1000)  # rough total estimate
        gather_partition_mem_mb = worst_partition_rows * mem_per_row_mb
        if gather_partition_mem_mb > node_mem_mb / 2:
            warnings.append(
                f"GATHER OOM RISK: worst-case partition ({worst_partition_rows} rows × "
                f"{mem_per_row_mb:.0f} MB/row) = {gather_partition_mem_mb/1000:.0f} GB. "
                f"Node has {node_mem_mb/1000:.0f} GB. Reduce pixel_threshold or increase workers."
            )

    # Gather walltime estimation
    if gather_elapsed and n_output_rows > 0:
        gather_per_row_sec = gather_elapsed / n_output_rows
        estimated_total_rows = n_output_rows * 1000  # rough
        estimated_gather_h = gather_per_row_sec * estimated_total_rows / 3600
        if estimated_gather_h > 4:
            warnings.append(
                f"GATHER SLOW: {gather_per_row_sec:.1f}s/row × ~{estimated_total_rows} rows "
                f"= ~{estimated_gather_h:.0f}h. Ensure walltime is sufficient."
            )

    return ProfileResult(
        dataset=dataset,
        n_sample=n_sample,
        n_output_rows=n_output_rows,
        scatter_peak_rss_mb=scatter_peak_rss,
        scatter_rss_per_row_mb=rss_per_row,
        scatter_rows_per_sec=rows_per_sec,
        scatter_elapsed_sec=scatter_elapsed,
        gather_peak_rss_mb=gather_peak_rss,
        gather_elapsed_sec=gather_elapsed,
        warnings=warnings,
        recommended_workers=recommended_workers,
        recommended_scatter_walltime_h=recommended_scatter_h,
        recommended_gather_walltime_h=recommended_gather_h,
        recommended_pixel_threshold=recommended_pixel_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Profile a HATS build script")
    parser.add_argument("dataset", help="Dataset name (e.g., manga, legacysurvey)")
    parser.add_argument("--n-sample", type=int, default=5)
    parser.add_argument("--no-gather", action="store_true", help="Skip mini gather test")
    parser.add_argument("--node-mem-mb", type=int, default=900_000)
    args = parser.parse_args()

    result = profile_build(
        args.dataset,
        n_sample=args.n_sample,
        node_mem_mb=args.node_mem_mb,
        run_gather=not args.no_gather,
    )
    print(result)


if __name__ == "__main__":
    main()
