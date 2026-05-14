"""Shared infrastructure for building HATS catalogs from MMU survey data.

The public surface is small:

- ``np_to_pyarrow_list`` — convert a 2D numpy array (one row per object) to a
  PyArrow ListArray suitable for use as a column in a HATS catalog.
- ``ArrowTableReader`` — adapter that lets ``hats-import`` consume in-memory
  PyArrow tables instead of reading from disk.
- ``write_hats`` — feed a list of PyArrow tables through the ``hats-import``
  pipeline to produce a HATS catalog (Parquet + healpix partitioning + margin
  cache + collection wrapper).

Per-survey ``build_parent_sample_hats.py`` scripts (under ``scripts/{survey}/``)
build their own PyArrow tables from the survey-native raw inputs and call
``write_hats``. This module deliberately knows nothing about HDF5.
"""

import contextlib
import glob
import logging
import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader, ParquetPyarrowReader
from hats_import.pipeline import pipeline_with_client

LOGGER = logging.getLogger(__name__)


# Sibling to the HATS output root on ceph. Per-shard parquet files from
# streaming build scripts go under here as ``<scratch>/<catalog>_<pid>/part-*.parquet``
# so peak RAM stays bounded by one shard instead of the full catalog.
# Kept on ceph (not /tmp) so the scratch dir survives compute-node failure
# and so we can see how far a failed build got.
DEFAULT_SCRATCH_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_scratch"


def default_scratch_dir(catalog_name: str) -> str:
    """Return the canonical per-catalog scratch directory on ceph.

    The directory is namespaced by catalog name and the current process ID,
    so concurrent builds of the same dataset don't stomp on each other.
    Callers are responsible for creating it and for cleaning it up once
    ``write_hats_from_parquet_dir`` has returned successfully.
    """
    return os.path.join(DEFAULT_SCRATCH_ROOT, f"{catalog_name}_{os.getpid()}")


def to_native_endian(array: np.ndarray) -> np.ndarray:
    """Return a copy of ``array`` in the machine's native byte order.

    Astropy reads FITS files in the file's byte order (usually big-endian) but
    PyArrow refuses to ingest byte-swapped arrays — convert before handing to
    ``pa.array``.
    """
    if array.dtype.byteorder in ("=", "|"):
        return array
    return array.byteswap().view(array.dtype.newbyteorder("="))


def np_to_pyarrow_list(array: np.ndarray) -> pa.Array:
    """Convert a 1D numpy array to a flat PyArrow array, or a 2D numpy array
    (shape ``(n_rows, length)``) to a PyArrow ListArray with one ``length``-long
    list per row.
    """
    if array.dtype.byteorder == ">":
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    values = pa.array(array.reshape(-1))
    if array.ndim == 1:
        return values
    if array.ndim != 2:
        raise ValueError(
            f"np_to_pyarrow_list expects 1D or 2D, got ndim={array.ndim}"
        )
    n_lists, length = array.shape
    offsets = np.arange(0, (n_lists + 1) * length, length, dtype=np.int32)
    return pa.ListArray.from_arrays(values=values, offsets=offsets)


class ArrowTableReader(InputReader):
    """``hats-import`` ``InputReader`` that yields pre-built PyArrow tables.

    Each "input file" is just an integer index into the in-memory table list.
    """

    def __init__(self, tables: list[pa.Table]):
        self.tables = tables

    def read(self, input_file, read_columns=None):
        idx = int(input_file)
        table = self.tables[idx]
        if read_columns:
            table = table.select(read_columns)
        yield table


@contextlib.contextmanager
def _client_ctx(client: Client | None, n_workers: int, debug: bool):
    """Yield a dask Client. If ``client`` is provided, use it as-is and DO NOT
    close it on exit (the caller owns its lifetime). Otherwise build a private
    LocalCluster client and close it on exit.
    """
    if client is not None:
        yield client
        return
    if debug:
        kwargs = {"n_workers": 1, "threads_per_worker": 1, "processes": False}
    else:
        kwargs = {"n_workers": n_workers, "threads_per_worker": 1}
    with Client(**kwargs, dashboard_address=None) as c:
        # Log memory pressure periodically so we can see OOM coming
        import threading
        import psutil

        def _mem_monitor():
            proc = psutil.Process()
            while not _stop_monitor.is_set():
                mem = proc.memory_info()
                children_rss = sum(
                    ch.memory_info().rss for ch in proc.children(recursive=True)
                )
                total_gb = (mem.rss + children_rss) / 1e9
                avail_gb = psutil.virtual_memory().available / 1e9
                LOGGER.info(
                    "Memory: process+children=%.1f GB, system_available=%.1f GB",
                    total_gb, avail_gb,
                )
                _stop_monitor.wait(60)  # log every 60s

        _stop_monitor = threading.Event()
        monitor = threading.Thread(target=_mem_monitor, daemon=True)
        monitor.start()
        try:
            yield c
        finally:
            _stop_monitor.set()
            monitor.join(timeout=5)


def write_hats(
    tables: list[pa.Table],
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 100_000,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    n_workers: int = 32,
    debug: bool = False,
    client: Client | None = None,
) -> str:
    """Write a list of PyArrow tables as a HATS catalog under ``output_path``.

    This is the simple in-memory path: ``tables`` are bundled by
    :class:`ArrowTableReader` and pickled to every dask worker, so total RAM
    demand scales as ``len(tables) * sum(table.nbytes) * n_workers``. Fine
    for small/medium datasets (~tens of GB total). For large datasets that
    would OOM, use :func:`write_hats_from_parquet_dir` instead — it streams
    via hats-import's native parquet directory reader.

    Args:
        tables: PyArrow tables to write. Must contain ``ra`` (float64) and
            ``dec`` (float64) columns.
        output_path: Root directory under which the catalog collection is written.
        catalog_name: Name of the output catalog (becomes both the collection
            and the inner catalog directory name).
        pixel_threshold: Max rows per HATS partition.
        lowest_healpix_order: Coarsest HEALPix order used for partitioning.
        margin_threshold_arcsec: Width of the margin cache in arcseconds.
        n_workers: Dask workers to use in production mode (ignored if ``client``
            is provided).
        debug: If True, run single-process / single-thread (ignored if ``client``
            is provided).
        client: Pre-built dask ``Client`` to reuse. If None, a private
            LocalCluster client is created and torn down per call.

    Returns:
        The path to the inner catalog directory ``output_path/{catalog_name}/{catalog_name}``.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"hats_import_{catalog_name}_")
    try:
        reader = ArrowTableReader(tables)
        import_args = (
            CollectionArguments(
                output_artifact_name=catalog_name,
                output_path=output_path,
                tmp_dir=tmp_dir,
            )
            .catalog(
                input_file_list=[str(i) for i in range(len(tables))],
                file_reader=reader,
                ra_column="ra",
                dec_column="dec",
                pixel_threshold=pixel_threshold,
                lowest_healpix_order=lowest_healpix_order,
            )
            .add_margin(margin_threshold=margin_threshold_arcsec, is_default=True)
        )
        with _client_ctx(client, n_workers, debug) as c:
            pipeline_with_client(import_args, c)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return os.path.join(output_path, catalog_name, catalog_name)


def write_hats_from_parquet_dir(
    parquet_dir: str,
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 100_000,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    chunksize: int = 500_000,
    n_workers: int = 32,
    debug: bool = False,
    client: Client | None = None,
    tmp_dir: str | None = None,
) -> str:
    """Write a HATS catalog by streaming parquet files from a directory.

    Unlike :func:`write_hats`, this never holds the input tables in RAM —
    hats-import's :class:`ParquetPyarrowReader` opens each parquet file in
    the directory and streams it via ``parquet_file.iter_batches`` with the
    given ``chunksize``. Use this for any dataset whose accumulated input
    would exceed available RAM (e.g. desi, gaia, allwise at full scale).

    Caller responsibilities:
        1. Pre-write per-shard parquet files into ``parquet_dir`` containing
           ``ra`` (float64) and ``dec`` (float64) columns. Each parquet file
           in the directory becomes one hats-import map/split task, so for
           parallelism aim for O(100+) files.
        2. Optionally pass an external dask ``client`` so multiple parquet
           directories can share one cluster across multiple write_hats calls.

    Args:
        parquet_dir: Directory containing the per-shard parquet files. Every
            ``*.parquet`` file under this directory will be ingested.
        output_path: Root directory under which the catalog collection is written.
        catalog_name: Name of the output catalog.
        pixel_threshold: Max rows per HATS partition.
        lowest_healpix_order: Coarsest HEALPix order used for partitioning.
        margin_threshold_arcsec: Width of the margin cache in arcseconds.
        chunksize: ``ParquetPyarrowReader.iter_batches`` batch size; controls
            map-stage memory per dask worker.
        n_workers: Dask workers (ignored if ``client`` is provided).
        debug: Single-process mode if no client provided.
        client: Pre-built dask ``Client`` to reuse.

    Returns:
        The path to the inner catalog directory.
    """
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"parquet_dir does not exist: {parquet_dir}")

    # Enumerate parquet files explicitly. We pass them via input_file_list
    # rather than relying on hats-import's input_path glob, because the
    # latter calls find_files_matching_path(input_path, "**/*.*") which uses
    # rglob with a `**/*.*` pattern that doesn't reliably match files at the
    # top of the directory across different filesystem backends.
    parquet_files = sorted(
        glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True)
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"no *.parquet files under {parquet_dir} (recursive search)"
        )

    LOGGER.info(
        "write_hats_from_parquet_dir: %d input parquets, pixel_threshold=%d, "
        "n_workers=%d, chunksize=%d, output=%s/%s",
        len(parquet_files), pixel_threshold, n_workers, chunksize,
        output_path, catalog_name,
    )

    # Clean stale output from a previous failed gather. hats-import's
    # Finishing stage scans the entire dataset/ dir for parquet files; if
    # a prior run left partial output at a different pixel_threshold, the
    # row counts won't match and the pipeline errors with a cryptic
    # "Number of rows does not match expectation" ValueError.
    inner_catalog = os.path.join(output_path, catalog_name, catalog_name)
    if os.path.isdir(inner_catalog):
        LOGGER.warning(
            "Cleaning stale output dir %s from a previous failed gather",
            inner_catalog,
        )
        shutil.rmtree(inner_catalog)

    if tmp_dir is not None:
        os.makedirs(tmp_dir, exist_ok=True)
    ingest_tmp = tempfile.mkdtemp(prefix=f"hats_import_{catalog_name}_", dir=tmp_dir)
    try:
        import_args = (
            CollectionArguments(
                output_artifact_name=catalog_name,
                output_path=output_path,
                tmp_dir=ingest_tmp,
            )
            .catalog(
                input_file_list=parquet_files,
                file_reader=ParquetPyarrowReader(chunksize=chunksize),
                ra_column="ra",
                dec_column="dec",
                pixel_threshold=pixel_threshold,
                lowest_healpix_order=lowest_healpix_order,
            )
            .add_margin(margin_threshold=margin_threshold_arcsec, is_default=True)
        )
        with _client_ctx(client, n_workers, debug) as c:
            pipeline_with_client(import_args, c)
    finally:
        shutil.rmtree(ingest_tmp, ignore_errors=True)

    return os.path.join(output_path, catalog_name, catalog_name)
