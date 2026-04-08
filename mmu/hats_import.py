"""Helpers for writing MMU survey data to HATS."""

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from inspect import signature

import numpy as np
import pyarrow as pa
from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader, ParquetPyarrowReader
from hats_import.pipeline import pipeline_with_client

@contextmanager
def numpy_unique_sorted_compat():
    """Backfill ``np.unique(sorted=...)`` for older NumPy versions."""

    if "sorted" in signature(np.unique).parameters:
        yield
        return

    original_unique = np.unique

    def compat_unique(
        ar,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        axis=None,
        *,
        equal_nan=True,
        sorted=True,
    ):
        return original_unique(
            ar,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
            axis=axis,
            equal_nan=equal_nan,
        )

    np.unique = compat_unique
    try:
        yield
    finally:
        np.unique = original_unique


def to_native_endian(array: np.ndarray) -> np.ndarray:
    """Return ``array`` in native byte order."""
    if array.dtype.byteorder in ("=", "|"):
        return array
    return array.byteswap().view(array.dtype.newbyteorder("="))


def np_to_pyarrow_list(array: np.ndarray) -> pa.Array:
    """Convert a 1D or 2D numpy array to Arrow."""
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
    """``hats-import`` reader for in-memory PyArrow tables."""

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
    """Yield a dask client, reusing ``client`` when one is provided."""
    if client is not None:
        yield client
        return

    if debug:
        kwargs = {
            "n_workers": 1,
            "threads_per_worker": 1,
            "processes": False,
            "dashboard_address": None,
        }
    else:
        kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": 1,
            "dashboard_address": None,
        }

    with Client(**kwargs) as c:
        yield c


def _run_hats_import(
    input_file_list: list[str],
    file_reader: InputReader,
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 8192,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    n_workers: int = 1,
    debug: bool = True,
    client: Client | None = None,
) -> str:
    """Write rows exposed by ``file_reader`` as a HATS catalog."""
    tmp_dir = tempfile.mkdtemp(prefix=f"hats_import_{catalog_name}_")
    try:
        with numpy_unique_sorted_compat():
            import_args = (
                CollectionArguments(
                    output_artifact_name=catalog_name,
                    output_path=output_path,
                    tmp_dir=tmp_dir,
                )
                .catalog(
                    input_file_list=input_file_list,
                    file_reader=file_reader,
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


def write_hats(
    tables: list[pa.Table],
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 8192,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    n_workers: int = 1,
    debug: bool = True,
    client: Client | None = None,
) -> str:
    """Write in-memory tables to HATS."""
    reader = ArrowTableReader(tables)
    return _run_hats_import(
        input_file_list=[str(i) for i in range(len(tables))],
        file_reader=reader,
        output_path=output_path,
        catalog_name=catalog_name,
        pixel_threshold=pixel_threshold,
        lowest_healpix_order=lowest_healpix_order,
        margin_threshold_arcsec=margin_threshold_arcsec,
        n_workers=n_workers,
        debug=debug,
        client=client,
    )


def write_hats_from_parquet(
    parquet_files: list[str],
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 8192,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    n_workers: int = 1,
    debug: bool = True,
    chunksize: int = 500_000,
    client: Client | None = None,
) -> str:
    """Write a HATS catalog from parquet shard files."""
    if not parquet_files:
        raise FileNotFoundError("no parquet shard files provided")

    reader = ParquetPyarrowReader(chunksize=chunksize)
    return _run_hats_import(
        input_file_list=parquet_files,
        file_reader=reader,
        output_path=output_path,
        catalog_name=catalog_name,
        pixel_threshold=pixel_threshold,
        lowest_healpix_order=lowest_healpix_order,
        margin_threshold_arcsec=margin_threshold_arcsec,
        n_workers=n_workers,
        debug=debug,
        client=client,
    )


def write_hats_from_parquet_dir(
    parquet_dir: str,
    output_path: str,
    catalog_name: str,
    *,
    pixel_threshold: int = 8192,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    chunksize: int = 500_000,
    n_workers: int = 1,
    debug: bool = True,
    client: Client | None = None,
) -> str:
    """Write a HATS catalog from a parquet directory."""
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"parquet_dir does not exist: {parquet_dir}")

    parquet_files = sorted(
        glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True)
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"no *.parquet files under {parquet_dir} (recursive search)"
        )

    return write_hats_from_parquet(
        parquet_files,
        output_path=output_path,
        catalog_name=catalog_name,
        pixel_threshold=pixel_threshold,
        lowest_healpix_order=lowest_healpix_order,
        margin_threshold_arcsec=margin_threshold_arcsec,
        n_workers=n_workers,
        debug=debug,
        chunksize=chunksize,
        client=client,
    )
