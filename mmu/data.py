"""HATS-native data loading for Multimodal Universe v2.

Provides PyTorch Dataset and Lightning DataModule backed by HATS catalogs
with lazy column loading, spatial filtering, and cross-matching.
"""

import typing as T
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

import hats
import lsdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_MAX_CACHED_PIXELS = 4


@dataclass(frozen=True)
class _ConeFilter:
    ra: float
    dec: float
    radius_arcsec: float

    def apply_hats(self, cat: "hats.catalog.Catalog") -> "hats.catalog.Catalog":
        return cat.filter_by_cone(self.ra, self.dec, self.radius_arcsec)

    def apply_lsdb(self, cat: "lsdb.Catalog") -> "lsdb.Catalog":
        return cat.cone_search(self.ra, self.dec, self.radius_arcsec)


@dataclass(frozen=True)
class _BoxFilter:
    ra_range: tuple[float, float]
    dec_range: tuple[float, float]

    def apply_hats(self, cat: "hats.catalog.Catalog") -> "hats.catalog.Catalog":
        return cat.filter_by_box(self.ra_range, self.dec_range)

    def apply_lsdb(self, cat: "lsdb.Catalog") -> "lsdb.Catalog":
        return cat.box_search(ra=self.ra_range, dec=self.dec_range)


class HATSDataset(Dataset):
    """PyTorch Dataset backed by a HATS catalog.

    Reads only the requested columns from Parquet files, caching loaded
    pixels in memory. Supports spatial filtering via cone/box and cross-matching
    via LSDB. All filter operations return a new HATSDataset that defers the
    same filter to both the underlying ``hats`` and ``lsdb`` views, so e.g.
    ``ds.filter_by_cone(...).crossmatch(other)`` matches against the cone, not
    the unfiltered catalog.

    Args:
        catalog_path: Path to a HATS catalog directory.
        columns: Columns to load. None means all columns.
        catalog: Pre-loaded hats.Catalog (alternative to catalog_path).
        max_cached_pixels: Maximum number of HEALPix tiles to keep in the
            in-memory LRU cache. Each tile can be up to ``pixel_threshold``
            rows wide so this caps memory at roughly
            ``max_cached_pixels * pixel_threshold * row_bytes``. Defaults to
            ``DEFAULT_MAX_CACHED_PIXELS`` (4).
        _filters: Internal — list of filter operations to apply lazily.
    """

    def __init__(
        self,
        catalog_path: str | None = None,
        columns: list[str] | None = None,
        catalog: hats.catalog.Catalog | None = None,
        max_cached_pixels: int = DEFAULT_MAX_CACHED_PIXELS,
        _filters: tuple = (),
    ):
        if catalog is not None:
            self.catalog = catalog
            self._catalog_base_dir = str(catalog.catalog_base_dir)
        elif catalog_path is not None:
            self.catalog = hats.read_hats(catalog_path)
            self._catalog_base_dir = catalog_path
        else:
            raise ValueError("Provide either catalog_path or catalog")

        self.columns = columns
        self.max_cached_pixels = max_cached_pixels
        self._filters = tuple(_filters)
        # OrderedDict gives us O(1) LRU eviction via move_to_end / popitem(last=False).
        self._pixel_cache: "OrderedDict[int, pa.Table]" = OrderedDict()

    @cached_property
    def _pixel_paths(self) -> list:
        return list(self.catalog.get_pixel_paths())

    @cached_property
    def _pixel_row_counts(self) -> list[int]:
        return [pq.read_metadata(p).num_rows for p in self._pixel_paths]

    @cached_property
    def _cumulative_counts(self) -> np.ndarray:
        return np.cumsum([0] + self._pixel_row_counts)

    def __len__(self) -> int:
        return int(self._cumulative_counts[-1])

    def _load_pixel(self, pixel_idx: int) -> pa.Table:
        cache = self._pixel_cache
        if pixel_idx in cache:
            cache.move_to_end(pixel_idx)
            return cache[pixel_idx]
        table = pq.read_table(self._pixel_paths[pixel_idx], columns=self.columns)
        cache[pixel_idx] = table
        while len(cache) > self.max_cached_pixels:
            cache.popitem(last=False)  # evict the least-recently-used pixel
        return table

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        """Map global index to (pixel_idx, row_within_pixel)."""
        pixel_idx = int(np.searchsorted(self._cumulative_counts[1:], idx, side="right"))
        row_idx = idx - int(self._cumulative_counts[pixel_idx])
        return pixel_idx, row_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pixel_idx, row_idx = self._resolve_index(idx)
        table = self._load_pixel(pixel_idx)
        row = table.slice(row_idx, 1)
        return _arrow_row_to_torch(row)

    def filter_by_cone(self, ra: float, dec: float, radius_arcsec: float) -> "HATSDataset":
        f = _ConeFilter(ra, dec, radius_arcsec)
        new = HATSDataset(
            columns=self.columns,
            catalog=f.apply_hats(self.catalog),
            _filters=self._filters + (f,),
        )
        new._catalog_base_dir = self._catalog_base_dir
        return new

    def filter_by_box(
        self, ra: tuple[float, float], dec: tuple[float, float]
    ) -> "HATSDataset":
        f = _BoxFilter(ra, dec)
        new = HATSDataset(
            columns=self.columns,
            catalog=f.apply_hats(self.catalog),
            _filters=self._filters + (f,),
        )
        new._catalog_base_dir = self._catalog_base_dir
        return new

    def clear_cache(self):
        self._pixel_cache.clear()

    @property
    def schema(self) -> pa.Schema:
        return self.catalog.schema

    @cached_property
    def lsdb_catalog(self) -> lsdb.Catalog:
        """Return an LSDB catalog with this dataset's column projection AND any
        spatial filters applied. Reads the same on-disk catalog the hats view
        was built from."""
        cat = lsdb.read_hats(self._catalog_base_dir, columns=self.columns)
        for f in self._filters:
            cat = f.apply_lsdb(cat)
        return cat

    def crossmatch(
        self,
        other: "HATSDataset",
        radius_arcsec: float = 1.0,
        n_neighbors: int = 1,
        suffixes: tuple[str, str] = ("_left", "_right"),
    ) -> "CrossMatchedHATSDataset":
        """Cross-match this catalog against another using LSDB.

        Honors ``self.columns`` / ``other.columns`` (passed through to
        ``lsdb.read_hats(columns=...)``) and any spatial filters previously
        applied via ``filter_by_cone`` / ``filter_by_box``.

        Args:
            other: The other HATSDataset to match against.
            radius_arcsec: Maximum match radius in arcseconds.
            n_neighbors: Number of nearest neighbors to find.
            suffixes: Suffixes appended to overlapping column names.
        """
        result = self.lsdb_catalog.crossmatch(
            other.lsdb_catalog,
            n_neighbors=n_neighbors,
            radius_arcsec=radius_arcsec,
            suffixes=suffixes,
        )
        df = result.compute()
        return CrossMatchedHATSDataset(df, suffixes=suffixes)


class CrossMatchedHATSDataset(Dataset):
    """PyTorch Dataset from a cross-matched result.

    Holds the materialized cross-match result as a DataFrame and provides
    indexed access to matched pairs.
    """

    def __init__(
        self,
        df,
        suffixes: tuple[str, str] = ("_left", "_right"),
    ):
        self.df = df.reset_index(drop=True)
        self.suffixes = suffixes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, T.Any]:
        row = self.df.iloc[idx]
        result = {}
        for col in self.df.columns:
            val = row[col]
            if hasattr(val, 'to_dict'):
                # Nested pandas DataFrame (e.g., spectrum struct) → dict of tensors
                d = val.to_dict(orient='list')
                result[col] = {k: _python_to_torch(v) for k, v in d.items()}
            elif hasattr(val, 'item'):
                result[col] = _python_to_torch(val.item())
            else:
                result[col] = _python_to_torch(val)
        return result

    def crossmatch(
        self,
        other: "HATSDataset",
        radius_arcsec: float = 1.0,
        n_neighbors: int = 1,
        suffixes: tuple[str, str] = ("_left", "_right"),
        ra_col: str | None = None,
        dec_col: str | None = None,
    ) -> "CrossMatchedHATSDataset":
        """Chain a crossmatch against another HATSDataset.

        The matched DataFrame is converted to a temporary HATS catalog via
        LSDB so we can run another spatial join. The ``ra_col``/``dec_col``
        args select which ra/dec columns from the current matched result to
        use as the spatial key (defaults to the first ``ra*``/``dec*`` column
        found).

        Args:
            other: The next HATSDataset to match against.
            radius_arcsec: Maximum match radius.
            n_neighbors: Nearest neighbors per object.
            suffixes: Suffixes for the new match (applied to overlapping cols
                between the current result and ``other``).
            ra_col: Which RA column to use from this result. Auto-detected
                if not given.
            dec_col: Which Dec column. Auto-detected if not given.
        """
        df = self.df.copy()

        if ra_col is None:
            ra_col = next(c for c in df.columns if c.startswith("ra"))
        if dec_col is None:
            dec_col = next(c for c in df.columns if c.startswith("dec"))

        # Rename to plain ra/dec so LSDB can use them as the spatial key
        df = df.rename(columns={ra_col: "ra", dec_col: "dec"})

        left_cat = lsdb.from_dataframe(
            df,
            ra_column="ra",
            dec_column="dec",
            partition_rows=100_000,
        )
        result = left_cat.crossmatch(
            other.lsdb_catalog,
            n_neighbors=n_neighbors,
            radius_arcsec=radius_arcsec,
            suffixes=suffixes,
        )
        out_df = result.compute()
        return CrossMatchedHATSDataset(out_df, suffixes=suffixes)

    @property
    def matched_count(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)


def _arrow_row_to_torch(row: pa.Table) -> dict[str, torch.Tensor]:
    """Convert a single-row Arrow table to a dict of tensors."""
    result = {}
    for field in row.schema:
        col = row.column(field.name)
        value = col[0]
        result[field.name] = _arrow_value_to_torch(value, field.type)
    return result


def _arrow_value_to_torch(value: pa.Scalar, arrow_type: pa.DataType) -> T.Any:
    """Convert a PyArrow scalar to a torch tensor or nested dict of tensors."""
    if pa.types.is_struct(arrow_type):
        struct = value.as_py()
        return {
            k: _python_to_torch(v) for k, v in struct.items()
        }
    if pa.types.is_list(arrow_type):
        arr = value.as_py()
        return _python_to_torch(arr)
    return _python_to_torch(value.as_py())


def _python_to_torch(value: T.Any) -> T.Any:
    """Convert a Python value to a torch tensor (or leave it as-is for str/None).

    NOTE on scalars: Python ``int`` / ``float`` / ``bool`` get wrapped as 0-D
    tensors so they batch cleanly through ``DataLoader``. This means downstream
    code reading e.g. ``item["healpix"]`` gets a 0-D torch tensor, NOT a plain
    int — call ``.item()`` if you want the underlying number for indexing.
    Strings and ``None`` are passed through unchanged.
    """
    if isinstance(value, dict):
        return {k: _python_to_torch(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == 0:
            return torch.tensor([])
        first = value[0]
        if isinstance(first, bool):
            return torch.tensor(value, dtype=torch.bool)
        if isinstance(first, (int, float)):
            return torch.tensor(value)
        if isinstance(first, list):
            return torch.tensor(value)
        return value
    if isinstance(value, bool):
        return torch.tensor(value)
    if isinstance(value, int):
        return torch.tensor(value)
    if isinstance(value, float):
        return torch.tensor(value)
    if isinstance(value, str):
        return value
    if value is None:
        return value
    return value


def mmu_collate(batch: list[dict]) -> dict:
    """Custom collate function for ``DataLoader`` that handles the mixed types
    produced by :func:`_arrow_row_to_torch`.

    PyTorch's ``default_collate`` chokes on strings, ``None``, and numpy
    object arrays (which arise from variable-length list columns). This
    collate:

    - Stacks tensors normally (like ``default_collate``)
    - Collects strings into plain Python lists
    - Recursively collates nested dicts (from struct columns)
    - Passes ``None`` through as a list of Nones
    """
    keys = batch[0].keys()
    result = {}
    for k in keys:
        values = [d[k] for d in batch]
        first = values[0]
        if isinstance(first, torch.Tensor):
            try:
                result[k] = torch.stack(values)
            except (RuntimeError, TypeError):
                result[k] = values
        elif isinstance(first, dict):
            result[k] = mmu_collate(values)
        elif isinstance(first, (str, type(None))):
            result[k] = values
        elif isinstance(first, np.ndarray):
            try:
                result[k] = torch.from_numpy(np.stack(values))
            except (ValueError, TypeError):
                result[k] = values
        else:
            result[k] = values
    return result
