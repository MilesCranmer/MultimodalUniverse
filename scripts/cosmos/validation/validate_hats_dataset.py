"""Validation CLI for COSMOS-Web MMU HATS catalogs.

Usage
-----
python -m scripts.cosmos.validation.validate_hats_dataset \\
    --dataset /n03data/huertas/mmu/cosmos/cosmos/cosmos \\
    --out-dir /n03data/huertas/mmu/cosmos_validation \\
    --hist MAG_MODEL_F277W \\
    --hist ZPHOT \\
    --bin-plot MAG_MODEL_F277W:0.5:17:27 \\
    --label-column obj_id \\
    --label-column MAG_MODEL_F277W
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    pa = None
    pq = None
    _PYARROW_IMPORT_ERROR = e
else:
    _PYARROW_IMPORT_ERROR = None


COSMOS_BANDS = ["F115W", "F150W", "F277W", "F444W", "F770W"]
# Indices into the (5, H, W) image.flux array
BAND_INDEX = {b: i for i, b in enumerate(COSMOS_BANDS)}
# Default single-band display: F277W (selection band)
DEFAULT_GRAY_BAND = "F277W"
# RGB composite: F444W=R, F277W=G, F115W=B
RGB_BANDS = ("F444W", "F277W", "F115W")


@dataclass(frozen=True)
class HatsCatalogLayout:
    root: Path
    catalog_dir: Path
    dataset_dir: Path
    common_metadata: Path
    parquet_files: tuple[Path, ...]
    skymap: Path | None


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    parquet_name: str
    dtype: str
    metadata: dict[str, str]
    is_nested: bool = False


@dataclass(frozen=True)
class HistSpec:
    column: str
    transform: str = "none"


@dataclass(frozen=True)
class BinPlotSpec:
    column: str
    width: float
    min_value: float
    max_value: float
    transform: str = "none"


@dataclass(frozen=True)
class RowLocation:
    path: Path
    row_index: int


@dataclass(frozen=True)
class ValidationContext:
    layout: HatsCatalogLayout
    columns: dict[str, ColumnInfo]
    n_rows: int


@dataclass(frozen=True)
class BinSummary:
    spec: BinPlotSpec
    bin_index: int
    lo: float
    hi: float
    total: int


@dataclass(frozen=True)
class BinnedSampleRecord:
    spec: BinPlotSpec
    bin_index: int
    lo: float
    hi: float
    slot_index: int
    location: RowLocation
    raw_value: float
    transformed_value: float
    label_values: dict[str, str]


@dataclass(frozen=True)
class BinnedModalitySample:
    record: BinnedSampleRecord
    image_flux: np.ndarray | None


@dataclass
class _BinState:
    total: int = 0
    image_seen: int = 0
    image: list = field(default_factory=list)


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise ImportError(
            "validate_hats_dataset requires pyarrow."
        ) from _PYARROW_IMPORT_ERROR


def _decode_metadata(metadata: dict | None) -> dict[str, str]:
    if not metadata:
        return {}
    out: dict[str, str] = {}
    for key, value in metadata.items():
        k = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
        v = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        out[k] = v
    return out


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "column"


def ordered_column_names(columns: Iterable[str]) -> list[str]:
    return sorted(columns)


def parse_hist_spec(raw: str) -> HistSpec:
    if ":" in raw:
        column, transform = raw.rsplit(":", 1)
    else:
        column, transform = raw, "none"
    column = column.strip()
    transform = transform.strip().lower()
    valid = {"none", "log10", "ln", "log1p", "asinh"}
    if not column:
        raise ValueError(f"Invalid histogram spec {raw!r}: missing column name.")
    if transform not in valid:
        raise ValueError(f"Invalid transform {transform!r}. Expected one of {sorted(valid)}.")
    return HistSpec(column=column, transform=transform)


def parse_bin_plot_spec(raw: str) -> BinPlotSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) not in (4, 5):
        raise ValueError(f"Invalid bin plot spec {raw!r}. Expected COLUMN:WIDTH:MIN:MAX[:TRANSFORM].")
    column = parts[0]
    if not column:
        raise ValueError(f"Invalid bin plot spec {raw!r}: missing column name.")
    try:
        width = float(parts[1])
        min_value = float(parts[2])
        max_value = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in bin plot spec {raw!r}.") from exc
    transform = parts[4].lower() if len(parts) == 5 else "none"
    valid = {"none", "log10", "ln", "log1p", "asinh"}
    if transform not in valid:
        raise ValueError(f"Invalid transform {transform!r}. Expected one of {sorted(valid)}.")
    if width <= 0:
        raise ValueError(f"Invalid bin width {width!r}; must be positive.")
    if max_value <= min_value:
        raise ValueError("Bin max must be greater than bin min.")
    return BinPlotSpec(column=column, width=width, min_value=min_value, max_value=max_value, transform=transform)


def discover_hats_catalog(dataset_root: str | Path) -> HatsCatalogLayout:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    common_files = sorted(root.glob("**/dataset/_common_metadata"))
    if not common_files:
        raise FileNotFoundError(f"No HATS _common_metadata found under {root}")
    candidates: list[tuple[int, Path]] = []
    for common in common_files:
        catalog_dir = common.parent.parent
        score = 0
        if catalog_dir.name.endswith("_margin") or "margin" in catalog_dir.name.lower():
            score += 100
        candidates.append((score, common))
    common = sorted(candidates, key=lambda item: (item[0], str(item[1])))[0][1]
    dataset_dir = common.parent
    catalog_dir = dataset_dir.parent
    parquet_files = tuple(sorted(dataset_dir.glob("**/Npix=*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No HATS Npix parquet files found under {dataset_dir}")
    skymap = catalog_dir / "skymap.fits"
    return HatsCatalogLayout(
        root=root,
        catalog_dir=catalog_dir,
        dataset_dir=dataset_dir,
        common_metadata=common,
        parquet_files=parquet_files,
        skymap=skymap if skymap.exists() else None,
    )


def load_validation_context(dataset_root: str | Path) -> ValidationContext:
    _require_pyarrow()
    layout = discover_hats_catalog(dataset_root)
    schema = pq.read_schema(layout.common_metadata)
    columns = schema_column_info(schema)
    return ValidationContext(layout=layout, columns=columns, n_rows=total_rows(layout.parquet_files))


def schema_column_info(schema: "pa.Schema") -> dict[str, ColumnInfo]:
    infos: dict[str, ColumnInfo] = {}
    for field in schema:
        if pa.types.is_struct(field.type) and field.name == "image":
            for child in field.type:
                logical = f"image_{child.name}"
                infos[logical] = ColumnInfo(
                    name=logical,
                    parquet_name=f"image.{child.name}",
                    dtype=str(child.type),
                    metadata=_decode_metadata(child.metadata),
                    is_nested=True,
                )
            continue
        infos[field.name] = ColumnInfo(
            name=field.name,
            parquet_name=field.name,
            dtype=str(field.type),
            metadata=_decode_metadata(field.metadata),
            is_nested=False,
        )
    return infos


def collect_schema_summary(columns: dict[str, ColumnInfo]) -> list[dict[str, str]]:
    rows = []
    for name in ordered_column_names(columns):
        info = columns[name]
        meta = info.metadata
        rows.append({
            "column": info.name,
            "parquet_name": info.parquet_name,
            "dtype": info.dtype,
            "unit": meta.get("unit", ""),
            "description": meta.get("description", ""),
            "inner_shape": meta.get("_inner_shape", ""),
        })
    return rows


def write_schema_summary(schema_rows: Sequence[dict[str, str]], out_dir: Path) -> Path:
    csv_path = out_dir / "schema_columns.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["column", "parquet_name", "dtype", "unit", "description", "inner_shape"]
        )
        writer.writeheader()
        writer.writerows(schema_rows)
    return csv_path


def print_schema_summary(schema_rows: Sequence[dict[str, str]]) -> None:
    print("\nColumns")
    for row in schema_rows:
        extras = []
        for key in ("unit", "inner_shape"):
            if row.get(key):
                extras.append(f"{key}={row[key]}")
        if row.get("description"):
            extras.append(f"description={row['description']}")
        suffix = f" | {'; '.join(extras)}" if extras else ""
        print(f"  {row['column']}: {row['dtype']}{suffix}")


def print_and_save_schema(columns: dict[str, ColumnInfo], out_dir: Path) -> None:
    schema_rows = collect_schema_summary(columns)
    print_schema_summary(schema_rows)
    csv_path = write_schema_summary(schema_rows, out_dir)
    print(f"\nWrote schema table: {csv_path}")


def total_rows(parquet_files: Sequence[Path]) -> int:
    _require_pyarrow()
    total = 0
    for path in parquet_files:
        total += pq.ParquetFile(path).metadata.num_rows
    return int(total)


def _top_level_scalar_columns(columns: dict[str, ColumnInfo]) -> list[str]:
    return [name for name, info in columns.items() if not info.is_nested]


def _iter_batches(parquet_files: Sequence[Path], column_names: Sequence[str], batch_size: int = 65_536):
    _require_pyarrow()
    if not column_names:
        return
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=list(column_names), batch_size=batch_size):
            yield batch


def scalar_null_counts_from_batches(
    batches: Iterable["pa.RecordBatch"], column_names: Sequence[str]
) -> dict[str, int]:
    counts = {name: 0 for name in column_names}
    for batch in batches:
        for name in column_names:
            counts[name] += int(batch.column(name).null_count)
    return counts


def compute_null_rates(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    n_rows: int,
) -> list[dict]:
    scalar_cols = _top_level_scalar_columns(columns)
    null_counts = scalar_null_counts_from_batches(
        _iter_batches(parquet_files, scalar_cols), scalar_cols
    )
    rows = []
    for name in ordered_column_names(columns):
        count = int(null_counts.get(name, 0))
        rate = float(count / n_rows) if n_rows else math.nan
        rows.append({"column": name, "missing": count, "missing_rate": rate})
    return rows


def write_null_rates(null_rows: Sequence[dict], out_dir: Path) -> Path:
    path = out_dir / "null_rates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["column", "missing", "missing_rate"])
        writer.writeheader()
        writer.writerows(null_rows)
    return path


def print_null_rates(null_rows: Sequence[dict], n_rows: int, path: Path | None = None) -> None:
    if path is not None:
        print(f"\nWrote null/missing-rate table: {path}")
    else:
        print("\nNull/missing-rate table")
    print("Highest missing-rate columns:")
    worst = sorted(null_rows, key=lambda row: float(row["missing_rate"]), reverse=True)[:10]
    for row in worst:
        print(f"  {row['column']}: {row['missing']} / {n_rows} ({float(row['missing_rate']):.3%})")


def print_image_availability(n_rows: int) -> None:
    print("\nImage availability")
    print(f"  All {n_rows} objects have image data (5 JWST bands: {', '.join(COSMOS_BANDS)})")


def print_modality_shapes(columns: dict[str, ColumnInfo]) -> None:
    print("\nImage shapes")
    for name in ordered_column_names(columns):
        if name.startswith("image_"):
            shape = columns[name].metadata.get("_inner_shape", "")
            dtype = columns[name].dtype
            print(f"  {name}: shape={shape or 'unknown'} dtype={dtype}")


def compute_unique_counts(parquet_files: Sequence[Path], column: str, include_missing: bool = True) -> Counter:
    counts: Counter = Counter()
    for batch in _iter_batches(parquet_files, [column]):
        for value in batch.column(column).to_pylist():
            if value is None:
                if include_missing:
                    counts["<MISSING>"] += 1
            else:
                counts[value] += 1
    return counts


def save_unique_counts(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    count_columns: Sequence[str],
    out_dir: Path,
) -> None:
    if not count_columns:
        print("\nNo unique-value count columns requested.")
        return
    print("\nUnique-value counts")
    for column in count_columns:
        _require_scalar_column(columns, column, "count")
        counts = compute_unique_counts(parquet_files, column)
        path = out_dir / f"counts_{_safe_filename(column)}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["value", "count"])
            writer.writeheader()
            for value, count in counts.most_common():
                writer.writerow({"value": value, "count": count})
        print(f"  {column}: wrote {path}")
        for value, count in counts.most_common(20):
            print(f"    {value!r}: {count}")


def _require_scalar_column(columns: dict[str, ColumnInfo], column: str, purpose: str) -> None:
    if column not in columns:
        raise ValueError(f"Cannot {purpose} unknown column {column!r}.")
    if columns[column].is_nested:
        raise ValueError(f"Cannot {purpose} nested tensor column {column!r}.")


def _array_to_numeric(array: "pa.Array") -> np.ndarray:
    return np.asarray(array.to_pandas(), dtype=float)


def _apply_transform(values: np.ndarray, transform: str) -> tuple[np.ndarray, int]:
    dropped = 0
    finite = values[np.isfinite(values)]
    if transform == "none":
        return finite, dropped
    if transform in {"log10", "ln"}:
        keep = finite > 0
        dropped = int(np.count_nonzero(~keep))
        finite = finite[keep]
        return (np.log10(finite) if transform == "log10" else np.log(finite)), dropped
    if transform == "log1p":
        keep = finite > -1
        dropped = int(np.count_nonzero(~keep))
        return np.log1p(finite[keep]), dropped
    if transform == "asinh":
        return np.arcsinh(finite), dropped
    raise ValueError(f"Unsupported transform: {transform}")


def collect_hist_values(
    parquet_files: Sequence[Path], spec: HistSpec
) -> tuple[np.ndarray, dict[str, int]]:
    chunks = []
    total = 0
    nulls = 0
    nonfinite = 0
    transform_dropped = 0
    for batch in _iter_batches(parquet_files, [spec.column]):
        arr = batch.column(spec.column)
        total += len(arr)
        nulls += arr.null_count
        raw = _array_to_numeric(arr)
        nonfinite += int(np.count_nonzero(~np.isfinite(raw) & ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)))
        transformed, dropped = _apply_transform(raw, spec.transform)
        transform_dropped += dropped
        if transformed.size:
            chunks.append(transformed)
    values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    return values, {
        "total": total,
        "nulls": int(nulls),
        "nonfinite": int(nonfinite),
        "transform_dropped": int(transform_dropped),
        "plotted": int(values.size),
    }


def plot_histogram(values: np.ndarray, spec: HistSpec, *, bins: int = 50, fig=None, ax=None, color: str = "#376996"):
    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bins, color=color, alpha=0.9)
    xlabel = spec.column if spec.transform == "none" else f"{spec.transform}({spec.column})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(xlabel)
    fig.tight_layout()
    return fig, ax


def save_histograms(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    hist_specs: Sequence[HistSpec],
    out_dir: Path,
    bins: int,
) -> None:
    if not hist_specs:
        print("\nNo histograms requested.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nHistograms")
    for spec in hist_specs:
        _require_scalar_column(columns, spec.column, "histogram")
        values, summary = collect_hist_values(parquet_files, spec)
        stem = _safe_filename(f"{spec.column}_{spec.transform}")
        png_path = out_dir / f"hist_{stem}.png"
        csv_path = out_dir / f"hist_{stem}_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["column", "transform", *summary.keys()])
            writer.writeheader()
            writer.writerow({"column": spec.column, "transform": spec.transform, **summary})
        if values.size:
            fig, ax = plot_histogram(values, spec, bins=bins)
            fig.savefig(png_path, dpi=180)
            plt.close(fig)
            print(f"  {spec.column}:{spec.transform}: wrote {png_path}")
        else:
            print(f"  {spec.column}:{spec.transform}: no finite values to plot.")
        if summary["transform_dropped"]:
            print(f"    dropped {summary['transform_dropped']} values during {spec.transform} transform")


# ── Image reading helpers ────────────────────────────────────────────────────


def _read_image_table(path: Path, extra_columns: Sequence[str] = ()) -> dict[str, "pa.Array"]:
    """Read image struct + optional scalar columns from a parquet file.

    Returns a dict mapping column name → pyarrow Array. The image struct is
    flattened: ``image.flux`` → ``image_flux``, ``image.band`` → ``image_band``, etc.
    """
    read_cols = ["image"] + [c for c in extra_columns if c not in ("image",)]
    table = pq.read_table(path, columns=list(dict.fromkeys(read_cols)))
    result: dict[str, pa.Array] = {}
    for name in table.schema.names:
        col = table.column(name)
        if pa.types.is_struct(col.type) and name == "image":
            for i in range(col.type.num_fields):
                f = col.type.field(i)
                result[f"image_{f.name}"] = col.field(f.name)
        else:
            result[name] = col
    return result


def _row_value(table: dict[str, "pa.Array"], column: str, index: int):
    if column not in table:
        return None
    try:
        return table[column][index].as_py()
    except Exception:
        return None


def _image_is_valid(value) -> bool:
    if value is None:
        return False
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return False
    return arr.size > 0 and not np.all(np.isnan(arr))


def _format_title(table: dict[str, "pa.Array"], index: int, label_columns: Sequence[str]) -> str:
    pieces = []
    for column in label_columns:
        if column not in table:
            continue
        value = _row_value(table, column, index)
        if value is None:
            continue
        text = str(value)
        if len(text) > 36:
            text = text[:33] + "..."
        pieces.append(f"{column}={text}")
    return "\n".join(pieces)


# ── Reservoir sampling ────────────────────────────────────────────────────────


def reservoir_sample_rows(
    parquet_files: Sequence[Path],
    sample_size: int,
    seed: int,
) -> list[RowLocation]:
    """Uniformly sample rows across the catalog."""
    if sample_size <= 0:
        return []
    rng = np.random.default_rng(seed)
    reservoir: list[RowLocation] = []
    seen = 0
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        local_offset = 0
        for batch in parquet.iter_batches(columns=["ra"], batch_size=65_536):
            for i in range(batch.num_rows):
                seen += 1
                location = RowLocation(path=path, row_index=int(local_offset + i))
                if len(reservoir) < sample_size:
                    reservoir.append(location)
                else:
                    replace = int(rng.integers(0, seen))
                    if replace < sample_size:
                        reservoir[replace] = location
            local_offset += batch.num_rows
    reservoir.sort(key=lambda loc: (str(loc.path), loc.row_index))
    return reservoir


def collect_plot_samples(
    parquet_files: Sequence[Path],
    sample_size: int,
    seed: int,
    label_columns: Sequence[str],
) -> list[dict]:
    """Collect random image samples from the catalog."""
    locations = reservoir_sample_rows(parquet_files, sample_size, seed)
    wanted_by_path: dict[Path, list[int]] = {}
    for loc in locations:
        wanted_by_path.setdefault(loc.path, []).append(loc.row_index)

    rows: list[dict] = []
    for path in sorted(wanted_by_path):
        table = _read_image_table(path, extra_columns=list(label_columns))
        for idx in wanted_by_path[path]:
            image_value = _row_value(table, "image_flux", idx)
            if _image_is_valid(image_value):
                rows.append({
                    "image_flux": np.asarray(image_value),
                    "title": _format_title(table, idx, label_columns),
                })
    return rows[:sample_size]


# ── Grid plot helpers ─────────────────────────────────────────────────────────


def _make_grid(n_items: int, ncols: int = 5):
    import matplotlib.pyplot as plt
    nrows = int(math.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    return fig, np.asarray(axes).reshape(-1)


def _render_gray(ax, image_2d):
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
    ax.imshow(
        image_2d, origin="lower", cmap="gray",
        norm=ImageNormalize(image_2d, interval=PercentileInterval(99.5), stretch=AsinhStretch()),
    )
    ax.axis("off")


def _render_rgb(ax, r, g, b):
    from astropy.visualization import PercentileInterval, make_lupton_rgb
    rgb = make_lupton_rgb(r, g, b, interval=PercentileInterval(99.5), stretch=5, Q=8)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")


def save_plot_grids(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    sample_size: int,
    seed: int,
    label_columns: Sequence[str],
    out_dir: Path,
) -> None:
    if sample_size <= 0:
        print("\nPlot sample size is 0; skipping modality grids.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nSampling image plots")
    valid_labels = [c for c in label_columns if c in columns and not columns[c].is_nested]
    image_rows = collect_plot_samples(parquet_files, sample_size, seed, valid_labels)

    if not image_rows:
        print("  no valid image rows found.")
        return

    n = len(image_rows)

    # Grayscale grid (F277W, index 2)
    fig, axes = _make_grid(n)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, row in zip(axes, image_rows):
        img = np.asarray(row["image_flux"], dtype=float)
        plane = img[BAND_INDEX[DEFAULT_GRAY_BAND]] if img.ndim == 3 else img
        _render_gray(ax, plane)
        ax.set_title(row["title"], fontsize=6)
    fig.suptitle(f"{DEFAULT_GRAY_BAND} cutouts ({n})")
    fig.tight_layout()
    path = out_dir / f"{DEFAULT_GRAY_BAND.lower()}_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    # RGB grid (F444W=R, F277W=G, F115W=B)
    fig, axes = _make_grid(n)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, row in zip(axes, image_rows):
        img = np.asarray(row["image_flux"], dtype=float)
        if img.ndim == 3 and img.shape[0] >= 5:
            r = img[BAND_INDEX[RGB_BANDS[0]]]
            g = img[BAND_INDEX[RGB_BANDS[1]]]
            b = img[BAND_INDEX[RGB_BANDS[2]]]
            _render_rgb(ax, r, g, b)
        else:
            ax.text(0.5, 0.5, "RGB unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        ax.set_title(row["title"], fontsize=6)
    fig.suptitle(f"RGB cutouts ({'/'.join(RGB_BANDS)}, n={n})")
    fig.tight_layout()
    path = out_dir / "rgb_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── Binned modality plots ─────────────────────────────────────────────────────


def _bin_edges(spec: BinPlotSpec) -> np.ndarray:
    edges = [float(spec.min_value)]
    value = float(spec.min_value)
    while value + spec.width < spec.max_value:
        value += spec.width
        edges.append(float(value))
    if edges[-1] != float(spec.max_value):
        edges.append(float(spec.max_value))
    return np.asarray(edges, dtype=float)


def _bin_index(value: float, edges: np.ndarray) -> int | None:
    if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def _transform_values_for_binning(values: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    transformed = np.full(raw.shape, np.nan, dtype=float)
    valid = np.isfinite(raw)
    if transform == "none":
        transformed[valid] = raw[valid]
        return transformed, valid
    if transform in {"log10", "ln"}:
        valid &= raw > 0
        transformed[valid] = np.log10(raw[valid]) if transform == "log10" else np.log(raw[valid])
        return transformed, valid
    if transform == "log1p":
        valid &= raw > -1
        transformed[valid] = np.log1p(raw[valid])
        return transformed, valid
    if transform == "asinh":
        transformed[valid] = np.arcsinh(raw[valid])
        return transformed, valid
    raise ValueError(f"Unsupported transform: {transform}")


def _reservoir_add(reservoir: list, seen: int, item: dict, limit: int, rng: np.random.Generator) -> int:
    seen += 1
    if limit <= 0:
        return seen
    if len(reservoir) < limit:
        reservoir.append(item)
        return seen
    replace = int(rng.integers(0, seen))
    if replace < limit:
        reservoir[replace] = item
    return seen


def collect_binned_sample_plan(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    spec: BinPlotSpec,
    *,
    sample_size: int = 25,
    seed: int = 42,
    label_columns: Sequence[str] = (),
) -> tuple[list[BinSummary], list[BinnedSampleRecord]]:
    _require_scalar_column(columns, spec.column, "bin")
    edges = _bin_edges(spec)
    states = [_BinState() for _ in range(len(edges) - 1)]
    rng = np.random.default_rng(seed)
    valid_labels = [col for col in label_columns if col in columns and not columns[col].is_nested]
    scan_columns = list(dict.fromkeys([spec.column, *valid_labels]))

    for path in parquet_files:
        local_offset = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=scan_columns, batch_size=65_536):
            raw_values = _array_to_numeric(batch.column(spec.column))
            transformed, valid_transform = _transform_values_for_binning(raw_values, spec.transform)
            label_values_by_col = {
                col: batch.column(col).to_pylist()
                for col in valid_labels
                if col in batch.schema.names
            }
            for i in range(batch.num_rows):
                if not valid_transform[i]:
                    continue
                bin_idx = _bin_index(float(transformed[i]), edges)
                if bin_idx is None:
                    continue
                state = states[bin_idx]
                state.total += 1
                labels = {
                    col: "" if label_values_by_col[col][i] is None else str(label_values_by_col[col][i])
                    for col in label_values_by_col
                }
                item = {
                    "location": RowLocation(path=path, row_index=int(local_offset + i)),
                    "raw_value": float(raw_values[i]),
                    "transformed_value": float(transformed[i]),
                    "label_values": labels,
                }
                state.image_seen = _reservoir_add(state.image, state.image_seen, item, sample_size, rng)
            local_offset += batch.num_rows

    summaries: list[BinSummary] = []
    samples: list[BinnedSampleRecord] = []
    for bin_idx, state in enumerate(states):
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        summaries.append(BinSummary(spec=spec, bin_index=bin_idx, lo=lo, hi=hi, total=state.total))
        for slot_index, item in enumerate(state.image[:sample_size]):
            samples.append(BinnedSampleRecord(
                spec=spec,
                bin_index=bin_idx,
                lo=lo,
                hi=hi,
                slot_index=slot_index,
                location=item["location"],
                raw_value=item["raw_value"],
                transformed_value=item["transformed_value"],
                label_values=item["label_values"],
            ))
    return summaries, samples


def read_binned_modality_samples(samples: Sequence[BinnedSampleRecord]) -> list[BinnedModalitySample]:
    by_path: dict[Path, list[BinnedSampleRecord]] = {}
    for sample in samples:
        by_path.setdefault(sample.location.path, []).append(sample)

    out: list[BinnedModalitySample] = []
    for path, records in sorted(by_path.items(), key=lambda item: str(item[0])):
        table = _read_image_table(path)
        for record in records:
            image_value = _row_value(table, "image_flux", record.location.row_index)
            image = np.asarray(image_value) if _image_is_valid(image_value) else None
            out.append(BinnedModalitySample(record=record, image_flux=image))
    out.sort(key=lambda sample: (sample.record.bin_index, sample.record.slot_index))
    return out


def _format_bin_suptitle(summary: BinSummary, label: str) -> str:
    spec = summary.spec
    transform = spec.transform if spec.transform != "none" else "raw"
    return (
        f"{label}: {spec.column} ({transform}) [{summary.lo:g}, {summary.hi:g}) | "
        f"total={summary.total}"
    )


def _format_sample_title(sample: BinnedModalitySample) -> str:
    return "\n".join(f"{key}={value}" for key, value in sample.record.label_values.items())


def _make_gridspec_axes(n_slots: int, ncols: int = 5, figsize_scale: float = 2.5, fig=None, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if axes is not None:
        return fig, np.asarray(axes).reshape(-1)
    nrows = max(1, int(math.ceil(max(1, n_slots) / ncols)))
    fig = plt.figure(figsize=(ncols * figsize_scale, nrows * figsize_scale)) if fig is None else fig
    grid = GridSpec(nrows, ncols, figure=fig)
    axes = np.array([fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(nrows * ncols)])
    return fig, axes


def _samples_by_slot(samples: Sequence[BinnedModalitySample], bin_index: int) -> dict[int, BinnedModalitySample]:
    return {s.record.slot_index: s for s in samples if s.record.bin_index == bin_index}


def plot_binned_gray_grid(
    samples: Sequence[BinnedModalitySample],
    summary: BinSummary,
    *,
    sample_size: int,
    band: str = DEFAULT_GRAY_BAND,
    ncols: int = 5,
    fig=None,
    axes=None,
):
    fig, axes = _make_gridspec_axes(sample_size, ncols=ncols, fig=fig, axes=axes)
    by_slot = _samples_by_slot(samples, summary.bin_index)
    band_idx = BAND_INDEX.get(band, BAND_INDEX[DEFAULT_GRAY_BAND])
    for slot, ax in enumerate(axes):
        if slot >= sample_size:
            ax.axis("off")
            continue
        sample = by_slot.get(slot)
        if sample is None or sample.image_flux is None:
            ax.axis("off")
            continue
        img = sample.image_flux
        plane = img[band_idx] if img.ndim == 3 else img
        _render_gray(ax, plane)
        ax.set_title(_format_sample_title(sample), fontsize=6)
    fig.suptitle(_format_bin_suptitle(summary, f"{band} grayscale"), fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_binned_rgb_grid(
    samples: Sequence[BinnedModalitySample],
    summary: BinSummary,
    *,
    sample_size: int,
    ncols: int = 5,
    fig=None,
    axes=None,
):
    """RGB composite: F444W=R, F277W=G, F115W=B."""
    fig, axes = _make_gridspec_axes(sample_size, ncols=ncols, fig=fig, axes=axes)
    by_slot = _samples_by_slot(samples, summary.bin_index)
    ri, gi, bi = (BAND_INDEX[b] for b in RGB_BANDS)
    for slot, ax in enumerate(axes):
        if slot >= sample_size:
            ax.axis("off")
            continue
        sample = by_slot.get(slot)
        if sample is None or sample.image_flux is None or sample.image_flux.ndim != 3 or sample.image_flux.shape[0] < 5:
            ax.axis("off")
            continue
        _render_rgb(ax, sample.image_flux[ri], sample.image_flux[gi], sample.image_flux[bi])
        ax.set_title(_format_sample_title(sample), fontsize=6)
    fig.suptitle(_format_bin_suptitle(summary, f"RGB ({'/'.join(RGB_BANDS)})"), fontsize=10)
    fig.tight_layout()
    return fig, axes


def _bin_file_stem(summary: BinSummary) -> str:
    return _safe_filename(f"bin_{summary.bin_index:03d}_{summary.lo:g}_{summary.hi:g}")


def _write_dict_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_binned_modality_plots(
    parquet_files: Sequence[Path],
    columns: dict[str, ColumnInfo],
    specs: Sequence[BinPlotSpec],
    out_dir: Path,
    *,
    label_columns: Sequence[str] = (),
    sample_size: int = 25,
    seed: int = 42,
    ncols: int = 5,
    dpi: int = 180,
) -> None:
    if not specs:
        print("\nNo binned modality plots requested.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nBinned modality plots")
    root = out_dir / "bin_plots"
    root.mkdir(parents=True, exist_ok=True)
    valid_labels = [col for col in label_columns if col in columns and not columns[col].is_nested]
    for spec_index, spec in enumerate(specs):
        spec_dir = root / _safe_filename(f"{spec.column}_{spec.transform}")
        spec_dir.mkdir(parents=True, exist_ok=True)
        summaries, sample_records = collect_binned_sample_plan(
            parquet_files,
            columns,
            spec,
            sample_size=sample_size,
            seed=seed + spec_index,
            label_columns=valid_labels,
        )
        summary_rows = [
            {
                "column": s.spec.column, "transform": s.spec.transform,
                "bin_index": s.bin_index, "lo": s.lo, "hi": s.hi, "total": s.total,
            }
            for s in summaries
        ]
        if summary_rows:
            _write_dict_rows(spec_dir / "bin_summary.csv", summary_rows, list(summary_rows[0].keys()))
        sample_rows = [
            {
                "column": sr.spec.column, "transform": sr.spec.transform,
                "bin_index": sr.bin_index, "lo": sr.lo, "hi": sr.hi,
                "slot_index": sr.slot_index,
                "parquet_path": str(sr.location.path), "row_index": sr.location.row_index,
                "raw_value": sr.raw_value, "transformed_value": sr.transformed_value,
                **{col: sr.label_values.get(col, "") for col in valid_labels},
            }
            for sr in sample_records
        ]
        if sample_rows:
            _write_dict_rows(
                spec_dir / "bin_samples.csv",
                sample_rows,
                ["column", "transform", "bin_index", "lo", "hi", "slot_index",
                 "parquet_path", "row_index", "raw_value", "transformed_value", *valid_labels],
            )
        modality_samples = read_binned_modality_samples(sample_records) if sample_records else []
        for summary in summaries:
            stem = _bin_file_stem(summary)
            fig, _ = plot_binned_gray_grid(modality_samples, summary, sample_size=sample_size, ncols=ncols)
            fig.savefig(spec_dir / f"{stem}_gray.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            fig, _ = plot_binned_rgb_grid(modality_samples, summary, sample_size=sample_size, ncols=ncols)
            fig.savefig(spec_dir / f"{stem}_rgb.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        print(f"  {spec.column}:{spec.transform}: wrote {spec_dir}")


# ── Sky coverage ──────────────────────────────────────────────────────────────


def moc_from_hats_skymap(skymap_path: Path):
    from astropy.io import fits
    from mocpy import MOC

    with fits.open(skymap_path) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        nside = int(header["NSIDE"])
        ordering = str(header.get("ORDERING", "NESTED")).upper()
        if ordering != "NESTED":
            raise ValueError(f"Expected NESTED skymap, got ORDERING={ordering!r}.")
        order = int(round(math.log2(nside)))
        ipix = np.asarray(data["PIXEL"], dtype=np.uint64)
    return MOC.from_healpix_cells(ipix=ipix, depth=order, max_depth=order)


def moc_from_points(parquet_files: Sequence[Path], max_order: int):
    from astropy import units as u
    from mocpy import MOC

    ra_chunks, dec_chunks = [], []
    for batch in _iter_batches(parquet_files, ["ra", "dec"]):
        ra_chunks.append(_array_to_numeric(batch.column("ra")))
        dec_chunks.append(_array_to_numeric(batch.column("dec")))
    if not ra_chunks:
        raise ValueError("Cannot build point MOC: no ra/dec rows found.")
    ra = np.concatenate(ra_chunks)
    dec = np.concatenate(dec_chunks)
    keep = np.isfinite(ra) & np.isfinite(dec)
    return MOC.from_lonlat(ra[keep] * u.deg, dec[keep] * u.deg, max_norder=max_order)


def build_coverage_moc(layout: HatsCatalogLayout, fallback_order: int = 10):
    if layout.skymap is not None:
        return moc_from_hats_skymap(layout.skymap), f"HATS skymap: {layout.skymap}"
    return moc_from_points(layout.parquet_files, fallback_order), "fallback point MOC from ra/dec"


def plot_coverage_moc(moc, *, fig=None, ax=None, title: str = "COSMOS-Web MMU HATS sky coverage"):
    import matplotlib.pyplot as plt
    from astropy.wcs import WCS

    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 5))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---AIT", "DEC--AIT"]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.crpix = [500.0, 250.0]
        wcs.wcs.cdelt = [-0.36, 0.36]
        ax = fig.add_subplot(111, projection=wcs)
    else:
        wcs = ax.wcs
    moc.fill(ax=ax, wcs=wcs, alpha=0.55, color="#376996")
    moc.border(ax=ax, wcs=wcs, color="#1f2d3a", linewidth=0.8)
    ax.coords.grid(color="0.7", linestyle=":", linewidth=0.6)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def save_moc(layout: HatsCatalogLayout, out_dir: Path, fallback_order: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nSky coverage MOC")
    moc, source = build_coverage_moc(layout, fallback_order)
    print(f"  built from {source}")
    fits_path = out_dir / "coverage_moc.fits"
    png_path = out_dir / "coverage_moc.png"
    moc.save(fits_path, format="fits", overwrite=True)
    fig, ax = plot_coverage_moc(moc)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fits_path}")
    print(f"  wrote {png_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HATS catalog root directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for validation artifacts.")
    parser.add_argument("--count-column", action="append", default=[], help="Column to unique-count. Repeatable.")
    parser.add_argument("--hist", action="append", default=[], help="Histogram spec COLUMN[:TRANSFORM]. Repeatable.")
    parser.add_argument("--hist-bins", type=int, default=50, help="Histogram bin count.")
    parser.add_argument("--bin-plot", action="append", default=[], help="Binned plot spec COLUMN:WIDTH:MIN:MAX[:TRANSFORM]. Repeatable.")
    parser.add_argument("--bin-sample-size", type=int, default=25, help="Grid slots per bin.")
    parser.add_argument("--label-column", action="append", default=[], help="Column for plot titles. Repeatable.")
    parser.add_argument("--sample-size", type=int, default=25, help="Random images to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fallback-moc-order", type=int, default=10, help="MOC order if skymap.fits absent.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_pyarrow()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_specs = [parse_hist_spec(raw) for raw in args.hist]
    bin_specs = [parse_bin_plot_spec(raw) for raw in args.bin_plot]

    context = load_validation_context(args.dataset)
    layout = context.layout
    columns = context.columns
    n_rows = context.n_rows

    print(f"Dataset root:       {layout.root}")
    print(f"HATS catalog:       {layout.catalog_dir}")
    print(f"Parquet partitions: {len(layout.parquet_files)}")
    print(f"Rows:               {n_rows}")
    print(f"Output directory:   {out_dir}")

    print_and_save_schema(columns, out_dir)
    print_modality_shapes(columns)
    print_image_availability(n_rows)
    null_rows = compute_null_rates(layout.parquet_files, columns, n_rows)
    null_path = write_null_rates(null_rows, out_dir)
    print_null_rates(null_rows, n_rows, null_path)
    save_unique_counts(layout.parquet_files, columns, args.count_column, out_dir)
    save_histograms(layout.parquet_files, columns, hist_specs, out_dir, args.hist_bins)
    save_plot_grids(layout.parquet_files, columns, args.sample_size, args.seed, args.label_column, out_dir)
    save_binned_modality_plots(
        layout.parquet_files,
        columns,
        bin_specs,
        out_dir,
        label_columns=args.label_column,
        sample_size=args.bin_sample_size,
        seed=args.seed,
    )
    save_moc(layout, out_dir, args.fallback_moc_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
