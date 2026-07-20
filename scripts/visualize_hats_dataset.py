"""Generate a visualization for a HATS catalog so a human (or LLM) can sanity-check it.

Picks a few rows and plots:
- For spectra: spectrum (flux vs wavelength) + a few catalog scalars
- For images: RGB-ish image cutout
- For time series: lightcurve (flux vs time)
- For tabular: a histogram of one feature

Usage:
    python scripts/visualize_hats_dataset.py \
        --catalog /path/to/sdss_sdss/sdss_sdss \
        --output sdss_sdss_preview.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def find_parquet_files(catalog_dir: str) -> list[str]:
    files = []
    for root, _, fs in os.walk(os.path.join(catalog_dir, "dataset")):
        for f in fs:
            if f.endswith(".parquet") and not f.startswith("_"):
                files.append(os.path.join(root, f))
    return sorted(files)


def detect_modality(schema_names: list[str]) -> str:
    if "spectrum" in schema_names:
        return "spectra"
    if "image_array" in schema_names and "image_array_shape" in schema_names:
        return "image"
    if "flux" in schema_names and "time" in schema_names:
        return "timeseries"
    if "spectrum_coefficients" in schema_names or "spectral_coefficients" in schema_names:
        return "gaia_coeffs"
    return "tabular"


def _resolve_x_axis(row: dict) -> tuple[np.ndarray, str]:
    """Find an x-axis array (wavelength/energy/index) and a label."""
    for key, label in [
        ("lambda", "Wavelength [Å]"),
        ("wavelength", "Wavelength [Å]"),
        ("ene", "Energy [keV]"),
        ("ene_center_bin", "Energy [keV]"),
    ]:
        if key in row:
            return np.asarray(row[key], dtype=np.float32), label
    # Fallback: use the index of the flux array
    flux = np.asarray(row.get("flux", []), dtype=np.float32)
    return np.arange(len(flux)), "Channel"


def plot_spectra(table, axes, n_show=4):
    spec_col = table.column("spectrum")
    z_col = table.column("Z") if "Z" in table.schema.names else None
    for i in range(min(n_show, table.num_rows)):
        row = spec_col[i].as_py()
        x, xlabel = _resolve_x_axis(row)
        flux = np.asarray(row["flux"], dtype=np.float32)
        if "mask" in row:
            mask = np.asarray(row["mask"], dtype=bool)
            flux_m = np.where(mask, np.nan, flux)
        else:
            flux_m = flux
        ax = axes[i]
        ax.plot(x, flux_m, linewidth=0.5, color="navy")
        title = f"row {i}"
        if z_col is not None:
            title += f"  z={z_col[i].as_py():.4f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Flux")


def plot_images(table, axes, n_show=4):
    n_show = min(n_show, table.num_rows)
    for i in range(n_show):
        flat = np.asarray(table.column("image_array")[i].as_py(), dtype=np.float32)
        shape = list(table.column("image_array_shape")[i].as_py())
        img = flat.reshape(shape)
        # Try to make a sensible RGB-ish view
        if img.ndim == 3:  # (bands, H, W)
            n_bands = img.shape[0]
            if n_bands >= 3:
                rgb = np.stack([img[min(2, n_bands - 1)], img[min(1, n_bands - 1)], img[0]], axis=-1)
            else:
                rgb = np.stack([img[0]] * 3, axis=-1)
        elif img.ndim == 2:
            rgb = np.stack([img] * 3, axis=-1)
        else:
            rgb = np.zeros((10, 10, 3))
        for c in range(3):
            lo, hi = np.percentile(rgb[..., c], [1, 99])
            rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
        rgb = rgb ** 0.5
        axes[i].imshow(rgb, origin="lower")
        axes[i].set_title(f"row {i}  shape={shape}", fontsize=9)
        axes[i].axis("off")


def plot_timeseries(table, axes, n_show=4):
    n_show = min(n_show, table.num_rows)
    for i in range(n_show):
        time = np.asarray(table.column("time")[i].as_py(), dtype=np.float64)
        flux = np.asarray(table.column("flux")[i].as_py(), dtype=np.float32)
        flux_err = None
        if "flux_err" in table.schema.names:
            flux_err = np.asarray(table.column("flux_err")[i].as_py(), dtype=np.float32)
        # MMU v1 stores fixed-length arrays with zero-padding at the end.
        # Drop padding rows (where time AND flux are both 0) for plotting.
        valid = ~((time == 0) & (flux == 0))
        time = time[valid]
        flux = flux[valid]
        if flux_err is not None:
            flux_err = flux_err[valid]
            axes[i].errorbar(time, flux, yerr=flux_err, fmt=".", markersize=2, color="navy", elinewidth=0.4)
        else:
            axes[i].plot(time, flux, ".", markersize=2, color="navy")
        axes[i].set_title(f"row {i}  ({len(time)} points)", fontsize=9)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Flux")


def plot_tabular(table, axes):
    """Show histograms of the first few numeric columns."""
    numeric_cols = []
    for f in table.schema:
        if f.name in ("ra", "dec", "object_id"):
            continue
        if f.name.startswith("_"):
            continue
        try:
            arr = table.column(f.name).to_numpy()
            if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1:
                numeric_cols.append(f.name)
        except Exception:
            pass
        if len(numeric_cols) >= len(axes):
            break

    for ax, name in zip(axes, numeric_cols):
        arr = table.column(name).to_numpy()
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            ax.hist(arr, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(name, fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="Path to inner HATS catalog dir (containing dataset/)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--n-show", type=int, default=4)
    args = parser.parse_args()

    files = find_parquet_files(args.catalog)
    if not files:
        print(f"ERROR: no parquet files in {args.catalog}", file=sys.stderr)
        return 1

    import pyarrow as pa
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables, promote_options="default")
    print(f"Loaded {table.num_rows} rows, {table.num_columns} columns from {len(files)} parquet file(s)")

    modality = detect_modality(table.schema.names)
    print(f"Detected modality: {modality}")

    n_show = min(args.n_show, table.num_rows)
    if modality == "tabular":
        n_show = 6
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.5 * n_show))
    if n_show == 1:
        axes = [axes]

    title = os.path.basename(os.path.normpath(args.catalog))
    fig.suptitle(f"{title}  ({modality}, {table.num_rows} rows)", fontsize=11, y=1.0)

    if modality == "spectra":
        plot_spectra(table, axes, n_show=n_show)
    elif modality == "image":
        plot_images(table, axes, n_show=n_show)
    elif modality == "timeseries":
        plot_timeseries(table, axes, n_show=n_show)
    else:
        plot_tabular(table, axes)

    plt.tight_layout()
    plt.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
