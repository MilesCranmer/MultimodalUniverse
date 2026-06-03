"""Visualisation helpers for COSMOS-Web MMU cutouts.

Band order in the (5, H, W) image.flux array:
    0  F115W  NIRCam 30 mas
    1  F150W  NIRCam 30 mas
    2  F277W  NIRCam 30 mas  ← selection band
    3  F444W  NIRCam 30 mas
    4  F770W  MIRI   60 mas

Quick-start (notebook)::

    import lsdb
    cat = lsdb.read_hats("/n03data/huertas/mmu/cosmos/cosmos/cosmos")
    df  = cat.head(64).compute()  # pandas DataFrame with nested image struct

    from scripts.cosmos.validation.viusalize import plot_all_modalities
    plot_all_modalities(df, max_plots=25, save=True, save_dir="/tmp/cosmos_vis")
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval, make_lupton_rgb


COSMOS_BANDS = ["F115W", "F150W", "F277W", "F444W", "F770W"]
BAND_INDEX = {b: i for i, b in enumerate(COSMOS_BANDS)}

# Default single-band display
DEFAULT_GRAY_BAND = "F277W"

# RGB composite channels
RGB_R = "F444W"
RGB_G = "F277W"
RGB_B = "F115W"


# ---------------------------------------------------------------------------
# Generic grid utilities
# ---------------------------------------------------------------------------


def compute_grid_shape(n_items: int, max_plots: int = 64) -> tuple[int, int, int]:
    """Return (n_select, nrows, ncols) for a near-square capped grid."""
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")
    n_select = min(int(n_items), int(max_plots))
    if n_select == 0:
        return 0, 0, 0
    ncols = int(math.ceil(math.sqrt(n_select)))
    nrows = int(math.ceil(n_select / ncols))
    return n_select, nrows, ncols


def flatten_axes(axs) -> np.ndarray:
    """Return axes as a 1D NumPy array regardless of original shape."""
    if isinstance(axs, np.ndarray):
        return axs.flatten()
    return np.array([axs])


def hide_unused_axes(axs, used_count: int) -> None:
    for ax in flatten_axes(axs)[int(used_count):]:
        ax.axis("off")


def _safe_object_id(row, object_id_col: str):
    try:
        return row[object_id_col]
    except Exception:
        return "unknown"


def _make_grid_axes_for_n(n_items: int, max_plots: int = 64, figsize_scale: float = 2.6):
    n_select, nrows, ncols = compute_grid_shape(n_items=n_items, max_plots=max_plots)
    if n_select == 0:
        raise ValueError("No rows available to plot.")
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * figsize_scale, nrows * figsize_scale))
    return fig, axs, n_select


# ---------------------------------------------------------------------------
# Low-level rendering primitives
# ---------------------------------------------------------------------------


def _render_grayscale(ax, image_2d: np.ndarray, title=None, fontsize: int = 8) -> None:
    ax.imshow(
        image_2d,
        cmap="gray",
        origin="lower",
        norm=ImageNormalize(image_2d, interval=PercentileInterval(99.5), stretch=AsinhStretch()),
    )
    if title is not None:
        ax.set_title(str(title), fontsize=fontsize)
    ax.axis("off")


def _render_rgb(ax, r: np.ndarray, g: np.ndarray, b: np.ndarray, title=None, fontsize: int = 8) -> None:
    """Lupton RGB composite."""
    rgb = make_lupton_rgb(r, g, b, interval=PercentileInterval(99.5), stretch=5, Q=8)
    ax.imshow(rgb, origin="lower")
    if title is not None:
        ax.set_title(str(title), fontsize=fontsize)
    ax.axis("off")


def _extract_band(image_flux_row, band_index: int) -> np.ndarray:
    """Extract one 2D plane from a (5, H, W) image_flux row."""
    if image_flux_row is None or isinstance(image_flux_row, np.ma.core.MaskedConstant):
        raise ValueError("Missing image_flux row (masked or None).")
    arr = np.asarray(image_flux_row)
    if arr.ndim == 3:
        if not (0 <= band_index < arr.shape[0]):
            raise ValueError(f"band_index {band_index} out of range for shape {arr.shape}.")
        return arr[band_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}.")


def _extract_rgb_channels(image_flux_row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, G, B) = (F444W, F277W, F115W) planes from a row."""
    if image_flux_row is None or isinstance(image_flux_row, np.ma.core.MaskedConstant):
        raise ValueError("Missing image_flux row (masked or None).")
    arr = np.asarray(image_flux_row)
    if arr.ndim != 3 or arr.shape[0] < 5:
        raise ValueError(f"Expected image_flux shape (5, H, W), got {arr.shape}.")
    return arr[BAND_INDEX[RGB_R]], arr[BAND_INDEX[RGB_G]], arr[BAND_INDEX[RGB_B]]


# ---------------------------------------------------------------------------
# Per-band grayscale grid
# ---------------------------------------------------------------------------


def plot_band_from_df(
    df,
    fig,
    axs,
    image_col: str = "image_flux",
    object_id_col: str = "obj_id",
    band: str = DEFAULT_GRAY_BAND,
    max_plots: int = 64,
) -> tuple:
    """Plot grayscale cutouts for one band from dataframe rows into caller-provided axes.

    Args:
        band: One of COSMOS_BANDS (default F277W).
    """
    band_index = BAND_INDEX[band]
    n_select = min(len(df), int(max_plots))
    axes = flatten_axes(axs)
    if len(axes) < n_select:
        raise ValueError(f"Not enough axes: need {n_select}, got {len(axes)}.")
    for i in range(n_select):
        row = df.iloc[i] if hasattr(df, "iloc") else df[i]
        obj_id = _safe_object_id(row, object_id_col)
        image_2d = _extract_band(row[image_col], band_index)
        _render_grayscale(axes[i], image_2d, title=obj_id)
    hide_unused_axes(axs, n_select)
    return fig, axs


# ---------------------------------------------------------------------------
# RGB composite grid
# ---------------------------------------------------------------------------


def plot_rgb_from_df(
    df,
    fig,
    axs,
    image_col: str = "image_flux",
    object_id_col: str = "obj_id",
    max_plots: int = 64,
) -> tuple:
    """Plot F444W/F277W/F115W RGB composites from dataframe rows."""
    n_select = min(len(df), int(max_plots))
    axes = flatten_axes(axs)
    if len(axes) < n_select:
        raise ValueError(f"Not enough axes: need {n_select}, got {len(axes)}.")
    for i in range(n_select):
        row = df.iloc[i] if hasattr(df, "iloc") else df[i]
        obj_id = _safe_object_id(row, object_id_col)
        r, g, b = _extract_rgb_channels(row[image_col])
        _render_rgb(axes[i], r, g, b, title=obj_id)
    hide_unused_axes(axs, n_select)
    return fig, axs


# ---------------------------------------------------------------------------
# 5-band strip: one row per object, 5 columns for F115W–F770W
# ---------------------------------------------------------------------------


def plot_multiband_strip(
    df,
    image_col: str = "image_flux",
    object_id_col: str = "obj_id",
    n_objects: int = 8,
    figsize_scale: float = 2.0,
    show: bool = True,
    save: bool = False,
    save_dir: str | Path | None = None,
    dpi: int = 180,
) -> tuple:
    """Plot a strip with one row per object and one column per JWST band.

    Args:
        n_objects: Number of objects to show (rows in the figure).
    """
    n = min(len(df), n_objects)
    n_bands = len(COSMOS_BANDS)
    fig, axs = plt.subplots(n, n_bands, figsize=(n_bands * figsize_scale, n * figsize_scale))
    axs = np.atleast_2d(axs)

    for row_idx in range(n):
        row = df.iloc[row_idx] if hasattr(df, "iloc") else df[row_idx]
        obj_id = _safe_object_id(row, object_id_col)
        arr = row[image_col]
        for col_idx, band in enumerate(COSMOS_BANDS):
            ax = axs[row_idx, col_idx]
            try:
                plane = _extract_band(arr, BAND_INDEX[band])
                _render_grayscale(ax, plane)
            except Exception:
                ax.axis("off")
            if row_idx == 0:
                ax.set_title(band, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(str(obj_id), fontsize=7, rotation=0, labelpad=40, va="center")

    fig.suptitle(f"COSMOS-Web 5-band strips (n={n})", fontsize=11)
    fig.tight_layout()

    if save:
        if save_dir is None:
            raise ValueError("save_dir must be provided when save=True.")
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "multiband_strip.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    return fig, axs


# ---------------------------------------------------------------------------
# Combined visualisation
# ---------------------------------------------------------------------------


def plot_all_modalities(
    df,
    max_plots: int = 64,
    object_id_col: str = "obj_id",
    show: bool = True,
    save: bool = False,
    save_dir: str | Path | None = None,
    dpi: int = 180,
) -> dict:
    """Plot grayscale (F277W) and RGB (F444W/F277W/F115W) grids.

    Returns a dict with keys 'gray' and 'rgb', each mapping to (fig, axs).
    """
    out = {}
    save_path = None
    if save:
        if save_dir is None:
            raise ValueError("save_dir must be provided when save=True.")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Grayscale (F277W)
    fig_g, axs_g, n_g = _make_grid_axes_for_n(len(df), max_plots=max_plots)
    plot_band_from_df(df=df, fig=fig_g, axs=axs_g, object_id_col=object_id_col, band=DEFAULT_GRAY_BAND, max_plots=max_plots)
    fig_g.suptitle(f"{DEFAULT_GRAY_BAND} cutouts (showing {n_g}/{len(df)})")
    out["gray"] = (fig_g, axs_g)
    if save_path is not None:
        fig_g.savefig(save_path / f"{DEFAULT_GRAY_BAND.lower()}_cutouts.png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)

    # RGB
    fig_r, axs_r, n_r = _make_grid_axes_for_n(len(df), max_plots=max_plots)
    plot_rgb_from_df(df=df, fig=fig_r, axs=axs_r, object_id_col=object_id_col, max_plots=max_plots)
    fig_r.suptitle(f"RGB ({RGB_R}/{RGB_G}/{RGB_B}) cutouts (showing {n_r}/{len(df)})")
    out["rgb"] = (fig_r, axs_r)
    if save_path is not None:
        fig_r.savefig(save_path / "rgb_cutouts.png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()
    return out
