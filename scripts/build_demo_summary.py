"""Build a summary figure of all v2 HATS catalogs from a JSON inventory.

Run on the local machine after copying /tmp/inventory.json from the cluster.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MOD_COLORS = {
    "spectra": "#1f77b4",
    "image": "#d62728",
    "timeseries": "#2ca02c",
    "tabular": "#7f7f7f",
    "ifu": "#9467bd",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.inventory) as f:
        inv = json.load(f)
    inv.sort(key=lambda c: (c["modality"], c["catalog_name"]))

    names = [c["catalog_name"] for c in inv]
    rows = [c["n_rows"] for c in inv]
    cols = [c["n_columns"] for c in inv]
    sizes = [c["bytes_on_disk"] / 1e6 for c in inv]
    mods = [c["modality"] for c in inv]
    colors = [MOD_COLORS[m] for m in mods]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Row counts
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(names))
    ax.barh(y, rows, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Rows")
    ax.set_title("Rows per HATS catalog (1 healpix tile, capped at 2k)")
    ax.grid(axis="x", alpha=0.3)

    # Column counts
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(y, cols, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Columns")
    ax.set_title("Columns per HATS catalog")
    ax.grid(axis="x", alpha=0.3)

    # Disk size (log scale)
    ax = fig.add_subplot(gs[1, 0])
    ax.barh(y, sizes, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("symlog")
    ax.set_xlabel("Parquet size [MB, log]")
    ax.set_title("Disk footprint (Parquet, after HATS partitioning)")
    ax.grid(axis="x", alpha=0.3)

    # Type breakdown summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    total_rows = sum(rows)
    total_size = sum(sizes)
    total_cols = sum(cols)
    n_modalities = len(set(mods))
    summary_lines = [
        f"MMU v2 HATS — fiducial healpix tile (1177)",
        "",
        f"Catalogs:        {len(inv)}",
        f"Modalities:      {n_modalities}  (spectra/image/timeseries/tabular/ifu)",
        f"Total rows:      {total_rows:,}",
        f"Total columns:   {total_cols:,}",
        f"Total disk:      {total_size:,.1f} MB ({total_size/1024:.2f} GB)",
        "",
        "Per-modality counts:",
    ]
    from collections import Counter
    mod_counts = Counter(mods)
    for m in ("spectra", "image", "timeseries", "tabular", "ifu"):
        if m in mod_counts:
            n = mod_counts[m]
            r = sum(rows[i] for i in range(len(inv)) if mods[i] == m)
            s = sum(sizes[i] for i in range(len(inv)) if mods[i] == m)
            summary_lines.append(f"  {m:<11s} {n:>2d} catalogs, {r:>7,} rows, {s:>8.1f} MB")
    summary_lines += [
        "",
        "All catalogs have:",
        "  - HATS healpix partitioning (Norder=4 typical)",
        "  - HATS margin cache (10 arcsec)",
        "  - Lazy column reads via Parquet",
        "  - Native LSDB cross-match support",
    ]
    ax.text(0.0, 1.0, "\n".join(summary_lines),
            family="monospace", fontsize=9, va="top", ha="left",
            transform=ax.transAxes)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k", lw=0.4) for c in MOD_COLORS.values()]
    fig.legend(handles, list(MOD_COLORS.keys()),
               loc="upper center", ncol=len(MOD_COLORS),
               bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=10)

    fig.suptitle("MultimodalUniverse v2 — HATS conversion summary",
                 fontsize=14, y=1.02)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
