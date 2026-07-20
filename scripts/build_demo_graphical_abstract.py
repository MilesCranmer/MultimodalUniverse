"""Build a clean horizontal pipeline diagram (graphical abstract).

Five stages: HDF5 input → auto-converter → PyArrow → HATS catalog → LSDB
cross-match. Headline metrics underneath.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def stage_box(ax, x, y, w, h, title, body, color):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.5,
        edgecolor="#222222",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 0.18,
        title,
        ha="center", va="top",
        fontsize=12, weight="bold", color="#0a1a3a",
    )
    ax.text(
        x + w / 2, y + h / 2 - 0.05,
        body,
        ha="center", va="center",
        fontsize=9, color="#1a1a1a",
        wrap=True,
    )


def arrow(ax, x0, x1, y):
    ax.annotate(
        "",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.6,
            color="#444",
            mutation_scale=18,
        ),
    )


def metric_badge(ax, x, y, label, value, color):
    box = mpatches.FancyBboxPatch(
        (x, y), 1.7, 0.85,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.2,
        edgecolor="#222",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 0.85, y + 0.55, value,
            ha="center", va="center",
            fontsize=18, weight="bold", color="#0a1a3a")
    ax.text(x + 0.85, y + 0.18, label,
            ha="center", va="center",
            fontsize=9, color="#333")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # Title strip at top
    ax.text(
        7.5, 4.2,
        "MultimodalUniverse v2 — HDF5 → HATS auto-converter",
        ha="center", va="center",
        fontsize=15, weight="bold", color="#0a1a3a",
    )
    ax.text(
        7.5, 3.85,
        "One pipeline for spectra, images, time series, IFU and tabular catalogs",
        ha="center", va="center",
        fontsize=10, style="italic", color="#444",
    )

    # Pipeline stages
    stages = [
        ("MMU v1\nHDF5 tiles", "/mnt/ceph/users/\npolymathic/MMU/\n(per-survey)", "#E8F0FE"),
        ("auto_arrow_\ntable_from_hdf5", "introspect dtypes\n+ shapes; alias\nra/dec/object_id", "#FFF4E1"),
        ("PyArrow\ntable", "spectrum struct,\nimage cube + shape,\ntime series, scalars", "#FFE6E6"),
        ("HATS catalog\n(Parquet)", "healpix partitioned\n+ margin cache\n+ lazy column reads", "#E6F4EA"),
        ("LSDB\ncross-match", "spatial joins via\npixel alignment\n+ Dask", "#EFE6FF"),
    ]
    n = len(stages)
    margin = 0.35
    box_w = (15 - 2 * margin - (n - 1) * 0.55) / n
    box_h = 1.7
    y_box = 1.6

    box_xs = []
    for i, (title, body, color) in enumerate(stages):
        x = margin + i * (box_w + 0.55)
        stage_box(ax, x, y_box, box_w, box_h, title, body, color)
        box_xs.append((x, x + box_w))

    for i in range(n - 1):
        arrow(ax, box_xs[i][1] + 0.05, box_xs[i + 1][0] - 0.05, y_box + box_h / 2)

    # Headline metrics
    metric_y = 0.35
    metrics = [
        ("Catalogs", "14", "#E8F0FE"),
        ("Modalities", "5", "#FFF4E1"),
        ("Objects", "17,884", "#FFE6E6"),
        ("Columns", "1,613", "#E6F4EA"),
        ("Cross-matches", "11/12", "#EFE6FF"),
    ]
    total_w = len(metrics) * 1.7 + (len(metrics) - 1) * 0.35
    start_x = (15 - total_w) / 2
    for i, (label, value, color) in enumerate(metrics):
        metric_badge(ax, start_x + i * (1.7 + 0.35), metric_y, label, value, color)

    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
