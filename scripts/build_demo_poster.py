"""Combine the four demo panels into a single landscape PNG poster."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dir", default="test_data/demo")
    p.add_argument("--output", default="test_data/demo/00_poster.png")
    args = p.parse_args()

    panels = [
        ("01_summary.png", "1) MMU v2 HATS conversion summary"),
        ("02_sky_map.png", "2) Sky coverage of converted catalogs"),
        ("03_xmatch_panel.png", "3) Pairwise LSDB cross-match results"),
        ("04_xmatch_galaxies.png", "4) Real cross-matched galaxies (SDSS spectra x LegacySurvey images)"),
    ]

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 1, hspace=0.12)

    for i, (fname, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        path = os.path.join(args.demo_dir, fname)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"missing: {fname}", ha="center", va="center")
            ax.axis("off")
            continue
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16, loc="left", weight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(
        "MultimodalUniverse v2 — HATS conversion demo",
        fontsize=22, weight="bold", y=0.995,
    )
    plt.savefig(args.output, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
