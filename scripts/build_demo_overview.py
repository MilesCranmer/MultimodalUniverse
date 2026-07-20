"""Single-page demo overview poster.

Layout:
   ┌──────────────────────────────────────────────────┐
   │  Graphical abstract (full width, banner)         │
   ├───────────────────────────┬──────────────────────┤
   │                           │  Sky map             │
   │  Lead visual              │                      │
   │  (matched galaxies)       ├──────────────────────┤
   │                           │  Cross-match table   │
   ├───────────────────────────┴──────────────────────┤
   │  Caption paragraph (full width)                  │
   └──────────────────────────────────────────────────┘
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


CAPTION = (
    "MMU v2 HATS conversion pipeline (top banner) and validation results. "
    "We auto-converted 14 surveys at the fiducial healpix tile (Norder=4, Npix=1177) "
    "into HATS Parquet collections under /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats. "
    "Catalogs span five modalities (spectra, image, time series, IFU, tabular). "
    "All 14 outputs pass row-count, RA/Dec, spectrum-struct and image-shape verification "
    "against the source HDF5. The right column shows where the catalogs land on the sky "
    "(top) and 11 of 12 LSDB cross-matches that succeed (bottom). The lead figure on the "
    "left shows six bright low-z galaxies recovered by an SDSS legacy x DESI Legacy Survey "
    "cross-match: each row pairs the LegacySurvey grz cutout with the SDSS spectrum from the "
    "matched object, with redshifted positions of common emission lines marked in red."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dir", default="test_data/demo")
    p.add_argument("--output", default="test_data/demo/01_overview.png")
    args = p.parse_args()

    abstract_path = os.path.join(args.demo_dir, "00_graphical_abstract.png")
    galaxies_path = os.path.join(args.demo_dir, "05_xmatch_galaxies.png")
    sky_path = os.path.join(args.demo_dir, "03_sky_map.png")
    table_path = os.path.join(args.demo_dir, "04_xmatch_panel.png")

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 5.0, 0.55],
        width_ratios=[1.4, 1.0],
        hspace=0.10, wspace=0.05,
    )

    # Title
    fig.suptitle(
        "MultimodalUniverse v2 — HATS conversion demo",
        fontsize=22, weight="bold", y=0.985,
    )

    # Banner: graphical abstract (spans both columns)
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.imshow(mpimg.imread(abstract_path))
    ax_banner.axis("off")

    # Lead visual: matched galaxies (large, left)
    ax_lead = fig.add_subplot(gs[1, 0])
    ax_lead.imshow(mpimg.imread(galaxies_path))
    ax_lead.set_title(
        "Lead result — bright low-z galaxies recovered by SDSS x LegacySurvey cross-match",
        fontsize=12, weight="bold", pad=8,
    )
    ax_lead.axis("off")

    # Right column: sky map on top, table below
    gs_right = gs[1, 1].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.16)

    ax_sky = fig.add_subplot(gs_right[0, 0])
    ax_sky.imshow(mpimg.imread(sky_path))
    ax_sky.set_title(
        "Sky positions of all 17,884 objects in the fiducial tile",
        fontsize=11, weight="bold", pad=6,
    )
    ax_sky.axis("off")

    ax_table = fig.add_subplot(gs_right[1, 0])
    ax_table.imshow(mpimg.imread(table_path))
    ax_table.set_title(
        "Pairwise LSDB cross-match results (11 of 12 pairs)",
        fontsize=11, weight="bold", pad=6,
    )
    ax_table.axis("off")

    # Caption strip (full width)
    ax_caption = fig.add_subplot(gs[2, :])
    ax_caption.axis("off")
    ax_caption.text(
        0.5, 0.6, CAPTION,
        ha="center", va="center",
        fontsize=10, color="#222",
        wrap=True,
        transform=ax_caption.transAxes,
        bbox=dict(facecolor="#f5f5f5", edgecolor="#bbb", boxstyle="round,pad=0.4"),
    )

    plt.savefig(args.output, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
