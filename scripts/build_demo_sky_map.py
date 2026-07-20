"""Plot ra/dec scatter for every v2 HATS catalog on one figure."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


MOD_COLORS = {
    "spectra": "#1f77b4",
    "image": "#d62728",
    "timeseries": "#2ca02c",
    "tabular": "#7f7f7f",
    "ifu": "#9467bd",
}

CATALOG_MODALITY = {
    "sdss_sdss": "spectra", "sdss_boss": "spectra", "desi_dr1_main": "spectra",
    "vipers_vipers_w1": "spectra", "galah_dr3": "spectra", "apogee_apogee": "spectra",
    "chandra_spectra": "spectra", "gaia_gaia": "spectra",
    "legacysurvey_dr10_south_21": "image",
    "manga_manga": "ifu",
    "tess_spoc": "timeseries",
    "allwise_allwise": "tabular", "twomass_psc": "tabular",
    "galex_ais": "tabular", "sages_dr1": "tabular",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Combined ra/dec parquet")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    table = pq.read_table(args.parquet)
    df = table.to_pandas()

    # Split into fiducial tile and MaNGA (different sky region)
    tile_mask = (df["ra"] > 340) & (df["ra"] < 350) & (df["dec"] > -5) & (df["dec"] < 5)
    tile = df[tile_mask]
    other = df[~tile_mask]

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Plot tile catalogs in stacking order: largest (tabular bg) first
    cat_order = sorted(
        tile["catalog"].unique(),
        key=lambda c: -len(tile[tile["catalog"] == c]),
    )
    for cat in cat_order:
        sub = tile[tile["catalog"] == cat]
        mod = CATALOG_MODALITY.get(cat, "tabular")
        color = MOD_COLORS[mod]
        ax.scatter(sub["ra"], sub["dec"], c=color, s=10, alpha=0.5,
                   edgecolors="none", label=f"{cat} ({len(sub):,})")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"Fiducial healpix tile (Norder=4, Npix=1177) — "
        f"{len(tile):,} objects, {tile['catalog'].nunique()} catalogs"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)

    # MaNGA inset (it's in a different sky region — healpix=0)
    if len(other):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = inset_axes(ax, width="22%", height="22%", loc="lower right",
                           borderpad=1.5)
        for cat in sorted(other["catalog"].unique()):
            sub = other[other["catalog"] == cat]
            color = MOD_COLORS[CATALOG_MODALITY.get(cat, "tabular")]
            inset.scatter(sub["ra"], sub["dec"], c=color, s=40,
                          edgecolors="black", linewidths=0.5, label=cat)
        inset.set_title(f"MaNGA (healpix=0)\n{len(other)} objects",
                        fontsize=7)
        inset.tick_params(axis="both", labelsize=6)
        inset.set_xlabel("RA", fontsize=6)
        inset.set_ylabel("Dec", fontsize=6)

    fig.suptitle("MMU v2 — sky coverage of HATS catalogs", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
