"""COSMOS HATS validation visualizations with literature reference overlays.

For every diagnostic, the data is plotted alongside an authoritative reference
(Padova-like main sequence, common galaxy emission lines, color-color stellar
locus, etc.) so that obvious bugs jump off the page: if our data doesn't sit
on top of the reference, something is wrong.

Generates 10 PNGs under OUT_DIR. The companion VIZ_GUIDE document (printed at
the end) describes what each plot SHOULD look like according to the science.
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


COSMOS_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_cosmos"
OUT_DIR = "/tmp/cosmos_viz"
COSMOS_RA = 150.0
COSMOS_DEC = 2.0
COSMOS_RADIUS = 0.5

DATASETS = ["allwise", "twomass", "galex", "sages", "sdss", "desi", "gaia"]
COLORS = {
    "allwise": "tab:purple",
    "twomass": "tab:orange",
    "galex":   "tab:cyan",
    "sages":   "tab:olive",
    "sdss":    "tab:red",
    "desi":    "tab:green",
    "gaia":    "tab:blue",
}
MARKERS = {
    "allwise": ".",
    "twomass": "+",
    "galex":   "x",
    "sages":   "v",
    "sdss":    "*",
    "desi":    "s",
    "gaia":    "o",
}


# ----------------------------------------------------------------------
# Reference data — literature anchor values used to overlay the plots.
# ----------------------------------------------------------------------

# Pecaut & Mamajek (2013) main-sequence color-magnitude relation in Gaia
# DR3 photometry, abridged. Each row is (BP-RP, M_G) for one spectral type.
# Source: http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
# (G_BP - G_RP, M_G columns)
PECAUT_MAMAJEK_MS = np.array([
    # spectype  BP-RP   M_G
    (-0.40, -2.50),  # B5V
    (-0.20, -0.30),  # B9V
    (-0.04,  0.71),  # A0V
    ( 0.07,  1.32),  # A5V
    ( 0.27,  2.16),  # F0V
    ( 0.50,  3.30),  # F5V
    ( 0.66,  4.13),  # G0V
    ( 0.78,  4.78),  # G5V
    ( 0.99,  5.71),  # K0V
    ( 1.31,  6.93),  # K5V
    ( 1.84,  9.41),  # M0V
    ( 2.43, 11.04),  # M3V
    ( 3.50, 13.40),  # M5V
])

# Common galaxy emission/absorption line rest wavelengths in Å.
# Used to draw vertical lines on each spectrum at obs_lam = lam_rest * (1 + z).
GALAXY_LINES = [
    ("Lyα",      1216),
    ("[OII]",    3727),
    ("Ca K",     3934),
    ("Ca H",     3969),
    ("Hδ",       4102),
    ("Hγ",       4341),
    ("Hβ",       4861),
    ("[OIII]",   5007),
    ("Mg b",     5175),
    ("Na D",     5895),
    ("Hα",       6563),
    ("[NII]",    6584),
    ("[SII]",    6716),
]

# Stern+ (2012) AGN cut in WISE colors: W1-W2 > 0.8 selects AGN/QSO.
# Stellar locus is at W1-W2 ~ 0, J-K ~ 0.3-0.8.
STERN_AGN_CUT = 0.8


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _load(name: str, columns=None) -> pa.Table | None:
    """Load all parquet partitions for one dataset, optionally column-pruned."""
    root = os.path.join(COSMOS_ROOT, name, name, name, "dataset")
    files = sorted(
        f for f in glob.glob(os.path.join(root, "**/*.parquet"), recursive=True)
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return None
    tables = []
    for f in files:
        try:
            tables.append(pq.read_table(f, columns=columns))
        except Exception as exc:
            print(f"  warn: could not read {f}: {exc}", file=sys.stderr)
    if not tables:
        return None
    return pa.concat_tables(tables, promote_options="default")


def _ra_dec(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    t = _load(name, columns=["ra", "dec"])
    if t is None or t.num_rows == 0:
        return None
    return t.column("ra").to_numpy(), t.column("dec").to_numpy()


def _haversine(ra, dec, ra0, dec0):
    ra_r, dec_r = np.deg2rad(ra), np.deg2rad(dec)
    ra0_r, dec0_r = np.deg2rad(ra0), np.deg2rad(dec0)
    a = (
        np.sin((dec_r - dec0_r) / 2) ** 2
        + np.cos(dec_r) * np.cos(dec0_r) * np.sin((ra_r - ra0_r) / 2) ** 2
    )
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _tangent_proj(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Tangent-plane projection around the cosmos cone center."""
    ra0 = np.deg2rad(COSMOS_RA)
    dec0 = np.deg2rad(COSMOS_DEC)
    ra_r = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)
    x = np.cos(dec_r) * np.sin(ra_r - ra0)
    y = np.sin(dec_r) * np.cos(dec0) - np.cos(dec_r) * np.sin(dec0) * np.cos(ra_r - ra0)
    return np.rad2deg(np.column_stack([x, y]))


def _cross_match(
    ra_a: np.ndarray, dec_a: np.ndarray,
    ra_b: np.ndarray, dec_b: np.ndarray,
    radius_arcsec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx_a, idx_b, dist_arcsec) of matched pairs within radius."""
    if len(ra_a) == 0 or len(ra_b) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    xa = _tangent_proj(ra_a, dec_a)
    xb = _tangent_proj(ra_b, dec_b)
    tree = cKDTree(xb)
    dists, idx_b = tree.query(xa, k=1)
    rad_deg = radius_arcsec / 3600.0
    keep = dists <= rad_deg
    return np.where(keep)[0], idx_b[keep], dists[keep] * 3600.0


# ----------------------------------------------------------------------
# Per-dataset figures
# ----------------------------------------------------------------------

def fig_sky_overlay(out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for name in DATASETS:
        rd = _ra_dec(name)
        if rd is None:
            continue
        ra, dec = rd
        ax.scatter(
            ra, dec,
            s=14, c=COLORS[name], marker=MARKERS[name],
            label=f"{name} ({len(ra)})",
            alpha=0.55, linewidths=0.5,
        )
    # Cone overlay (cosmos cone, slightly distorted by cos(dec) ≈ 0.999).
    theta = np.linspace(0, 2 * np.pi, 200)
    cos_d = np.cos(np.deg2rad(COSMOS_DEC))
    ax.plot(
        COSMOS_RA + (COSMOS_RADIUS / cos_d) * np.cos(theta),
        COSMOS_DEC + COSMOS_RADIUS * np.sin(theta),
        "k--", lw=1.8, label=f"{COSMOS_RADIUS}° cone (reference)",
    )
    ax.scatter([COSMOS_RA], [COSMOS_DEC], marker="+", s=300, c="black", linewidths=2)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"COSMOS sky overlay — center ({COSMOS_RA}, {COSMOS_DEC}), radius {COSMOS_RADIUS}°\n"
        "Reference: dashed circle = exact cone boundary"
    )
    ax.set_aspect("equal")
    ax.invert_xaxis()  # astronomical convention
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_distance_histograms(out: str) -> None:
    """Each dataset: histogram of distance from cone center.

    REFERENCE: a uniform sky density inside the cone gives a histogram
    proportional to r (linear ramp from 0 at r=0 to peak at r=R), so we
    overlay the linear ramp expectation, normalized to the data's total.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    axes = axes.flatten()
    bins = np.linspace(0, COSMOS_RADIUS * 1.05, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    for ax, name in zip(axes, DATASETS):
        rd = _ra_dec(name)
        if rd is None:
            ax.set_visible(False)
            continue
        ra, dec = rd
        d = _haversine(ra, dec, COSMOS_RA, COSMOS_DEC)
        counts, _ = np.histogram(d, bins=bins)
        ax.bar(
            bin_centers, counts, width=bin_width * 0.9,
            color=COLORS[name], edgecolor="black", linewidth=0.4,
            label=f"data (n={len(d)})",
        )
        # Reference: uniform density expectation = N_total * 2*r / R^2 * dr
        # (area-element dA = 2*pi*r*dr; histogram = (N/A_total) * 2*pi*r*dr).
        ref = len(d) * (2.0 * bin_centers / (COSMOS_RADIUS ** 2)) * bin_width
        ref[bin_centers > COSMOS_RADIUS] = 0
        ax.plot(
            bin_centers, ref, "k-", lw=1.5,
            label="uniform-density reference",
        )
        ax.axvline(
            COSMOS_RADIUS, color="red", linestyle="--", lw=1,
            label=f"{COSMOS_RADIUS}° cone",
        )
        ax.set_xlabel("dist from cone center [deg]")
        ax.set_ylabel("# rows")
        ax.set_title(f"{name} (n={len(d)})")
        ax.set_xlim(0, COSMOS_RADIUS * 1.05)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(alpha=0.3)

    for ax in axes[len(DATASETS):]:
        ax.set_visible(False)
    fig.suptitle(
        "Radial distribution from cosmos cone center — should track linear ramp "
        "and cut off cleanly at 0.5°",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_spectra(name: str, out: str, n_examples: int = 6) -> None:
    """Spectra with rest-frame galaxy lines redshifted to each source's Z.

    REFERENCE: vertical dashed lines at common galaxy emission/absorption
    line wavelengths shifted by (1+z). Real spectra of galaxies at the
    cataloged Z should show actual features at these positions.
    """
    cols_needed = ["spectrum", "ra", "dec", "Z"]
    t = _load(name, columns=cols_needed)
    if t is None or t.num_rows == 0:
        print(f"  skip {name} spectra (no data)")
        return
    n = min(n_examples, t.num_rows)
    rows = t.slice(0, n).to_pylist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        spec = rows[i]["spectrum"]
        z = rows[i].get("Z", 0.0)
        if z is None or not np.isfinite(z) or z < -0.05 or z > 7:
            z = 0.0
        flux = np.asarray(spec["flux"], dtype=float)
        lam = np.asarray(spec["lambda"], dtype=float)
        mask = np.asarray(spec["mask"])
        good = ~mask if mask.dtype == bool else (mask == 0)
        flux_good = flux[good]
        lam_good = lam[good]
        if len(lam_good) == 0:
            ax.text(0.5, 0.5, "no good pixels", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.plot(lam_good, flux_good, lw=0.5, c=COLORS[name])
        # Reference: redshifted galaxy emission lines.
        ymin, ymax = np.nanpercentile(flux_good, [1, 99])
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax == ymin:
            ymin, ymax = float(np.nanmin(flux_good)), float(np.nanmax(flux_good)) + 1
        ax.set_ylim(ymin - 0.5 * (ymax - ymin), ymax + 0.5 * (ymax - ymin))
        for label, rest in GALAXY_LINES:
            obs = rest * (1 + z)
            if lam_good.min() <= obs <= lam_good.max():
                ax.axvline(obs, color="gray", lw=0.5, alpha=0.6)
                ax.text(obs, ax.get_ylim()[1], label,
                        ha="center", va="top", fontsize=6,
                        rotation=90, color="gray")
        ax.set_xlabel("wavelength [Å]")
        ax.set_ylabel("flux")
        ax.set_title(
            f"{name} #{i}: ra={rows[i]['ra']:.4f}, dec={rows[i]['dec']:.4f}, z={z:.4f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(
        f"{name.upper()} spectra in COSMOS — vertical lines = galaxy lines "
        "redshifted by each source's measured Z",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_redshifts(out: str) -> None:
    """SDSS and DESI redshift distributions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, name in zip(axes, ["sdss", "desi"]):
        t = _load(name, columns=["Z"])
        if t is None or t.num_rows == 0:
            ax.set_visible(False)
            continue
        z = t.column("Z").to_numpy().astype(float)
        z = z[np.isfinite(z) & (z > -0.01) & (z < 7)]
        ax.hist(z, bins=40, color=COLORS[name], edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("spectroscopic redshift Z")
        ax.set_ylabel("# sources")
        ax.set_title(
            f"{name} in COSMOS (n={len(z)}, median z = {np.median(z):.3f})"
        )
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Spectroscopic redshift distribution — real surveys peak at z~0.1-0.5 "
        "with a long high-z tail",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gaia_cmd(out: str) -> None:
    """Gaia CMD with Pecaut & Mamajek main-sequence overlay.

    REFERENCE: Pecaut & Mamajek (2013) main-sequence M_G vs BP-RP. Compute
    M_G from G + 5 + 5*log10(parallax/1000), drop sources with bad parallax.
    Real stars should sit on the main sequence locus or above it (giants).
    """
    t = _load("gaia", columns=["photometry", "astrometry"])
    if t is None or t.num_rows == 0:
        print("  skip gaia CMD (no data)")
        return
    rows = t.to_pylist()
    g = np.array([r["photometry"]["phot_g_mean_mag"] for r in rows])
    bp_rp = np.array([r["photometry"]["bp_rp"] for r in rows])
    parallax = np.array([r["astrometry"]["parallax"] for r in rows])
    parallax_err = np.array([r["astrometry"]["parallax_error"] for r in rows])
    valid = (
        np.isfinite(g) & np.isfinite(bp_rp)
        & np.isfinite(parallax) & (parallax > 0)
        & (parallax_err < 0.2 * parallax)  # >5σ parallax
    )
    g_v = g[valid]
    bp_rp_v = bp_rp[valid]
    parallax_v = parallax[valid]
    abs_g = g_v + 5 + 5 * np.log10(parallax_v / 1000)

    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.scatter(bp_rp_v, abs_g, s=20, c="tab:blue", alpha=0.7, label=f"Gaia data (n={len(abs_g)})")
    # Pecaut-Mamajek reference main sequence
    ax.plot(
        PECAUT_MAMAJEK_MS[:, 0], PECAUT_MAMAJEK_MS[:, 1],
        "k-", lw=2, label="Pecaut & Mamajek (2013) main sequence",
    )
    ax.set_xlabel("BP − RP [mag]")
    ax.set_ylabel("M_G [absolute mag]")
    ax.invert_yaxis()
    ax.set_title(
        f"Gaia DR3 CMD in COSMOS — data should sit on or above the\n"
        f"main sequence (giants are brighter than the line)"
    )
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(15, -3)  # inverted; brighter at top
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------
# Cross-survey science checks
# ----------------------------------------------------------------------

def fig_pairwise_crossmatch(out: str, radius_arcsec: float = 1.0) -> None:
    """Bar chart of pairwise 1\" cross-match counts for all dataset pairs."""
    coords: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in DATASETS:
        rd = _ra_dec(name)
        if rd is not None:
            coords[name] = rd
    pairs = []
    counts = []
    names = list(coords.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ix_a, _, _ = _cross_match(*coords[a], *coords[b], radius_arcsec=radius_arcsec)
            pairs.append(f"{a}\n×\n{b}")
            counts.append(len(ix_a))
    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.5), 6))
    bars = ax.bar(range(len(pairs)), counts, color="tab:gray", edgecolor="black")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=6.5, rotation=0)
    ax.set_ylabel(f"# matches within {radius_arcsec}\"")
    ax.set_title(
        f"Pairwise cross-match counts in COSMOS ({radius_arcsec}\" radius)\n"
        "AllWISE × 2MASS should give MANY matches (overlapping all-sky catalogs)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_allwise_2mass_color_color(out: str) -> None:
    """W1-W2 vs J-K color-color diagram for AllWISE × 2MASS matches.

    REFERENCE: Stern+ (2012) AGN cut at W1-W2 > 0.8. Stellar locus sits at
    J-K ~ 0.3-0.8, W1-W2 ~ -0.1 to +0.1. QSOs/AGN at high W1-W2.
    """
    aw = _load("allwise", columns=["ra", "dec", "w1mpro", "w2mpro"])
    tm = _load("twomass", columns=["ra", "dec", "j_m", "k_m"])
    if aw is None or tm is None:
        print("  skip allwise×2mass (missing dataset)")
        return
    ra_a = aw.column("ra").to_numpy()
    dec_a = aw.column("dec").to_numpy()
    w1 = aw.column("w1mpro").to_numpy()
    w2 = aw.column("w2mpro").to_numpy()
    ra_t = tm.column("ra").to_numpy()
    dec_t = tm.column("dec").to_numpy()
    j = tm.column("j_m").to_numpy()
    k = tm.column("k_m").to_numpy()

    ix_a, ix_t, _ = _cross_match(ra_a, dec_a, ra_t, dec_t, radius_arcsec=2.0)
    n_match = len(ix_a)
    if n_match == 0:
        print("  no allwise×2mass matches")
        return
    w1_m = w1[ix_a]
    w2_m = w2[ix_a]
    j_m = j[ix_t]
    k_m = k[ix_t]
    valid = np.isfinite(w1_m) & np.isfinite(w2_m) & np.isfinite(j_m) & np.isfinite(k_m)
    j_k = (j_m - k_m)[valid]
    w1_w2 = (w1_m - w2_m)[valid]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(j_k, w1_w2, s=14, c="tab:purple", alpha=0.55,
               label=f"matches (n={valid.sum()})")
    # Stern+ 2012 AGN cut.
    ax.axhline(STERN_AGN_CUT, color="red", lw=1.5, linestyle="--",
               label="Stern+ 2012 AGN cut (W1−W2 > 0.8)")
    # Stellar locus rectangle.
    ax.add_patch(plt.Rectangle(
        (0.2, -0.2), 0.6, 0.3,
        facecolor="none", edgecolor="green", lw=1.5,
        label="stellar locus (Wright+ 2010)",
    ))
    ax.set_xlabel("J − K [2MASS]")
    ax.set_ylabel("W1 − W2 [AllWISE]")
    ax.set_title(
        f"AllWISE × 2MASS color–color in COSMOS (2\" matches)\n"
        f"data should populate stellar locus (green box) and AGN cloud above red line"
    )
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_sdss_desi_redshift(out: str) -> None:
    """Z_sdss vs Z_desi for SDSS×DESI matches. Should fall on y=x line."""
    sd = _load("sdss", columns=["ra", "dec", "Z"])
    de = _load("desi", columns=["ra", "dec", "Z"])
    if sd is None or de is None:
        print("  skip sdss×desi redshift (missing dataset)")
        return
    ra_s = sd.column("ra").to_numpy()
    dec_s = sd.column("dec").to_numpy()
    z_s = sd.column("Z").to_numpy().astype(float)
    ra_d = de.column("ra").to_numpy()
    dec_d = de.column("dec").to_numpy()
    z_d = de.column("Z").to_numpy().astype(float)

    ix_s, ix_d, dists = _cross_match(ra_s, dec_s, ra_d, dec_d, radius_arcsec=1.5)
    n_match = len(ix_s)
    fig, ax = plt.subplots(figsize=(7, 7))
    if n_match == 0:
        ax.text(0.5, 0.5, "no SDSS×DESI matches in COSMOS",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("SDSS × DESI redshift agreement (NO matches)")
    else:
        zs_m = z_s[ix_s]
        zd_m = z_d[ix_d]
        valid = np.isfinite(zs_m) & np.isfinite(zd_m) & (zs_m > -0.01) & (zd_m > -0.01)
        ax.scatter(zs_m[valid], zd_m[valid], s=20, c="tab:red", alpha=0.7,
                   label=f"matches (n={valid.sum()})")
        zmin = min(0, float(min(zs_m[valid].min(), zd_m[valid].min())) - 0.05) if valid.any() else 0
        zmax = float(max(zs_m[valid].max(), zd_m[valid].max())) + 0.05 if valid.any() else 1
        # y = x reference
        ax.plot([zmin, zmax], [zmin, zmax], "k--", lw=1.5, label="y = x reference")
        ax.set_xlim(zmin, zmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel("Z (SDSS)")
        ax.set_ylabel("Z (DESI)")
        ax.set_title(
            f"SDSS × DESI spectroscopic redshift agreement ({n_match} matches)\n"
            "Real matches should fall on the y=x line"
        )
        ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gaia_desi_brightness(out: str) -> None:
    """DESI Z (redshift) vs Gaia G mag for matched sources.

    REFERENCE: Gaia bright limit at G ~ 21; DESI extragalactic targets are
    typically G ~ 17-21. Bright stellar sources at low G have spectroscopic
    Z ~ 0; faint extragalactic sources tail to higher z.
    """
    ga = _load("gaia", columns=["ra", "dec", "photometry"])
    de = _load("desi", columns=["ra", "dec", "Z"])
    if ga is None or de is None:
        print("  skip gaia×desi (missing dataset)")
        return
    rows_g = ga.to_pylist()
    ra_g = np.array([r["ra"] for r in rows_g])
    dec_g = np.array([r["dec"] for r in rows_g])
    g_mag = np.array([r["photometry"]["phot_g_mean_mag"] for r in rows_g])
    ra_d = de.column("ra").to_numpy()
    dec_d = de.column("dec").to_numpy()
    z_d = de.column("Z").to_numpy().astype(float)

    ix_g, ix_d, dists = _cross_match(ra_g, dec_g, ra_d, dec_d, radius_arcsec=1.5)
    n_match = len(ix_g)
    fig, ax = plt.subplots(figsize=(8, 6))
    if n_match == 0:
        ax.text(0.5, 0.5, "no Gaia×DESI matches in COSMOS",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gaia × DESI brightness vs redshift (NO matches)")
    else:
        gm = g_mag[ix_g]
        zm = z_d[ix_d]
        valid = np.isfinite(gm) & np.isfinite(zm) & (zm > -0.01) & (zm < 7)
        ax.scatter(gm[valid], zm[valid], s=20, c="tab:olive", alpha=0.7,
                   label=f"matches (n={valid.sum()})")
        ax.axvline(21, color="red", linestyle="--", lw=1, label="Gaia G=21 limit")
        ax.set_xlabel("Gaia G [mag]")
        ax.set_ylabel("DESI spectroscopic Z")
        ax.set_title(
            f"Gaia × DESI: Z vs Gaia G mag ({n_match} matches)\n"
            "Bright sources (G<18) tend to be stars (Z≈0); faint = extragalactic"
        )
        ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing visualizations to {OUT_DIR}/")
    fig_sky_overlay(os.path.join(OUT_DIR, "01_sky_overlay.png"))
    fig_distance_histograms(os.path.join(OUT_DIR, "02_distance_histograms.png"))
    fig_spectra("desi", os.path.join(OUT_DIR, "03_desi_spectra.png"))
    fig_spectra("sdss", os.path.join(OUT_DIR, "04_sdss_spectra.png"))
    fig_redshifts(os.path.join(OUT_DIR, "05_redshift_histograms.png"))
    fig_gaia_cmd(os.path.join(OUT_DIR, "06_gaia_cmd.png"))
    fig_pairwise_crossmatch(os.path.join(OUT_DIR, "07_pairwise_crossmatch.png"))
    fig_allwise_2mass_color_color(os.path.join(OUT_DIR, "08_allwise_2mass_colors.png"))
    fig_sdss_desi_redshift(os.path.join(OUT_DIR, "09_sdss_desi_redshift.png"))
    fig_gaia_desi_brightness(os.path.join(OUT_DIR, "10_gaia_desi_brightness.png"))
    print(f"\nFiles in {OUT_DIR}/:")
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        print(f"  {os.path.getsize(path):>10} bytes  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
