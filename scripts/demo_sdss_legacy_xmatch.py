"""End-to-end demo: HATS-ify SDSS spectra and LegacySurvey images for one healpix tile,
cross-match them, and plot matched galaxies (image + spectrum side by side).

Run on the Flatiron cluster where MMU v1 HDF5 data is mounted at
/mnt/ceph/users/polymathic/MultimodalUniverse/.
"""

import argparse
import os

import h5py
import numpy as np
import pyarrow as pa
from astropy.table import Table as AstropyTable

from mmu.hats_import import build_arrow_table, np_to_pyarrow_list, write_hats

MMU_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse"

SDSS_FLOAT = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
SDSS_BOOL = ["ZWARNING"]
SDSS_FLUX = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
SDSS_FILTERS = ["U", "G", "R", "I", "Z"]

LEGACY_FLOAT = [
    "FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_I", "FIBERFLUX_Z",
    "EBV",
]


def load_sdss_tile(healpix: int) -> pa.Table:
    path = f"{MMU_ROOT}/sdss/sdss/healpix={healpix}/001-of-001.hdf5"
    print(f"Loading SDSS tile: {path}")
    with h5py.File(path, "r") as f:
        cols = [
            "ra", "dec", "object_id",
            "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
            "spectrum_lsf_sigma", "spectrum_mask",
        ] + SDSS_FLOAT + SDSS_BOOL + SDSS_FLUX
        data = AstropyTable({k: f[k][:] for k in cols})
    table = build_arrow_table(
        data,
        float_features=SDSS_FLOAT,
        bool_features=SDSS_BOOL,
        flux_features=SDSS_FLUX,
        flux_filters=SDSS_FILTERS,
    )
    print(f"  SDSS: {table.num_rows} objects with spectra")
    return table


def load_legacy_tile_lazy(healpix: int, max_rows: int) -> pa.Table:
    """Read only the first `max_rows` from a LegacySurvey HDF5 tile, including image cutouts."""
    path = f"{MMU_ROOT}/legacysurvey/dr10_south_21/healpix={healpix}/001-of-001.hdf5"
    print(f"Loading LegacySurvey tile (first {max_rows} rows): {path}")
    with h5py.File(path, "r") as f:
        n = min(max_rows, f["object_id"].shape[0])

        columns = {
            "ra": pa.array(f["ra"][:n].astype(np.float64)),
            "dec": pa.array(f["dec"][:n].astype(np.float64)),
            "object_id": pa.array([str(o.decode() if isinstance(o, bytes) else o)
                                    for o in f["object_id"][:n]]),
        }

        for feat in LEGACY_FLOAT:
            if feat in f:
                columns[feat] = pa.array(f[feat][:n].astype(np.float32))

        # Read image cutouts as nested list-of-list (band, height, width)
        # Shape: (n, 4, 160, 160) — flatten last two dims to a list per band
        if "image_array" in f:
            imgs = f["image_array"][:n].astype(np.float32)
            n_obj, n_band, h, w = imgs.shape
            # Build a struct with one list-of-floats per row (band, h, w flattened)
            # Simpler: store as 1D list and remember shape
            flat = imgs.reshape(n_obj, -1)
            columns["image_flat"] = np_to_pyarrow_list(flat)
            columns["image_shape"] = pa.array(
                [[n_band, h, w]] * n_obj,
                type=pa.list_(pa.int32()),
            )

    table = pa.table(columns)
    print(f"  LegacySurvey: {table.num_rows} objects with image cutouts")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healpix", type=int, default=2300,
                        help="Healpix tile to process")
    parser.add_argument("--legacy-rows", type=int, default=2000,
                        help="Max LegacySurvey rows to read (limits memory)")
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser("~/ceph/general_data/mmu_hats_demo_out"),
                        help="Output directory for HATS catalogs and plot")
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--n-show", type=int, default=4,
                        help="Number of matched galaxies to plot")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    sdss_dir = os.path.join(args.output, "sdss_demo", "sdss_demo")
    legacy_dir = os.path.join(args.output, "legacy_demo", "legacy_demo")

    # Build SDSS HATS catalog (skip if already built)
    if not os.path.exists(sdss_dir):
        sdss_table = load_sdss_tile(args.healpix)
        write_hats([sdss_table], args.output, "sdss_demo", debug=True)
    else:
        print(f"Reusing existing SDSS catalog at {sdss_dir}")

    # Build LegacySurvey HATS catalog (skip if already built)
    if not os.path.exists(legacy_dir):
        legacy_table = load_legacy_tile_lazy(args.healpix, args.legacy_rows)
        write_hats([legacy_table], args.output, "legacy_demo", debug=True)
    else:
        print(f"Reusing existing LegacySurvey catalog at {legacy_dir}")

    # Cross-match using LSDB
    from mmu.data import HATSDataset

    sdss_ds = HATSDataset(
        os.path.join(args.output, "sdss_demo", "sdss_demo"),
        columns=["ra", "dec", "Z", "object_id", "spectrum"],
    )
    legacy_ds = HATSDataset(
        os.path.join(args.output, "legacy_demo", "legacy_demo"),
        columns=["ra", "dec", "object_id", "image_flat", "image_shape",
                 "FLUX_G", "FLUX_R", "FLUX_Z"],
    )

    print(f"\nCross-matching SDSS x LegacySurvey at radius={args.radius_arcsec} arcsec...")
    matched = sdss_ds.crossmatch(
        legacy_ds,
        radius_arcsec=args.radius_arcsec,
        suffixes=("_sdss", "_legacy"),
    )
    print(f"Matched {matched.matched_count} pairs")

    # Visualize: pick the brightest low-z galaxies for clearest spectra/images
    if matched.matched_count > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = matched.df.copy()
        # Filter to nearby galaxies (z < 0.3) where features are obvious
        df = df[(df["Z_sdss"] > 0.005) & (df["Z_sdss"] < 0.3)]
        # Sort by brightness (LegacySurvey r-band flux, descending)
        df = df.sort_values("FLUX_R_legacy", ascending=False).reset_index(drop=True)
        print(f"After filtering to bright low-z galaxies: {len(df)} pairs")

        n_show = min(args.n_show, len(df))
        fig, axes = plt.subplots(n_show, 2, figsize=(11, 3 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_show):
            row = df.iloc[i]

            # Image: reconstruct from flat list + shape
            img_flat = np.asarray(row["image_flat_legacy"], dtype=np.float32)
            shape = list(row["image_shape_legacy"])
            img = img_flat.reshape(shape)  # (4, 160, 160) — bands g, r, i, z

            # RGB from z, r, g (red, green, blue)
            rgb = np.stack([img[3], img[1], img[0]], axis=-1)
            # Per-band percentile stretch for visibility
            for c in range(3):
                lo, hi = np.percentile(rgb[..., c], [1, 99.5])
                rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
            rgb = rgb ** 0.5  # gamma stretch

            axes[i, 0].imshow(rgb, origin="lower")
            axes[i, 0].set_title(
                f"LegacySurvey grz  flux_r={row['FLUX_R_legacy']:.1f} nMgy"
            )
            axes[i, 0].axis("off")

            # Spectrum
            spec = row["spectrum_sdss"]
            # spec is a nested pandas DataFrame with columns flux/ivar/lambda/mask
            lam = np.asarray(spec["lambda"].values, dtype=np.float32)
            flux = np.asarray(spec["flux"].values, dtype=np.float32)
            mask = np.asarray(spec["mask"].values, dtype=bool)
            flux_masked = np.where(mask, np.nan, flux)

            axes[i, 1].plot(lam, flux_masked, linewidth=0.6, color="navy")
            axes[i, 1].set_title(
                f"SDSS spectrum  z={row['Z_sdss']:.4f}  sep={row['_dist_arcsec']:.2f}\""
            )
            axes[i, 1].set_xlabel("Observed wavelength [Å]")
            axes[i, 1].set_ylabel("Flux")
            axes[i, 1].set_xlim(3800, 9200)

            # Mark common emission lines at observed wavelength (z corrected)
            z = row["Z_sdss"]
            lines = {
                "Hβ": 4861, "[OIII]": 5007, "Mg": 5175, "Na": 5893,
                "Hα": 6563, "[SII]": 6724,
            }
            ymin, ymax = axes[i, 1].get_ylim()
            for name, rest in lines.items():
                obs = rest * (1 + z)
                if 3800 < obs < 9200:
                    axes[i, 1].axvline(obs, color="red", alpha=0.25, linewidth=0.5)
                    axes[i, 1].text(obs, ymax * 0.92, name, fontsize=7,
                                    rotation=90, ha="right", va="top", color="red", alpha=0.6)

        plt.tight_layout()
        plot_path = os.path.join(args.output, "xmatch_demo.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"\nSaved visualization to {plot_path}")


if __name__ == "__main__":
    main()
