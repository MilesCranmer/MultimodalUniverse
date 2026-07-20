"""Render the LSDB cross-match panel as a heatmap-style plot."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xmatch-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = json.load(open(args.xmatch_json))
    matches = data["matches"]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(matches) + 2)))
    ax.axis("off")

    headers = [
        "Left catalog", "Right catalog", "Radius",
        "Left N", "Right N", "Matches", "% of L", "Median sep", "Status",
    ]
    rows = []
    for m in matches:
        n = m["n_matches"]
        pct = 100 * n / m["left_n"] if m["left_n"] else 0
        med = (
            f"{m['median_dist_arcsec']*1000:.1f} mas"
            if m.get("median_dist_arcsec") else "—"
        )
        status = "ERROR" if m.get("error") else ("OK" if n > 0 else "no matches")
        rows.append([
            m["left"],
            m["right"],
            f"{m['radius_arcsec']:.1f}\"",
            f"{m['left_n']:,}",
            f"{m['right_n']:,}",
            f"{n:,}",
            f"{pct:.1f}%",
            med,
            status,
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header styling
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white", weight="bold")

    # Status coloring
    for ri, row in enumerate(rows):
        status = row[-1]
        color = {
            "OK": "#e1f5e1",
            "no matches": "#fff5cc",
            "ERROR": "#ffcccc",
        }.get(status, "white")
        for ci in range(len(headers)):
            table[(ri + 1, ci)].set_facecolor(color)

    fig.suptitle(
        f"LSDB cross-matches across {len(data['catalogs'])} HATS catalogs "
        f"(fiducial healpix tile)",
        fontsize=13, y=0.98,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
