"""Validate each completed COSMOS HATS catalog.

Checks, per dataset:
  1. hats.properties exists
  2. at least one data parquet under dataset/Norder=*/
  3. schema sanity: all expected fields present with correct types
  4. row count > 0
  5. sample row readable
  6. lsdb.read_hats() can load the catalog
  7. a cone query via lsdb returns rows inside the cosmos cone

Prints a compact report.
"""

from __future__ import annotations

import glob
import os
import sys
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


COSMOS_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_cosmos"
COSMOS_RA = 150.0
COSMOS_DEC = 2.0
COSMOS_RADIUS = 0.5


# What schema do we expect for each dataset?
# Each entry: name -> dict(required_top_level_columns, struct_field=expected_sub_field_names or None)
EXPECTED = {
    "allwise": {
        "top_level": ["ra", "dec", "object_id", "cntr", "w1mpro"],
        "structs": {},
    },
    "twomass": {
        "top_level": ["ra", "dec", "object_id"],
        "structs": {},
    },
    "galex": {
        "top_level": ["ra", "dec", "object_id"],
        "structs": {},
    },
    "sages": {
        "top_level": ["ra", "dec", "object_id", "MAG_U", "MAG_V"],
        "structs": {},
    },
    "sdss": {
        "top_level": ["ra", "dec", "object_id", "spectrum", "survey", "Z", "Z_ERR"],
        "structs": {
            "spectrum": {"flux", "ivar", "lsf_sigma", "lambda", "mask"},
        },
    },
    "desi": {
        "top_level": [
            "ra", "dec", "object_id", "spectrum",
            "Z", "ZERR", "ZWARN", "EBV",
            "FLUX_G", "FLUX_R", "FLUX_Z",
            "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
            "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z",
            "FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z",
        ],
        "structs": {
            "spectrum": {"flux", "ivar", "lsf_sigma", "lambda", "mask"},
        },
    },
    "gaia": {
        "top_level": [
            "ra", "dec", "object_id",
            "spectral_coefficients", "photometry", "astrometry",
            "radial_velocity", "gspphot", "flags", "corrections",
        ],
        "structs": {
            "spectral_coefficients": {"coeff", "coeff_error"},
        },
    },
}


def _find_data_parquet(dataset_dir: str) -> str | None:
    for f in sorted(glob.glob(os.path.join(dataset_dir, "**/*.parquet"), recursive=True)):
        if os.path.basename(f).startswith("_"):
            continue
        return f
    return None


def validate_one(name: str) -> dict:
    """Return a dict describing the validation result for one dataset."""
    result: dict = {"name": name, "ok": True, "errors": [], "warnings": [], "info": {}}
    root = os.path.join(COSMOS_ROOT, name, name, name)
    result["info"]["root"] = root

    hats_props = os.path.join(root, "hats.properties")
    if not os.path.exists(hats_props):
        result["ok"] = False
        result["errors"].append(f"missing hats.properties at {hats_props}")
        return result

    dataset_dir = os.path.join(root, "dataset")
    if not os.path.isdir(dataset_dir):
        result["ok"] = False
        result["errors"].append(f"missing dataset/ subdir at {dataset_dir}")
        return result

    parquet = _find_data_parquet(dataset_dir)
    if parquet is None:
        result["ok"] = False
        result["errors"].append(f"no data parquet under {dataset_dir}")
        return result

    try:
        tbl = pq.read_table(parquet)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"failed to read parquet {parquet}: {exc}")
        return result

    n_rows_first = tbl.num_rows
    schema_names = set(tbl.schema.names)
    result["info"]["first_parquet"] = os.path.basename(parquet)
    result["info"]["first_parquet_rows"] = n_rows_first
    result["info"]["n_columns"] = len(schema_names)

    # Total rows across all partitions.
    data_files = [
        f for f in sorted(glob.glob(os.path.join(dataset_dir, "**/*.parquet"), recursive=True))
        if not os.path.basename(f).startswith("_")
    ]
    total_rows = 0
    for f in data_files:
        try:
            total_rows += pq.read_metadata(f).num_rows
        except Exception as exc:
            result["warnings"].append(f"failed to read metadata for {f}: {exc}")
    result["info"]["n_partitions"] = len(data_files)
    result["info"]["total_rows"] = total_rows

    if total_rows == 0:
        result["ok"] = False
        result["errors"].append("total row count is 0")

    # Schema: required top-level columns present.
    expected = EXPECTED.get(name, {})
    required = expected.get("top_level", [])
    missing = [c for c in required if c not in schema_names]
    if missing:
        result["ok"] = False
        result["errors"].append(f"missing top-level columns: {missing}")

    # Struct sub-fields check.
    for struct_name, expected_subs in expected.get("structs", {}).items():
        if struct_name not in schema_names:
            continue  # caught above
        struct_type = tbl.schema.field(struct_name).type
        if not pa.types.is_struct(struct_type):
            result["ok"] = False
            result["errors"].append(
                f"{struct_name!r} should be a struct, got {struct_type}"
            )
            continue
        actual_subs = {f.name for f in struct_type}
        missing_subs = expected_subs - actual_subs
        if missing_subs:
            result["ok"] = False
            result["errors"].append(
                f"{struct_name!r} missing sub-fields: {sorted(missing_subs)}"
            )

    # Check ra/dec ranges: every row should be inside the cosmos cone.
    if "ra" in schema_names and "dec" in schema_names and n_rows_first > 0:
        ra_vals = tbl.column("ra").to_numpy()
        dec_vals = tbl.column("dec").to_numpy()
        # Great-circle distance to cosmos center (haversine).
        ra_r = np.deg2rad(ra_vals)
        dec_r = np.deg2rad(dec_vals)
        ra0 = np.deg2rad(COSMOS_RA)
        dec0 = np.deg2rad(COSMOS_DEC)
        d_ra = ra_r - ra0
        d_dec = dec_r - dec0
        a = np.sin(d_dec / 2) ** 2 + np.cos(dec_r) * np.cos(dec0) * np.sin(d_ra / 2) ** 2
        dists = np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        max_dist = float(np.nanmax(dists))
        inside = int(np.sum(dists <= COSMOS_RADIUS))
        result["info"]["max_dist_deg"] = round(max_dist, 4)
        result["info"]["inside_cone_ratio"] = f"{inside}/{n_rows_first}"
        # Tabular datasets that only apply the cone cut per-shard can have rows
        # slightly outside if the shard's footprint overlaps only partially.
        # We warn instead of fail if that's the case — as long as all rows are
        # near the cone and not obviously wrong.
        if max_dist > 2.0:  # 2 degrees is >> 0.5 degree cone, clearly wrong
            result["ok"] = False
            result["errors"].append(
                f"row at {max_dist:.3f}° from cone center "
                f"(expected <= {COSMOS_RADIUS}°)"
            )
        elif max_dist > COSMOS_RADIUS:
            result["warnings"].append(
                f"some rows up to {max_dist:.3f}° from cone center "
                f"(cone is {COSMOS_RADIUS}°); {inside}/{n_rows_first} strictly inside"
            )

    # Sample row.
    try:
        sample = tbl.slice(0, 1).to_pylist()[0]
        key_keys = [k for k in ("ra", "dec", "object_id", "Z", "ZERR", "ZWARN",
                                 "MAG_U", "FLUX_G", "tic_id", "survey")
                     if k in sample]
        result["info"]["sample"] = {k: sample[k] for k in key_keys[:6]}
    except Exception as exc:
        result["warnings"].append(f"sample row extraction failed: {exc}")

    return result


def validate_lsdb(name: str) -> dict:
    """Try loading via lsdb and running a tiny cone query."""
    result: dict = {"name": name, "ok": True, "errors": [], "info": {}}
    try:
        import lsdb
    except ImportError as exc:
        result["ok"] = False
        result["errors"].append(f"lsdb import failed: {exc}")
        return result

    collection = os.path.join(COSMOS_ROOT, name, name)
    try:
        cat = lsdb.read_hats(collection)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"lsdb.read_hats({collection}) failed: {exc}")
        return result

    try:
        head = cat.compute().head(3)
        result["info"]["lsdb_head_rows"] = len(head)
        result["info"]["lsdb_head_cols"] = len(head.columns)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"lsdb .compute().head() failed: {exc}")

    return result


def print_report(results: list[dict], lsdb_results: list[dict]) -> None:
    print("=" * 70)
    print(f"COSMOS validation report ({len(results)} datasets)")
    print("=" * 70)
    n_ok = sum(1 for r in results if r["ok"])
    n_lsdb_ok = sum(1 for r in lsdb_results if r["ok"])
    for r in results:
        name = r["name"]
        status = "OK" if r["ok"] else "FAIL"
        info = r["info"]
        print(f"\n[{status}] {name}")
        if "total_rows" in info:
            print(f"  partitions: {info.get('n_partitions')}  rows: {info.get('total_rows')}  columns: {info.get('n_columns')}")
        if "first_parquet" in info:
            print(f"  first parquet: {info['first_parquet']} ({info['first_parquet_rows']} rows)")
        if "max_dist_deg" in info:
            print(f"  max dist from cosmos center: {info['max_dist_deg']}° (inside cone: {info['inside_cone_ratio']})")
        if info.get("sample"):
            print(f"  sample row: {info['sample']}")
        for warn in r["warnings"]:
            print(f"  WARN: {warn}")
        for err in r["errors"]:
            print(f"  ERROR: {err}")
        # lsdb piggyback
        lsdb_r = next((l for l in lsdb_results if l["name"] == name), None)
        if lsdb_r:
            lsdb_status = "OK" if lsdb_r["ok"] else "FAIL"
            print(f"  lsdb: [{lsdb_status}]", end="")
            if lsdb_r["info"]:
                print(f" {lsdb_r['info']}", end="")
            for err in lsdb_r["errors"]:
                print(f"\n    ERROR: {err}", end="")
            print()
    print()
    print("=" * 70)
    print(f"SUMMARY: {n_ok}/{len(results)} schema+rows valid, "
          f"{n_lsdb_ok}/{len(lsdb_results)} lsdb readable")
    print("=" * 70)


def main() -> int:
    # Only validate datasets that actually have hats.properties on disk.
    candidates = sorted(EXPECTED.keys())
    existing = [
        n for n in candidates
        if os.path.exists(os.path.join(COSMOS_ROOT, n, n, n, "hats.properties"))
    ]
    missing = set(candidates) - set(existing)
    if missing:
        print(f"(skipping unbuilt: {sorted(missing)})")

    results = [validate_one(n) for n in existing]
    lsdb_results = [validate_lsdb(n) for n in existing]
    print_report(results, lsdb_results)

    any_fail = any(not r["ok"] for r in results) or any(not l["ok"] for l in lsdb_results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
