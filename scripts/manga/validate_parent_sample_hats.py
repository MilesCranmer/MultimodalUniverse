"""Validate direct MaNGA HATS output against the existing MMU v1 HDF5 view.

This script treats the v1 HDF5 sample as the semantic reference and adapts the
native-shape HATS rows back onto the old padded ``96x96`` canvas so the two can
be compared exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyarrow.dataset as ds


V1_IMAGE_SIZE = 96
SPAXEL_MASK_FILL_VALUE = 1024
MAP_MASK_FILL_VALUE = 1073741824.0


def _center_pad_2d(array: np.ndarray, fill_value=0):
    ny, nx = array.shape
    pad_y = (V1_IMAGE_SIZE - ny) // 2
    pad_x = (V1_IMAGE_SIZE - nx) // 2
    out = np.full((V1_IMAGE_SIZE, V1_IMAGE_SIZE), fill_value, dtype=array.dtype)
    out[pad_y : pad_y + ny, pad_x : pad_x + nx] = array
    return out


def _center_pad_3d_last(array: np.ndarray, fill_value=0):
    ny, nx, nz = array.shape
    pad_y = (V1_IMAGE_SIZE - ny) // 2
    pad_x = (V1_IMAGE_SIZE - nx) // 2
    out = np.full((V1_IMAGE_SIZE, V1_IMAGE_SIZE, nz), fill_value, dtype=array.dtype)
    out[pad_y : pad_y + ny, pad_x : pad_x + nx, :] = array
    return out


def _center_pad_3d_first(array: np.ndarray, fill_value=0):
    n0, ny, nx = array.shape
    pad_y = (V1_IMAGE_SIZE - ny) // 2
    pad_x = (V1_IMAGE_SIZE - nx) // 2
    out = np.full((n0, V1_IMAGE_SIZE, V1_IMAGE_SIZE), fill_value, dtype=array.dtype)
    out[:, pad_y : pad_y + ny, pad_x : pad_x + nx] = array
    return out


def _adapt_hats_row_to_v1(row: dict) -> dict:
    ny = int(row["spatial_shape_y"])
    nx = int(row["spatial_shape_x"])

    flux = np.asarray(row["spaxels"]["flux"], dtype=np.float32)
    ivar = np.asarray(row["spaxels"]["ivar"], dtype=np.float32)
    mask = np.asarray(row["spaxels"]["mask"], dtype=np.int64)
    lsf = np.asarray(row["spaxels"]["lsf"], dtype=np.float32)
    wave = np.asarray(row["spaxels"]["lambda"], dtype=np.float32)

    flux = _center_pad_3d_last(flux, 0.0).reshape(V1_IMAGE_SIZE * V1_IMAGE_SIZE, flux.shape[-1])
    ivar = _center_pad_3d_last(ivar, 0.0).reshape(V1_IMAGE_SIZE * V1_IMAGE_SIZE, ivar.shape[-1])
    mask = _center_pad_3d_last(mask, SPAXEL_MASK_FILL_VALUE).reshape(
        V1_IMAGE_SIZE * V1_IMAGE_SIZE,
        mask.shape[-1],
    )
    lsf = _center_pad_3d_last(lsf, 0.0).reshape(V1_IMAGE_SIZE * V1_IMAGE_SIZE, lsf.shape[-1])
    wave = np.repeat(wave[np.newaxis, :], V1_IMAGE_SIZE * V1_IMAGE_SIZE, axis=0)

    yy, xx = np.indices((V1_IMAGE_SIZE, V1_IMAGE_SIZE))
    spaxels = {
        "flux": flux,
        "ivar": ivar,
        "mask": mask,
        "lsf_sigma": lsf,
        "lambda": wave,
        "x": xx.reshape(-1).astype(np.int8),
        "y": yy.reshape(-1).astype(np.int8),
        "spaxel_idx": np.arange(V1_IMAGE_SIZE * V1_IMAGE_SIZE, dtype=np.int16),
        "flux_units": np.array(
            [row["spaxels"]["flux_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "lambda_units": np.array(
            [row["spaxels"]["lambda_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "skycoo_x": _center_pad_2d(np.asarray(row["spaxels"]["skycoo_x"], dtype=np.float32), 0.0).reshape(-1),
        "skycoo_y": _center_pad_2d(np.asarray(row["spaxels"]["skycoo_y"], dtype=np.float32), 0.0).reshape(-1),
        "ellcoo_r": _center_pad_2d(np.asarray(row["spaxels"]["ellcoo_r"], dtype=np.float32), 0.0).reshape(-1),
        "ellcoo_rre": _center_pad_2d(np.asarray(row["spaxels"]["ellcoo_rre"], dtype=np.float32), 0.0).reshape(-1),
        "ellcoo_rkpc": _center_pad_2d(np.asarray(row["spaxels"]["ellcoo_rkpc"], dtype=np.float32), 0.0).reshape(-1),
        "ellcoo_theta": _center_pad_2d(np.asarray(row["spaxels"]["ellcoo_theta"], dtype=np.float32), 0.0).reshape(-1),
        "skycoo_units": np.array(
            [row["spaxels"]["skycoo_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "ellcoo_r_units": np.array(
            [row["spaxels"]["ellcoo_r_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "ellcoo_rre_units": np.array(
            [row["spaxels"]["ellcoo_rre_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "ellcoo_rkpc_units": np.array(
            [row["spaxels"]["ellcoo_rkpc_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
        "ellcoo_theta_units": np.array(
            [row["spaxels"]["ellcoo_theta_units"]] * (V1_IMAGE_SIZE * V1_IMAGE_SIZE),
            dtype=object,
        ),
    }

    images_flux = _center_pad_3d_first(np.asarray(row["images"]["flux"], dtype=np.float32), 0.0)
    images_psf = _center_pad_3d_first(np.asarray(row["images"]["psf"], dtype=np.float32), 0.0)
    images = {
        "image_band": np.asarray(row["images"]["filter"], dtype=object),
        "image_array": images_flux,
        "image_array_units": np.asarray(row["images"]["flux_units"], dtype=object),
        "image_psf": images_psf,
        "image_psf_units": np.asarray(row["images"]["psf_units"], dtype=object),
        "image_scale": np.asarray(row["images"]["scale"], dtype=np.float32),
        "image_scale_units": np.asarray(row["images"]["scale_units"], dtype=object),
    }

    maps = {
        "group": np.asarray(row["maps"]["group"], dtype=object),
        "label": np.asarray(row["maps"]["label"], dtype=object),
        "array": _center_pad_3d_first(np.asarray(row["maps"]["flux"], dtype=np.float32), 0.0),
        "ivar": _center_pad_3d_first(np.asarray(row["maps"]["ivar"], dtype=np.float32), 0.0),
        "mask": _center_pad_3d_first(
            np.asarray(row["maps"]["mask"], dtype=np.float32),
            MAP_MASK_FILL_VALUE,
        ),
        "array_units": np.asarray(row["maps"]["array_units"], dtype=object),
    }

    return {
        "object_id": row["object_id"],
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "healpix": int(row["healpix"]),
        "z": float(row["z"]),
        "spaxel_size": float(row["spaxel_size"]),
        "spaxel_size_unit": row["spaxel_size_units"],
        "spaxels": spaxels,
        "images": images,
        "maps": maps,
    }


def _load_v1_rows(v1_root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(v1_root.rglob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            for object_id in handle.keys():
                group = handle[object_id]
                rows[object_id] = {
                    "object_id": group["object_id"].asstr()[()],
                    "ra": float(group["ra"][()]),
                    "dec": float(group["dec"][()]),
                    "healpix": int(group["healpix"][()]),
                    "z": float(group["z"][()]),
                    "spaxel_size": float(group["spaxel_size"][()]),
                    "spaxel_size_unit": group["spaxel_size_unit"].asstr()[()],
                    "spaxels": group["spaxels"][()],
                    "images": group["images"][()],
                    "maps": group["maps"][()],
                }
    return rows


def _load_hats_rows(hats_root: Path) -> dict[str, dict]:
    dataset = ds.dataset(str(hats_root / "dataset"), format="parquet")
    table = dataset.to_table()
    rows = {}
    for row in table.to_pylist():
        rows[row["object_id"]] = row
    return rows


def _assert_allclose(name: str, lhs: np.ndarray, rhs: np.ndarray, atol=1e-6):
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name}: shape mismatch {lhs.shape} != {rhs.shape}")
    if not np.allclose(lhs, rhs, atol=atol, equal_nan=True):
        diff = np.max(np.abs(lhs - rhs))
        raise AssertionError(f"{name}: max abs diff {diff}")


def validate(v1_root: Path, hats_root: Path, object_ids: list[str] | None = None) -> None:
    v1_rows = _load_v1_rows(v1_root)
    hats_rows = _load_hats_rows(hats_root)

    ids = sorted(object_ids or v1_rows.keys())
    if set(ids) != set(hats_rows):
        missing = sorted(set(ids) - set(hats_rows))
        extra = sorted(set(hats_rows) - set(ids))
        raise AssertionError(f"object_id mismatch: missing={missing} extra={extra}")

    for object_id in ids:
        expected = v1_rows[object_id]
        actual = _adapt_hats_row_to_v1(hats_rows[object_id])

        assert expected["object_id"] == actual["object_id"]
        assert expected["healpix"] == actual["healpix"]
        _assert_allclose(f"{object_id} ra", np.array(expected["ra"]), np.array(actual["ra"]), atol=1e-9)
        _assert_allclose(f"{object_id} dec", np.array(expected["dec"]), np.array(actual["dec"]), atol=1e-9)
        _assert_allclose(f"{object_id} z", np.array(expected["z"]), np.array(actual["z"]), atol=1e-9)
        _assert_allclose(
            f"{object_id} spaxel_size",
            np.array(expected["spaxel_size"]),
            np.array(actual["spaxel_size"]),
            atol=1e-9,
        )
        assert expected["spaxel_size_unit"] == actual["spaxel_size_unit"]

        for field in [
            "flux",
            "ivar",
            "mask",
            "lsf_sigma",
            "lambda",
            "x",
            "y",
            "spaxel_idx",
            "skycoo_x",
            "skycoo_y",
            "ellcoo_r",
            "ellcoo_rre",
            "ellcoo_rkpc",
            "ellcoo_theta",
        ]:
            _assert_allclose(
                f"{object_id} spaxels.{field}",
                np.asarray(expected["spaxels"][field]),
                np.asarray(actual["spaxels"][field]),
            )

        for field in [
            "flux_units",
            "lambda_units",
            "skycoo_units",
            "ellcoo_r_units",
            "ellcoo_rre_units",
            "ellcoo_rkpc_units",
            "ellcoo_theta_units",
        ]:
            if list(expected["spaxels"][field].astype(str)) != list(actual["spaxels"][field].astype(str)):
                raise AssertionError(f"{object_id} spaxels.{field} mismatch")

        for field in [
            "image_band",
            "image_array_units",
            "image_psf_units",
            "image_scale_units",
        ]:
            if list(expected["images"][field].astype(str)) != list(actual["images"][field].astype(str)):
                raise AssertionError(f"{object_id} images.{field} mismatch")
        for field in ["image_array", "image_psf", "image_scale"]:
            _assert_allclose(
                f"{object_id} images.{field}",
                np.asarray(expected["images"][field]),
                np.asarray(actual["images"][field]),
            )

        for field in ["group", "label", "array_units"]:
            if list(expected["maps"][field].astype(str)) != list(actual["maps"][field].astype(str)):
                raise AssertionError(f"{object_id} maps.{field} mismatch")
        for field in ["array", "ivar", "mask"]:
            _assert_allclose(
                f"{object_id} maps.{field}",
                np.asarray(expected["maps"][field]),
                np.asarray(actual["maps"][field]),
            )

        print(f"{object_id}: parity OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, required=True, help="Root containing v1 MaNGA HDF5 files.")
    parser.add_argument("--hats-root", type=Path, required=True, help="Inner HATS catalog dir containing dataset/*.parquet.")
    parser.add_argument("--object-id", action="append", default=[], help="Restrict validation to specific object IDs.")
    args = parser.parse_args(argv)

    validate(args.v1_root, args.hats_root, object_ids=args.object_id or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
