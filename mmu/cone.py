"""Spherical cone-cut helper used by every ``build_parent_sample_hats.py``
script for the MMU v2 HATS port.

Every build script accepts ``--ra-center/--dec-center/--radius`` optional args.
When all three are set, the script applies :func:`apply_cone_filter` to its
master catalog (or per-file bounding box) as early as possible in the pipeline,
so that a 1° test slice runs in seconds to minutes instead of hours.

The default test slice is the COSMOS field: ra=150°, dec=+2°, radius=0.5°
(1° diameter). See ``PORT_STATUS.md`` for progress against that slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConeCut:
    """Immutable cone-cut parameters. ``None`` radius means no cut."""

    ra_center: float | None
    dec_center: float | None
    radius: float | None

    @property
    def is_active(self) -> bool:
        return (
            self.ra_center is not None
            and self.dec_center is not None
            and self.radius is not None
        )

    def __post_init__(self) -> None:
        partial = [
            v is not None
            for v in (self.ra_center, self.dec_center, self.radius)
        ]
        if any(partial) and not all(partial):
            raise ValueError(
                "ConeCut requires all of ra_center, dec_center, radius "
                "to be set, or all to be None."
            )
        if self.radius is not None and self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.radius is not None and self.radius > 180:
            raise ValueError(f"radius must be <= 180 degrees, got {self.radius}")


def apply_cone_filter(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_center: float,
    dec_center: float,
    radius: float,
) -> np.ndarray:
    """Return a boolean mask selecting rows within a spherical cone.

    Uses the haversine great-circle distance so it handles RA wrap-around
    at 0°/360° and polar regions correctly.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination in degrees. Must be broadcastable
        to a common shape.
    ra_center, dec_center : float
        Cone center in degrees.
    radius : float
        Cone radius in degrees (great-circle distance from the center).

    Returns
    -------
    mask : ndarray[bool]
        Boolean mask with the same shape as the broadcast of ``ra``/``dec``.
        ``True`` where the point lies within ``radius`` degrees of the center.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    ra0 = np.deg2rad(ra_center)
    dec0 = np.deg2rad(dec_center)

    d_ra = ra_rad - ra0
    d_dec = dec_rad - dec0
    a = (
        np.sin(d_dec / 2.0) ** 2
        + np.cos(dec_rad) * np.cos(dec0) * np.sin(d_ra / 2.0) ** 2
    )
    # Clamp to avoid floating-point drift above 1.0 causing NaN in arcsin.
    a = np.clip(a, 0.0, 1.0)
    central_angle = 2.0 * np.arcsin(np.sqrt(a))
    return np.rad2deg(central_angle) <= radius


def cone_bounding_box(
    ra_center: float,
    dec_center: float,
    radius: float,
) -> tuple[float, float, float, float]:
    """Return a ``(ra_min, ra_max, dec_min, dec_max)`` bounding box that
    conservatively contains the cone.

    Useful for pre-filtering file lists by header RA/Dec ranges before
    the more expensive per-row cone cut. The RA range widens near the
    poles to account for the cos(dec) shortening of small circles of
    latitude.

    Returns ``ra_min, ra_max`` in [0, 360); note that if the cone wraps
    around the RA=0 meridian, ``ra_min`` may be > ``ra_max`` (indicating
    two separate intervals [ra_min, 360) and [0, ra_max]).
    """
    dec_min = max(dec_center - radius, -90.0)
    dec_max = min(dec_center + radius, 90.0)

    # Cone covers the whole RA range if it touches a pole.
    if dec_max >= 90.0 - 1e-9 or dec_min <= -90.0 + 1e-9:
        return 0.0, 360.0, dec_min, dec_max

    # Widening factor: at declination |d|, a great-circle distance r on the
    # sphere corresponds to Δra = r / cos(d) (for small r). To be safe use
    # the maximum |d| hit by the cone.
    max_abs_dec = max(abs(dec_min), abs(dec_max))
    cos_d = np.cos(np.deg2rad(max_abs_dec))
    # Prevent division by ~zero near poles (already handled above, but belt
    # and suspenders).
    if cos_d < 1e-6:
        return 0.0, 360.0, dec_min, dec_max
    d_ra = radius / cos_d
    # A wide cone near the equator can blow Δra past 180° even though the
    # cone itself doesn't touch a pole. In that case the cone already
    # covers the whole RA circle; return the full range rather than a
    # bogus wraparound interval that would silently drop real data.
    if d_ra >= 180.0:
        return 0.0, 360.0, dec_min, dec_max
    ra_min = (ra_center - d_ra) % 360.0
    ra_max = (ra_center + d_ra) % 360.0
    return ra_min, ra_max, dec_min, dec_max
