"""Small analytical checks for the thermal reduction and coupling protocol."""

from __future__ import annotations

from .thermal_driver import centered_first_moment


def linear_profile_driver(gradient: float, thickness: float, *, quadrature_points: int = 20) -> float:
    """For ``DeltaT(z)=gradient*z``, the centered driver equals ``gradient``."""

    return centered_first_moment(
        lambda z: float(gradient) * z,
        thickness,
        quadrature_points=quadrature_points,
    )
