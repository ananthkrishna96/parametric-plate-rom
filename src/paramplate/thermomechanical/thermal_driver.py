"""Centered through-thickness thermal-driver reduction."""

from __future__ import annotations

from typing import Callable

import numpy as np


def centered_first_moment(
    temperature_increment: Callable[[float], float],
    thickness: float,
    *,
    quadrature_points: int = 20,
) -> float:
    r"""Return ``theta = 12/h^3 int_{-h/2}^{h/2} z DeltaT(z) dz``.

    ``theta`` is a temperature-gradient-like plate driver with units K/m; it is
    not a temperature.
    """

    h = float(thickness)
    if h <= 0.0:
        raise ValueError("thickness must be positive.")
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least two.")
    xi, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    z = 0.5 * h * xi
    values = np.asarray([temperature_increment(float(point)) for point in z], dtype=float)
    integral = 0.5 * h * float(np.dot(weights, z * values))
    return float(12.0 * integral / h**3)


def centered_first_moment_samples(
    z: np.ndarray,
    temperature_increment: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Integrate sampled through-thickness profiles along the last axis."""

    coordinates = np.asarray(z, dtype=float).reshape(-1)
    values = np.asarray(temperature_increment, dtype=float)
    if values.shape[-1] != coordinates.size:
        raise ValueError("The last temperature axis must match z.")
    h = float(thickness)
    if h <= 0.0:
        raise ValueError("thickness must be positive.")
    if coordinates[0] < -0.5 * h - 1e-12 or coordinates[-1] > 0.5 * h + 1e-12:
        raise ValueError("z coordinates must lie in the centered plate thickness.")
    return 12.0 * np.trapezoid(values * coordinates, coordinates, axis=-1) / h**3


def thermal_bending_moments(
    theta: np.ndarray | float,
    *,
    Dx: float,
    Dy: float,
    Dxy: float,
    alpha_x: float,
    alpha_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned orthotropic thermal bending moments ``(Mx^T, My^T)``."""

    driver = np.asarray(theta, dtype=float)
    mx = (float(Dx) * float(alpha_x) + float(Dxy) * float(alpha_y)) * driver
    my = (float(Dxy) * float(alpha_x) + float(Dy) * float(alpha_y)) * driver
    return mx, my
