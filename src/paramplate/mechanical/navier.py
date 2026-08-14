"""Truncated Navier reference for simply supported orthotropic plates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from paramplate.core.parameters import OrthotropicRigidity


@dataclass(frozen=True)
class NavierPlateSolution:
    x: np.ndarray
    y: np.ndarray
    displacement: np.ndarray
    modes_m: np.ndarray
    modes_n: np.ndarray
    coefficients: np.ndarray

    @property
    def maximum_absolute_displacement(self) -> float:
        return float(np.max(np.abs(self.displacement)))


def orthotropic_denominator(
    m: np.ndarray,
    n: np.ndarray,
    *,
    length: float,
    width: float,
    rigidity: OrthotropicRigidity,
    foundation_stiffness: float = 0.0,
) -> np.ndarray:
    alpha = np.asarray(m, dtype=float) * np.pi / float(length)
    beta = np.asarray(n, dtype=float) * np.pi / float(width)
    return (
        rigidity.Dx * alpha**4
        + 2.0 * (rigidity.Dxy + 2.0 * rigidity.Ds) * alpha**2 * beta**2
        + rigidity.Dy * beta**4
        + float(foundation_stiffness)
    )


def uniform_load_coefficient(amplitude: float, m: int, n: int) -> float:
    if m % 2 == 0 or n % 2 == 0:
        return 0.0
    return float(16.0 * amplitude / (np.pi**2 * m * n))


def patch_load_coefficient(
    amplitude: float,
    m: int,
    n: int,
    *,
    length: float,
    width: float,
    x0: float,
    y0: float,
    size_x: float,
    size_y: float,
) -> float:
    ax = m * np.pi / length
    ay = n * np.pi / width
    xa, xb = max(0.0, x0 - size_x / 2.0), min(length, x0 + size_x / 2.0)
    ya, yb = max(0.0, y0 - size_y / 2.0), min(width, y0 + size_y / 2.0)
    ix = (np.cos(ax * xa) - np.cos(ax * xb)) / ax
    iy = (np.cos(ay * ya) - np.cos(ay * yb)) / ay
    return float(4.0 * amplitude * ix * iy / (length * width))


def solve_ssss_navier(
    rigidity: OrthotropicRigidity,
    *,
    length: float = 2.0,
    width: float = 1.0,
    foundation_stiffness: float = 0.0,
    load_amplitude: float = -1000.0,
    load: str = "uniform",
    patch: tuple[float, float, float, float] = (1.0, 0.5, 0.351, 0.351),
    n_terms: int = 31,
    nx: int = 121,
    ny: int = 61,
    load_coefficient: Callable[[int, int], float] | None = None,
) -> NavierPlateSolution:
    """Evaluate a finite double-sine series on a regular midsurface grid."""

    if length <= 0.0 or width <= 0.0 or n_terms < 1:
        raise ValueError("Plate dimensions and n_terms must be positive.")
    x = np.linspace(0.0, length, int(nx))
    y = np.linspace(0.0, width, int(ny))
    modes = np.arange(1, int(n_terms) + 1, dtype=int)
    coefficients = np.zeros((len(modes), len(modes)), dtype=float)
    field = np.zeros((len(y), len(x)), dtype=float)

    for i, m in enumerate(modes):
        sx = np.sin(m * np.pi * x / length)
        for j, n in enumerate(modes):
            if load_coefficient is not None:
                qmn = float(load_coefficient(int(m), int(n)))
            elif load == "uniform":
                qmn = uniform_load_coefficient(load_amplitude, int(m), int(n))
            elif load == "patch":
                qmn = patch_load_coefficient(
                    load_amplitude,
                    int(m),
                    int(n),
                    length=length,
                    width=width,
                    x0=patch[0],
                    y0=patch[1],
                    size_x=patch[2],
                    size_y=patch[3],
                )
            else:
                raise ValueError("load must be uniform or patch unless load_coefficient is supplied.")
            denominator = float(
                orthotropic_denominator(
                    np.asarray([m]),
                    np.asarray([n]),
                    length=length,
                    width=width,
                    rigidity=rigidity,
                    foundation_stiffness=foundation_stiffness,
                )[0]
            )
            coefficients[i, j] = qmn / denominator
            if qmn != 0.0:
                field += coefficients[i, j] * np.outer(
                    np.sin(n * np.pi * y / width),
                    sx,
                )
    return NavierPlateSolution(x, y, field, modes, modes.copy(), coefficients)
