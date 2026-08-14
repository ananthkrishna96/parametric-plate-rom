"""Sampling helpers for comparing finite-element fields on regular grids."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_scalar_field(
    field: Callable[..., float],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a scalar callable on the tensor-product grid ``y x x``.

    DOLFIN scalar functions accept positional coordinates in the supported
    legacy environment. Plain Python callables are accepted as well, which
    keeps the grid-contract test independent of FEniCS.
    """

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("A comparison grid requires at least two points per direction.")
    if np.any(np.diff(x_values) <= 0.0) or np.any(np.diff(y_values) <= 0.0):
        raise ValueError("Grid coordinates must be strictly increasing.")
    sampled = np.empty((y_values.size, x_values.size), dtype=float)
    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            try:
                value = field(float(x_value), float(y_value))
            except TypeError:
                value = field((float(x_value), float(y_value)))
            sampled[row, column] = float(value)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("The sampled field contains non-finite values.")
    return sampled
