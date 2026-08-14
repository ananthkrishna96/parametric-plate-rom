"""Reference-field loading, interpolation, and finite-element sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass(frozen=True)
class ReferenceField:
    """Scalar field represented by in-plane sample coordinates."""

    coordinates: np.ndarray
    values: np.ndarray
    name: str
    unit: str = ""

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Reference coordinates must have shape (n_points, 2).")
        if coordinates.shape[0] != values.size:
            raise ValueError("Reference coordinates and values have different row counts.")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(values)):
            raise ValueError("Reference fields must contain finite coordinates and values.")
        if not self.name:
            raise ValueError("ReferenceField.name cannot be empty.")
        object.__setattr__(self, "coordinates", coordinates.copy())
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class RegularGridField:
    """Scalar field on a strictly increasing tensor-product grid."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    name: str
    unit: str = ""

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        values = np.asarray(self.values, dtype=float)
        if x.size < 2 or y.size < 2:
            raise ValueError("Reference grids require at least two points per direction.")
        if values.shape != (y.size, x.size):
            raise ValueError(
                f"Reference field {self.name!r} has shape {values.shape}; "
                f"expected {(y.size, x.size)}."
            )
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("Reference coordinates must be strictly increasing.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Reference coordinates must be finite.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Reference field contains non-finite values.")
        if not self.name:
            raise ValueError("RegularGridField.name cannot be empty.")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "values", values.copy())


def function_dof_coordinates(function: Any) -> np.ndarray:
    """Return in-plane coordinates for a scalar legacy-FEniCS function."""

    space = function.function_space()
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    geometric_dimension = int(space.mesh().geometry().dim())
    coordinates = coordinates.reshape((-1, geometric_dimension))
    if geometric_dimension < 2:
        raise ValueError("Plate output spaces must have at least two geometric coordinates.")
    return coordinates[:, :2].copy()


def sample_function(function: Any, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate a scalar legacy-FEniCS function at in-plane coordinates."""

    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_points, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must be finite.")
    try:
        function.set_allow_extrapolation(False)
    except AttributeError:
        pass
    values: list[float] = []
    for x, y in points:
        try:
            value = function(float(x), float(y))
        except TypeError:
            value = function((float(x), float(y)))
        values.append(float(value))
    sampled = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("The sampled function contains non-finite values.")
    return sampled


def load_regular_grid_field(
    path: str | Path,
    *,
    key: str = "displacement",
    x_key: str = "x",
    y_key: str = "y",
    unit: str = "",
) -> RegularGridField:
    """Load a numeric regular-grid field without enabling pickle support."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        missing = [name for name in (x_key, y_key, key) if name not in data.files]
        if missing:
            raise KeyError(f"Reference archive {source} is missing {missing}.")
        x = np.asarray(data[x_key], dtype=float).reshape(-1)
        y = np.asarray(data[y_key], dtype=float).reshape(-1)
        values = np.asarray(data[key], dtype=float)
    return RegularGridField(x=x, y=y, values=values, name=key, unit=unit)


def load_reference_field(
    path: str | Path,
    *,
    value_key: str,
    unit: str = "",
) -> ReferenceField:
    """Load a scattered or regular-grid reference from a pickle-free NPZ archive.

    Coordinates may be field-specific, generic, or represented by one-dimensional
    ``x`` and ``y`` arrays paired with a rectangular field. Field-specific
    coordinates allow displacement and thermal-driver references to retain their
    distinct output spaces.
    """

    archive = Path(path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with np.load(archive, allow_pickle=False) as data:
        if value_key not in data.files:
            raise KeyError(f"{value_key!r} is missing from {archive}.")
        values = np.asarray(data[value_key], dtype=float)
        coordinate_key = f"{value_key}_coordinates"
        x_key = f"{value_key}_x"
        y_key = f"{value_key}_y"
        if coordinate_key in data.files:
            coordinates = np.asarray(data[coordinate_key], dtype=float)
            flattened = values.reshape(-1)
        elif "coordinates" in data.files:
            coordinates = np.asarray(data["coordinates"], dtype=float)
            flattened = values.reshape(-1)
        elif x_key in data.files and y_key in data.files:
            coordinates, flattened = _flatten_regular_grid(
                np.asarray(data[x_key], dtype=float),
                np.asarray(data[y_key], dtype=float),
                values,
                value_key=value_key,
                x_label=x_key,
                y_label=y_key,
            )
        elif "x" in data.files and "y" in data.files:
            coordinates, flattened = _flatten_regular_grid(
                np.asarray(data["x"], dtype=float),
                np.asarray(data["y"], dtype=float),
                values,
                value_key=value_key,
                x_label="x",
                y_label="y",
            )
        else:
            raise ValueError(
                f"{archive} must contain field-specific coordinates, generic coordinates, "
                "or one-dimensional x and y arrays."
            )
    return ReferenceField(coordinates, flattened, value_key, unit)


def _flatten_regular_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    value_key: str,
    x_label: str,
    y_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    field = np.asarray(values, dtype=float)
    expected = (y_values.size, x_values.size)
    if field.shape != expected:
        raise ValueError(
            f"{value_key!r} must have shape (len({y_label}), len({x_label})); "
            f"found {field.shape}."
        )
    if np.any(np.diff(x_values) <= 0.0) or np.any(np.diff(y_values) <= 0.0):
        raise ValueError("Reference coordinates must be strictly increasing.")
    xx, yy = np.meshgrid(x_values, y_values)
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)]), field.reshape(-1)


def interpolate_reference(field: ReferenceField, target_coordinates: np.ndarray) -> np.ndarray:
    """Interpolate a scattered reference, using nearest values outside its hull."""

    target = np.asarray(target_coordinates, dtype=float)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_coordinates must have shape (n_points, 2).")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_coordinates must be finite.")
    linear = LinearNDInterpolator(field.coordinates, field.values, fill_value=np.nan)
    values = np.asarray(linear(target), dtype=float).reshape(-1)
    missing = ~np.isfinite(values)
    if np.any(missing):
        nearest = NearestNDInterpolator(field.coordinates, field.values)
        values[missing] = np.asarray(nearest(target[missing]), dtype=float).reshape(-1)
    return values
