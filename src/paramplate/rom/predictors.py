"""Interpolation and Gaussian-process predictors of standardized POD coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, interp1d
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def maximin_subset(values: np.ndarray, n_select: int, *, seed: int = 100) -> np.ndarray:
    """Select a deterministic space-filling subset in Euclidean coordinates."""

    X = _matrix(values, "values")
    n = len(X)
    k = min(max(int(n_select), 1), n)
    if k == n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(int(seed))
    selected = [int(rng.integers(n))]
    distances = np.linalg.norm(X - X[selected[0]], axis=1)
    for _ in range(1, k):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.linalg.norm(X - X[index], axis=1))
    return np.sort(np.asarray(selected, dtype=int))


@dataclass
class PODIRBF:
    smoothing: float = 0.0
    neighbors: int | None = None
    kernel: str = "thin_plate_spline"
    degree: int = 1
    _model: RBFInterpolator | None = None
    _tree: cKDTree | None = None
    _targets: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, coordinates: np.ndarray) -> "PODIRBF":
        X = _matrix(inputs, "inputs")
        Y = _matrix(coordinates, "coordinates")
        if len(X) != len(Y):
            raise ValueError("Input and coordinate row counts differ.")
        neighbors = None if self.neighbors is None else min(int(self.neighbors), len(X))
        self._model = RBFInterpolator(
            X,
            Y,
            smoothing=float(self.smoothing),
            kernel=self.kernel,
            degree=int(self.degree),
            neighbors=neighbors,
        )
        self._tree = cKDTree(X)
        self._targets = Y.copy()
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._model is None or self._tree is None or self._targets is None:
            raise RuntimeError("PODIRBF has not been fitted.")
        X = _matrix(inputs, "inputs")
        try:
            values = np.asarray(self._model(X), dtype=float)
        except Exception:
            values = np.full((len(X), self._targets.shape[1]), np.nan)
        invalid = ~np.all(np.isfinite(values), axis=1)
        if np.any(invalid):
            _, nearest = self._tree.query(X[invalid], k=1)
            values[invalid] = self._targets[np.asarray(nearest, dtype=int)]
        return values

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target


@dataclass
class PODILinear:
    fit_limit: int | None = None
    subset_seed: int = 100
    _interpolator: object | None = None
    _tree: cKDTree | None = None
    _inputs: np.ndarray | None = None
    _targets: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, coordinates: np.ndarray) -> "PODILinear":
        X = _matrix(inputs, "inputs")
        Y = _matrix(coordinates, "coordinates")
        if len(X) != len(Y):
            raise ValueError("Input and coordinate row counts differ.")
        if self.fit_limit is not None and len(X) > int(self.fit_limit):
            chosen = maximin_subset(X, int(self.fit_limit), seed=self.subset_seed)
            X, Y = X[chosen], Y[chosen]
        if X.shape[1] == 1:
            order = np.argsort(X[:, 0])
            self._interpolator = interp1d(
                X[order, 0],
                Y[order],
                axis=0,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
        else:
            self._interpolator = LinearNDInterpolator(X, Y, fill_value=np.nan, rescale=False)
        self._inputs = X.copy()
        self._targets = Y.copy()
        self._tree = cKDTree(X)
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._interpolator is None or self._tree is None or self._targets is None:
            raise RuntimeError("PODILinear has not been fitted.")
        X = _matrix(inputs, "inputs")
        if self._inputs is not None and self._inputs.shape[1] == 1:
            values = np.asarray(self._interpolator(X[:, 0]), dtype=float)
        else:
            values = np.asarray(self._interpolator(X), dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        invalid = ~np.all(np.isfinite(values), axis=1)
        if np.any(invalid):
            _, nearest = self._tree.query(X[invalid], k=1)
            values[invalid] = self._targets[np.asarray(nearest, dtype=int)]
        return values

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target


@dataclass
class PODGPR:
    kernel: Literal["rbf", "matern52"] = "rbf"
    alpha: float = 1.0e-9
    white_noise: float = 1.0e-8
    optimizer_restarts: int = 2
    fit_limit: int | None = None
    subset_seed: int = 100
    normalize_y: bool = False
    models: list[GaussianProcessRegressor] | None = None
    subset_indices: np.ndarray | None = None

    def _kernel(self, n_features: int):
        length = np.ones(n_features, dtype=float)
        if self.kernel == "rbf":
            base = RBF(length_scale=length, length_scale_bounds=(1.0e-2, 1.0e2))
        elif self.kernel == "matern52":
            base = Matern(length_scale=length, length_scale_bounds=(1.0e-2, 1.0e2), nu=2.5)
        else:
            raise ValueError(f"Unknown GP kernel {self.kernel!r}.")
        return ConstantKernel(1.0, (1.0e-3, 1.0e3)) * base + WhiteKernel(
            noise_level=float(self.white_noise),
            noise_level_bounds=(1.0e-12, 1.0e-3),
        )

    def fit(self, inputs: np.ndarray, coordinates: np.ndarray) -> "PODGPR":
        X = _matrix(inputs, "inputs")
        Y = _matrix(coordinates, "coordinates")
        if len(X) != len(Y):
            raise ValueError("Input and coordinate row counts differ.")
        chosen = np.arange(len(X), dtype=int)
        if self.fit_limit is not None and len(X) > int(self.fit_limit):
            chosen = maximin_subset(X, int(self.fit_limit), seed=self.subset_seed)
            X, Y = X[chosen], Y[chosen]
        self.subset_indices = chosen
        self.models = []
        for column in range(Y.shape[1]):
            model = GaussianProcessRegressor(
                kernel=self._kernel(X.shape[1]),
                alpha=float(self.alpha),
                normalize_y=bool(self.normalize_y),
                n_restarts_optimizer=int(self.optimizer_restarts),
                random_state=int(self.subset_seed + column),
            )
            model.fit(X, Y[:, column])
            self.models.append(model)
        return self

    def predict(self, inputs: np.ndarray, *, return_std: bool = False):
        if not self.models:
            raise RuntimeError("PODGPR has not been fitted.")
        X = _matrix(inputs, "inputs")
        means, standard_deviations = [], []
        for model in self.models:
            mean, std = model.predict(X, return_std=True)
            means.append(mean)
            standard_deviations.append(std)
        mean_matrix = np.column_stack(means)
        std_matrix = np.column_stack(standard_deviations)
        return (mean_matrix, std_matrix) if return_std else mean_matrix

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target
