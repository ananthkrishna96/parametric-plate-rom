"""Non-intrusive ROM suite for POD coefficient prediction.

All models in this module operate on scaled parameter matrices and POD
coefficient matrices.  They are independent of FEniCS and can be trained from
saved ``pod_bases/<case>/`` artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor

from .intrusive import LoadedPOD
from .pod import reconstruct_snapshots, relative_frobenius_error


@dataclass(frozen=True)
class NonIntrusiveMetrics:
    """Prediction metrics for one non-intrusive field surrogate."""

    field_name: str
    method: str
    rank: int
    coefficient_relative_error_train: float
    coefficient_relative_error_test: float
    field_relative_error_train: float
    field_relative_error_test: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "method": self.method,
            "rank": self.rank,
            "coefficient_relative_error_train": self.coefficient_relative_error_train,
            "coefficient_relative_error_test": self.coefficient_relative_error_test,
            "field_relative_error_train": self.field_relative_error_train,
            "field_relative_error_test": self.field_relative_error_test,
        }


def _as_2d(A: np.ndarray) -> np.ndarray:
    X = np.asarray(A, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Array contains NaN or infinite values.")
    return X


def _relative_matrix_error(reference: np.ndarray, approximation: np.ndarray, eps: float = 1e-14) -> float:
    A = _as_2d(reference)
    B = _as_2d(approximation)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}.")
    denom = max(float(np.linalg.norm(A, ord="fro")), float(eps))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


class CoefficientRegressor:
    """Base interface for non-intrusive coefficient regressors."""

    method_name = "base"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CoefficientRegressor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RBFInterpolatorRegressor(CoefficientRegressor):
    """RBF interpolation for POD coefficients."""

    method_name = "podi-rbf"

    def __init__(self, *, kernel: str = "thin_plate_spline", smoothing: float = 0.0, neighbors: int | None = None):
        self.kernel = kernel
        self.smoothing = float(smoothing)
        self.neighbors = neighbors
        self.model: RBFInterpolator | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RBFInterpolatorRegressor":
        self.model = RBFInterpolator(
            _as_2d(X),
            _as_2d(Y),
            kernel=self.kernel,
            smoothing=self.smoothing,
            neighbors=self.neighbors,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RBFInterpolatorRegressor has not been fitted.")
        return _as_2d(self.model(_as_2d(X)))


class LinearNearestRegressor(CoefficientRegressor):
    """Linear interpolation with nearest fallback outside the convex hull."""

    method_name = "podi-linear"

    def __init__(self):
        self.linear = None
        self.nearest = None
        self._x1 = None
        self._y1 = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearNearestRegressor":
        X2 = _as_2d(X)
        Y2 = _as_2d(Y)
        if X2.shape[1] == 1:
            order = np.argsort(X2[:, 0])
            self._x1 = X2[order, 0]
            self._y1 = Y2[order]
        else:
            self.linear = LinearNDInterpolator(X2, Y2)
            self.nearest = NearestNDInterpolator(X2, Y2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2 = _as_2d(X)
        if self._x1 is not None and self._y1 is not None:
            cols = [np.interp(X2[:, 0], self._x1, self._y1[:, j]) for j in range(self._y1.shape[1])]
            return np.vstack(cols).T
        if self.linear is None or self.nearest is None:
            raise RuntimeError("LinearNearestRegressor has not been fitted.")
        Y = np.asarray(self.linear(X2), dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        missing = ~np.all(np.isfinite(Y), axis=1)
        if np.any(missing):
            Y[missing] = np.asarray(self.nearest(X2[missing]), dtype=np.float64)
        return _as_2d(Y)


class NearestRegressor(CoefficientRegressor):
    """Nearest-neighbour interpolation for POD coefficients."""

    method_name = "nearest"

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NearestRegressor":
        self.model = NearestNDInterpolator(_as_2d(X), _as_2d(Y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NearestRegressor has not been fitted.")
        return _as_2d(np.asarray(self.model(_as_2d(X)), dtype=np.float64))


class GPRCoefficientRegressor(CoefficientRegressor):
    """One Gaussian-process regressor per POD coefficient."""

    method_name = "pod-gpr"

    def __init__(self, *, kernel: str = "rbf", alpha: float = 1e-10, normalize_y: bool = True):
        self.kernel = kernel
        self.alpha = float(alpha)
        self.normalize_y = bool(normalize_y)
        self.models: list[GaussianProcessRegressor] = []

    def _kernel(self):
        if self.kernel == "matern":
            return ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-8)
        return ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-8)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GPRCoefficientRegressor":
        X2 = _as_2d(X)
        Y2 = _as_2d(Y)
        self.models = []
        for j in range(Y2.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=self._kernel(),
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                optimizer=None,
            )
            gp.fit(X2, Y2[:, j])
            self.models.append(gp)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("GPRCoefficientRegressor has not been fitted.")
        X2 = _as_2d(X)
        return np.column_stack([m.predict(X2) for m in self.models])


class NNRegressor(CoefficientRegressor):
    """Feed-forward neural-network regressor using scikit-learn MLPRegressor."""

    method_name = "pod-nn"

    def __init__(self, *, hidden_layer_sizes: tuple[int, ...] = (64, 64), seed: int = 42, max_iter: int = 2000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="tanh",
            solver="adam",
            alpha=1e-6,
            random_state=self.seed,
            max_iter=self.max_iter,
            early_stopping=False,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NNRegressor":
        self.model.fit(_as_2d(X), _as_2d(Y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _as_2d(np.asarray(self.model.predict(_as_2d(X)), dtype=np.float64))


class PODAELatentRegressor(CoefficientRegressor):
    """Initial POD-AE-style latent baseline.

    This is a lightweight linear-bottleneck autoencoding baseline in POD
    coefficient space: PCA encodes coefficients to a low-dimensional latent
    variable and an MLP maps parameters to that latent variable.  A nonlinear
    dense autoencoder can later replace this backend behind the same API.
    """

    method_name = "pod-ae"

    def __init__(self, *, latent_dim: int | None = None, seed: int = 42, max_iter: int = 2000):
        self.latent_dim = latent_dim
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.pca: PCA | None = None
        self.regressor: MLPRegressor | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PODAELatentRegressor":
        X2 = _as_2d(X)
        Y2 = _as_2d(Y)
        dim = int(self.latent_dim or min(max(1, Y2.shape[1] // 2), Y2.shape[1]))
        dim = max(1, min(dim, Y2.shape[1], Y2.shape[0]))
        self.pca = PCA(n_components=dim, random_state=self.seed)
        Z = self.pca.fit_transform(Y2)
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="tanh",
            solver="adam",
            alpha=1e-6,
            random_state=self.seed,
            max_iter=self.max_iter,
            early_stopping=False,
        )
        self.regressor.fit(X2, Z)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None or self.regressor is None:
            raise RuntimeError("PODAELatentRegressor has not been fitted.")
        Z = _as_2d(np.asarray(self.regressor.predict(_as_2d(X)), dtype=np.float64))
        return _as_2d(self.pca.inverse_transform(Z))


def make_coefficient_regressor(method: str, **kwargs: Any) -> CoefficientRegressor:
    """Create a coefficient regressor by method name."""

    key = method.lower().strip().replace("_", "-")
    if key == "podi-rbf":
        return RBFInterpolatorRegressor(
            kernel=str(kwargs.get("kernel", "thin_plate_spline")),
            smoothing=float(kwargs.get("smoothing", 0.0)),
            neighbors=kwargs.get("neighbors"),
        )
    if key == "podi-linear":
        return LinearNearestRegressor()
    if key in {"nearest", "podi-nearest"}:
        return NearestRegressor()
    if key == "pod-gpr":
        return GPRCoefficientRegressor(kernel=str(kwargs.get("kernel", "rbf")))
    if key == "pod-nn":
        return NNRegressor(seed=int(kwargs.get("seed", 42)), max_iter=int(kwargs.get("max_iter", 2000)))
    if key == "pod-ae":
        latent = kwargs.get("latent_dim")
        return PODAELatentRegressor(
            latent_dim=None if latent is None else int(latent),
            seed=int(kwargs.get("seed", 42)),
            max_iter=int(kwargs.get("max_iter", 2000)),
        )
    raise ValueError(f"Unknown non-intrusive method {method!r}.")


def evaluate_coefficient_regressor(
    *,
    method: str,
    field_name: str,
    regressor: CoefficientRegressor,
    parameters_train: np.ndarray,
    parameters_test: np.ndarray,
    coefficients_train: np.ndarray,
    coefficients_test: np.ndarray,
    pod: LoadedPOD,
    snapshots_train: np.ndarray,
    snapshots_test: np.ndarray,
) -> tuple[NonIntrusiveMetrics, np.ndarray, np.ndarray]:
    """Fit/evaluate a coefficient regressor and return metrics + predictions."""

    Ctr = _as_2d(coefficients_train)
    Cte = _as_2d(coefficients_test)
    regressor.fit(parameters_train, Ctr)
    Ctr_pred = _as_2d(regressor.predict(parameters_train))
    Cte_pred = _as_2d(regressor.predict(parameters_test))

    pod_result = pod.as_pod_result()
    Xtr_pred = reconstruct_snapshots(Ctr_pred, pod_result)
    Xte_pred = reconstruct_snapshots(Cte_pred, pod_result)

    metrics = NonIntrusiveMetrics(
        field_name=field_name,
        method=method,
        rank=pod.rank,
        coefficient_relative_error_train=_relative_matrix_error(Ctr, Ctr_pred),
        coefficient_relative_error_test=_relative_matrix_error(Cte, Cte_pred),
        field_relative_error_train=relative_frobenius_error(snapshots_train, Xtr_pred),
        field_relative_error_test=relative_frobenius_error(snapshots_test, Xte_pred),
    )
    return metrics, Ctr_pred, Cte_pred


def write_nonintrusive_result(
    output_dir: str | Path,
    *,
    case_name: str,
    method: str,
    source_archive: str | Path,
    w_metrics: NonIntrusiveMetrics,
    w_predictions: tuple[np.ndarray, np.ndarray],
    w_model: CoefficientRegressor,
    theta_metrics: NonIntrusiveMetrics | None = None,
    theta_predictions: tuple[np.ndarray, np.ndarray] | None = None,
    theta_model: CoefficientRegressor | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a non-intrusive method result to disk."""

    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Use overwrite=True.")
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "suite": "nonintrusive",
        "method": method,
        "case_name": case_name,
        "source_archive": str(source_archive),
        "w": w_metrics.to_dict(),
        "theta": theta_metrics.to_dict() if theta_metrics is not None else None,
    }
    (out / "nonintrusive_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        out / "w_predictions.npz",
        train_coefficients_pred=w_predictions[0],
        test_coefficients_pred=w_predictions[1],
    )
    joblib.dump(w_model, out / "w_model.joblib")

    if theta_metrics is not None and theta_predictions is not None and theta_model is not None:
        np.savez_compressed(
            out / "theta_predictions.npz",
            train_coefficients_pred=theta_predictions[0],
            test_coefficients_pred=theta_predictions[1],
        )
        joblib.dump(theta_model, out / "theta_model.joblib")

    return out / "nonintrusive_summary.json"
