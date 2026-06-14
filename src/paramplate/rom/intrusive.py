"""Intrusive ROM suite helpers.

This module currently provides the robust, data-driven intrusive-adjacent
projection checks that can run from saved archives without importing FEniCS.
The POD-Galerkin entry is deliberately represented as a disabled capability
object until the legacy residual/Jacobian path is wrapped safely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pod import PODResult, project_snapshots, reconstruct_snapshots, relative_frobenius_error


@dataclass(frozen=True)
class LoadedPOD:
    """POD basis loaded from a ``pod_w.npz`` or ``pod_theta.npz`` artifact."""

    basis: np.ndarray
    mean: np.ndarray
    singular_values: np.ndarray
    energy: np.ndarray
    cumulative_energy: np.ndarray
    train_coefficients: np.ndarray
    test_coefficients: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    centered: bool

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def as_pod_result(self) -> PODResult:
        return PODResult(
            basis=self.basis,
            singular_values=self.singular_values,
            energy=self.energy,
            cumulative_energy=self.cumulative_energy,
            mean=self.mean,
            rank=self.rank,
            centered=self.centered,
        )


@dataclass(frozen=True)
class PODProjectedMetrics:
    """Projection/reconstruction metrics for a field."""

    field_name: str
    rank: int
    n_train: int
    n_test: int
    relative_error_train: float
    relative_error_test: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "rank": self.rank,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "relative_error_train": self.relative_error_train,
            "relative_error_test": self.relative_error_test,
        }


@dataclass(frozen=True)
class PODGalerkinCapability:
    """Explicit record that POD-Galerkin is available conceptually but off by default."""

    enabled: bool = False
    reason: str = (
        "POD-Galerkin needs case-specific FEniCS residual/Jacobian assembly from the legacy "
        "solver and is intentionally disabled in the pure-data suite runner."
    )

    def require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(self.reason)


def load_pod_artifact(path: str | Path) -> LoadedPOD:
    """Load a POD artifact written by ``save_pod_preprocessing``."""

    p = Path(path).expanduser().resolve()
    with np.load(p, allow_pickle=True) as z:
        return LoadedPOD(
            basis=np.asarray(z["basis"], dtype=np.float64),
            mean=np.asarray(z["mean"], dtype=np.float64),
            singular_values=np.asarray(z["singular_values"], dtype=np.float64),
            energy=np.asarray(z["energy"], dtype=np.float64),
            cumulative_energy=np.asarray(z["cumulative_energy"], dtype=np.float64),
            train_coefficients=np.asarray(z["train_coefficients"], dtype=np.float64),
            test_coefficients=np.asarray(z["test_coefficients"], dtype=np.float64),
            train_indices=np.asarray(z["train_indices"], dtype=int),
            test_indices=np.asarray(z["test_indices"], dtype=int),
            centered=bool(np.asarray(z["centered"]).item()),
        )


def evaluate_pod_projection(
    snapshots: np.ndarray,
    pod: LoadedPOD,
    *,
    field_name: str,
) -> PODProjectedMetrics:
    """Evaluate projection and reconstruction error using an existing POD artifact."""

    X = np.asarray(snapshots, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D snapshots for {field_name}, got shape {X.shape}.")

    pod_result = pod.as_pod_result()
    X_train = X[pod.train_indices]
    X_test = X[pod.test_indices]

    C_train = project_snapshots(X_train, pod_result)
    C_test = project_snapshots(X_test, pod_result)
    X_train_rec = reconstruct_snapshots(C_train, pod_result)
    X_test_rec = reconstruct_snapshots(C_test, pod_result)

    return PODProjectedMetrics(
        field_name=field_name,
        rank=pod.rank,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        relative_error_train=relative_frobenius_error(X_train, X_train_rec),
        relative_error_test=relative_frobenius_error(X_test, X_test_rec),
    )


def write_intrusive_projection_summary(
    output_dir: str | Path,
    *,
    case_name: str,
    source_archive: str | Path,
    w_metrics: PODProjectedMetrics,
    theta_metrics: PODProjectedMetrics | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a compact intrusive-suite projection summary."""

    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Use overwrite=True.")
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "suite": "intrusive",
        "method": "pod-projected",
        "case_name": case_name,
        "source_archive": str(source_archive),
        "pod_galerkin": PODGalerkinCapability().reason,
        "w": w_metrics.to_dict(),
        "theta": theta_metrics.to_dict() if theta_metrics is not None else None,
    }
    path = out / "intrusive_projection_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
