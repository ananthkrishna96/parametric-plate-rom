"""ROM preprocessing utilities for snapshot archives.

This module builds POD-ready artifacts from a :class:`ROMDataset`.  It is
deliberately independent of FEniCS, so it can be run on saved snapshot
archives without rebuilding finite-element objects.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .datasets import ROMDataset
from .pod import PODResult, compute_pod, project_snapshots, relative_frobenius_error
from .splits import TrainTestSplit, make_train_test_split


@dataclass(frozen=True)
class ParameterScaler:
    """Simple standardization fitted on training parameters."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        return (A - self.mean) / self.scale

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        A = np.asarray(Z, dtype=np.float64)
        return A * self.scale + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "standard",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def fit_parameter_scaler(parameters: np.ndarray) -> ParameterScaler:
    """Fit a robust standard scaler on parameter rows."""

    P = np.asarray(parameters, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.ndim != 2:
        raise ValueError(f"Expected 2D parameter matrix, got shape {P.shape}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("Parameter matrix contains NaN or infinite values.")

    mean = P.mean(axis=0)
    scale = P.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return ParameterScaler(mean=mean, scale=scale)


@dataclass
class PODPreprocessingResult:
    """Complete POD preprocessing output for one ROM dataset."""

    dataset: ROMDataset
    split: TrainTestSplit
    parameter_scaler: ParameterScaler
    parameters_train_scaled: np.ndarray
    parameters_test_scaled: np.ndarray
    pod_w: PODResult
    coeffs_w_train: np.ndarray
    coeffs_w_test: np.ndarray
    relerr_w_train: float
    relerr_w_test: float
    pod_theta: Optional[PODResult] = None
    coeffs_theta_train: Optional[np.ndarray] = None
    coeffs_theta_test: Optional[np.ndarray] = None
    relerr_theta_train: Optional[float] = None
    relerr_theta_test: Optional[float] = None

    @property
    def case_name(self) -> str:
        return self.dataset.source_path.parent.name


def build_pod_preprocessing(
    dataset: ROMDataset,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
    energy_tol: float = 0.9999,
    max_rank: Optional[int] = None,
    center: bool = True,
    include_theta: bool = True,
) -> PODPreprocessingResult:
    """Build POD preprocessing artifacts in memory."""

    split = make_train_test_split(
        dataset.n_samples,
        train_fraction=train_fraction,
        seed=seed,
        shuffle=True,
    )

    P = np.asarray(dataset.parameters, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    scaler = fit_parameter_scaler(P[split.train_indices])
    P_train_scaled = scaler.transform(P[split.train_indices])
    P_test_scaled = scaler.transform(P[split.test_indices])

    X = np.asarray(dataset.snapshots, dtype=np.float64)
    X_train = X[split.train_indices]
    X_test = X[split.test_indices]

    pod_w = compute_pod(
        X_train,
        energy_tol=energy_tol,
        max_rank=max_rank,
        center=center,
    )
    Cw_train = project_snapshots(X_train, pod_w)
    Cw_test = project_snapshots(X_test, pod_w)
    X_train_rec = Cw_train @ pod_w.basis.T + pod_w.mean
    X_test_rec = Cw_test @ pod_w.basis.T + pod_w.mean
    err_w_train = relative_frobenius_error(X_train, X_train_rec)
    err_w_test = relative_frobenius_error(X_test, X_test_rec)

    pod_theta = None
    Ct_train = Ct_test = None
    err_t_train = err_t_test = None
    if include_theta and dataset.theta_snapshots is not None:
        T = np.asarray(dataset.theta_snapshots, dtype=np.float64)
        T_train = T[split.train_indices]
        T_test = T[split.test_indices]
        pod_theta = compute_pod(
            T_train,
            energy_tol=energy_tol,
            max_rank=max_rank,
            center=center,
        )
        Ct_train = project_snapshots(T_train, pod_theta)
        Ct_test = project_snapshots(T_test, pod_theta)
        T_train_rec = Ct_train @ pod_theta.basis.T + pod_theta.mean
        T_test_rec = Ct_test @ pod_theta.basis.T + pod_theta.mean
        err_t_train = relative_frobenius_error(T_train, T_train_rec)
        err_t_test = relative_frobenius_error(T_test, T_test_rec)

    return PODPreprocessingResult(
        dataset=dataset,
        split=split,
        parameter_scaler=scaler,
        parameters_train_scaled=P_train_scaled,
        parameters_test_scaled=P_test_scaled,
        pod_w=pod_w,
        coeffs_w_train=Cw_train,
        coeffs_w_test=Cw_test,
        relerr_w_train=err_w_train,
        relerr_w_test=err_w_test,
        pod_theta=pod_theta,
        coeffs_theta_train=Ct_train,
        coeffs_theta_test=Ct_test,
        relerr_theta_train=err_t_train,
        relerr_theta_test=err_t_test,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_energy_csv(path: Path, pod: PODResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "singular_value", "energy", "cumulative_energy"],
        )
        writer.writeheader()
        for i, (s, e, ce) in enumerate(zip(pod.singular_values, pod.energy, pod.cumulative_energy), start=1):
            writer.writerow({
                "mode": i,
                "singular_value": float(s),
                "energy": float(e),
                "cumulative_energy": float(ce),
            })


def save_pod_preprocessing(result: PODPreprocessingResult, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Save POD preprocessing artifacts to an external output directory."""

    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Use overwrite=True.")
    out.mkdir(parents=True, exist_ok=True)

    split = result.split
    np.savez_compressed(
        out / "pod_w.npz",
        basis=result.pod_w.basis,
        singular_values=result.pod_w.singular_values,
        energy=result.pod_w.energy,
        cumulative_energy=result.pod_w.cumulative_energy,
        mean=result.pod_w.mean,
        rank=np.array(result.pod_w.rank, dtype=int),
        centered=np.array(result.pod_w.centered, dtype=bool),
        train_coefficients=result.coeffs_w_train,
        test_coefficients=result.coeffs_w_test,
        train_indices=split.train_indices,
        test_indices=split.test_indices,
    )
    _write_energy_csv(out / "energy_w.csv", result.pod_w)

    if result.pod_theta is not None:
        np.savez_compressed(
            out / "pod_theta.npz",
            basis=result.pod_theta.basis,
            singular_values=result.pod_theta.singular_values,
            energy=result.pod_theta.energy,
            cumulative_energy=result.pod_theta.cumulative_energy,
            mean=result.pod_theta.mean,
            rank=np.array(result.pod_theta.rank, dtype=int),
            centered=np.array(result.pod_theta.centered, dtype=bool),
            train_coefficients=result.coeffs_theta_train,
            test_coefficients=result.coeffs_theta_test,
            train_indices=split.train_indices,
            test_indices=split.test_indices,
        )
        _write_energy_csv(out / "energy_theta.csv", result.pod_theta)

    np.savez_compressed(
        out / "parameters_scaled.npz",
        train_scaled=result.parameters_train_scaled,
        test_scaled=result.parameters_test_scaled,
        train_indices=split.train_indices,
        test_indices=split.test_indices,
        raw_parameters=result.dataset.parameters,
    )

    _write_json(out / "parameter_scaler.json", result.parameter_scaler.to_dict())
    _write_json(out / "train_test_split.json", {
        "seed": split.seed,
        "train_fraction": split.train_fraction,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
    })

    summary: dict[str, Any] = {
        "source_archive": str(result.dataset.source_path),
        "project": result.dataset.project,
        "case_name": result.case_name,
        "n_samples": result.dataset.n_samples,
        "parameter_shape": list(result.dataset.parameters.shape),
        "snapshot_shape": list(result.dataset.snapshots.shape),
        "theta_shape": list(result.dataset.theta_snapshots.shape) if result.dataset.theta_snapshots is not None else None,
        "train_fraction": split.train_fraction,
        "seed": split.seed,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "w_rank": result.pod_w.rank,
        "w_retained_energy": result.pod_w.retained_energy,
        "w_relative_error_train": result.relerr_w_train,
        "w_relative_error_test": result.relerr_w_test,
        "theta_rank": result.pod_theta.rank if result.pod_theta is not None else None,
        "theta_retained_energy": result.pod_theta.retained_energy if result.pod_theta is not None else None,
        "theta_relative_error_train": result.relerr_theta_train,
        "theta_relative_error_test": result.relerr_theta_test,
    }
    _write_json(out / "pod_summary.json", summary)
    return out
