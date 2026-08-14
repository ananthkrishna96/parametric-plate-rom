"""Metric-aware POD and the transient Euclidean-SVD/reorthonormalization path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import Metric, apply_metric, metric_gram, relative_metric_errors


@dataclass(frozen=True)
class PODBasis:
    modes: np.ndarray
    singular_values: np.ndarray
    eigenvalues: np.ndarray
    energy: np.ndarray
    cumulative_energy: np.ndarray
    mean: np.ndarray
    centered: bool
    metric_name: str
    extraction: str

    @property
    def rank(self) -> int:
        return int(self.modes.shape[1])

    @property
    def n_dofs(self) -> int:
        return int(self.modes.shape[0])

    def project(self, snapshots: np.ndarray, metric: Metric = None) -> np.ndarray:
        X = _snapshot_matrix(snapshots)
        if X.shape[1] != self.n_dofs:
            raise ValueError(f"Snapshot width {X.shape[1]} does not match POD width {self.n_dofs}.")
        return apply_metric(X - self.mean, metric) @ self.modes

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        C = np.asarray(coefficients, dtype=float)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        if C.ndim != 2 or C.shape[1] != self.rank:
            raise ValueError(f"Expected coefficient shape (*, {self.rank}), got {C.shape}.")
        return C @ self.modes.T + self.mean

    def projection_errors(self, snapshots: np.ndarray, metric: Metric = None) -> np.ndarray:
        X = _snapshot_matrix(snapshots)
        return relative_metric_errors(X, self.reconstruct(self.project(X, metric)), metric)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            modes=self.modes,
            singular_values=self.singular_values,
            eigenvalues=self.eigenvalues,
            energy=self.energy,
            cumulative_energy=self.cumulative_energy,
            mean=self.mean,
            centered=np.asarray(self.centered),
            metric_name=np.asarray(self.metric_name),
            extraction=np.asarray(self.extraction),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PODBasis":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                modes=np.asarray(data["modes"], dtype=float),
                singular_values=np.asarray(data["singular_values"], dtype=float),
                eigenvalues=np.asarray(data["eigenvalues"], dtype=float),
                energy=np.asarray(data["energy"], dtype=float),
                cumulative_energy=np.asarray(data["cumulative_energy"], dtype=float),
                mean=np.asarray(data["mean"], dtype=float),
                centered=bool(data["centered"].item()),
                metric_name=str(data["metric_name"].item()),
                extraction=str(data["extraction"].item()),
            )


def _snapshot_matrix(snapshots: np.ndarray) -> np.ndarray:
    X = np.asarray(snapshots, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Expected a nonempty two-dimensional snapshot matrix, got {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Snapshot matrix contains NaN or infinite values.")
    return X


def _energy(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(eigenvalues, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.zeros_like(values), np.zeros_like(values)
    energy = values / total
    cumulative = np.cumsum(energy)
    if cumulative.size:
        cumulative[-1] = 1.0
    return energy, cumulative


def select_rank(
    cumulative_energy: np.ndarray,
    *,
    rank: int | None = None,
    energy_tolerance: float = 0.9999,
    maximum_rank: int | None = None,
) -> int:
    cumulative = np.asarray(cumulative_energy, dtype=float)
    if cumulative.ndim != 1 or cumulative.size == 0:
        raise ValueError("No nonzero POD spectrum is available.")
    if rank is not None:
        selected = int(rank)
    else:
        if not 0.0 < energy_tolerance <= 1.0:
            raise ValueError("energy_tolerance must lie in (0, 1].")
        selected = int(np.searchsorted(cumulative, energy_tolerance, side="left") + 1)
    if maximum_rank is not None:
        selected = min(selected, int(maximum_rank))
    if selected < 1 or selected > cumulative.size:
        raise ValueError(f"Selected rank {selected} is outside [1, {cumulative.size}].")
    return selected


def reorthonormalize(modes: np.ndarray, metric: Metric = None, *, tolerance: float = 1.0e-12) -> np.ndarray:
    """Return a metric-orthonormal basis spanning the supplied columns."""

    B = np.asarray(modes, dtype=float)
    if B.ndim != 2 or B.shape[1] == 0:
        raise ValueError("modes must be a nonempty two-dimensional array.")
    gram = 0.5 * (metric_gram(B, metric) + metric_gram(B, metric).T)
    values, vectors = np.linalg.eigh(gram)
    keep = values > tolerance * max(float(values.max(initial=0.0)), 1.0)
    if not np.any(keep):
        raise ValueError("The supplied modes have zero rank in the declared metric.")
    transform = vectors[:, keep] @ np.diag(1.0 / np.sqrt(values[keep]))
    return B @ transform


def compute_metric_pod(
    snapshots: np.ndarray,
    *,
    metric: Metric = None,
    rank: int | None = None,
    energy_tolerance: float = 0.9999,
    maximum_rank: int | None = None,
    center: bool = False,
    metric_name: str = "Euclidean",
    tolerance: float = 1.0e-12,
) -> PODBasis:
    """Compute an uncentered or centered POD in a declared coefficient metric.

    The method of snapshots is used for non-Euclidean metrics. Only the
    training rows supplied to this function enter the basis.
    """

    X = _snapshot_matrix(snapshots)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean

    if metric is None:
        _, singular, vt = np.linalg.svd(Xc, full_matrices=False)
        eigenvalues = singular**2
        positive = eigenvalues > tolerance * max(float(eigenvalues.max(initial=0.0)), 1.0)
        singular = singular[positive]
        eigenvalues = eigenvalues[positive]
        raw_modes = vt[positive].T
    else:
        correlation = apply_metric(Xc, metric) @ Xc.T
        correlation = 0.5 * (correlation + correlation.T)
        values, vectors = np.linalg.eigh(correlation)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        positive = values > tolerance * max(float(values.max(initial=0.0)), 1.0)
        eigenvalues = values[positive]
        vectors = vectors[:, positive]
        singular = np.sqrt(eigenvalues)
        raw_modes = Xc.T @ (vectors / singular.reshape(1, -1))
        raw_modes = reorthonormalize(raw_modes, metric, tolerance=tolerance)
        # Reorthonormalization can remove numerically dependent columns.
        n = raw_modes.shape[1]
        singular = singular[:n]
        eigenvalues = eigenvalues[:n]

    if eigenvalues.size == 0:
        raise ValueError("The snapshot matrix has zero numerical rank.")
    energy, cumulative = _energy(eigenvalues)
    selected = select_rank(
        cumulative,
        rank=rank,
        energy_tolerance=energy_tolerance,
        maximum_rank=maximum_rank,
    )
    modes = raw_modes[:, :selected].copy()
    if metric is not None:
        modes = reorthonormalize(modes, metric, tolerance=tolerance)
    return PODBasis(
        modes=modes,
        singular_values=singular.copy(),
        eigenvalues=eigenvalues.copy(),
        energy=energy,
        cumulative_energy=cumulative,
        mean=mean,
        centered=bool(center),
        metric_name=metric_name,
        extraction="metric method of snapshots" if metric is not None else "Euclidean SVD",
    )


def compute_transient_pod(
    training_snapshots: np.ndarray,
    *,
    finite_element_metric: Metric,
    rank: int,
    metric_name: str = "displacement H1-type",
    center: bool = False,
) -> PODBasis:
    """Apply the transient thesis path: Euclidean SVD then FE reorthonormalization."""

    X = _snapshot_matrix(training_snapshots)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean
    _, singular, vt = np.linalg.svd(Xc, full_matrices=False)
    eigenvalues = singular**2
    energy, cumulative = _energy(eigenvalues)
    selected = select_rank(cumulative, rank=rank)
    modes = reorthonormalize(vt[:selected].T, finite_element_metric)
    if modes.shape[1] != selected:
        raise ValueError("FE reorthonormalization reduced the requested transient POD rank.")
    return PODBasis(
        modes=modes,
        singular_values=singular,
        eigenvalues=eigenvalues,
        energy=energy,
        cumulative_energy=cumulative,
        mean=mean,
        centered=bool(center),
        metric_name=metric_name,
        extraction="training-only Euclidean SVD followed by FE metric reorthonormalization",
    )


def orthogonality_error(basis: PODBasis, metric: Metric = None) -> float:
    identity = np.eye(basis.rank)
    return float(np.linalg.norm(metric_gram(basis.modes, metric) - identity, ord=np.inf))
