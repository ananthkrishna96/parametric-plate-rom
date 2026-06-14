"""POD/SVD utilities for ROM preprocessing.

The functions in this module are intentionally NumPy-only.  They do not
import FEniCS and can therefore be tested quickly on any machine.
Snapshot matrices are assumed to have shape ``(n_samples, n_dofs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PODResult:
    """Result of a snapshot-matrix POD computation."""

    basis: np.ndarray
    singular_values: np.ndarray
    energy: np.ndarray
    cumulative_energy: np.ndarray
    mean: np.ndarray
    rank: int
    centered: bool

    @property
    def retained_energy(self) -> float:
        if self.cumulative_energy.size == 0 or self.rank <= 0:
            return 0.0
        return float(self.cumulative_energy[self.rank - 1])


def as_snapshot_matrix(snapshots: np.ndarray) -> np.ndarray:
    """Return snapshots as a 2D ``float64`` matrix.

    Parameters
    ----------
    snapshots:
        Array with first dimension interpreted as sample index.
    """

    X = np.asarray(snapshots, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D snapshot matrix, got shape {X.shape}.")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"Snapshot matrix cannot be empty, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Snapshot matrix contains NaN or infinite values.")
    return X


def cumulative_energy_from_singular_values(singular_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-mode and cumulative POD energy from singular values."""

    s = np.asarray(singular_values, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("singular_values must be a one-dimensional array.")
    if s.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    squared = s**2
    total = float(np.sum(squared))
    if total <= 0.0:
        energy = np.zeros_like(squared)
        cumulative = np.zeros_like(squared)
    else:
        energy = squared / total
        cumulative = np.cumsum(energy)
        # Numerical guard: make the final value exactly one for readable summaries.
        cumulative[-1] = min(1.0, max(float(cumulative[-1]), 0.0))
    return energy, cumulative


def select_rank(
    cumulative_energy: np.ndarray,
    *,
    energy_tol: float = 0.9999,
    max_rank: Optional[int] = None,
    min_rank: int = 1,
) -> int:
    """Select POD rank from cumulative energy."""

    ce = np.asarray(cumulative_energy, dtype=np.float64)
    if ce.ndim != 1:
        raise ValueError("cumulative_energy must be one-dimensional.")
    if ce.size == 0:
        return 0
    if not (0.0 < float(energy_tol) <= 1.0):
        raise ValueError("energy_tol must be in (0, 1].")
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative.")

    r = int(np.searchsorted(ce, float(energy_tol), side="left") + 1)
    r = max(int(min_rank), r)
    if max_rank is not None:
        if int(max_rank) <= 0:
            raise ValueError("max_rank must be positive when provided.")
        r = min(r, int(max_rank))
    return min(r, int(ce.size))


def compute_pod(
    snapshots: np.ndarray,
    *,
    energy_tol: float = 0.9999,
    max_rank: Optional[int] = None,
    center: bool = True,
    min_rank: int = 1,
) -> PODResult:
    """Compute a POD basis using economy SVD.

    Parameters
    ----------
    snapshots:
        Snapshot matrix with shape ``(n_samples, n_dofs)``.
    energy_tol:
        Cumulative energy threshold used for rank selection.
    max_rank:
        Optional cap on retained rank.
    center:
        If true, subtract the sample mean before SVD.
    min_rank:
        Minimum selected rank, unless the matrix has zero available rank.
    """

    X = as_snapshot_matrix(snapshots)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean if center else X.copy()

    # Economy SVD: Xc = U diag(s) Vt.  POD basis vectors are columns of V.
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    energy, cumulative = cumulative_energy_from_singular_values(s)
    rank = select_rank(cumulative, energy_tol=energy_tol, max_rank=max_rank, min_rank=min_rank)

    basis = vt[:rank].T.copy()
    return PODResult(
        basis=basis,
        singular_values=s.copy(),
        energy=energy,
        cumulative_energy=cumulative,
        mean=mean.astype(np.float64, copy=False),
        rank=rank,
        centered=bool(center),
    )


def project_snapshots(snapshots: np.ndarray, pod: PODResult) -> np.ndarray:
    """Project snapshots onto a POD basis, returning coefficient rows."""

    X = as_snapshot_matrix(snapshots)
    if X.shape[1] != pod.basis.shape[0]:
        raise ValueError(
            f"Snapshot DOFs {X.shape[1]} do not match basis DOFs {pod.basis.shape[0]}."
        )
    return (X - pod.mean) @ pod.basis


def reconstruct_snapshots(coefficients: np.ndarray, pod: PODResult) -> np.ndarray:
    """Reconstruct snapshots from POD coefficients."""

    C = np.asarray(coefficients, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError(f"Expected coefficient matrix, got shape {C.shape}.")
    if C.shape[1] != pod.basis.shape[1]:
        raise ValueError(
            f"Coefficient rank {C.shape[1]} does not match basis rank {pod.basis.shape[1]}."
        )
    return C @ pod.basis.T + pod.mean


def relative_frobenius_error(reference: np.ndarray, approximation: np.ndarray, eps: float = 1e-14) -> float:
    """Return relative Frobenius reconstruction error."""

    A = as_snapshot_matrix(reference)
    B = as_snapshot_matrix(approximation)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: reference {A.shape}, approximation {B.shape}.")
    denom = max(float(np.linalg.norm(A, ord="fro")), float(eps))
    return float(np.linalg.norm(A - B, ord="fro") / denom)
