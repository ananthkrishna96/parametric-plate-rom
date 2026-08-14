"""Finite-dimensional metric operations for POD and error evaluation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

Metric = np.ndarray | sparse.spmatrix | None


def apply_metric(vectors: np.ndarray, metric: Metric) -> np.ndarray:
    """Apply a DOF metric to row vectors."""

    array = np.asarray(vectors, dtype=float)
    one_dimensional = array.ndim == 1
    rows = array.reshape(1, -1) if one_dimensional else array
    if rows.ndim != 2:
        raise ValueError("vectors must be one- or two-dimensional.")
    if metric is None:
        out = rows.copy()
    elif sparse.issparse(metric):
        out = np.asarray((metric @ rows.T).T)
    else:
        G = np.asarray(metric, dtype=float)
        out = rows @ G
    return out[0] if one_dimensional else out


def metric_gram(basis: np.ndarray, metric: Metric) -> np.ndarray:
    B = np.asarray(basis, dtype=float)
    if B.ndim != 2:
        raise ValueError("basis must be two-dimensional.")
    return B.T @ apply_metric(B.T, metric).T


def metric_inner(a: np.ndarray, b: np.ndarray, metric: Metric = None) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Metric inner-product vectors have different shapes.")
    return float(x @ apply_metric(y, metric))


def metric_norm(values: np.ndarray, metric: Metric = None) -> np.ndarray:
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        return np.asarray(np.sqrt(max(metric_inner(rows, rows, metric), 0.0)))
    if rows.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    weighted = apply_metric(rows, metric)
    squared = np.einsum("ij,ij->i", rows, weighted)
    return np.sqrt(np.maximum(squared, 0.0))


def relative_metric_errors(
    reference: np.ndarray,
    approximation: np.ndarray,
    metric: Metric = None,
    *,
    zero_policy: str = "nan",
    epsilon: float = 1.0e-14,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(approximation, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} != {pred.shape}.")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
        pred = pred.reshape(1, -1)
    denom = metric_norm(ref, metric)
    numer = metric_norm(ref - pred, metric)
    result = np.empty_like(denom)
    nonzero = denom > epsilon
    result[nonzero] = numer[nonzero] / denom[nonzero]
    if zero_policy == "nan":
        result[~nonzero] = np.nan
    elif zero_policy == "zero":
        result[~nonzero] = 0.0
    elif zero_policy == "absolute":
        result[~nonzero] = numer[~nonzero]
    else:
        raise ValueError("zero_policy must be nan, zero, or absolute.")
    return result
