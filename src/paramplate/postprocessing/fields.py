"""Restrained plotting helpers for stored plate fields and ROM errors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_field_image(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output: str | Path,
    *,
    label: str,
    title: str | None = None,
) -> Path:
    values = np.asarray(field, dtype=float)
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if values.shape != (yy.size, xx.size):
        raise ValueError("field shape must be (len(y), len(x)).")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.0, 3.1))
    image = axis.pcolormesh(xx, yy, values, shading="auto")
    fig.colorbar(image, ax=axis, label=label)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    if title:
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def save_pod_spectrum(
    eigenvalues: np.ndarray,
    output: str | Path,
    *,
    title: str = "POD spectrum",
) -> Path:
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 3.4))
    axis.semilogy(np.arange(1, len(values) + 1), np.maximum(values, np.finfo(float).tiny), marker="o")
    axis.set_xlabel("mode")
    axis.set_ylabel("POD eigenvalue")
    axis.set_title(title)
    axis.grid(True, which="both", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def save_error_histogram(errors: np.ndarray, output: str | Path, *, label: str = "relative error") -> Path:
    values = np.asarray(errors, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.4, 3.3))
    axis.hist(values, bins=min(24, max(5, int(np.sqrt(max(len(values), 1))))))
    axis.set_xlabel(label)
    axis.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
