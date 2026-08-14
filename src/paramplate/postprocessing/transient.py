"""Transient-history plots from stored numerical arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_probe_history(
    coordinates: np.ndarray,
    values: np.ndarray,
    output: str | Path,
    *,
    xlabel: str = "time [s]",
    ylabel: str = "displacement [m]",
) -> Path:
    x = np.asarray(coordinates, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("coordinates and values must have one common shape.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    axis.plot(x, y)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def save_energy_history(times: np.ndarray, energy: np.ndarray, output: str | Path) -> Path:
    return save_probe_history(times, energy, output, ylabel="mechanical energy [J]")
