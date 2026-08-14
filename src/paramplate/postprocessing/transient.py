"""Transient-history plots from stored numerical arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_probe_history(
    times: np.ndarray,
    values: np.ndarray,
    output: str | Path,
    *,
    ylabel: str = "displacement [m]",
) -> Path:
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("times and values must have one common shape.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    axis.plot(t, y)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def save_energy_history(times: np.ndarray, energy: np.ndarray, output: str | Path) -> Path:
    return save_probe_history(times, energy, output, ylabel="mechanical energy [J]")
