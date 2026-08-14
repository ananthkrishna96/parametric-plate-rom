"""Deterministic parameter designs for snapshot campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterRange:
    name: str
    lower: float
    upper: float
    unit: str = ""
    scale: str = "linear"
    role: str = "parameter"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParameterRange.name cannot be empty.")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.upper <= self.lower:
            raise ValueError(f"Invalid interval for {self.name}: [{self.lower}, {self.upper}].")
        if self.scale not in {"linear", "log"}:
            raise ValueError("scale must be 'linear' or 'log'.")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError("Logarithmic ranges require a positive lower bound.")

    def decode(self, u: np.ndarray) -> np.ndarray:
        a = np.asarray(u, dtype=float)
        if self.scale == "linear":
            return self.lower + (self.upper - self.lower) * a
        lo, hi = np.log(self.lower), np.log(self.upper)
        return np.exp(lo + (hi - lo) * a)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def latin_hypercube(
    ranges: Sequence[ParameterRange],
    n_samples: int,
    *,
    seed: int = 100,
    optimization: str | None = None,
) -> np.ndarray:
    """Sample a declared physical parameter design.

    Log-scaled variables are stratified in logarithmic coordinates. The return
    shape is ``(n_samples, n_parameters)`` and column order follows ``ranges``.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not ranges:
        raise ValueError("At least one parameter range is required.")
    sampler = qmc.LatinHypercube(d=len(ranges), seed=int(seed), optimization=optimization)
    unit = sampler.random(n=int(n_samples))
    return np.column_stack([spec.decode(unit[:, j]) for j, spec in enumerate(ranges)])


def parameter_records(ranges: Iterable[ParameterRange], samples: np.ndarray) -> list[dict[str, float]]:
    specs = list(ranges)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(specs):
        raise ValueError("samples do not match the declared parameter ranges.")
    return [dict(zip((s.name for s in specs), row, strict=True)) for row in values]
