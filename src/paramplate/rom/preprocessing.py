"""Training-only affine preprocessing for physical inputs and ROM coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Standardizer.fit expects a nonempty two-dimensional array.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Training data contain NaN or infinite values.")
        mean = array.mean(axis=0)
        scale = array.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array * self.scale + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Standardizer":
        return cls(np.asarray(payload["mean"], dtype=float), np.asarray(payload["scale"], dtype=float))


@dataclass(frozen=True)
class MinMaxTransform:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxTransform":
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return cls(np.nanmin(array, axis=0), np.nanmax(array, axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        span = np.maximum(self.maximum - self.minimum, 1.0e-30)
        return (np.asarray(values, dtype=float) - self.minimum) / span
