"""Train-fitted parameter--time features for the transient POD--DL-ROM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TransientFeatureMap:
    """Feature map used by the current time-conditioned transient model.

    It concatenates raw inputs, train-min/max-normalized inputs, clipped
    log-normalized inputs, four normalized-time powers, and six sine/cosine
    time harmonics. With active inputs ``(A_q, time)`` and
    ``(D_x, A_q, time)``, the dimensions are 22 and 25 respectively.
    """

    time_column: int = -1
    harmonics: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    use_log_features: bool = True
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def fit(self, inputs: np.ndarray) -> "TransientFeatureMap":
        X = np.asarray(inputs, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Feature fitting requires a nonempty two-dimensional array.")
        index = self.time_column if self.time_column >= 0 else X.shape[1] + self.time_column
        if not 0 <= index < X.shape[1]:
            raise ValueError("time_column is outside the input width.")
        self.time_column = index
        self.minimum = np.nanmin(X, axis=0)
        self.maximum = np.nanmax(X, axis=0)
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("TransientFeatureMap must be fitted on training data first.")
        X = np.asarray(inputs, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.minimum.size:
            raise ValueError("Input width differs from the fitted feature map.")
        eps = 1.0e-30
        span = np.maximum(self.maximum - self.minimum, eps)
        normalized = (X - self.minimum) / span
        features = [X, normalized]
        if self.use_log_features:
            positive = np.maximum(X, eps)
            low = np.maximum(self.minimum, eps)
            high = np.maximum(self.maximum, eps)
            log_span = np.maximum(np.log(high) - np.log(low), eps)
            features.append((np.log(positive) - np.log(low)) / log_span)

        tau = normalized[:, self.time_column]
        features.extend([(tau**power).reshape(-1, 1) for power in (1, 2, 3, 4)])
        for k in self.harmonics:
            phase = 2.0 * np.pi * float(k) * tau
            features.append(np.sin(phase).reshape(-1, 1))
            features.append(np.cos(phase).reshape(-1, 1))
        return np.hstack(features).astype(float, copy=False)

    def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
        return self.fit(inputs).transform(inputs)

    @property
    def output_dimension(self) -> int:
        if self.minimum is None:
            raise RuntimeError("Feature map has not been fitted.")
        d = int(self.minimum.size)
        return 2 * d + (d if self.use_log_features else 0) + 4 + 2 * len(self.harmonics)


def impose_zero_time(
    coordinates: np.ndarray,
    inputs: np.ndarray,
    *,
    time_column: int = -1,
    tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Set predicted coordinates to zero at the prescribed initial time."""

    coeffs = np.asarray(coordinates, dtype=float).copy()
    X = np.asarray(inputs, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(1, -1)
    if len(coeffs) != len(X):
        raise ValueError("Coordinate and input row counts differ.")
    index = time_column if time_column >= 0 else X.shape[1] + time_column
    coeffs[np.abs(X[:, index]) <= tolerance, :] = 0.0
    return coeffs
