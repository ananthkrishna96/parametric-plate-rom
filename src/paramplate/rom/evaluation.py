"""ROM reconstruction, field-error, and online timing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .metrics import Metric, relative_metric_errors
from .pod import PODBasis


@dataclass(frozen=True)
class EvaluationSummary:
    mean: float
    median: float
    maximum: float
    minimum: float
    finite_count: int
    total_count: int


def summarize_errors(errors: np.ndarray) -> EvaluationSummary:
    values = np.asarray(errors, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return EvaluationSummary(np.nan, np.nan, np.nan, np.nan, 0, len(values))
    return EvaluationSummary(
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        maximum=float(np.max(finite)),
        minimum=float(np.min(finite)),
        finite_count=int(finite.size),
        total_count=int(values.size),
    )


def evaluate_coordinate_predictor(
    predictor,
    inputs: np.ndarray,
    reference_fields: np.ndarray,
    pod: PODBasis,
    *,
    metric: Metric = None,
    inverse_coordinate_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    zero_policy: str = "nan",
) -> tuple[np.ndarray, np.ndarray, EvaluationSummary]:
    predicted_coordinates = np.asarray(predictor.predict(inputs), dtype=float)
    if inverse_coordinate_transform is not None:
        predicted_coordinates = inverse_coordinate_transform(predicted_coordinates)
    predicted_fields = pod.reconstruct(predicted_coordinates)
    errors = relative_metric_errors(reference_fields, predicted_fields, metric, zero_policy=zero_policy)
    return predicted_coordinates, predicted_fields, summarize_errors(errors)


def timed_prediction(
    operation: Callable[[], object],
    *,
    repeats: int = 10,
    warmup: int = 1,
) -> tuple[object, np.ndarray]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative.")
    result = None
    for _ in range(warmup):
        result = operation()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        result = operation()
        timings.append(perf_counter() - start)
    return result, np.asarray(timings, dtype=float)
