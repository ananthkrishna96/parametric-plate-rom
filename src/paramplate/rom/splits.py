"""Reproducible train/test split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainTestSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    seed: int
    train_fraction: float

    @property
    def n_train(self) -> int:
        return int(self.train_indices.size)

    @property
    def n_test(self) -> int:
        return int(self.test_indices.size)


def make_train_test_split(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> TrainTestSplit:
    """Create reproducible train/test indices."""

    n = int(n_samples)
    if n < 2:
        raise ValueError("At least two samples are required for a train/test split.")
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")

    n_train = int(round(n * float(train_fraction)))
    n_train = max(1, min(n_train, n - 1))

    indices = np.arange(n, dtype=int)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)

    train = np.sort(indices[:n_train])
    test = np.sort(indices[n_train:])
    return TrainTestSplit(
        train_indices=train,
        test_indices=test,
        seed=int(seed),
        train_fraction=float(train_fraction),
    )
