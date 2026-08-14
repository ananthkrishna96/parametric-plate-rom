"""Training/validation/test partitions with trajectory leakage control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paramplate.io.manifest import ensure_unique_memberships


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_seed: int = 100
    holdout_seed: int = 101
    level: str = "sample"

    def __post_init__(self) -> None:
        ensure_unique_memberships((self.train, self.validation, self.test))
        if self.level not in {"sample", "trajectory"}:
            raise ValueError("Split level must be sample or trajectory.")

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.train), len(self.validation), len(self.test)


def _three_way_ids(
    ids: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    train_seed: int,
    holdout_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.asarray(ids, dtype=int).reshape(-1)
    if unique.size < 3:
        raise ValueError("At least three unique records are required for a three-way split.")
    if np.unique(unique).size != unique.size:
        raise ValueError("ids passed to the splitter must be unique.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie in (0, 1).")
    if not (0.0 < validation_fraction < 1.0 - train_fraction):
        raise ValueError("validation_fraction must be positive and leave a nonempty test fraction.")

    rng_train = np.random.default_rng(int(train_seed))
    shuffled = unique.copy()
    rng_train.shuffle(shuffled)
    n_train = int(np.floor(train_fraction * len(shuffled)))
    n_train = min(max(n_train, 1), len(shuffled) - 2)
    train = np.sort(shuffled[:n_train])
    remainder = shuffled[n_train:].copy()

    rng_holdout = np.random.default_rng(int(holdout_seed))
    rng_holdout.shuffle(remainder)
    target_val_fraction = validation_fraction / (1.0 - train_fraction)
    n_val = int(np.floor(target_val_fraction * len(remainder)))
    # Equal 10/10 holdouts use exactly half of the remainder; floor gives the
    # thesis counts for odd totals while preserving at least one test record.
    n_val = min(max(n_val, 1), len(remainder) - 1)
    validation = np.sort(remainder[:n_val])
    test = np.sort(remainder[n_val:])
    return train, validation, test


def make_sample_split(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    train_seed: int = 100,
    holdout_seed: int = 101,
) -> DataSplit:
    ids = np.arange(int(n_samples), dtype=int)
    train, validation, test = _three_way_ids(
        ids,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        train_seed=train_seed,
        holdout_seed=holdout_seed,
    )
    return DataSplit(train, validation, test, train_seed, holdout_seed, "sample")


def make_trajectory_split(
    trajectory_ids: np.ndarray,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    train_seed: int = 100,
    holdout_seed: int = 101,
) -> DataSplit:
    """Split unique trajectories before expanding to state-row indices."""

    ids = np.asarray(trajectory_ids, dtype=int).reshape(-1)
    unique = np.unique(ids)
    train, validation, test = _three_way_ids(
        unique,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        train_seed=train_seed,
        holdout_seed=holdout_seed,
    )
    return DataSplit(train, validation, test, train_seed, holdout_seed, "trajectory")


def rows_for_trajectories(trajectory_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(trajectory_ids, dtype=int).reshape(-1)
    chosen = np.asarray(selected_ids, dtype=int).reshape(-1)
    return np.flatnonzero(np.isin(ids, chosen)).astype(int)


def validate_no_trajectory_leakage(trajectory_ids: np.ndarray, split: DataSplit) -> None:
    if split.level != "trajectory":
        raise ValueError("validate_no_trajectory_leakage requires a trajectory-level split.")
    present = set(np.unique(np.asarray(trajectory_ids, dtype=int)).tolist())
    assigned = set(np.concatenate([split.train, split.validation, split.test]).tolist())
    if assigned != present:
        missing = sorted(present - assigned)
        extra = sorted(assigned - present)
        raise ValueError(f"Trajectory assignment mismatch; missing={missing[:10]}, extra={extra[:10]}.")
