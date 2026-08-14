from __future__ import annotations

import numpy as np

from paramplate.rom.preprocessing import Standardizer
from paramplate.rom.splits import (
    make_sample_split,
    make_trajectory_split,
    rows_for_trajectories,
    validate_no_trajectory_leakage,
)


def test_thesis_sample_split_counts_are_exact() -> None:
    assert make_sample_split(450).counts == (360, 45, 45)
    assert make_sample_split(378).counts == (302, 38, 38)
    assert make_sample_split(500).counts == (400, 50, 50)


def test_split_is_deterministic_and_disjoint() -> None:
    first = make_sample_split(60)
    second = make_sample_split(60)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    all_ids = np.concatenate([first.train, first.validation, first.test])
    assert np.array_equal(np.sort(all_ids), np.arange(60))


def test_complete_trajectory_split_precedes_flattening() -> None:
    trajectory_ids = np.repeat(np.arange(20), 11)
    split = make_trajectory_split(trajectory_ids)
    assert split.counts == (16, 2, 2)
    validate_no_trajectory_leakage(trajectory_ids, split)
    train_rows = rows_for_trajectories(trajectory_ids, split.train)
    val_rows = rows_for_trajectories(trajectory_ids, split.validation)
    test_rows = rows_for_trajectories(trajectory_ids, split.test)
    assert (len(train_rows), len(val_rows), len(test_rows)) == (176, 22, 22)
    assert not set(trajectory_ids[train_rows]).intersection(trajectory_ids[val_rows])
    assert not set(trajectory_ids[train_rows]).intersection(trajectory_ids[test_rows])


def test_standardizer_is_fitted_only_to_supplied_rows() -> None:
    training = np.array([[0.0], [2.0], [4.0]])
    scaler = Standardizer.fit(training)
    assert scaler.mean[0] == 2.0
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(training)), training, atol=1.0e-14
    )
