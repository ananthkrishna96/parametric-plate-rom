"""Common training-only POD and coordinate-predictor preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import SnapshotDataset
from .metrics import Metric
from .pod import PODBasis, compute_metric_pod, compute_transient_pod
from .preprocessing import Standardizer
from .splits import DataSplit, make_sample_split, make_trajectory_split, rows_for_trajectories


@dataclass(frozen=True)
class PreparedROMData:
    dataset: SnapshotDataset
    split: DataSplit
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    pod: PODBasis
    input_scaler: Standardizer
    coordinate_scaler: Standardizer
    inputs_train: np.ndarray
    inputs_validation: np.ndarray
    inputs_test: np.ndarray
    coordinates_train: np.ndarray
    coordinates_validation: np.ndarray
    coordinates_test: np.ndarray


def prepare_rom_data(
    dataset: SnapshotDataset,
    *,
    metric: Metric,
    rank: int,
    metric_name: str,
    transient_protocol: bool | None = None,
) -> PreparedROMData:
    """Apply the thesis split before fitting POD, scalers, or models."""

    transient = dataset.is_trajectory_dataset if transient_protocol is None else bool(transient_protocol)
    if transient:
        if dataset.trajectory_ids is None:
            raise ValueError("The transient protocol requires trajectory IDs.")
        split = make_trajectory_split(dataset.trajectory_ids)
        train_rows = rows_for_trajectories(dataset.trajectory_ids, split.train)
        validation_rows = rows_for_trajectories(dataset.trajectory_ids, split.validation)
        test_rows = rows_for_trajectories(dataset.trajectory_ids, split.test)
        pod = compute_transient_pod(
            dataset.snapshots[train_rows],
            finite_element_metric=metric,
            rank=rank,
            metric_name=metric_name,
            center=False,
        )
    else:
        split = make_sample_split(dataset.n_rows)
        train_rows, validation_rows, test_rows = split.train, split.validation, split.test
        pod = compute_metric_pod(
            dataset.snapshots[train_rows],
            metric=metric,
            rank=rank,
            center=False,
            metric_name=metric_name,
        )

    input_scaler = Standardizer.fit(dataset.parameters[train_rows])
    coordinates_all = pod.project(dataset.snapshots, metric)
    coordinate_scaler = Standardizer.fit(coordinates_all[train_rows])
    scaled_inputs = input_scaler.transform(dataset.parameters)
    scaled_coordinates = coordinate_scaler.transform(coordinates_all)
    return PreparedROMData(
        dataset=dataset,
        split=split,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        pod=pod,
        input_scaler=input_scaler,
        coordinate_scaler=coordinate_scaler,
        inputs_train=scaled_inputs[train_rows],
        inputs_validation=scaled_inputs[validation_rows],
        inputs_test=scaled_inputs[test_rows],
        coordinates_train=scaled_coordinates[train_rows],
        coordinates_validation=scaled_coordinates[validation_rows],
        coordinates_test=scaled_coordinates[test_rows],
    )
