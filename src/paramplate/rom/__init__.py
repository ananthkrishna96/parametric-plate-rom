"""Reduced-order modelling utilities."""

from .datasets import ROMDataset, load_rom_dataset, load_project2_dataset_by_name
from .pod import (
    PODResult,
    as_snapshot_matrix,
    compute_pod,
    cumulative_energy_from_singular_values,
    project_snapshots,
    reconstruct_snapshots,
    relative_frobenius_error,
    select_rank,
)
from .preprocessing import (
    PODPreprocessingResult,
    ParameterScaler,
    build_pod_preprocessing,
    fit_parameter_scaler,
    save_pod_preprocessing,
)
from .splits import TrainTestSplit, make_train_test_split

__all__ = [
    "ROMDataset",
    "load_rom_dataset",
    "load_project2_dataset_by_name",
    "PODResult",
    "as_snapshot_matrix",
    "compute_pod",
    "cumulative_energy_from_singular_values",
    "project_snapshots",
    "reconstruct_snapshots",
    "relative_frobenius_error",
    "select_rank",
    "PODPreprocessingResult",
    "ParameterScaler",
    "build_pod_preprocessing",
    "fit_parameter_scaler",
    "save_pod_preprocessing",
    "TrainTestSplit",
    "make_train_test_split",
]
