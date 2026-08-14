"""Field-aware views of mechanical, thermomechanical, and transient archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paramplate.io.archives import archive_keys


@dataclass(frozen=True)
class SnapshotDataset:
    source: Path
    study: str
    output_name: str
    parameters: np.ndarray
    snapshots: np.ndarray
    parameter_names: tuple[str, ...]
    output_unit: str
    trajectory_ids: np.ndarray | None = None
    times: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.snapshots.shape[0])

    @property
    def n_dofs(self) -> int:
        return int(self.snapshots.shape[1])

    @property
    def is_trajectory_dataset(self) -> bool:
        return self.trajectory_ids is not None


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(x) for x in np.asarray(values, dtype=object).reshape(-1).tolist())


def _first_present(files: set[str], candidates: tuple[str, ...], label: str) -> str:
    for key in candidates:
        if key in files:
            return key
    raise KeyError(f"None of the supported {label} keys are present: {candidates}.")


def load_snapshot_dataset(
    path: str | Path,
    *,
    study: str,
    output: str = "displacement",
) -> SnapshotDataset:
    """Load one active output while preserving complete-trajectory identifiers.

    The loader accepts both the cleaned repository schema and the current thesis
    snapshot-export names.  It never combines displacement and thermal-driver
    fields into one target.
    """

    archive = Path(path).expanduser().resolve()
    keys = set(archive_keys(archive))
    normalized_study = study.lower().strip()
    normalized_output = output.lower().strip()
    supported_studies = {"mechanical", "thermomechanical", "thermo", "dynamics", "transient"}
    if normalized_study not in supported_studies:
        raise ValueError(
            f"Unsupported study {study!r}; expected mechanical, thermomechanical, or dynamics."
        )
    if normalized_study in {"mechanical", "dynamics", "transient"} and normalized_output not in {
        "displacement",
        "w",
    }:
        raise ValueError(f"{normalized_study} archives expose displacement as the only reduced output.")
    if normalized_study in {"thermomechanical", "thermo"} and normalized_output not in {
        "displacement",
        "w",
        "thermal_driver",
        "theta",
    }:
        raise ValueError(
            "Thermomechanical archives expose separate displacement and thermal_driver outputs."
        )
    with np.load(archive, allow_pickle=True) as data:
        if normalized_study in {"thermomechanical", "thermo"}:
            if normalized_output in {"thermal_driver", "theta"}:
                snapshot_key = _first_present(
                    keys,
                    ("thermal_driver_snapshots", "theta_snapshots", "theta", "T1_snapshots"),
                    "thermal-driver snapshot",
                )
                unit = "K/m"
            else:
                snapshot_key = _first_present(
                    keys,
                    ("displacement_snapshots", "snapshots", "fom_snapshots", "w_snapshots"),
                    "displacement snapshot",
                )
                unit = "m"
        elif normalized_study in {"dynamics", "transient"}:
            snapshot_key = _first_present(keys, ("snapshots", "fom_snapshots"), "transient snapshot")
            unit = "m"
        else:
            snapshot_key = _first_present(keys, ("snapshots", "fom_snapshots"), "mechanical snapshot")
            unit = "m"

        parameter_key = _first_present(
            keys,
            ("query_parameters", "parameters", "parameters_flat"),
            "parameter",
        )
        snapshots = np.asarray(data[snapshot_key], dtype=float)
        parameters = np.asarray(data[parameter_key], dtype=float)
        names = (
            _strings(data["parameter_names"])
            if "parameter_names" in data.files
            else tuple(f"mu_{j}" for j in range(parameters.shape[1]))
        )
        trajectory_ids = None
        for candidate in ("trajectory_ids", "trajectory_ids_flat"):
            if candidate in data.files:
                trajectory_ids = np.asarray(data[candidate], dtype=int).reshape(-1)
                break
        times = np.asarray(data["times"], dtype=float) if "times" in data.files else None

    if snapshots.ndim > 2 and normalized_study in {"dynamics", "transient"}:
        snapshots = snapshots.reshape(-1, snapshots.shape[-1])
    if snapshots.ndim != 2 or parameters.ndim != 2 or len(snapshots) != len(parameters):
        raise ValueError("Archive parameters and selected snapshots must be aligned two-dimensional arrays.")
    if trajectory_ids is not None and len(trajectory_ids) != len(snapshots):
        raise ValueError("Trajectory identifiers are not aligned with flattened snapshot rows.")
    return SnapshotDataset(
        source=archive,
        study=normalized_study,
        output_name="thermal_driver" if normalized_output in {"thermal_driver", "theta"} else "displacement",
        parameters=parameters,
        snapshots=snapshots,
        parameter_names=names,
        output_unit=unit,
        trajectory_ids=trajectory_ids,
        times=times,
    )
