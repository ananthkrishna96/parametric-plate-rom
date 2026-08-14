"""Lightweight inspection and validation of thesis snapshot archives."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ArrayHeader:
    key: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool


def npz_array_header(path: str | Path, key: str) -> ArrayHeader:
    """Read a member header without loading its numerical payload."""

    archive = Path(path)
    member = f"{key}.npy"
    with zipfile.ZipFile(archive, "r") as zf:
        if member not in zf.namelist():
            raise KeyError(f"{key!r} is missing from {archive}.")
        with zf.open(member, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version {version} for {member}.")
    return ArrayHeader(key, tuple(int(x) for x in shape), str(np.dtype(dtype)), bool(fortran_order))


def archive_keys(path: str | Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path, "r") as zf:
        return tuple(sorted(name[:-4] for name in zf.namelist() if name.endswith(".npy")))


def small_array(path: str | Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{key!r} is missing from {path}.")
        return np.asarray(data[key])


def metadata_json(path: str | Path, key: str = "metadata_json") -> dict[str, Any]:
    try:
        raw = small_array(path, key)
    except KeyError:
        return {}
    try:
        return json.loads(str(raw.item()))
    except Exception:
        return {}


def _choose(keys: set[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    raise ValueError(f"Archive is missing all supported members: {candidates}.")


def validate_thermomechanical_archive(path: str | Path) -> dict[str, Any]:
    keys = set(archive_keys(path))
    w_key = _choose(keys, "displacement_snapshots", "snapshots", "w_snapshots")
    theta_key = _choose(keys, "thermal_driver_snapshots", "theta_snapshots", "T1_snapshots")
    parameter_key = _choose(keys, "parameters")
    w = npz_array_header(path, w_key)
    theta = npz_array_header(path, theta_key)
    params = npz_array_header(path, parameter_key)
    if len(w.shape) != 2 or len(theta.shape) != 2 or len(params.shape) != 2:
        raise ValueError("Thermomechanical snapshots and parameters must be two-dimensional.")
    if not (w.shape[0] == theta.shape[0] == params.shape[0]):
        raise ValueError("Displacement, thermal-driver, and parameter row counts differ.")
    names = [str(x) for x in small_array(path, "parameter_names").tolist()] if "parameter_names" in keys else []
    return {
        "kind": "thermomechanical",
        "rows": w.shape[0],
        "displacement_key": w_key,
        "thermal_driver_key": theta_key,
        "displacement_shape": w.shape,
        "thermal_driver_shape": theta.shape,
        "parameter_shape": params.shape,
        "parameter_names": names,
    }


def validate_dynamic_archive(path: str | Path) -> dict[str, Any]:
    keys = set(archive_keys(path))
    snapshot_key = _choose(keys, "snapshots", "fom_snapshots")
    parameter_key = _choose(keys, "query_parameters", "parameters", "parameters_flat")
    trajectory_key = _choose(keys, "trajectory_ids", "trajectory_ids_flat")
    snapshots = npz_array_header(path, snapshot_key)
    parameters = npz_array_header(path, parameter_key)
    ids = np.asarray(small_array(path, trajectory_key), dtype=int).reshape(-1)
    times = np.asarray(small_array(path, "times"), dtype=float).reshape(-1)
    n_rows = int(np.prod(snapshots.shape[:-1])) if len(snapshots.shape) > 2 else snapshots.shape[0]
    if n_rows != parameters.shape[0] or ids.size != n_rows:
        raise ValueError("Dynamic flattened arrays have inconsistent row counts.")
    unique_ids, counts = np.unique(ids, return_counts=True)
    if counts.size and np.any(counts != len(times)):
        raise ValueError("Not every trajectory contains the complete stored-time vector.")
    names = [str(x) for x in small_array(path, "parameter_names").tolist()] if "parameter_names" in keys else []
    return {
        "kind": "dynamics",
        "snapshot_shape": snapshots.shape,
        "parameter_shape": parameters.shape,
        "n_trajectories": int(unique_ids.size),
        "n_times": int(len(times)),
        "trajectory_id_key": trajectory_key,
        "parameter_names": names,
    }


def validate_archive(path: str | Path, kind: str = "auto") -> dict[str, Any]:
    archive = Path(path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    keys = set(archive_keys(archive))
    selected = kind
    if selected == "auto":
        if {"trajectory_ids", "trajectory_ids_flat"}.intersection(keys):
            selected = "dynamics"
        elif {"thermal_driver_snapshots", "theta_snapshots", "T1_snapshots"}.intersection(keys):
            selected = "thermomechanical"
        else:
            selected = "mechanical"
    if selected == "thermomechanical":
        return validate_thermomechanical_archive(archive)
    if selected == "dynamics":
        return validate_dynamic_archive(archive)
    if selected == "mechanical":
        snapshot_key = _choose(keys, "snapshots", "fom_snapshots")
        parameter_key = _choose(keys, "parameters")
        snapshots = npz_array_header(archive, snapshot_key)
        parameters = npz_array_header(archive, parameter_key)
        if len(snapshots.shape) != 2 or len(parameters.shape) != 2 or snapshots.shape[0] != parameters.shape[0]:
            raise ValueError("Mechanical snapshots and parameters must have matching rows.")
        return {
            "kind": "mechanical",
            "snapshot_key": snapshot_key,
            "snapshot_shape": snapshots.shape,
            "parameter_shape": parameters.shape,
        }
    raise ValueError(f"Unknown archive kind {kind!r}.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a paramplate snapshot archive without loading large field arrays."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--kind",
        choices=["auto", "mechanical", "thermomechanical", "dynamics"],
        default="auto",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(validate_archive(args.archive, args.kind), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
