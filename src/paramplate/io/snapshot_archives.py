"""Snapshot archive detection and loading utilities.

The loader intentionally supports both historical Paper-1 archive conventions
and the newer Project-2 thermomechanical SNAPGEN archive convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


PAPER2_PARAMETER_KEYS = ("parameters",)
PAPER2_SNAPSHOT_KEYS = ("snapshots",)
PAPER2_THETA_KEYS = ("theta_snapshots", "T1_snapshots", "thermal_snapshots")

PAPER1_PARAMETER_KEYS = (
    "mus",
    "mu",
    "parameters",
    "training_parameters",
    "parameter_matrix",
)
PAPER1_SNAPSHOT_KEYS = (
    "fom_snapshots",
    "snapshots",
    "nonlinear_snapshots",
    "solution_snapshots",
    "reprojected_snapshots",
    "projected_snapshots",
    "S",
    "X",
)


@dataclass(frozen=True)
class SnapshotArchiveInfo:
    """Lightweight metadata summary of a ``.npz`` snapshot archive."""

    path: Path
    project: str
    parameter_key: Optional[str]
    snapshot_key: Optional[str]
    theta_key: Optional[str]
    n_samples: Optional[int]
    parameter_shape: Optional[tuple[int, ...]]
    snapshot_shape: Optional[tuple[int, ...]]
    theta_shape: Optional[tuple[int, ...]]
    keys: tuple[str, ...]

    @property
    def is_project2(self) -> bool:
        return self.project == "paper2"

    @property
    def is_paper1(self) -> bool:
        return self.project == "paper1"


def _first_existing(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_set = set(keys)
    for name in candidates:
        if name in key_set:
            return name
    return None


def _shape_of(npz: np.lib.npyio.NpzFile, key: Optional[str]) -> Optional[tuple[int, ...]]:
    if key is None:
        return None
    try:
        return tuple(int(x) for x in npz[key].shape)
    except Exception:
        return None


def _infer_n_samples(parameter_shape: Optional[tuple[int, ...]], snapshot_shape: Optional[tuple[int, ...]]) -> Optional[int]:
    if parameter_shape and len(parameter_shape) >= 1:
        return int(parameter_shape[0])
    if snapshot_shape and len(snapshot_shape) >= 1:
        return int(snapshot_shape[0])
    return None


def detect_archive_kind(keys: Iterable[str], path: str | Path | None = None) -> str:
    """Detect whether an archive is ``paper1``, ``paper2``, or ``unknown``."""

    key_set = set(keys)
    name = Path(path).name.lower() if path is not None else ""

    has_p2_core = {"parameters", "snapshots"}.issubset(key_set)
    has_p2_extra = bool(key_set.intersection({"theta_snapshots", "thermal_parameters", "mech_parameters"}))
    if has_p2_core and (has_p2_extra or name.startswith("snapshots__") or "thermo" in name):
        return "paper2"

    has_p1_params = bool(key_set.intersection(PAPER1_PARAMETER_KEYS))
    has_p1_snaps = bool(key_set.intersection(PAPER1_SNAPSHOT_KEYS))
    if has_p1_params and has_p1_snaps:
        return "paper1"

    if has_p2_core:
        return "paper2"
    return "unknown"


def summarize_snapshot_archive(path: str | Path) -> SnapshotArchiveInfo:
    """Open a snapshot archive lazily and return key/shape metadata."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz archive, got: {p}")

    with np.load(p, allow_pickle=True) as z:
        keys = tuple(str(k) for k in z.files)
        project = detect_archive_kind(keys, p)

        if project == "paper2":
            parameter_key = _first_existing(keys, PAPER2_PARAMETER_KEYS)
            snapshot_key = _first_existing(keys, PAPER2_SNAPSHOT_KEYS)
            theta_key = _first_existing(keys, PAPER2_THETA_KEYS)
        elif project == "paper1":
            parameter_key = _first_existing(keys, PAPER1_PARAMETER_KEYS)
            snapshot_key = _first_existing(keys, PAPER1_SNAPSHOT_KEYS)
            theta_key = None
        else:
            parameter_key = _first_existing(keys, PAPER2_PARAMETER_KEYS + PAPER1_PARAMETER_KEYS)
            snapshot_key = _first_existing(keys, PAPER2_SNAPSHOT_KEYS + PAPER1_SNAPSHOT_KEYS)
            theta_key = _first_existing(keys, PAPER2_THETA_KEYS)

        parameter_shape = _shape_of(z, parameter_key)
        snapshot_shape = _shape_of(z, snapshot_key)
        theta_shape = _shape_of(z, theta_key)
        n_samples = _infer_n_samples(parameter_shape, snapshot_shape)

    return SnapshotArchiveInfo(
        path=p,
        project=project,
        parameter_key=parameter_key,
        snapshot_key=snapshot_key,
        theta_key=theta_key,
        n_samples=n_samples,
        parameter_shape=parameter_shape,
        snapshot_shape=snapshot_shape,
        theta_shape=theta_shape,
        keys=keys,
    )


def _decode_optional_metadata(value: Any) -> Any:
    """Decode metadata arrays that may contain JSON strings or Python objects."""

    try:
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    except Exception:
        return None


def load_npz_metadata(path: str | Path) -> dict[str, Any]:
    """Return non-large metadata-like keys from an archive.

    This intentionally avoids returning the large solution matrices.
    """

    p = Path(path).expanduser().resolve()
    meta: dict[str, Any] = {}
    with np.load(p, allow_pickle=True) as z:
        for key in z.files:
            if key in {"snapshots", "theta_snapshots", "fom_snapshots", "nonlinear_snapshots"}:
                continue
            if key.endswith("_names") or key.endswith("_units") or key.endswith("_roles") or key in {
                "metadata",
                "metadata_json",
                "case_name",
                "case_id",
                "parameter_names",
                "parameter_units",
                "parameter_roles",
                "mech_parameter_names",
                "thermal_parameter_names",
            }:
                try:
                    meta[key] = _decode_optional_metadata(z[key])
                except Exception:
                    meta[key] = None
    return meta
