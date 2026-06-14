"""ROM dataset abstraction for Paper-1 and Project-2 snapshot archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from paramplate.io.data_locations import DataLocations, load_data_locations
from paramplate.io.snapshot_archives import (
    SnapshotArchiveInfo,
    load_npz_metadata,
    summarize_snapshot_archive,
)
from paramplate.io.snapshot_manifest import paper1_archives, project2_final_archives


@dataclass
class ROMDataset:
    """Standard in-memory representation of one ROM training archive."""

    source_path: Path
    project: str
    parameters: np.ndarray
    snapshots: np.ndarray
    theta_snapshots: Optional[np.ndarray] = None
    metadata: dict[str, Any] | None = None
    info: SnapshotArchiveInfo | None = None

    @property
    def n_samples(self) -> int:
        return int(self.parameters.shape[0])

    @property
    def n_dofs(self) -> int:
        return int(self.snapshots.shape[1]) if self.snapshots.ndim >= 2 else int(self.snapshots.shape[0])

    @property
    def has_theta(self) -> bool:
        return self.theta_snapshots is not None


def load_rom_dataset(path: str | Path, *, load_theta: bool = True) -> ROMDataset:
    """Load one Paper-1 or Paper-2 archive as a :class:`ROMDataset`.

    This function reads the large snapshot arrays into memory.  For lightweight
    checks, use ``summarize_snapshot_archive`` instead.
    """

    p = Path(path).expanduser().resolve()
    info = summarize_snapshot_archive(p)
    if info.parameter_key is None or info.snapshot_key is None:
        raise ValueError(f"Cannot identify parameter/snapshot arrays in {p}. Keys: {info.keys}")

    with np.load(p, allow_pickle=True) as z:
        parameters = np.asarray(z[info.parameter_key])
        snapshots = np.asarray(z[info.snapshot_key])
        theta = None
        if load_theta and info.theta_key is not None:
            theta = np.asarray(z[info.theta_key])

    metadata = load_npz_metadata(p)
    return ROMDataset(
        source_path=p,
        project=info.project,
        parameters=parameters,
        snapshots=snapshots,
        theta_snapshots=theta,
        metadata=metadata,
        info=info,
    )


def iter_project2_archive_infos(locations: DataLocations | None = None) -> Iterator[SnapshotArchiveInfo]:
    """Yield summaries for all Project-2 final archives."""

    loc = locations or load_data_locations()
    for p in project2_final_archives(loc):
        yield summarize_snapshot_archive(p)


def iter_paper1_archive_infos(locations: DataLocations | None = None) -> Iterator[SnapshotArchiveInfo]:
    """Yield summaries for all Paper-1 archive candidates."""

    loc = locations or load_data_locations()
    for p in paper1_archives(loc):
        yield summarize_snapshot_archive(p)


def load_project2_dataset_by_name(
    name_contains: str,
    *,
    locations: DataLocations | None = None,
    load_theta: bool = True,
) -> ROMDataset:
    """Load the first Project-2 archive whose parent folder contains text."""

    loc = locations or load_data_locations()
    needle = str(name_contains).lower()
    for p in project2_final_archives(loc):
        if needle in p.parent.name.lower() or needle in p.name.lower():
            return load_rom_dataset(p, load_theta=load_theta)
    raise FileNotFoundError(f"No Project-2 archive matching {name_contains!r} found in {loc.paper2_baseline_root}")
