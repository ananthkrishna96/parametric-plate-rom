"""Dataset manifests and leakage-sensitive schema validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DatasetManifest:
    study: str
    archive: str
    snapshot_key: str
    parameter_key: str
    parameter_names: tuple[str, ...]
    output_name: str
    output_unit: str
    split_level: str
    expected_shape: tuple[int, ...] | None = None
    trajectory_id_key: str | None = None
    time_key: str | None = None
    metric: str | None = None
    sha256: str | None = None
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_names"] = list(self.parameter_names)
        payload["expected_shape"] = None if self.expected_shape is None else list(self.expected_shape)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DatasetManifest":
        data = dict(values)
        data["parameter_names"] = tuple(data.get("parameter_names", ()))
        if data.get("expected_shape") is not None:
            data["expected_shape"] = tuple(int(x) for x in data["expected_shape"])
        return cls(**data)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_manifest(manifest: DatasetManifest, *, archive_root: str | Path | None = None) -> list[str]:
    """Return a list of manifest problems; an empty list means valid."""

    errors: list[str] = []
    if manifest.study not in {"mechanical", "thermomechanical", "dynamics"}:
        errors.append(f"Unknown study {manifest.study!r}.")
    if not manifest.snapshot_key or not manifest.parameter_key:
        errors.append("snapshot_key and parameter_key are required.")
    if manifest.split_level not in {"sample", "trajectory"}:
        errors.append("split_level must be sample or trajectory.")
    if manifest.split_level == "trajectory" and not manifest.trajectory_id_key:
        errors.append("trajectory_id_key is required for trajectory-level splitting.")
    if not manifest.parameter_names:
        errors.append("parameter_names cannot be empty.")

    archive = Path(manifest.archive)
    if archive_root is not None and not archive.is_absolute():
        archive = Path(archive_root) / archive
    if archive.exists() and manifest.sha256 and sha256_file(archive) != manifest.sha256:
        errors.append(f"SHA-256 mismatch for {archive}.")
    return errors


def ensure_unique_memberships(groups: Iterable[Iterable[int]]) -> None:
    """Raise when an index occurs in more than one statistical subset."""

    seen: set[int] = set()
    for group in groups:
        current = {int(x) for x in group}
        overlap = seen.intersection(current)
        if overlap:
            raise ValueError(f"Split leakage detected for indices: {sorted(overlap)[:10]}")
        seen.update(current)
