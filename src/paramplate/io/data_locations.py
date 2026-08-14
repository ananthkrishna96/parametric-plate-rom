"""External research-data locations without machine-specific defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DataLocations:
    root: Path
    mechanical: Path
    thermomechanical: Path
    dynamics: Path

    def for_study(self, study: str) -> Path:
        key = study.lower().strip()
        if key in {"mechanical", "static"}:
            return self.mechanical
        if key in {"thermomechanical", "thermo"}:
            return self.thermomechanical
        if key in {"dynamics", "transient"}:
            return self.dynamics
        raise KeyError(f"Unknown study {study!r}.")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_data_locations(path: str | Path | None = None) -> DataLocations:
    """Load the optional local data-location file.

    Resolution order is an explicit path, ``PARAMPLATE_DATA_CONFIG``, then
    ``configs/data_locations.local.yml``. The local file is intentionally
    ignored by Git. When no file is present, ``PARAMPLATE_DATA_ROOT`` is used;
    otherwise an informative error is raised.
    """

    candidate = path or os.environ.get("PARAMPLATE_DATA_CONFIG")
    if candidate is None:
        candidate = Path.cwd() / "configs" / "data_locations.local.yml"
    cfg_path = Path(candidate).expanduser().resolve()

    if cfg_path.is_file():
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Data-location file must be a mapping: {cfg_path}")
        root_raw = payload.get("data_root") or os.environ.get("PARAMPLATE_DATA_ROOT")
        if root_raw is None:
            raise ValueError(f"data_root is missing from {cfg_path}.")
        root = Path(str(root_raw)).expanduser().resolve()
        return DataLocations(
            root=root,
            mechanical=_resolve(root, payload.get("mechanical", "mechanical")),
            thermomechanical=_resolve(root, payload.get("thermomechanical", "thermomechanical")),
            dynamics=_resolve(root, payload.get("dynamics", "dynamics")),
        )

    root_env = os.environ.get("PARAMPLATE_DATA_ROOT")
    if root_env is None:
        raise FileNotFoundError(
            "No external-data location is configured. Copy "
            "configs/data_locations.template.yml to configs/data_locations.local.yml "
            "or set PARAMPLATE_DATA_ROOT."
        )
    root = Path(root_env).expanduser().resolve()
    return DataLocations(root, root / "mechanical", root / "thermomechanical", root / "dynamics")
