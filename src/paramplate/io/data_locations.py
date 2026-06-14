"""Machine-independent data-location helpers for :mod:`paramplate`.

The source repository should not contain large snapshot archives.  This module
locates the external data root from, in order of priority:

1. an explicit ``--data-root`` / function argument,
2. the ``PARAMPLATE_DATA_ROOT`` environment variable,
3. ``configs/data_locations.local.yml`` in the repository root,
4. the default sibling path ``~/Documents/PHD_WORKS/parametric-plate-rom-data``.

The committed file ``configs/data_locations.template.yml`` documents the
expected structure.  The local YAML file is intentionally ignored by Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

try:  # PyYAML is a project dependency, but keep the error message explicit.
    import yaml
except Exception:  # pragma: no cover - exercised only if dependency is absent
    yaml = None


DEFAULT_DATA_ROOT = Path.home() / "Documents" / "PHD_WORKS" / "parametric-plate-rom-data"


def repo_root_from_module() -> Path:
    """Return the repository root for an editable/source checkout.

    For this source layout the file is ``src/paramplate/io/data_locations.py``;
    therefore ``parents[3]`` is the repository root.  A fallback upward search is
    included so the helper still works if the layout changes slightly.
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class DataLocations:
    """Resolved external data locations used by ROM and SNAPGEN workflows."""

    data_root: Path
    paper1_snapshots_root: Path
    paper1_rom_root: Path
    paper1_results_root: Path
    paper1_logs_root: Path
    paper2_snapshots_root: Path
    paper2_campaigns_root: Path
    paper2_baseline_root: Path
    paper2_addon_root: Path
    paper2_merged_root: Path
    paper2_rom_root: Path
    paper2_convergence_root: Path
    paper2_results_root: Path
    paper2_logs_root: Path
    shared_manifests_root: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}

    def ensure_dirs(self) -> None:
        """Create the standard external directory tree if it does not exist."""

        for path in self.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configuration files.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def _path_from(root: Path, value: str | os.PathLike[str] | None, default_rel: str) -> Path:
    raw = Path(os.path.expanduser(str(value if value is not None else default_rel)))
    return raw if raw.is_absolute() else root / raw


def load_data_locations(
    *,
    data_root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> DataLocations:
    """Load and resolve all external data paths.

    Parameters
    ----------
    data_root:
        Optional explicit root.  Overrides environment variables and YAML.
    config_path:
        Optional YAML path.  Defaults to ``configs/data_locations.local.yml``
        when it exists, otherwise ``configs/data_locations.template.yml``.
    repo_root:
        Optional repository root.  Mainly useful in tests.
    """

    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else repo_root_from_module()

    if config_path is None:
        local = repo / "configs" / "data_locations.local.yml"
        template = repo / "configs" / "data_locations.template.yml"
        config = local if local.exists() else template
    else:
        config = Path(config_path).expanduser().resolve()

    cfg = _read_yaml(config)

    cfg_root = cfg.get("data_root")
    if isinstance(cfg_root, str) and cfg_root.strip().startswith("/absolute/path"):
        cfg_root = None

    root_raw = (
        data_root
        if data_root is not None
        else os.environ.get("PARAMPLATE_DATA_ROOT")
        or cfg_root
        or DEFAULT_DATA_ROOT
    )
    root = Path(os.path.expanduser(str(root_raw)))
    if not root.is_absolute():
        root = repo / root
    root = root.resolve()

    paper1: Mapping[str, Any] = cfg.get("paper1", {}) if isinstance(cfg.get("paper1", {}), Mapping) else {}
    paper2: Mapping[str, Any] = cfg.get("paper2", {}) if isinstance(cfg.get("paper2", {}), Mapping) else {}

    paper1_snapshots = _path_from(root, paper1.get("snapshots_root"), "paper1_mechanical/snapshots")
    paper2_campaigns = _path_from(root, paper2.get("campaigns_root"), "paper2_thermomechanical/campaigns")

    return DataLocations(
        data_root=root,
        paper1_snapshots_root=paper1_snapshots,
        paper1_rom_root=_path_from(root, paper1.get("rom_root"), "paper1_mechanical/rom_training"),
        paper1_results_root=_path_from(root, paper1.get("results_root"), "paper1_mechanical/results"),
        paper1_logs_root=_path_from(root, paper1.get("logs_root"), "paper1_mechanical/logs"),
        paper2_snapshots_root=_path_from(root, paper2.get("snapshots_root"), "paper2_thermomechanical/snapshots"),
        paper2_campaigns_root=paper2_campaigns,
        paper2_baseline_root=_path_from(root, paper2.get("baseline_root"), "paper2_thermomechanical/campaigns/baseline"),
        paper2_addon_root=_path_from(root, paper2.get("addon_root"), "paper2_thermomechanical/campaigns/addon_gapfill"),
        paper2_merged_root=_path_from(root, paper2.get("merged_root"), "paper2_thermomechanical/campaigns/merged"),
        paper2_rom_root=_path_from(root, paper2.get("rom_root"), "paper2_thermomechanical/rom_training"),
        paper2_convergence_root=_path_from(root, paper2.get("convergence_root"), "paper2_thermomechanical/convergence"),
        paper2_results_root=_path_from(root, paper2.get("results_root"), "paper2_thermomechanical/results"),
        paper2_logs_root=_path_from(root, paper2.get("logs_root"), "paper2_thermomechanical/logs"),
        shared_manifests_root=root / "shared" / "manifests",
    )
