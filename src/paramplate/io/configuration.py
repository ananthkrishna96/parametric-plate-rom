"""YAML configuration loading with explicit repository-relative paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a workflow configuration is malformed."""


def repository_root(start: str | Path | None = None) -> Path:
    """Locate the repository root from a path inside the working tree."""

    current = Path(start or __file__).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "paramplate").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the parametric-plate-rom repository root.")


def resolve_repo_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a repository-relative path without accepting shell expansion."""

    raw = str(value)
    if raw.startswith("~"):
        raise ConfigurationError("Home-directory paths are not permitted in shared configuration files.")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return ((Path(root).resolve() if root is not None else repository_root()) / path).resolve()


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_environment(v) for k, v in value.items()}
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge workflow mappings without mutating either input."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_path(raw: str | Path, current: Path) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate.resolve()
    adjacent = (current.parent / candidate).resolve()
    if adjacent.is_file():
        return adjacent
    return (repository_root(current) / candidate).resolve()


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ConfigurationError(f"Cyclic base_config chain: {chain}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"Configuration must be a mapping: {config_path}")
    current = dict(_expand_environment(dict(payload)))
    base_reference = current.pop("base_config", None)
    if base_reference is None:
        merged = current
    else:
        base = _load_config_tree(_base_path(base_reference, config_path), (*stack, config_path))
        base.pop("_config_path", None)
        merged = _deep_merge(base, current)
    merged["_config_path"] = str(config_path)
    return merged


def load_config(path: str | Path, *, expected_study: str | None = None) -> dict[str, Any]:
    """Read, compose, and minimally validate a workflow YAML file.

    A configuration may declare ``base_config``.  Nested mappings are merged so a
    case file can override only the fields that differ from the shared thesis setup.
    The base path may be repository-relative or adjacent to the case file.
    """

    config_path = Path(path).expanduser().resolve()
    config = _load_config_tree(config_path)
    for key in ("study", "mode"):
        if key not in config:
            raise ConfigurationError(f"Missing required key {key!r} in {config_path}.")
    if expected_study is not None and str(config["study"]).lower() != str(expected_study).lower():
        raise ConfigurationError(
            f"Expected study {expected_study!r}, found {config['study']!r} in {config_path}."
        )
    if str(config["mode"]).lower() not in {"smoke", "thesis", "full"}:
        raise ConfigurationError("mode must be smoke, thesis, or full.")
    return config
