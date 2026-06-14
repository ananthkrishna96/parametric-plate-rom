"""Manifest utilities for external snapshot archives."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .data_locations import DataLocations, load_data_locations
from .snapshot_archives import SnapshotArchiveInfo, summarize_snapshot_archive


def discover_npz_archives(root: str | Path, *, skip_tmp: bool = True) -> list[Path]:
    """Recursively discover ``.npz`` archives under ``root``.

    ``skip_tmp`` ignores repository/data subfolders named ``tmp`` or starting
    with ``tmp_``.  It deliberately checks paths relative to ``root`` so a
    system path like ``/tmp/pytest-...`` does not cause false exclusion.
    """

    r = Path(root).expanduser().resolve()
    if not r.exists():
        return []
    paths = sorted(p for p in r.rglob("*.npz") if p.is_file())
    if skip_tmp:
        filtered = []
        for p in paths:
            try:
                rel_parts = p.relative_to(r).parts
            except Exception:
                rel_parts = p.parts
            lowered = [part.lower() for part in rel_parts]
            if any(part == "tmp" or part.startswith("tmp_") for part in lowered):
                continue
            filtered.append(p)
        paths = filtered
    return paths


def project2_final_archives(locations: DataLocations | None = None, *, root: str | Path | None = None) -> list[Path]:
    """Return Project-2 final ``SNAPSHOTS__*.npz`` archives."""

    loc = locations or load_data_locations()
    base = Path(root).expanduser().resolve() if root is not None else loc.paper2_baseline_root
    if not base.exists():
        return []
    return sorted(base.glob("*/SNAPSHOTS__*.npz"))


def paper1_archives(locations: DataLocations | None = None, *, root: str | Path | None = None) -> list[Path]:
    """Return Paper-1 archive candidates, excluding temporary recovery files."""

    loc = locations or load_data_locations()
    base = Path(root).expanduser().resolve() if root is not None else loc.paper1_snapshots_root
    return discover_npz_archives(base, skip_tmp=True)


def build_snapshot_manifest(
    locations: DataLocations | None = None,
    *,
    project: str = "all",
) -> list[SnapshotArchiveInfo]:
    """Build a fresh in-memory manifest by inspecting archive headers."""

    loc = locations or load_data_locations()
    paths: list[Path] = []
    if project in {"all", "paper1"}:
        paths.extend(paper1_archives(loc))
    if project in {"all", "paper2"}:
        paths.extend(project2_final_archives(loc))

    out: list[SnapshotArchiveInfo] = []
    for p in sorted(set(paths)):
        try:
            out.append(summarize_snapshot_archive(p))
        except Exception:
            # Keep manifest generation robust; a corrupted archive can be diagnosed separately.
            continue
    return out


def write_snapshot_manifest(
    rows: Iterable[SnapshotArchiveInfo],
    out_csv: str | Path,
    out_json: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Write manifest rows to CSV and optionally JSON."""

    rows_list = list(rows)
    csv_path = Path(out_csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "project",
        "path",
        "n_samples",
        "parameter_key",
        "parameter_shape",
        "snapshot_key",
        "snapshot_shape",
        "theta_key",
        "theta_shape",
        "keys",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({
                "project": row.project,
                "path": str(row.path),
                "n_samples": row.n_samples,
                "parameter_key": row.parameter_key,
                "parameter_shape": row.parameter_shape,
                "snapshot_key": row.snapshot_key,
                "snapshot_shape": row.snapshot_shape,
                "theta_key": row.theta_key,
                "theta_shape": row.theta_shape,
                "keys": ";".join(row.keys),
            })

    json_path: Path | None = None
    if out_json is not None:
        json_path = Path(out_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for row in rows_list:
            d = asdict(row)
            d["path"] = str(row.path)
            serializable.append(d)
        json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    return csv_path, json_path


def read_snapshot_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV manifest into dictionaries."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
