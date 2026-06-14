#!/usr/bin/env python3
"""Check that Paper-1 and Project-2 ROM snapshot archives are readable.

Examples
--------

    python scripts/project2/check_rom_dataset.py --project paper2
    python scripts/project2/check_rom_dataset.py --project all --write-manifest
    PARAMPLATE_DATA_ROOT=/path/to/parametric-plate-rom-data \
        python scripts/project2/check_rom_dataset.py --project paper2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None

from paramplate.io.data_locations import load_data_locations
from paramplate.io.snapshot_archives import SnapshotArchiveInfo, summarize_snapshot_archive
from paramplate.io.snapshot_manifest import (
    build_snapshot_manifest,
    paper1_archives,
    project2_final_archives,
    write_snapshot_manifest,
)


def _fmt_shape(shape):
    return "" if shape is None else "x".join(str(x) for x in shape)


def _rows(infos: Iterable[SnapshotArchiveInfo], data_root: Path):
    for info in infos:
        try:
            rel = info.path.relative_to(data_root)
        except Exception:
            rel = info.path
        yield [
            info.project,
            info.n_samples if info.n_samples is not None else "?",
            _fmt_shape(info.parameter_shape),
            _fmt_shape(info.snapshot_shape),
            _fmt_shape(info.theta_shape),
            str(rel),
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", choices=["paper1", "paper2", "all"], default="paper2")
    ap.add_argument("--data-root", default=None, help="Override external data root.")
    ap.add_argument("--config", default=None, help="Optional data_locations YAML file.")
    ap.add_argument("--write-manifest", action="store_true", help="Write shared/manifests/rom_snapshot_manifest.*")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if no matching archives are found.")
    args = ap.parse_args()

    loc = load_data_locations(data_root=args.data_root, config_path=args.config)
    print(f"data_root: {loc.data_root}")

    paths: list[Path] = []
    if args.project in {"paper1", "all"}:
        p1 = paper1_archives(loc)
        print(f"paper1 archive candidates: {len(p1)}")
        paths.extend(p1)
    if args.project in {"paper2", "all"}:
        p2 = project2_final_archives(loc)
        print(f"paper2 final archives: {len(p2)}")
        paths.extend(p2)

    infos: list[SnapshotArchiveInfo] = []
    failures: list[tuple[Path, str]] = []
    for p in sorted(paths):
        try:
            infos.append(summarize_snapshot_archive(p))
        except Exception as exc:
            failures.append((p, str(exc)))

    headers = ["project", "n", "parameters", "snapshots", "theta", "archive"]
    table_rows = list(_rows(infos, loc.data_root))
    if tabulate is not None:
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
    else:
        print("\t".join(headers))
        for row in table_rows:
            print("\t".join(map(str, row)))

    if failures:
        print("\nFailed archives:")
        for p, msg in failures:
            print(f"  - {p}: {msg}")

    if args.write_manifest:
        rows = build_snapshot_manifest(loc, project=args.project)
        out_csv = loc.shared_manifests_root / "rom_snapshot_manifest.csv"
        out_json = loc.shared_manifests_root / "rom_snapshot_manifest.json"
        csv_path, json_path = write_snapshot_manifest(rows, out_csv, out_json)
        print(f"\nwrote manifest rows: {len(rows)}")
        print(f"csv : {csv_path}")
        print(f"json: {json_path}")

    if args.strict and not infos:
        return 2
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
