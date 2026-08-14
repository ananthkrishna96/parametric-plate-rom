#!/usr/bin/env python3
"""Train and assess one field-specific ROM on an existing snapshot archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paramplate.postprocessing import save_error_histogram
from paramplate.rom.datasets import load_snapshot_dataset
from paramplate.rom.suite import run_rom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--study", choices=["mechanical", "thermomechanical", "dynamics"], required=True)
    parser.add_argument("--output-field", choices=["displacement", "thermal_driver"], default="displacement")
    parser.add_argument(
        "--method",
        choices=["pod-proj", "podi-rbf", "podi-linear", "pod-gpr", "pod-nn", "pod-dl-rom", "dl-rom"],
        required=True,
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--metric", type=Path, help="Optional .npy finite-element metric matrix")
    parser.add_argument("--options", type=Path, help="Optional JSON method options")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    metric = np.load(args.metric, allow_pickle=False) if args.metric else None
    options = json.loads(args.options.read_text(encoding="utf-8")) if args.options else {}
    dataset = load_snapshot_dataset(args.archive, study=args.study, output=args.output_field)
    result = run_rom(
        dataset,
        method=args.method,
        rank=args.rank,
        metric=metric,
        metric_name="external finite-element metric" if metric is not None else "Euclidean",
        options=options,
    )
    report = result.report()
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_error_histogram(result.errors, args.report.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
