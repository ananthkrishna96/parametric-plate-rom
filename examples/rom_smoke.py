#!/usr/bin/env python3
"""Run POD projection or a lightweight predictive ROM on bundled smoke data."""

from __future__ import annotations

import argparse
from pathlib import Path

from paramplate.rom.datasets import load_snapshot_dataset
from paramplate.rom.suite import run_rom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["mechanical", "thermomechanical", "dynamics"], default="mechanical")
    parser.add_argument("--output-field", choices=["displacement", "thermal_driver"], default="displacement")
    parser.add_argument("--method", choices=["pod-proj", "podi-rbf", "podi-linear", "pod-gpr"], default="pod-proj")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    archive = args.data_root / f"{args.study}_smoke.npz"
    dataset = load_snapshot_dataset(archive, study=args.study, output=args.output_field)
    result = run_rom(
        dataset,
        method=args.method,
        rank=args.rank,
        options={"neighbors": 16, "smoothing": 1.0e-10, "optimizer_restarts": 0, "gpr_fit_limit": 64},
    )
    print(
        f"{result.method}: mean held-out relative error "
        f"{result.summary.mean:.6e} over {result.n_test} states"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
