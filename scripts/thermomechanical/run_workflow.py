#!/usr/bin/env python3
"""Run the steady coupled thermomechanical FOM or an accepted-state campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.thermomechanical.fem import ThermomechanicalPlateFOM
from paramplate.thermomechanical.snapshots import case_lhs, generate_case_archive
from paramplate.workflows import thermomechanical_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["solve", "snapshots"], default="solve")
    parser.add_argument(
        "--case",
        type=int,
        choices=[2, 3, 5],
        help="Override the case declared by the configuration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = load_config(args.config, expected_study="thermomechanical")
    config = thermomechanical_config(raw)
    sampling = raw.get("sampling", {})
    selected_case = int(args.case if args.case is not None else raw.get("case", 2))
    requested_samples = int(
        sampling.get("requested_samples", sampling.get("n_samples", 6))
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "thermomechanical",
                    "mode": raw["mode"],
                    "case": selected_case,
                    "requested_samples": requested_samples,
                    "config": asdict(config),
                },
                indent=2,
                default=str,
            )
        )
        return 0
    output = args.output or resolve_repo_path(raw.get("output", "results/thermomechanical"))
    if args.action == "snapshots":
        samples = case_lhs(
            selected_case,
            requested_samples,
            seed=int(sampling.get("seed", 100)),
        )
        target = (
            output
            if output.suffix == ".npz"
            else output / f"thermomechanical_case{selected_case}.npz"
        )
        print(generate_case_archive(config, selected_case, samples, target))
        return 0
    solution = ThermomechanicalPlateFOM(config).solve()
    target = output if output.suffix == ".npz" else output / "thermomechanical_solution.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        displacement=solution.displacement_vector(),
        thermal_driver=solution.theta_vector(),
        coupling_iterations=np.asarray(solution.coupled_state.iterations),
        coupling_history=np.asarray(
            [[x.iteration, x.displacement_update, x.thermal_driver_update] for x in solution.coupled_state.history]
        ),
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
