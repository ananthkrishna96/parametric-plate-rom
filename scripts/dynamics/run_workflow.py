#!/usr/bin/env python3
"""Run one transient trajectory, modal check, or complete-trajectory campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from paramplate.core.sampling import configured_parameter_ranges
from paramplate.dynamics.fem import TransientPlateFOM
from paramplate.dynamics.model import parameter_ranges
from paramplate.dynamics.trajectories import campaign_lhs, generate_trajectory_archive
from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.workflows import transient_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["solve", "modal", "snapshots"], default="solve")
    parser.add_argument(
        "--campaign",
        choices=["case2_monolithic", "case2_one_interface", "case3_monolithic"],
        help="Override the campaign declared by the configuration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = load_config(args.config, expected_study="dynamics")
    config = transient_config(raw)
    selected_campaign = str(args.campaign or raw.get("campaign", "case2_monolithic"))
    sampling = raw.get("sampling", {})
    ranges = configured_parameter_ranges(sampling, parameter_ranges(selected_campaign))
    requested_trajectories = int(sampling.get("n_trajectories", 4))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "dynamics",
                    "mode": raw["mode"],
                    "campaign": selected_campaign,
                    "requested_trajectories": requested_trajectories,
                    "sampling_ranges": [item.to_dict() for item in ranges],
                    "config": asdict(config),
                },
                indent=2,
                default=str,
            )
        )
        return 0
    output = args.output or resolve_repo_path(raw.get("output", "results/dynamics"))
    if args.action == "snapshots":
        samples = campaign_lhs(
            selected_campaign,
            requested_trajectories,
            seed=int(sampling.get("seed", 100)),
            ranges=ranges,
        )
        target = (
            output
            if output.suffix == ".npz"
            else output / f"{selected_campaign}_trajectories.npz"
        )
        print(generate_trajectory_archive(config, selected_campaign, samples, target))
        return 0
    solver = TransientPlateFOM(config)
    if args.action == "modal":
        frequencies = solver.modal_frequencies(int(raw.get("verification", {}).get("n_modes", 8)))
        print(json.dumps({"frequencies_hz": frequencies.tolist()}, indent=2))
        return 0
    solution = solver.solve()
    target = output if output.suffix == ".npz" else output / "transient_trajectory.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        times=solution.stored_times,
        snapshots=solution.stored_output_snapshots,
        full_times=solution.active_result.times,
        mechanical_energy=solution.active_result.mechanical_energy,
        active_dofs=solution.active_system.active_dofs,
        removed_zero_mass_dofs=solution.active_system.removed_zero_mass_dofs,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
