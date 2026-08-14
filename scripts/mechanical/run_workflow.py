#!/usr/bin/env python3
"""Run a static-mechanics solve, sample campaign, or Navier smoke reference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.mechanical.fem import MechanicalPlateFOM
from paramplate.mechanical.navier import solve_ssss_navier
from paramplate.mechanical.snapshots import direct_lhs, generate_snapshot_archive
from paramplate.postprocessing import save_field_image
from paramplate.workflows import mechanical_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["solve", "snapshots", "navier"], default="solve")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = load_config(args.config, expected_study="mechanical")
    config = mechanical_config(raw)
    if args.dry_run:
        print(json.dumps({"study": "mechanical", "mode": raw["mode"], "config": asdict(config)}, indent=2, default=str))
        return 0
    output = args.output or resolve_repo_path(raw.get("output", "results/mechanical"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "navier":
        solution = solve_ssss_navier(
            config.rigidity,
            length=config.geometry.length,
            width=config.geometry.width,
            foundation_stiffness=config.foundation.stiffness,
            load_amplitude=config.load.amplitude,
            load=config.load.kind if config.load.kind in {"uniform", "patch"} else "patch",
            patch=(config.load.x0, config.load.y0, config.load.size_x, config.load.size_y),
            n_terms=int(raw.get("verification", {}).get("navier_terms", 21)),
            nx=81,
            ny=41,
        )
        target = output if output.suffix == ".npz" else output / "navier_reference.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, x=solution.x, y=solution.y, displacement=solution.displacement)
        save_field_image(
            solution.displacement,
            solution.x,
            solution.y,
            target.with_suffix(".png"),
            label="displacement [m]",
            title="Static Navier smoke reference",
        )
        print(target)
        return 0
    if args.action == "snapshots":
        count = int(raw.get("sampling", {}).get("n_samples", 8))
        samples = direct_lhs(count, seed=int(raw.get("sampling", {}).get("seed", 100)))
        target = output if output.suffix == ".npz" else output / "mechanical_snapshots.npz"
        print(generate_snapshot_archive(config, samples, target))
        return 0
    solution = MechanicalPlateFOM(config).solve()
    target = output if output.suffix == ".npz" else output / "mechanical_solution.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        displacement=solution.vector("global"),
        native_state=solution.vector("native"),
        parameters=np.asarray(list(solution.parameters.values())),
        newton_iterations=np.asarray(solution.newton.iterations),
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
