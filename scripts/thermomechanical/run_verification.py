#!/usr/bin/env python3
"""Run steady thermomechanical analytical or external-reference checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.postprocessing import save_field_image, save_probe_history
from paramplate.thermomechanical.verification import (
    compare_coupled_fom_with_auxiliary_navier,
    compare_coupled_fom_with_external_fields,
    linear_profile_driver,
)
from paramplate.workflows import thermomechanical_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["thermal-driver", "navier", "external"],
        default="navier",
    )
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = load_config(args.config, expected_study="thermomechanical")
    config = thermomechanical_config(raw)
    verification = dict(raw.get("verification", {}))
    grid = dict(verification.get("grid", {}))
    projection = dict(verification.get("projection_grid", grid))
    plan = {
        "study": "thermomechanical",
        "mode": args.mode,
        "config": str(args.config),
        "reference_archive": None if args.reference_archive is None else str(args.reference_archive),
        "grid": {"nx": int(grid.get("nx", 121)), "ny": int(grid.get("ny", 61))},
        "projection_grid": {
            "nx": int(projection.get("nx", grid.get("nx", 121))),
            "ny": int(projection.get("ny", grid.get("ny", 61))),
        },
        "navier_terms": int(verification.get("navier_terms", 31)),
        "effective_foundation": float(
            verification.get("effective_foundation", config.mechanical.foundation.stiffness)
        ),
        "thermal_driver_gradient": float(verification.get("thermal_driver_gradient", 18.0)),
        "theta_quadrature_points": int(config.theta_quadrature_points),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output = args.output or resolve_repo_path(
        raw.get("verification_output", "results/thermomechanical/verification")
    )
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "thermal-driver":
        gradient = plan["thermal_driver_gradient"]
        recovered = linear_profile_driver(
            gradient,
            config.mechanical.geometry.thickness,
            quadrature_points=config.theta_quadrature_points,
        )
        summary = {
            "reference_gradient_K_per_m": gradient,
            "recovered_theta_K_per_m": recovered,
            "absolute_error_K_per_m": abs(recovered - gradient),
        }
        (output / "thermal_driver_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    if args.mode == "external":
        if args.reference_archive is None:
            parser.error("--reference-archive is required for --mode external")
        result = compare_coupled_fom_with_external_fields(config, args.reference_archive)
        np.savez_compressed(
            output / "external_comparison.npz",
            x=result.x,
            y=result.y,
            reference_displacement=result.reference_displacement,
            finite_element_displacement=result.finite_element_displacement,
            displacement_absolute_error=np.abs(
                result.finite_element_displacement - result.reference_displacement
            ),
            reference_thermal_driver=result.reference_thermal_driver,
            finite_element_thermal_driver=result.finite_element_thermal_driver,
            thermal_driver_absolute_error=np.abs(
                result.finite_element_thermal_driver - result.reference_thermal_driver
            ),
        )
        summary = result.metadata()
        (output / "external_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    result = compare_coupled_fom_with_auxiliary_navier(
        config,
        n_terms=plan["navier_terms"],
        nx=plan["grid"]["nx"],
        ny=plan["grid"]["ny"],
        projection_nx=plan["projection_grid"]["nx"],
        projection_ny=plan["projection_grid"]["ny"],
        foundation_stiffness=plan["effective_foundation"],
    )
    np.savez_compressed(
        output / "navier_comparison.npz",
        x=result.x,
        y=result.y,
        thermal_driver=result.thermal_driver,
        navier_reference=result.reference_displacement,
        finite_element_displacement=result.finite_element_displacement,
        absolute_error=result.absolute_error,
        displacement_updates=result.displacement_updates,
        thermal_driver_updates=result.thermal_driver_updates,
    )
    summary = result.metadata()
    (output / "navier_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_field_image(
        result.reference_displacement,
        result.x,
        result.y,
        output / "navier_reference.png",
        label="displacement [m]",
        title="Auxiliary Navier reference",
    )
    save_field_image(
        result.finite_element_displacement,
        result.x,
        result.y,
        output / "finite_element_displacement.png",
        label="displacement [m]",
        title="Coupled finite-element displacement",
    )
    save_field_image(
        result.absolute_error,
        result.x,
        result.y,
        output / "absolute_error.png",
        label="absolute error [m]",
        title="Absolute displacement difference",
    )
    iterations = np.arange(1, result.displacement_updates.size + 1, dtype=float)
    save_probe_history(
        iterations,
        result.displacement_updates,
        output / "displacement_updates.png",
        xlabel="coupling iteration",
        ylabel="relative displacement update",
    )
    finite_theta = np.where(np.isfinite(result.thermal_driver_updates), result.thermal_driver_updates, np.nan)
    save_probe_history(
        iterations,
        finite_theta,
        output / "thermal_driver_updates.png",
        xlabel="coupling iteration",
        ylabel="relative thermal-driver update",
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
