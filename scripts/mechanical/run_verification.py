#!/usr/bin/env python3
"""Run static Navier, external-grid, or panel-to-monolithic verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.mechanical.verification import (
    compare_fom_with_external_grid,
    compare_fom_with_navier,
    compare_panel_with_monolithic,
)
from paramplate.postprocessing import save_field_image
from paramplate.workflows import mechanical_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["navier", "external", "panel"], default="navier")
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = load_config(args.config, expected_study="mechanical")
    config = mechanical_config(raw)
    verification = raw.get("verification", {})
    plan = {
        "study": "mechanical",
        "mode": args.mode,
        "config": str(args.config),
        "reference_config": None if args.reference_config is None else str(args.reference_config),
        "reference_archive": None if args.reference_archive is None else str(args.reference_archive),
        "grid": {
            "nx": int(verification.get("nx", 121)),
            "ny": int(verification.get("ny", 61)),
        },
        "navier_terms": int(verification.get("navier_terms", 31)),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if args.mode == "navier":
        result = compare_fom_with_navier(
            config,
            n_terms=plan["navier_terms"],
            nx=plan["grid"]["nx"],
            ny=plan["grid"]["ny"],
        )
    elif args.mode == "external":
        if args.reference_archive is None:
            parser.error("--reference-archive is required for --mode external")
        result = compare_fom_with_external_grid(config, args.reference_archive)
    else:
        if args.reference_config is None:
            parser.error("--reference-config is required for --mode panel")
        reference_raw = load_config(args.reference_config, expected_study="mechanical")
        result = compare_panel_with_monolithic(
            config,
            mechanical_config(reference_raw),
            nx=plan["grid"]["nx"],
            ny=plan["grid"]["ny"],
        )

    output = args.output or resolve_repo_path(raw.get("verification_output", "results/mechanical/verification"))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "comparison.npz",
        x=result.x,
        y=result.y,
        reference=result.reference,
        approximation=result.approximation,
        absolute_error=result.absolute_error,
    )
    summary = {"reference": result.reference_name, **result.summary.to_dict()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_field_image(
        result.reference,
        result.x,
        result.y,
        output / "reference.png",
        label="displacement [m]",
        title=f"Reference: {result.reference_name}",
    )
    save_field_image(
        result.approximation,
        result.x,
        result.y,
        output / "approximation.png",
        label="displacement [m]",
        title="Mechanical FOM",
    )
    save_field_image(
        result.absolute_error,
        result.x,
        result.y,
        output / "absolute_error.png",
        label="absolute error [m]",
        title="Absolute displacement error",
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
