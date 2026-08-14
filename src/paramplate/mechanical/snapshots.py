"""Static-mechanics sampling and snapshot storage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from paramplate.core.parameters import FoundationParameters, LoadParameters, OrthotropicRigidity
from paramplate.core.sampling import latin_hypercube
from paramplate.io.manifest import DatasetManifest, write_manifest

from .fem import MechanicalFEMConfig, MechanicalPlateFOM
from .model import MechanicalParameters, direct_parameter_ranges


def direct_lhs(n_samples: int, *, seed: int = 100) -> np.ndarray:
    return latin_hypercube(direct_parameter_ranges(), int(n_samples), seed=seed)


def solve_direct_sample(base: MechanicalFEMConfig, values: Iterable[float]):
    parameters = MechanicalParameters.from_direct_vector(values)
    config = replace(
        base,
        rigidity=parameters.rigidity,
        foundation=parameters.foundation,
        load=replace(base.load, amplitude=parameters.load_amplitude),
    )
    return MechanicalPlateFOM(config).solve()


def generate_snapshot_archive(
    base: MechanicalFEMConfig,
    parameters: np.ndarray,
    output: str | Path,
    *,
    include_native: bool = True,
) -> Path:
    """Run a parameter table and store output-space fields and solve metadata."""

    table = np.asarray(parameters, dtype=float)
    if table.ndim != 2 or table.shape[1] != 6:
        raise ValueError("Static direct samples must have six columns.")
    outputs, native, iterations, residuals = [], [], [], []
    for row in table:
        result = solve_direct_sample(base, row)
        outputs.append(result.vector("global"))
        if include_native:
            native.append(result.vector("native"))
        iterations.append(result.newton.iterations)
        residuals.append(result.newton.final_residual)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parameters": table,
        "snapshots": np.asarray(outputs),
        "newton_iterations": np.asarray(iterations, dtype=int),
        "newton_residuals": np.asarray(residuals, dtype=float),
        "parameter_names": np.asarray(["Dx", "Dy", "Dxy", "Ds", "ks", "q"]),
        "output_name": np.asarray("displacement"),
        "output_units": np.asarray("m"),
    }
    if include_native:
        payload["native_snapshots"] = np.asarray(native)
    np.savez_compressed(target, **payload)
    manifest = DatasetManifest(
        study="mechanical",
        archive=target.name,
        snapshot_key="snapshots",
        parameter_key="parameters",
        parameter_names=("Dx", "Dy", "Dxy", "Ds", "ks", "q"),
        output_name="displacement",
        output_unit="m",
        split_level="sample",
        expected_shape=tuple(np.asarray(outputs).shape),
        metric="H1-type",
        notes="Upward-positive displacement; downward loads are negative.",
        metadata={"n_samples": int(table.shape[0]), "native_snapshots": bool(include_native)},
    )
    write_manifest(manifest, target.with_suffix(".manifest.json"))
    return target
