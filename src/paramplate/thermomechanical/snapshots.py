"""Thermomechanical parameter sampling and accepted-state archives."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from paramplate.core.sampling import latin_hypercube
from paramplate.io.manifest import DatasetManifest, write_manifest

from .fem import ThermomechanicalFEMConfig, ThermomechanicalPlateFOM
from .model import CASES, parameter_mapping, parameter_ranges


def case_lhs(case: int, n_samples: int, *, seed: int = 100) -> np.ndarray:
    return latin_hypercube(parameter_ranges(case), int(n_samples), seed=seed)


def pack_coupling_histories(histories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Store variable-length coupling traces without pickle-backed object arrays."""

    lengths = np.asarray([np.asarray(history).shape[0] for history in histories], dtype=int)
    if not histories:
        return np.empty((0, 0, 3), dtype=float), lengths
    maximum = int(lengths.max(initial=0))
    packed = np.full((len(histories), maximum, 3), np.nan, dtype=float)
    for index, history in enumerate(histories):
        values = np.asarray(history, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Each coupling history must have columns (iteration, error_w, error_theta).")
        packed[index, : values.shape[0], :] = values
    return packed, lengths


def apply_case_parameters(
    base: ThermomechanicalFEMConfig,
    case: int,
    values: np.ndarray,
) -> ThermomechanicalFEMConfig:
    mapping = parameter_mapping(case, values)
    mechanical = base.mechanical
    environment = base.environment
    if "foundation_stiffness" in mapping:
        mechanical = replace(
            mechanical,
            foundation=replace(mechanical.foundation, stiffness=mapping["foundation_stiffness"]),
        )
    if "load_amplitude" in mapping:
        mechanical = replace(mechanical, load=replace(mechanical.load, amplitude=mapping["load_amplitude"]))
    environment_fields = {
        key: value
        for key, value in mapping.items()
        if key
        in {
            "absorbed_solar_flux",
            "ambient_temperature",
            "substrate_temperature",
            "contact_conductance",
            "gap_conductance",
        }
    }
    if environment_fields:
        environment = replace(environment, **environment_fields)
    return replace(base, mechanical=mechanical, environment=environment)


def generate_case_archive(
    base: ThermomechanicalFEMConfig,
    case: int,
    parameters: np.ndarray,
    output: str | Path,
) -> Path:
    table = np.asarray(parameters, dtype=float)
    expected = len(CASES[int(case)].parameter_names)
    if table.ndim != 2 or table.shape[1] != expected:
        raise ValueError(f"Thermomechanical case {case} expects {expected} parameter columns.")
    displacement, theta, iterations, histories = [], [], [], []
    accepted_parameters = []
    rejected: list[int] = []
    rejection_reasons: list[str] = []
    for index, row in enumerate(table):
        try:
            result = ThermomechanicalPlateFOM(apply_case_parameters(base, case, row)).solve()
        except RuntimeError as exc:
            if "converg" not in str(exc).lower():
                raise
            rejected.append(index)
            rejection_reasons.append(str(exc))
            continue
        displacement.append(result.displacement_vector())
        theta.append(result.theta_vector())
        iterations.append(result.coupled_state.iterations)
        histories.append(
            np.asarray(
                [
                    [entry.iteration, entry.displacement_update, entry.thermal_driver_update]
                    for entry in result.coupled_state.history
                ],
                dtype=float,
            )
        )
        accepted_parameters.append(row)
    if not accepted_parameters:
        raise RuntimeError("No thermomechanical sample converged; the archive was not written.")
    packed_histories, history_lengths = pack_coupling_histories(histories)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        parameters=np.asarray(accepted_parameters, dtype=float).reshape(-1, expected),
        displacement_snapshots=np.asarray(displacement, dtype=float),
        thermal_driver_snapshots=np.asarray(theta, dtype=float),
        coupling_iterations=np.asarray(iterations, dtype=int),
        coupling_histories=packed_histories,
        coupling_history_lengths=history_lengths,
        rejected_indices=np.asarray(rejected, dtype=int),
        rejection_reasons=np.asarray(rejection_reasons, dtype=str),
        parameter_names=np.asarray(CASES[int(case)].parameter_names),
        displacement_units=np.asarray("m"),
        thermal_driver_units=np.asarray("K/m"),
    )
    for output_name, key, unit in (
        ("displacement", "displacement_snapshots", "m"),
        ("thermal_driver", "thermal_driver_snapshots", "K/m"),
    ):
        manifest = DatasetManifest(
            study="thermomechanical",
            archive=target.name,
            snapshot_key=key,
            parameter_key="parameters",
            parameter_names=CASES[int(case)].parameter_names,
            output_name=output_name,
            output_unit=unit,
            split_level="sample",
            metric="field-specific finite-element metric",
            notes="Accepted converged coupled states only; displacement and theta remain separate outputs.",
            metadata={
                "case": int(case),
                "rejected_indices": rejected,
                "rejection_reasons": rejection_reasons,
            },
        )
        write_manifest(manifest, target.with_name(f"{target.stem}.{output_name}.manifest.json"))
    return target
