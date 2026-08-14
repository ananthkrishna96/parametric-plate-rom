#!/usr/bin/env python3
"""Run transient frequency, time-integration, energy, and panel checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from paramplate.core.parameters import LoadParameters, RayleighDamping
from paramplate.dynamics.fem import TransientPlateFOM
from paramplate.dynamics.verification import (
    compare_frequencies,
    compare_histories,
    modal_harmonic_response,
    ordered_navier_frequencies,
    relative_energy_drift,
    relative_energy_range,
    single_mode_step_reference,
)
from paramplate.io.configuration import load_config, resolve_repo_path
from paramplate.postprocessing import save_energy_history, save_probe_history
from paramplate.workflows import transient_config


def _point(values: object, default: tuple[float, float]) -> tuple[float, float]:
    raw = default if values is None else tuple(values)  # type: ignore[arg-type]
    if len(raw) != 2:
        raise ValueError("Probe coordinates must contain exactly two values.")
    return float(raw[0]), float(raw[1])


def _modal_load(raw: dict[str, object], *, amplitude: float) -> LoadParameters:
    load = dict(raw.get("load", {}))
    return LoadParameters(
        kind="modal_sine",
        amplitude=float(amplitude),
        x0=float(load.get("x0", 1.0)),
        y0=float(load.get("y0", 0.5)),
        size_x=float(load.get("size_x", 0.35)),
        size_y=float(load.get("size_y", 0.35)),
    )


def _patch_load(verification: dict[str, object]) -> LoadParameters:
    patch = dict(verification.get("patch_load", {}))
    return LoadParameters(
        kind="patch",
        amplitude=float(patch.get("amplitude", 1500.0)),
        x0=float(patch.get("x0", 1.0)),
        y0=float(patch.get("y0", 0.5)),
        size_x=float(patch.get("size_x", 0.35)),
        size_y=float(patch.get("size_y", 0.35)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["frequencies", "single-mode", "modal", "energy", "harmonic", "patch-modal", "panel"],
        default="frequencies",
    )
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--foundation-branch", choices=["none", "compression", "uplift"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = load_config(args.config, expected_study="dynamics")
    base = transient_config(raw)
    verification = dict(raw.get("verification", {}))
    center_probe = _point(verification.get("center_probe"), (1.0, 0.5))
    panel_probe = _point(verification.get("panel_probe"), (0.73, 0.37))
    n_modes = int(verification.get("n_modes", 8))
    modal_modes = int(verification.get("patch_modal_modes", 12))
    search_order = int(verification.get("navier_search_order", 12))
    branch = str(args.foundation_branch or verification.get("default_branch", "compression"))
    regimes = tuple(str(item) for item in verification.get("frequency_regimes", ["none", "compression", "uplift"]))
    plan = {
        "study": "dynamics",
        "mode": args.mode,
        "config": str(args.config),
        "reference_config": None if args.reference_config is None else str(args.reference_config),
        "foundation_branch": branch,
        "frequency_regimes": list(regimes),
        "n_modes": n_modes,
        "patch_modal_modes": modal_modes,
        "center_probe": list(center_probe),
        "panel_probe": list(panel_probe),
        "time_step": base.time_step,
        "final_time": base.final_time,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output = args.output or resolve_repo_path(raw.get("verification_output", "results/dynamics/verification"))
    output.mkdir(parents=True, exist_ok=True)
    geometry = base.mechanical.geometry

    if args.mode == "frequencies":
        summaries: dict[str, object] = {}
        arrays: dict[str, np.ndarray] = {}
        for regime in regimes:
            config = replace(base, foundation_branch=regime)
            solver = TransientPlateFOM(config)
            finite_element = solver.modal_frequencies(n_modes)
            effective = config.mechanical.foundation.effective_dynamic_stiffness(regime)
            reference = ordered_navier_frequencies(
                n_modes,
                length=geometry.length,
                width=geometry.width,
                thickness=geometry.thickness,
                density=config.density,
                rigidity=config.mechanical.rigidity,
                foundation_stiffness=effective,
                search_order=search_order,
            )
            comparison = compare_frequencies(finite_element, reference)
            summaries[regime] = {
                "effective_foundation_N_per_m3": effective,
                "active_dofs": int(solver.active.stiffness.shape[0]),
                **comparison.to_dict(),
            }
            arrays[f"{regime}_finite_element_hz"] = comparison.finite_element_hz
            arrays[f"{regime}_reference_hz"] = comparison.reference_hz
            arrays[f"{regime}_relative_errors"] = comparison.relative_errors
            arrays[f"{regime}_mode_pairs"] = comparison.mode_pairs
        np.savez_compressed(output / "frequency_comparison.npz", **arrays)
        (output / "frequency_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(output), "regimes": summaries}, indent=2))
        return 0

    if args.mode == "panel":
        if args.reference_config is None:
            parser.error("--reference-config is required for --mode panel")
        monolithic_raw = load_config(args.reference_config, expected_study="dynamics")
        panel_config = replace(base, foundation_branch=branch)
        monolithic_config = replace(transient_config(monolithic_raw), foundation_branch=branch)
        panel_solver = TransientPlateFOM(panel_config)
        monolithic_solver = TransientPlateFOM(monolithic_config)
        load = _patch_load(verification)
        panel_solution = panel_solver.solve(load=load)
        monolithic_solution = monolithic_solver.solve(load=load)
        panel_history = panel_solver.active_probe_history(
            panel_solution.active_result.displacement, panel_probe
        )
        monolithic_history = monolithic_solver.active_probe_history(
            monolithic_solution.active_result.displacement, panel_probe
        )
        comparison = compare_histories(monolithic_history, panel_history)
        np.savez_compressed(
            output / "panel_comparison.npz",
            times=monolithic_solution.active_result.times,
            monolithic=monolithic_history,
            panel=panel_history,
        )
        summary = {
            **comparison.to_dict(),
            "probe_m": list(panel_probe),
            "monolithic_active_dofs": int(monolithic_solver.active.stiffness.shape[0]),
            "panel_active_dofs": int(panel_solver.active.stiffness.shape[0]),
            "panel_removed_zero_mass_dofs": int(panel_solver.active.removed_zero_mass_dofs.size),
        }
        (output / "panel_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        save_probe_history(
            monolithic_solution.active_result.times,
            monolithic_history,
            output / "monolithic_history.png",
        )
        save_probe_history(
            panel_solution.active_result.times,
            panel_history,
            output / "panel_history.png",
        )
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    config = replace(base, foundation_branch=branch)
    if args.mode == "energy":
        config = replace(
            config,
            foundation_branch="none",
            damping=RayleighDamping(0.0, 0.0),
        )
        solver = TransientPlateFOM(config)
        amplitude = float(verification.get("energy_initial_amplitude", 1.0e-4))
        solution = solver.free_vibration(mode_amplitude=amplitude)
        energy = solution.active_result.mechanical_energy
        summary = {
            "relative_energy_drift_from_initial": relative_energy_drift(energy),
            "relative_energy_range": relative_energy_range(energy),
            "initial_amplitude_m": amplitude,
            "active_dofs": int(solver.active.stiffness.shape[0]),
        }
        np.savez_compressed(
            output / "energy_history.npz",
            times=solution.active_result.times,
            kinetic=solution.active_result.kinetic_energy,
            potential=solution.active_result.potential_energy,
            mechanical=energy,
        )
        (output / "energy_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        save_energy_history(solution.active_result.times, energy, output / "energy_history.png")
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    solver = TransientPlateFOM(config)
    if args.mode == "harmonic":
        analysis = solver.modal_analysis(n_modes)
        amplitude = float(verification.get("single_mode_load", 1000.0))
        load = _modal_load(raw, amplitude=amplitude)
        spatial_load = solver.spatial_load(load)
        reduced_load = analysis.modes.T @ spatial_load
        probe_values = solver.mode_probe_values(analysis, center_probe)
        ratio = verification.get("harmonic_ratio", [0.20, 2.50])
        ratio_low, ratio_high = float(ratio[0]), float(ratio[1])  # type: ignore[index]
        count = int(verification.get("harmonic_points", 180))
        excitation = np.linspace(ratio_low, ratio_high, count) * analysis.angular_frequencies[0]
        damping_ratio = float(verification.get("harmonic_damping_ratio", 0.02))
        response = modal_harmonic_response(
            analysis.angular_frequencies,
            reduced_load,
            probe_values,
            excitation,
            damping_ratio=damping_ratio,
        )
        peak = int(np.argmax(response))
        peak_hz = float(excitation[peak] / (2.0 * np.pi))
        first_hz = float(analysis.frequencies_hz[0])
        summary = {
            "first_frequency_hz": first_hz,
            "sampled_peak_hz": peak_hz,
            "relative_peak_location_error": abs(peak_hz - first_hz) / first_hz,
            "damping_ratio": damping_ratio,
            "n_modes": n_modes,
        }
        np.savez_compressed(
            output / "harmonic_response.npz",
            excitation_hz=excitation / (2.0 * np.pi),
            probe_amplitude=response,
            natural_frequencies_hz=analysis.frequencies_hz,
        )
        (output / "harmonic_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        save_probe_history(
            excitation / (2.0 * np.pi),
            response,
            output / "harmonic_response.png",
            xlabel="excitation frequency [Hz]",
            ylabel="probe amplitude [m]",
        )
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    if args.mode == "patch-modal":
        load = _patch_load(verification)
        direct = solver.solve(load=load)
        modal = solver.solve_modal(n_modes=modal_modes, load=load)
        direct_history = solver.active_probe_history(
            direct.active_result.displacement, center_probe
        )
        modal_history = solver.active_probe_history(modal.active_displacement, center_probe)
        comparison = compare_histories(direct_history, modal_history)
        np.savez_compressed(
            output / "patch_modal_comparison.npz",
            times=direct.active_result.times,
            direct=direct_history,
            modal=modal_history,
        )
        summary = {
            **comparison.to_dict(),
            "probe_m": list(center_probe),
            "n_modes": modal_modes,
            "load_amplitude_N_per_m2": load.amplitude,
            "patch_size_m": [load.size_x, load.size_y],
        }
        (output / "patch_modal_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        save_probe_history(direct.active_result.times, direct_history, output / "patch_direct.png")
        save_probe_history(modal.modal_result.times, modal_history, output / "patch_modal.png")
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 0

    amplitude = float(verification.get("single_mode_load", 1000.0))
    modal_load = _modal_load(raw, amplitude=amplitude)
    direct = solver.solve(load=modal_load)
    direct_history = solver.active_probe_history(direct.active_result.displacement, center_probe)

    if args.mode == "single-mode":
        effective = config.mechanical.foundation.effective_dynamic_stiffness(branch)
        reference = single_mode_step_reference(
            direct.active_result.times,
            m=1,
            n=1,
            load_amplitude=amplitude,
            probe=center_probe,
            length=geometry.length,
            width=geometry.width,
            thickness=geometry.thickness,
            density=config.density,
            rigidity=config.mechanical.rigidity,
            foundation_stiffness=effective,
        )
        comparison = compare_histories(reference.probe_history, direct_history)
        comparison_history = reference.probe_history
        label = "navier_reference"
        extra = {
            "reference_frequency_hz": reference.angular_frequency / (2.0 * np.pi),
            "modal_mass": reference.modal_mass,
            "modal_stiffness": reference.modal_stiffness,
        }
    else:
        modal = solver.solve_modal(n_modes=n_modes, load=modal_load)
        comparison_history = solver.active_probe_history(modal.active_displacement, center_probe)
        comparison = compare_histories(direct_history, comparison_history)
        label = "finite_element_modal"
        extra = {"n_modes": n_modes}

    np.savez_compressed(
        output / f"{args.mode}_comparison.npz",
        times=direct.active_result.times,
        direct=direct_history,
        comparison=comparison_history,
    )
    summary = {
        **comparison.to_dict(),
        "probe_m": list(center_probe),
        "foundation_branch": branch,
        "load_amplitude_N_per_m2": amplitude,
        "comparison": label,
        **extra,
    }
    (output / f"{args.mode}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_probe_history(direct.active_result.times, direct_history, output / "direct_history.png")
    save_probe_history(direct.active_result.times, comparison_history, output / f"{label}.png")
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
