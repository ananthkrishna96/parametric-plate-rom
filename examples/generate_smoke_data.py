#!/usr/bin/env python3
"""Generate small synthetic archives that exercise the ROM data paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paramplate.io.manifest import DatasetManifest, sha256_file, write_manifest


def _spatial_basis(nx: int = 12, ny: int = 8) -> np.ndarray:
    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    fields = [
        np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y),
        np.sin(2.0 * np.pi * X / 2.0) * np.sin(np.pi * Y),
        np.sin(np.pi * X / 2.0) * np.sin(2.0 * np.pi * Y),
        np.exp(-8.0 * ((X - 1.0) ** 2 + (Y - 0.5) ** 2)),
        (X - 1.0) * np.sin(np.pi * Y),
    ]
    matrix = np.column_stack([field.reshape(-1) for field in fields])
    q, _ = np.linalg.qr(matrix)
    return q


def _mechanical(path: Path, basis: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 60
    lower = np.array([5.0e3, 5.0e3, 1.0e3, 1.0e3, 1.0e6, -1.0e4])
    upper = np.array([2.0e4, 2.0e4, 5.0e3, 5.0e3, 1.0e7, -2.0e3])
    parameters = lower + (upper - lower) * rng.random((n, 6))
    u = (parameters - lower) / (upper - lower)
    coordinates = np.column_stack(
        [
            -0.010 * (0.4 + u[:, 5]) / (0.7 + u[:, 0]),
            0.002 * (u[:, 1] - u[:, 0]),
            -0.0015 * np.sin(np.pi * u[:, 4]),
            0.0007 * (u[:, 2] + u[:, 3] - 1.0),
        ]
    )
    snapshots = coordinates @ basis[:, :4].T
    np.savez_compressed(
        path,
        parameters=parameters,
        snapshots=snapshots,
        parameter_names=np.asarray(["Dx", "Dy", "Dxy", "Ds", "ks", "q"]),
        output_name=np.asarray("displacement"),
        output_units=np.asarray("m"),
        provenance=np.asarray("synthetic workflow smoke data; not a thesis result"),
    )
    write_manifest(
        DatasetManifest(
            study="mechanical",
            archive=path.name,
            snapshot_key="snapshots",
            parameter_key="parameters",
            parameter_names=("Dx", "Dy", "Dxy", "Ds", "ks", "q"),
            output_name="displacement",
            output_unit="m",
            split_level="sample",
            expected_shape=snapshots.shape,
            metric="Euclidean smoke metric",
            sha256=sha256_file(path),
            notes="Synthetic low-rank fields for software-path checks only.",
        ),
        path.with_suffix(".manifest.json"),
    )


def _thermomechanical(path: Path, basis: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed + 1)
    n = 60
    load = rng.uniform(-6500.0, -1500.0, n)
    solar = rng.uniform(150.0, 850.0, n)
    parameters = np.column_stack([load, solar])
    load_n = (load + 6500.0) / 5000.0
    solar_n = (solar - 150.0) / 700.0
    w_coordinates = np.column_stack(
        [
            -0.008 * (0.35 + load_n),
            0.0015 * solar_n,
            -0.0008 * load_n * solar_n,
        ]
    )
    theta_coordinates = np.column_stack(
        [12.0 + 35.0 * solar_n, 4.0 * (load_n - 0.5) * solar_n]
    )
    displacement = w_coordinates @ basis[:, :3].T
    theta = theta_coordinates @ basis[:, 3:5].T
    np.savez_compressed(
        path,
        parameters=parameters,
        displacement_snapshots=displacement,
        thermal_driver_snapshots=theta,
        parameter_names=np.asarray(["load_amplitude", "absorbed_solar_flux"]),
        displacement_units=np.asarray("m"),
        thermal_driver_units=np.asarray("K/m"),
        provenance=np.asarray("synthetic workflow smoke data; not a thesis result"),
    )
    for output, key, unit in (
        ("displacement", "displacement_snapshots", "m"),
        ("thermal_driver", "thermal_driver_snapshots", "K/m"),
    ):
        write_manifest(
            DatasetManifest(
                study="thermomechanical",
                archive=path.name,
                snapshot_key=key,
                parameter_key="parameters",
                parameter_names=("load_amplitude", "absorbed_solar_flux"),
                output_name=output,
                output_unit=unit,
                split_level="sample",
                expected_shape=(displacement if output == "displacement" else theta).shape,
                metric="field-specific Euclidean smoke metric",
                sha256=sha256_file(path),
                notes="Synthetic one-field ROM data from matched sample assignments.",
            ),
            path.with_name(f"{path.stem}.{output}.manifest.json"),
        )


def _dynamics(path: Path, basis: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed + 2)
    n_trajectories = 20
    times = np.linspace(0.0, 0.20, 11)
    amplitudes = rng.uniform(2.0e3, 1.0e4, n_trajectories)
    rows: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    trajectory_ids: list[int] = []
    for trajectory_id, amplitude in enumerate(amplitudes):
        scale = amplitude / 1.0e4
        for time in times:
            tau = time / times[-1]
            coefficients = np.array(
                [
                    -0.006 * scale * np.sin(2.5 * np.pi * tau),
                    -0.002 * scale * (1.0 - np.cos(2.0 * np.pi * tau)),
                    0.001 * scale**2 * np.sin(4.0 * np.pi * tau),
                ]
            )
            if np.isclose(time, 0.0):
                coefficients[:] = 0.0
            rows.append(np.array([amplitude, time]))
            snapshots.append(coefficients @ basis[:, :3].T)
            trajectory_ids.append(trajectory_id)
    query = np.asarray(rows)
    fields = np.asarray(snapshots)
    ids = np.asarray(trajectory_ids, dtype=int)
    np.savez_compressed(
        path,
        trajectory_parameters=amplitudes.reshape(-1, 1),
        times=times,
        snapshots=fields,
        query_parameters=query,
        trajectory_ids=ids,
        parameter_names=np.asarray(["load_amplitude", "time"]),
        output_name=np.asarray("displacement"),
        output_units=np.asarray("m"),
        provenance=np.asarray("synthetic workflow smoke data; not a thesis result"),
    )
    write_manifest(
        DatasetManifest(
            study="dynamics",
            archive=path.name,
            snapshot_key="snapshots",
            parameter_key="query_parameters",
            parameter_names=("load_amplitude", "time"),
            output_name="displacement",
            output_unit="m",
            split_level="trajectory",
            expected_shape=fields.shape,
            trajectory_id_key="trajectory_ids",
            time_key="times",
            metric="Euclidean smoke metric",
            sha256=sha256_file(path),
            notes="Synthetic complete trajectories; split trajectory IDs before flattening.",
            metadata={"n_trajectories": n_trajectories, "states_per_trajectory": len(times)},
        ),
        path.with_suffix(".manifest.json"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sample"))
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    basis = _spatial_basis()
    _mechanical(output / "mechanical_smoke.npz", basis, args.seed)
    _thermomechanical(output / "thermomechanical_smoke.npz", basis, args.seed)
    _dynamics(output / "dynamics_smoke.npz", basis, args.seed)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
