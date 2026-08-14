"""Complete-trajectory generation and leakage-safe archive layout."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from paramplate.core.parameters import LoadParameters, OrthotropicRigidity
from paramplate.core.sampling import latin_hypercube
from paramplate.io.manifest import DatasetManifest, write_manifest

from .fem import TransientFEMConfig, TransientPlateFOM
from .model import CAMPAIGNS, parameter_ranges


def campaign_lhs(campaign: str, n_trajectories: int | None = None, *, seed: int = 100) -> np.ndarray:
    spec = CAMPAIGNS[campaign]
    count = spec.n_trajectories if n_trajectories is None else int(n_trajectories)
    return latin_hypercube(parameter_ranges(campaign), count, seed=seed)


def configuration_for_row(base: TransientFEMConfig, campaign: str, row: np.ndarray) -> tuple[TransientFEMConfig, LoadParameters]:
    values = np.asarray(row, dtype=float).reshape(-1)
    if campaign in {"case2_monolithic", "case2_one_interface"}:
        amplitude = float(values[0])
        mechanical = base.mechanical
    elif campaign == "case3_monolithic":
        Dx, amplitude = map(float, values)
        mechanical = replace(
            base.mechanical,
            rigidity=OrthotropicRigidity.linked_from_Dx(Dx, dy_ratio=1.0, dxy_ratio=0.30, ds_ratio=0.35),
        )
    else:
        raise ValueError(f"Unknown campaign {campaign!r}.")
    # The source campaign samples positive amplitudes and applies downward q=-Aq.
    load = replace(mechanical.load, amplitude=-amplitude)
    return replace(base, mechanical=mechanical), load


def generate_trajectory_archive(
    base: TransientFEMConfig,
    campaign: str,
    parameters: np.ndarray,
    output: str | Path,
) -> Path:
    table = np.asarray(parameters, dtype=float)
    expected = len(CAMPAIGNS[campaign].parameter_names)
    if table.ndim != 2 or table.shape[1] != expected:
        raise ValueError(f"Campaign {campaign} expects {expected} parameter columns.")
    trajectories, times = [], None
    for row in table:
        config, load = configuration_for_row(base, campaign, row)
        solution = TransientPlateFOM(config).solve(load=load, time_profile="step")
        if times is None:
            times = solution.stored_times
        elif not np.allclose(times, solution.stored_times):
            raise RuntimeError("Stored trajectory time grids differ.")
        trajectories.append(solution.stored_output_snapshots)
    assert times is not None
    states = np.asarray(trajectories)
    trajectory_ids = np.repeat(np.arange(table.shape[0], dtype=int), times.size)
    parameter_rows = np.repeat(table, times.size, axis=0)
    time_rows = np.tile(times, table.shape[0])
    query_rows = np.column_stack([parameter_rows, time_rows])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        trajectory_parameters=table,
        times=times,
        trajectories=states,
        snapshots=states.reshape(-1, states.shape[-1]),
        query_parameters=query_rows,
        trajectory_ids=trajectory_ids,
        parameter_names=np.asarray((*CAMPAIGNS[campaign].parameter_names, "time")),
        output_name=np.asarray("displacement"),
        output_units=np.asarray("m"),
    )
    manifest = DatasetManifest(
        study="dynamics",
        archive=target.name,
        snapshot_key="snapshots",
        parameter_key="query_parameters",
        parameter_names=(*CAMPAIGNS[campaign].parameter_names, "time"),
        output_name="displacement",
        output_unit="m",
        split_level="trajectory",
        trajectory_id_key="trajectory_ids",
        time_key="times",
        metric="Euclidean snapshot SVD followed by FE H1-type reorthonormalization",
        notes="Complete trajectories must be split before row flattening.",
        metadata={"campaign": campaign, "stored_states_per_trajectory": int(times.size)},
    )
    write_manifest(manifest, target.with_suffix(".manifest.json"))
    return target
