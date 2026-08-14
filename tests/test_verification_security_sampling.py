from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paramplate.core.parameters import (
    FoundationParameters,
    LoadParameters,
    OrthotropicRigidity,
    PlateGeometry,
)
from paramplate.core.sampling import (
    ParameterRange,
    configured_parameter_ranges,
    parameter_ranges_from_config,
)
from paramplate.dynamics.model import parameter_ranges as dynamic_parameter_ranges
from paramplate.dynamics.trajectories import campaign_lhs
from paramplate.dynamics.verification import (
    compare_histories,
    ordered_navier_frequencies,
    relative_energy_range,
    single_mode_step_reference,
)
from paramplate.io.archives import small_array
from paramplate.mechanical.fem import MechanicalFEMConfig
from paramplate.mechanical.model import direct_parameter_ranges
from paramplate.mechanical.snapshots import direct_lhs
from paramplate.mechanical.navier import solve_ssss_navier
from paramplate.rom.datasets import load_snapshot_dataset
from paramplate.thermomechanical.fem import ThermomechanicalFEMConfig
from paramplate.thermomechanical.model import parameter_ranges as thermo_parameter_ranges
from paramplate.thermomechanical.snapshots import case_lhs
from paramplate.thermomechanical.verification import (
    auxiliary_navier_displacement,
    double_sine_coefficients,
    evaluate_double_sine_series,
)


def test_pickle_backed_npz_members_are_rejected_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "legacy_object.npz"
    np.savez_compressed(
        archive,
        parameters=np.asarray([[1.0]]),
        snapshots=np.asarray([[2.0]]),
        parameter_names=np.asarray(["mu"], dtype=object),
    )
    with pytest.raises(ValueError, match="pickle-backed"):
        small_array(archive, "parameter_names")
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_snapshot_dataset(archive, study="mechanical")
    trusted = small_array(archive, "parameter_names", allow_unsafe_pickle=True)
    assert trusted.tolist() == ["mu"]


def test_invalid_or_unknown_sampling_ranges_are_rejected() -> None:
    defaults = (ParameterRange("mu", 1.0, 2.0),)
    with pytest.raises(ValueError, match="Unknown sampling range"):
        configured_parameter_ranges({"ranges": {"other": [0.0, 1.0]}}, defaults)
    with pytest.raises(ValueError, match="must have two values"):
        configured_parameter_ranges({"ranges": {"mu": [1.0]}}, defaults)
    with pytest.raises(ValueError, match="Unknown range fields"):
        configured_parameter_ranges(
            {"ranges": {"mu": {"lower": 1.0, "upper": 2.0, "typo": 3.0}}},
            defaults,
        )
    with pytest.raises(ValueError, match="sampling.ranges must be a mapping"):
        configured_parameter_ranges({"ranges": [1.0, 2.0]}, defaults)


def test_legacy_sampling_alias_api_remains_compatible() -> None:
    defaults = (ParameterRange("ks", 1.0, 2.0),)
    ranges = parameter_ranges_from_config(
        {"ranges": {"foundation_stiffness": [3.0, 4.0]}},
        defaults,
        aliases={"foundation_stiffness": "ks"},
    )
    assert ranges[0].lower == 3.0
    assert ranges[0].upper == 4.0


def test_configured_ranges_drive_all_three_campaign_samplers() -> None:
    mechanical_ranges = configured_parameter_ranges(
        {"ranges": {"ks": [10.0, 20.0], "q": [-2.0, -1.0]}},
        direct_parameter_ranges(),
    )
    mechanical = direct_lhs(24, seed=5, ranges=mechanical_ranges)
    assert np.all((mechanical[:, 4] >= 10.0) & (mechanical[:, 4] <= 20.0))
    assert np.all((mechanical[:, 5] >= -2.0) & (mechanical[:, 5] <= -1.0))

    thermo_ranges = configured_parameter_ranges(
        {"ranges": {"load_amplitude": [-5.0, -4.0], "absorbed_solar_flux": [20.0, 30.0]}},
        thermo_parameter_ranges(2),
    )
    thermo = case_lhs(2, 16, seed=6, ranges=thermo_ranges)
    assert np.all((thermo[:, 0] >= -5.0) & (thermo[:, 0] <= -4.0))
    assert np.all((thermo[:, 1] >= 20.0) & (thermo[:, 1] <= 30.0))

    transient_ranges = configured_parameter_ranges(
        {"ranges": {"load_amplitude": [1.0, 2.0]}},
        dynamic_parameter_ranges("case2_monolithic"),
    )
    transient = campaign_lhs("case2_monolithic", 16, seed=7, ranges=transient_ranges)
    assert np.all((transient[:, 0] >= 1.0) & (transient[:, 0] <= 2.0))


def test_auxiliary_thermomechanical_navier_reduces_to_mechanical_mode() -> None:
    geometry = PlateGeometry(length=2.0, width=1.0, thickness=0.05, resolution=8, degree=2)
    rigidity = OrthotropicRigidity(10_000.0, 10_000.0, 3_000.0, 3_500.0)
    load = LoadParameters(kind="modal_sine", amplitude=-1_000.0)
    mechanical = MechanicalFEMConfig(
        geometry=geometry,
        rigidity=rigidity,
        foundation=FoundationParameters(1_000.0, 1.0e-3),
        load=load,
        boundary_condition="simply_supported",
        fixed_foundation_factor=1.0,
    )
    config = ThermomechanicalFEMConfig(mechanical=mechanical)
    x = np.linspace(0.0, geometry.length, 81)
    y = np.linspace(0.0, geometry.width, 41)
    theta = np.zeros((y.size, x.size))
    thermo = auxiliary_navier_displacement(config, x, y, theta, n_terms=5)
    mechanical_reference = solve_ssss_navier(
        rigidity,
        length=geometry.length,
        width=geometry.width,
        foundation_stiffness=1_000.0,
        load_coefficient=lambda m, n: -1_000.0 if (m, n) == (1, 1) else 0.0,
        n_terms=5,
        nx=x.size,
        ny=y.size,
    )
    np.testing.assert_allclose(thermo, mechanical_reference.displacement, rtol=1.0e-12, atol=1.0e-14)


def test_double_sine_projection_preserves_m_n_orientation() -> None:
    length, width = 2.0, 1.0
    x = np.linspace(0.0, length, 401)
    y = np.linspace(0.0, width, 201)
    xx, yy = np.meshgrid(x, y)
    field = 2.75 * np.sin(2.0 * np.pi * xx / length) * np.sin(np.pi * yy / width)
    coefficients = double_sine_coefficients(field, x, y, n_terms=3)
    assert coefficients[1, 0] == pytest.approx(2.75, rel=2.0e-5, abs=2.0e-7)
    assert abs(coefficients[0, 1]) < 1.0e-10
    reconstructed = evaluate_double_sine_series(
        coefficients, x, y, length=length, width=width
    )
    np.testing.assert_allclose(reconstructed, field, rtol=2.0e-5, atol=2.0e-7)


def test_ordered_dynamic_reference_and_history_metrics() -> None:
    rigidity = OrthotropicRigidity(10_302.20, 10_302.20, 3_090.659, 3_605.769)
    reference = ordered_navier_frequencies(
        8,
        length=2.0,
        width=1.0,
        thickness=0.05,
        density=1200.0,
        rigidity=rigidity,
        foundation_stiffness=0.0,
        search_order=12,
    )
    assert np.all(np.diff(reference.frequencies_hz) >= 0.0)
    assert reference.mode_pairs.shape == (8, 2)
    assert reference.mode_pairs[0].tolist() == [1, 1]

    history = np.sin(np.linspace(0.0, 1.0, 20))
    metrics = compare_histories(history, history.copy())
    assert metrics.relative_l2 == 0.0
    assert metrics.relative_linf == 0.0
    assert relative_energy_range(np.ones(10)) == 0.0


def test_single_mode_reference_has_consistent_shapes_and_initial_acceleration() -> None:
    times = np.linspace(0.0, 0.02, 21)
    rigidity = OrthotropicRigidity(10_302.20, 10_302.20, 3_090.659, 3_605.769)
    reference = single_mode_step_reference(
        times,
        m=1,
        n=1,
        load_amplitude=1000.0,
        probe=(1.0, 0.5),
        length=2.0,
        width=1.0,
        thickness=0.05,
        density=1200.0,
        rigidity=rigidity,
        foundation_stiffness=100_000.0,
    )
    assert reference.probe_history.shape == times.shape
    assert reference.result.acceleration[0, 0] == pytest.approx(
        reference.modal_load / reference.modal_mass
    )
