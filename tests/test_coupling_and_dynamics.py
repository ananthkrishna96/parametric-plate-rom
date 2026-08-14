from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from paramplate.dynamics.active_dofs import restrict_active_dofs
from paramplate.dynamics.newmark import average_acceleration_newmark
from paramplate.dynamics.verification import navier_angular_frequency, relative_energy_drift
from paramplate.core.parameters import OrthotropicRigidity
from paramplate.thermomechanical.coupling import relaxed_partitioned_solve


def test_partitioned_acceptance_keeps_last_relaxed_w_and_same_iteration_theta() -> None:
    thermal_inputs: list[np.ndarray] = []

    def thermal(w: np.ndarray):
        thermal_inputs.append(w.copy())
        theta = 2.0 * w + 1.0
        return {"input": w.copy()}, theta

    def mechanical(theta: np.ndarray):
        return 0.25 * theta

    result = relaxed_partitioned_solve(
        np.zeros(2),
        thermal_solve=thermal,
        mechanical_solve=mechanical,
        relaxation=0.7,
        tolerance=1.0e-7,
        maximum_iterations=100,
    )
    assert result.converged
    np.testing.assert_allclose(result.thermal_driver, 2.0 * thermal_inputs[-1] + 1.0)
    previous_w = thermal_inputs[-1]
    expected_relaxed = 0.3 * previous_w + 0.7 * (0.25 * result.thermal_driver)
    np.testing.assert_allclose(result.displacement, expected_relaxed)


def test_partitioned_solver_accepts_field_specific_update_measures() -> None:
    displacement_calls: list[tuple[np.ndarray, np.ndarray]] = []
    theta_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def update_w(new: np.ndarray, old: np.ndarray) -> float:
        displacement_calls.append((new.copy(), old.copy()))
        return 0.0

    def update_theta(new: np.ndarray, old: np.ndarray) -> float:
        theta_calls.append((new.copy(), old.copy()))
        return 0.0

    result = relaxed_partitioned_solve(
        np.zeros(2),
        thermal_solve=lambda w: ({"w": w.copy()}, w + 1.0),
        mechanical_solve=lambda theta: 0.5 * theta,
        displacement_update_measure=update_w,
        thermal_driver_update_measure=update_theta,
        maximum_iterations=3,
    )
    assert result.converged and result.iterations == 2
    assert len(displacement_calls) == 2
    assert len(theta_calls) == 1


def test_active_dof_restriction_removes_zero_mass_entries() -> None:
    K = sparse.diags([2.0, 3.0, 4.0, 5.0])
    M = sparse.diags([1.0, 0.0, 2.0, 3.0])
    active = restrict_active_dofs(K, M, constrained_dofs=np.array([3]))
    assert active.active_dofs.tolist() == [0, 2]
    assert active.removed_zero_mass_dofs.tolist() == [1]
    assert active.stiffness.shape == active.mass.shape == (2, 2)


def test_average_acceleration_newmark_conserves_undamped_energy() -> None:
    times = np.linspace(0.0, 2.0, 2001)
    result = average_acceleration_newmark(
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[16.0]]),
        times,
        lambda _time: np.zeros(1),
        initial_displacement=np.array([1.0]),
    )
    exact = np.cos(4.0 * times)
    assert np.max(np.abs(result.displacement[:, 0] - exact)) < 2.0e-5
    assert relative_energy_drift(result.mechanical_energy) < 2.0e-12


def test_navier_dynamic_frequency_is_positive() -> None:
    omega = navier_angular_frequency(
        1,
        1,
        length=2.0,
        width=1.0,
        thickness=0.05,
        density=1200.0,
        rigidity=OrthotropicRigidity(6000.0, 6000.0, 1800.0, 2100.0),
        foundation_stiffness=500_000.0,
    )
    assert omega > 0.0
