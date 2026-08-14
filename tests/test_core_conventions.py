from __future__ import annotations

import numpy as np
import pytest

from paramplate.core.panels import interface_coordinates, panel_boxes
from paramplate.core.parameters import (
    FoundationParameters,
    OrthotropicRigidity,
    PlateGeometry,
)
from paramplate.mechanical.foundation import (
    active_foundation_stiffness,
    compression_mask,
    foundation_energy_density,
    foundation_reaction,
)
from paramplate.mechanical.model import direct_parameter_ranges, linked_rigidity
from paramplate.mechanical.navier import solve_ssss_navier
from paramplate.thermomechanical.thermal_driver import (
    centered_first_moment,
    centered_first_moment_samples,
    thermal_bending_moments,
)


def test_upward_positive_unilateral_foundation() -> None:
    w = np.array([-2.0, 0.0, 3.0])
    stiffness = active_foundation_stiffness(w, 100.0, 1.0e-3)
    assert np.allclose(stiffness, [100.0, 0.1, 0.1])
    assert compression_mask(w).tolist() == [True, False, False]
    assert np.allclose(foundation_reaction(w, 100.0, 1.0e-3), [-200.0, 0.0, 0.3])
    assert np.all(foundation_energy_density(w, 100.0, 1.0e-3) >= 0.0)


def test_dynamic_foundation_branches() -> None:
    f = FoundationParameters(500_000.0, uplift_factor=1.0e-3)
    assert f.effective_dynamic_stiffness("compression") == 500_000.0
    assert f.effective_dynamic_stiffness("uplift") == 500.0
    assert f.effective_dynamic_stiffness("none") == 0.0


def test_panel_geometry_bookkeeping() -> None:
    geometry = PlateGeometry(resolution=8, n_vertical_interfaces=1, n_horizontal_interfaces=1)
    boxes = panel_boxes(geometry)
    assert geometry.n_panels == 4
    assert len(boxes) == 4
    assert boxes[0].contains(0.25, 0.25)
    interfaces = interface_coordinates(geometry)
    assert interfaces == (("vertical", 1.0, 5), ("horizontal", 0.5, 6))


def test_rigidity_records_and_static_ranges() -> None:
    rigidity = linked_rigidity(10_000.0, 0.8, 1.2)
    assert rigidity.Dx == 10_000.0
    assert rigidity.Dy == 8_000.0
    assert rigidity.Dxy == 0.0
    assert rigidity.Ds == pytest.approx(0.6 * np.sqrt(80_000_000.0))
    ranges = direct_parameter_ranges()
    assert [item.name for item in ranges] == ["Dx", "Dy", "Dxy", "Ds", "ks", "q"]
    assert ranges[-1].upper < 0.0


def test_thermal_driver_is_gradient_like() -> None:
    gradient = 18.0
    theta = centered_first_moment(lambda z: gradient * z, 0.05, quadrature_points=20)
    assert theta == pytest.approx(gradient, rel=1.0e-13, abs=1.0e-13)
    z = np.linspace(-0.025, 0.025, 1001)
    sampled = centered_first_moment_samples(z, gradient * z, 0.05)
    assert sampled == pytest.approx(gradient, rel=5.0e-6)
    mx, my = thermal_bending_moments(
        np.array([1.0, 2.0]),
        Dx=10.0,
        Dy=12.0,
        Dxy=2.0,
        alpha_x=0.1,
        alpha_y=0.2,
    )
    assert np.allclose(mx, [1.4, 2.8])
    assert np.allclose(my, [2.6, 5.2])


def test_navier_reference_has_expected_sign_and_boundary_values() -> None:
    solution = solve_ssss_navier(
        OrthotropicRigidity(9500.0, 9500.0, 2850.0, 3320.0),
        foundation_stiffness=1000.0,
        load_amplitude=-1000.0,
        n_terms=15,
        nx=31,
        ny=17,
    )
    assert solution.displacement.shape == (17, 31)
    assert np.min(solution.displacement) < 0.0
    assert np.allclose(solution.displacement[[0, -1], :], 0.0, atol=1.0e-14)
    assert np.allclose(solution.displacement[:, [0, -1]], 0.0, atol=1.0e-14)
