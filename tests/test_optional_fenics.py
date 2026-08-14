from __future__ import annotations

import pytest


pytestmark = pytest.mark.fenics


def _fenics_modules():
    dolfin = pytest.importorskip("dolfin")
    pytest.importorskip("mshr")
    return dolfin


def test_static_fom_constructs_in_legacy_environment() -> None:
    _fenics_modules()
    from paramplate.core.parameters import PlateGeometry
    from paramplate.mechanical.fem import MechanicalFEMConfig, MechanicalPlateFOM

    solver = MechanicalPlateFOM(
        MechanicalFEMConfig(geometry=PlateGeometry(resolution=4, degree=2))
    )
    assert solver.n_native_dofs > 0
    assert solver.n_output_dofs > 0


def test_thermomechanical_and_transient_classes_are_constructible() -> None:
    _fenics_modules()
    from paramplate.core.parameters import PlateGeometry
    from paramplate.dynamics.fem import TransientFEMConfig, TransientPlateFOM
    from paramplate.mechanical.fem import MechanicalFEMConfig
    from paramplate.thermomechanical.fem import ThermomechanicalFEMConfig, ThermomechanicalPlateFOM

    mechanics = MechanicalFEMConfig(geometry=PlateGeometry(resolution=4, degree=2))
    thermo = ThermomechanicalPlateFOM(
        ThermomechanicalFEMConfig(mechanical=mechanics, heat_nx=2, heat_ny=2, heat_nz=2)
    )
    assert thermo.heat_solver.V.dim() > 0
    transient = TransientPlateFOM(
        TransientFEMConfig(mechanical=mechanics, final_time=0.002, time_step=0.001, store_every=1)
    )
    assert transient.active.mass.shape[0] > 0
