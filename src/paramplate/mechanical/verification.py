"""Mechanical analytical, external-reference, and panel comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from paramplate.core.field_sampling import sample_scalar_field
from paramplate.io.reference_fields import load_regular_grid_field

from .fem import MechanicalFEMConfig, MechanicalPlateFOM
from .navier import NavierPlateSolution, solve_ssss_navier


@dataclass(frozen=True)
class FieldComparison:
    relative_l2: float
    maximum_absolute_error: float
    reference_maximum: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GridVerification:
    x: np.ndarray
    y: np.ndarray
    reference: np.ndarray
    approximation: np.ndarray
    summary: FieldComparison
    reference_name: str

    @property
    def absolute_error(self) -> np.ndarray:
        return np.abs(self.approximation - self.reference)


def compare_arrays(reference: np.ndarray, approximation: np.ndarray) -> FieldComparison:
    ref = np.asarray(reference, dtype=float)
    app = np.asarray(approximation, dtype=float)
    if ref.shape != app.shape:
        raise ValueError(f"Field shapes differ: {ref.shape} and {app.shape}.")
    if ref.size == 0 or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(app)):
        raise ValueError("Comparison fields must be finite and nonempty.")
    denominator = max(float(np.linalg.norm(ref)), np.finfo(float).eps)
    return FieldComparison(
        relative_l2=float(np.linalg.norm(app - ref) / denominator),
        maximum_absolute_error=float(np.max(np.abs(app - ref))),
        reference_maximum=float(np.max(np.abs(ref))),
    )


def navier_smoke_reference(*, n_terms: int = 21) -> NavierPlateSolution:
    from paramplate.core.parameters import OrthotropicRigidity

    return solve_ssss_navier(
        OrthotropicRigidity(9.5e3, 9.5e3, 2.85e3, 3.32e3),
        foundation_stiffness=1.0e3,
        load_amplitude=-1000.0,
        load="uniform",
        n_terms=n_terms,
        nx=41,
        ny=21,
    )


def compare_fom_with_navier(
    config: MechanicalFEMConfig,
    *,
    n_terms: int = 31,
    nx: int = 121,
    ny: int = 61,
) -> GridVerification:
    """Solve the monolithic FOM and compare it on the Navier grid."""

    if config.geometry.n_panels != 1:
        raise ValueError("The Navier comparison requires a monolithic plate configuration.")
    if config.boundary_condition != "simply_supported":
        raise ValueError("The Navier comparison requires simply supported exterior edges.")
    if config.fixed_foundation_factor is None:
        raise ValueError(
            "The Navier comparison requires a prescribed linear foundation factor."
        )
    if config.load.kind not in {"uniform", "patch"}:
        raise ValueError("The Navier comparison supports uniform or patch loading.")
    solution = MechanicalPlateFOM(config).solve()
    reference = solve_ssss_navier(
        config.rigidity,
        length=config.geometry.length,
        width=config.geometry.width,
        foundation_stiffness=(
            config.foundation.stiffness * float(config.fixed_foundation_factor)
        ),
        load_amplitude=config.load.amplitude,
        load=config.load.kind,
        patch=(config.load.x0, config.load.y0, config.load.size_x, config.load.size_y),
        n_terms=n_terms,
        nx=nx,
        ny=ny,
    )
    approximation = sample_scalar_field(solution.displacement, reference.x, reference.y)
    return GridVerification(
        x=reference.x,
        y=reference.y,
        reference=reference.displacement,
        approximation=approximation,
        summary=compare_arrays(reference.displacement, approximation),
        reference_name="Navier series",
    )


def compare_fom_with_external_grid(
    config: MechanicalFEMConfig,
    reference_archive: str | Path,
    *,
    key: str = "displacement",
) -> GridVerification:
    """Compare a solved FOM field with a numeric regular-grid reference archive."""

    reference = load_regular_grid_field(reference_archive, key=key, unit="m")
    solution = MechanicalPlateFOM(config).solve()
    approximation = sample_scalar_field(solution.displacement, reference.x, reference.y)
    return GridVerification(
        x=reference.x,
        y=reference.y,
        reference=reference.values,
        approximation=approximation,
        summary=compare_arrays(reference.values, approximation),
        reference_name=Path(reference_archive).name,
    )


def compare_panel_with_monolithic(
    panel_config: MechanicalFEMConfig,
    monolithic_config: MechanicalFEMConfig,
    *,
    nx: int = 121,
    ny: int = 61,
) -> GridVerification:
    """Compare a panel-partitioned solve with a monolithic reference solve."""

    if panel_config.geometry.n_panels <= 1:
        raise ValueError("panel_config must contain at least one physical interface.")
    if monolithic_config.geometry.n_panels != 1:
        raise ValueError("monolithic_config must contain one physical panel.")
    for name in ("length", "width", "thickness"):
        if not np.isclose(
            getattr(panel_config.geometry, name),
            getattr(monolithic_config.geometry, name),
        ):
            raise ValueError(f"Panel and monolithic geometries differ in {name}.")
    for name in ("rigidity", "foundation", "load", "boundary_condition", "fixed_foundation_factor"):
        if getattr(panel_config, name) != getattr(monolithic_config, name):
            raise ValueError(f"Panel and monolithic configurations differ in {name}.")
    panel = MechanicalPlateFOM(panel_config).solve()
    monolithic = MechanicalPlateFOM(monolithic_config).solve()
    x = np.linspace(0.0, panel_config.geometry.length, int(nx))
    y = np.linspace(0.0, panel_config.geometry.width, int(ny))
    panel_field = sample_scalar_field(panel.displacement, x, y)
    monolithic_field = sample_scalar_field(monolithic.displacement, x, y)
    return GridVerification(
        x=x,
        y=y,
        reference=monolithic_field,
        approximation=panel_field,
        summary=compare_arrays(monolithic_field, panel_field),
        reference_name="monolithic FOM",
    )
