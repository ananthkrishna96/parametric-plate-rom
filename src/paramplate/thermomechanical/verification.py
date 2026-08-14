"""Analytical and external-reference checks for steady thermomechanics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from paramplate.core.field_sampling import sample_scalar_field
from paramplate.core.parameters import LoadParameters
from paramplate.io.reference_fields import load_regular_grid_field
from paramplate.mechanical.verification import FieldComparison, compare_arrays

from .fem import ThermomechanicalFEMConfig, ThermomechanicalPlateFOM
from .thermal_driver import centered_first_moment



@dataclass(frozen=True)
class ThermomechanicalNavierVerification:
    x: np.ndarray
    y: np.ndarray
    thermal_driver: np.ndarray
    reference_displacement: np.ndarray
    finite_element_displacement: np.ndarray
    summary: FieldComparison
    coupling_iterations: int
    displacement_updates: np.ndarray
    thermal_driver_updates: np.ndarray

    @property
    def absolute_error(self) -> np.ndarray:
        return np.abs(self.finite_element_displacement - self.reference_displacement)

    def metadata(self) -> dict[str, object]:
        return {
            "reference": "finite auxiliary thermomechanical Navier series",
            "coupling_iterations": int(self.coupling_iterations),
            **self.summary.to_dict(),
        }


@dataclass(frozen=True)
class ThermomechanicalExternalVerification:
    x: np.ndarray
    y: np.ndarray
    reference_displacement: np.ndarray
    finite_element_displacement: np.ndarray
    displacement_summary: FieldComparison
    reference_thermal_driver: np.ndarray
    finite_element_thermal_driver: np.ndarray
    thermal_driver_summary: FieldComparison
    coupling_iterations: int

    def metadata(self) -> dict[str, object]:
        return {
            "reference": "external regular-grid fields",
            "coupling_iterations": int(self.coupling_iterations),
            "displacement": self.displacement_summary.to_dict(),
            "thermal_driver": self.thermal_driver_summary.to_dict(),
        }


def linear_profile_driver(
    gradient: float,
    thickness: float,
    *,
    quadrature_points: int = 20,
) -> float:
    """For ``DeltaT(z)=gradient*z``, the centered driver equals ``gradient``."""

    return centered_first_moment(
        lambda z: float(gradient) * z,
        thickness,
        quadrature_points=quadrature_points,
    )


def load_field(load: LoadParameters, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a configured transverse load on a regular midsurface grid."""

    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if load.kind == "uniform":
        return np.full_like(xx, float(load.amplitude), dtype=float)
    if load.kind == "patch":
        mask = (
            (np.abs(xx - float(load.x0)) <= float(load.size_x) / 2.0)
            & (np.abs(yy - float(load.y0)) <= float(load.size_y) / 2.0)
        )
        return np.where(mask, float(load.amplitude), 0.0)
    if load.kind == "linear_x":
        length = max(float(np.max(x) - np.min(x)), np.finfo(float).eps)
        return float(load.linear_start) + (
            float(load.amplitude) - float(load.linear_start)
        ) * (xx - float(np.min(x))) / length
    if load.kind == "modal_sine":
        length = float(np.max(x) - np.min(x))
        width = float(np.max(y) - np.min(y))
        return float(load.amplitude) * np.sin(np.pi * xx / length) * np.sin(np.pi * yy / width)
    if load.kind == "multi_patch":
        field = np.zeros_like(xx, dtype=float)
        for patch in load.patches:
            size_x = float(patch.get("size_x", patch.get("patch_size_x", load.size_x)))
            size_y = float(patch.get("size_y", patch.get("patch_size_y", load.size_y)))
            x0, y0 = float(patch["x0"]), float(patch["y0"])
            amplitude = float(patch.get("amplitude", patch.get("load_value", load.amplitude)))
            mask = (np.abs(xx - x0) <= size_x / 2.0) & (np.abs(yy - y0) <= size_y / 2.0)
            field += np.where(mask, amplitude, 0.0)
        return field
    raise ValueError(f"Unsupported load kind: {load.kind!r}.")


def _trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float).reshape(-1)
    if points.size < 2 or np.any(np.diff(points) <= 0.0):
        raise ValueError("Integration coordinates must be strictly increasing.")
    increments = np.diff(points)
    weights = np.empty_like(points)
    weights[0] = increments[0] / 2.0
    weights[-1] = increments[-1] / 2.0
    if points.size > 2:
        weights[1:-1] = (increments[:-1] + increments[1:]) / 2.0
    return weights


def double_sine_coefficients(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_terms: int,
) -> np.ndarray:
    """Project a regular-grid field onto the rectangular double-sine basis.

    ``values`` must have shape ``(len(y), len(x))`` and include both boundary
    lines. Composite trapezoidal weights are applied in matrix form, which
    keeps the 401-by-401, 159-mode thesis reference practical.
    """

    field = np.asarray(values, dtype=float)
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if field.shape != (yy.size, xx.size):
        raise ValueError("values must have shape (len(y), len(x)).")
    if n_terms < 1:
        raise ValueError("n_terms must be positive.")
    length = float(xx[-1] - xx[0])
    width = float(yy[-1] - yy[0])
    if length <= 0.0 or width <= 0.0:
        raise ValueError("x and y must span a nondegenerate rectangular plate.")

    modes = np.arange(1, int(n_terms) + 1, dtype=float)
    sin_x = np.sin(np.pi * np.outer(modes, (xx - xx[0]) / length))
    sin_y = np.sin(np.pi * np.outer(modes, (yy - yy[0]) / width))
    weighted = _trapezoid_weights(yy)[:, None] * field * _trapezoid_weights(xx)[None, :]
    # Rows index x modes m; columns index y modes n.
    return 4.0 * (sin_x @ weighted.T @ sin_y.T) / (length * width)


def auxiliary_navier_coefficients(
    config: ThermomechanicalFEMConfig,
    x: np.ndarray,
    y: np.ndarray,
    thermal_driver: np.ndarray,
    *,
    n_terms: int,
    foundation_stiffness: float | None = None,
) -> np.ndarray:
    """Return finite thermomechanical Navier displacement coefficients."""

    mechanical = config.mechanical
    if mechanical.geometry.n_panels != 1:
        raise ValueError("The auxiliary Navier reference requires a monolithic plate.")
    if mechanical.boundary_condition != "simply_supported":
        raise ValueError("The auxiliary Navier reference requires simply supported edges.")

    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(thermal_driver, dtype=float)
    if theta.shape != (yy.size, xx.size):
        raise ValueError("thermal_driver must have shape (len(y), len(x)).")
    length, width = mechanical.geometry.length, mechanical.geometry.width
    q_coefficients = double_sine_coefficients(
        load_field(mechanical.load, xx, yy), xx, yy, n_terms=n_terms
    )
    theta_coefficients = double_sine_coefficients(theta, xx, yy, n_terms=n_terms)

    rigidity = mechanical.rigidity
    alpha_x = float(config.environment.thermal_expansion_x)
    alpha_y = float(config.environment.thermal_expansion_y)
    effective_foundation = (
        float(mechanical.foundation.stiffness)
        if foundation_stiffness is None
        else float(foundation_stiffness)
    )
    modes = np.arange(1, int(n_terms) + 1, dtype=float)
    lambda_m = modes[:, None] * np.pi / float(length)
    mu_n = modes[None, :] * np.pi / float(width)
    thermal_factor = (
        (rigidity.Dx * alpha_x + rigidity.Dxy * alpha_y) * lambda_m**2
        + (rigidity.Dxy * alpha_x + rigidity.Dy * alpha_y) * mu_n**2
    )
    denominator = (
        rigidity.Dx * lambda_m**4
        + 2.0 * (rigidity.Dxy + 2.0 * rigidity.Ds) * lambda_m**2 * mu_n**2
        + rigidity.Dy * mu_n**4
        + effective_foundation
    )
    return (q_coefficients + thermal_factor * theta_coefficients) / denominator


def evaluate_double_sine_series(
    coefficients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    length: float,
    width: float,
) -> np.ndarray:
    """Evaluate coefficients indexed by positive ``(m,n)`` sine modes."""

    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1]:
        raise ValueError("coefficients must be a square (n_terms, n_terms) array.")
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    modes = np.arange(1, coeff.shape[0] + 1, dtype=float)
    sin_x = np.sin(np.pi * np.outer(modes, (xx - xx[0]) / float(length)))
    sin_y = np.sin(np.pi * np.outer(modes, (yy - yy[0]) / float(width)))
    return sin_y.T @ coeff.T @ sin_x


def auxiliary_navier_displacement(
    config: ThermomechanicalFEMConfig,
    x: np.ndarray,
    y: np.ndarray,
    thermal_driver: np.ndarray,
    *,
    n_terms: int,
    foundation_stiffness: float | None = None,
) -> np.ndarray:
    """Evaluate the finite auxiliary thermomechanical Navier series.

    The reference assumes a monolithic simply supported rectangle, a prescribed
    spatially constant foundation coefficient, and the supplied fixed thermal
    driver. It is not a solution of the displacement-dependent coupled problem.
    """

    coefficients = auxiliary_navier_coefficients(
        config,
        x,
        y,
        thermal_driver,
        n_terms=n_terms,
        foundation_stiffness=foundation_stiffness,
    )
    return evaluate_double_sine_series(
        coefficients,
        x,
        y,
        length=config.mechanical.geometry.length,
        width=config.mechanical.geometry.width,
    )


def compare_coupled_fom_with_auxiliary_navier(
    config: ThermomechanicalFEMConfig,
    *,
    n_terms: int = 31,
    nx: int = 121,
    ny: int = 61,
    projection_nx: int | None = None,
    projection_ny: int | None = None,
    foundation_stiffness: float | None = None,
) -> ThermomechanicalNavierVerification:
    """Solve the coupled FOM and compare with the fixed-driver Navier reference."""

    solution = ThermomechanicalPlateFOM(config).solve()
    pnx = int(nx if projection_nx is None else projection_nx)
    pny = int(ny if projection_ny is None else projection_ny)
    projection_x = np.linspace(0.0, config.mechanical.geometry.length, pnx)
    projection_y = np.linspace(0.0, config.mechanical.geometry.width, pny)
    theta_projection = sample_scalar_field(solution.thermal_driver, projection_x, projection_y)
    coefficients = auxiliary_navier_coefficients(
        config,
        projection_x,
        projection_y,
        theta_projection,
        n_terms=n_terms,
        foundation_stiffness=foundation_stiffness,
    )

    x = np.linspace(0.0, config.mechanical.geometry.length, int(nx))
    y = np.linspace(0.0, config.mechanical.geometry.width, int(ny))
    theta = (
        theta_projection
        if pnx == int(nx) and pny == int(ny)
        else sample_scalar_field(solution.thermal_driver, x, y)
    )
    reference = evaluate_double_sine_series(
        coefficients,
        x,
        y,
        length=config.mechanical.geometry.length,
        width=config.mechanical.geometry.width,
    )
    approximation = sample_scalar_field(solution.displacement, x, y)
    history = solution.coupled_state.history
    return ThermomechanicalNavierVerification(
        x=x,
        y=y,
        thermal_driver=theta,
        reference_displacement=reference,
        finite_element_displacement=approximation,
        summary=compare_arrays(reference, approximation),
        coupling_iterations=solution.coupled_state.iterations,
        displacement_updates=np.asarray([item.displacement_update for item in history], dtype=float),
        thermal_driver_updates=np.asarray(
            [item.thermal_driver_update for item in history], dtype=float
        ),
    )


def compare_coupled_fom_with_external_fields(
    config: ThermomechanicalFEMConfig,
    reference_archive: str | Path,
    *,
    displacement_key: str = "displacement",
    thermal_driver_key: str = "thermal_driver",
) -> ThermomechanicalExternalVerification:
    """Compare one coupled solve with trusted numeric regular-grid fields."""

    displacement_reference = load_regular_grid_field(
        reference_archive,
        key=displacement_key,
        unit="m",
    )
    theta_reference = load_regular_grid_field(
        reference_archive,
        key=thermal_driver_key,
        unit="K/m",
    )
    if not np.array_equal(displacement_reference.x, theta_reference.x) or not np.array_equal(
        displacement_reference.y, theta_reference.y
    ):
        raise ValueError("Displacement and thermal-driver references must share one grid.")
    solution = ThermomechanicalPlateFOM(config).solve()
    x, y = displacement_reference.x, displacement_reference.y
    displacement = sample_scalar_field(solution.displacement, x, y)
    theta = sample_scalar_field(solution.thermal_driver, x, y)
    return ThermomechanicalExternalVerification(
        x=x,
        y=y,
        reference_displacement=displacement_reference.values,
        finite_element_displacement=displacement,
        displacement_summary=compare_arrays(displacement_reference.values, displacement),
        reference_thermal_driver=theta_reference.values,
        finite_element_thermal_driver=theta,
        thermal_driver_summary=compare_arrays(theta_reference.values, theta),
        coupling_iterations=solution.coupled_state.iterations,
    )
