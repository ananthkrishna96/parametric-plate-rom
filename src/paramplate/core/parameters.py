"""Validated parameter records shared by the three plate studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PlateGeometry:
    length: float = 2.0
    width: float = 1.0
    thickness: float = 0.05
    resolution: int = 64
    degree: int = 2
    n_vertical_interfaces: int = 0
    n_horizontal_interfaces: int = 0

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0 or self.thickness <= 0:
            raise ValueError("Plate dimensions and thickness must be positive.")
        if self.resolution < 2:
            raise ValueError("resolution must be at least 2.")
        if self.degree < 2:
            raise ValueError("C0-IPG Kirchhoff--Love runs require degree >= 2.")
        if self.n_vertical_interfaces < 0 or self.n_horizontal_interfaces < 0:
            raise ValueError("Interface counts must be nonnegative.")

    @property
    def n_panels(self) -> int:
        return (self.n_vertical_interfaces + 1) * (self.n_horizontal_interfaces + 1)

    @property
    def panel_shape(self) -> tuple[int, int]:
        return self.n_vertical_interfaces + 1, self.n_horizontal_interfaces + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrthotropicRigidity:
    Dx: float
    Dy: float
    Dxy: float
    Ds: float

    def __post_init__(self) -> None:
        vals = np.asarray([self.Dx, self.Dy, self.Ds], dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise ValueError("Dx, Dy, and Ds must be finite and strictly positive.")
        if not np.isfinite(self.Dxy):
            raise ValueError("Dxy must be finite.")
        # Positive definiteness of the aligned bending block.
        if self.Dx * self.Dy - self.Dxy**2 <= 0.0:
            raise ValueError("The aligned orthotropic bending block must be positive definite.")

    @property
    def effective(self) -> float:
        return float(np.sqrt(self.Dx * self.Dy))

    @classmethod
    def linked_from_Dx(
        cls,
        Dx: float,
        *,
        dy_ratio: float = 1.0,
        dxy_ratio: float = 0.30,
        ds_ratio: float = 0.35,
    ) -> "OrthotropicRigidity":
        dy = float(dy_ratio) * float(Dx)
        dgeo = float(np.sqrt(float(Dx) * dy))
        return cls(float(Dx), dy, float(dxy_ratio) * dgeo, float(ds_ratio) * dgeo)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationParameters:
    stiffness: float
    uplift_factor: float = 1.0e-3
    dynamic_branch: str = "compression"

    def __post_init__(self) -> None:
        if self.stiffness < 0.0 or not np.isfinite(self.stiffness):
            raise ValueError("Foundation stiffness must be finite and nonnegative.")
        if self.uplift_factor < 0.0 or not np.isfinite(self.uplift_factor):
            raise ValueError("uplift_factor must be finite and nonnegative.")
        if self.dynamic_branch not in {"compression", "uplift", "none"}:
            raise ValueError("dynamic_branch must be compression, uplift, or none.")

    def effective_dynamic_stiffness(self, branch: str | None = None) -> float:
        active = self.dynamic_branch if branch is None else str(branch).lower()
        if active == "compression":
            return float(self.stiffness)
        if active == "uplift":
            return float(self.uplift_factor * self.stiffness)
        if active == "none":
            return 0.0
        raise ValueError(f"Unknown dynamic foundation branch: {active!r}.")


@dataclass(frozen=True)
class LoadParameters:
    kind: str = "patch"
    amplitude: float = -5000.0
    x0: float = 1.0
    y0: float = 0.5
    size_x: float = 0.351
    size_y: float = 0.351
    linear_start: float = 0.0
    patches: tuple[Mapping[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in {"uniform", "patch", "multi_patch", "linear_x", "modal_sine"}:
            raise ValueError(f"Unsupported load kind {self.kind!r}.")
        if self.size_x <= 0.0 or self.size_y <= 0.0:
            raise ValueError("Patch dimensions must be positive.")
        if not np.isfinite(self.amplitude):
            raise ValueError("Load amplitude must be finite.")


@dataclass(frozen=True)
class SolverTolerances:
    absolute: float = 1.0e-8
    relative: float = 1.0e-8
    maximum_iterations: int = 25

    def __post_init__(self) -> None:
        if self.absolute <= 0.0 or self.relative <= 0.0:
            raise ValueError("Solver tolerances must be positive.")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive.")


@dataclass(frozen=True)
class RayleighDamping:
    alpha_M: float = 0.0
    alpha_K: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha_M < 0.0 or self.alpha_K < 0.0:
            raise ValueError("Rayleigh coefficients must be nonnegative.")


@dataclass(frozen=True)
class ThermalEnvironment:
    ambient_temperature: float = 305.15
    substrate_temperature: float = 298.15
    reference_temperature: float = 293.15
    top_convection: float = 10.0
    emissivity: float = 0.90
    absorbed_solar_flux: float = 700.0
    contact_conductance: float = 200.0
    gap_conductance: float = 5.0
    contact_transition: float = 50.0
    conductivity_x: float = 0.35
    conductivity_y: float = 0.35
    conductivity_z: float = 0.35
    thermal_expansion_x: float = 1.3e-4
    thermal_expansion_y: float = 1.3e-4
    density: float = 950.0
    radiation: bool = True

    def __post_init__(self) -> None:
        positive = [
            self.top_convection,
            self.contact_conductance,
            self.gap_conductance,
            self.contact_transition,
            self.conductivity_x,
            self.conductivity_y,
            self.conductivity_z,
            self.density,
        ]
        if any((not np.isfinite(v)) or v <= 0.0 for v in positive):
            raise ValueError("Thermal conductivities, conductances, transition, and density must be positive.")
        if not (0.0 <= self.emissivity <= 1.0):
            raise ValueError("emissivity must lie in [0, 1].")


def from_mapping(cls: type, values: Mapping[str, Any]) -> Any:
    """Construct a dataclass from a mapping while rejecting unknown keys."""

    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(values))
