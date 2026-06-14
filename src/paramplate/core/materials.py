"""Material and foundation parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrthotropicPlateRigidity:
    """Orthotropic bending and twisting rigidity parameters."""

    Dx: float
    Dy: float
    Dxy: float
    Ds: float

    def validate(self) -> None:
        if self.Dx <= 0.0:
            raise ValueError("Dx must be positive.")
        if self.Dy <= 0.0:
            raise ValueError("Dy must be positive.")
        if self.Ds <= 0.0:
            raise ValueError("Ds must be positive.")


@dataclass(frozen=True)
class WinklerFoundation:
    """Winkler-type elastic foundation parameterization."""

    stiffness: float
    tension_factor: float = 1.0e-3

    def validate(self) -> None:
        if self.stiffness < 0.0:
            raise ValueError("Foundation stiffness must be non-negative.")
        if self.tension_factor < 0.0:
            raise ValueError("Foundation tension factor must be non-negative.")


@dataclass(frozen=True)
class ThermalExpansion:
    """Orthotropic thermal expansion coefficients."""

    alpha1: float = 1.3e-4
    alpha2: float = 1.3e-4

    def validate(self) -> None:
        # Allow negative CTEs mathematically, but require finite values.
        if not all(abs(v) < float("inf") for v in (self.alpha1, self.alpha2)):
            raise ValueError("Thermal expansion coefficients must be finite.")
