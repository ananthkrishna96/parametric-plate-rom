"""Geometric definitions for parameterized plate models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateGeometry:
    """Rectangular Kirchhoff--Love plate geometry."""

    length: float = 2.0
    width: float = 1.0
    thickness: float = 0.05

    def validate(self) -> None:
        if self.length <= 0.0:
            raise ValueError("Plate length must be positive.")
        if self.width <= 0.0:
            raise ValueError("Plate width must be positive.")
        if self.thickness <= 0.0:
            raise ValueError("Plate thickness must be positive.")
