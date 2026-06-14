"""Load definitions for plate models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LoadType(str, Enum):
    """Supported mechanical load types."""

    UNIFORM = "uniform"
    PATCH = "patch"
    MULTI_PATCH = "multi_patch"
    LINEAR = "linear"


@dataclass(frozen=True)
class PatchLoad:
    """Localized rectangular patch load."""

    x0: float = 1.0
    y0: float = 0.5
    size_x: float = 0.351
    size_y: float = 0.351
    value: float = -3000.0

    def validate(self) -> None:
        if self.size_x <= 0.0:
            raise ValueError("Patch size_x must be positive.")
        if self.size_y <= 0.0:
            raise ValueError("Patch size_y must be positive.")


@dataclass(frozen=True)
class LinearLoad:
    """Linearly varying load along x."""

    x_start: float = 0.0
    x_end: float = 2.0
    value_start: float = 0.0
    value_end: float = -3000.0

    def validate(self) -> None:
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be larger than x_start.")
