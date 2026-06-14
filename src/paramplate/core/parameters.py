"""Parameter-space utilities for many-query and ROM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ParameterScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"


class ParameterRole(str, Enum):
    MECHANICAL = "MP"
    THERMAL = "TP"


@dataclass(frozen=True)
class ParameterRange:
    """One scalar parameter range for snapshot generation."""

    name: str
    lower: float
    upper: float
    unit: str = ""
    scale: ParameterScale = ParameterScale.LINEAR
    role: ParameterRole = ParameterRole.MECHANICAL
    description: str = ""

    def validate(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError(f"Non-finite range for {self.name}.")
        if self.upper <= self.lower:
            raise ValueError(f"Invalid range for {self.name}: upper must exceed lower.")
        if self.scale == ParameterScale.LOG and self.lower <= 0.0:
            raise ValueError(f"Log-scale parameter {self.name} must have lower > 0.")
