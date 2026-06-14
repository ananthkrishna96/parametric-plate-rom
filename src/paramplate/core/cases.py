"""High-level case descriptors for plate simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

from paramplate.core.boundary_conditions import BoundaryConditionType
from paramplate.core.loads import LoadType
from paramplate.core.parameters import ParameterRange


@dataclass(frozen=True)
class CaseDefinition:
    """Descriptor for a parameterized simulation case."""

    case_id: int
    name: str
    description: str
    parameter_ranges: tuple[ParameterRange, ...] = field(default_factory=tuple)
    boundary_condition: BoundaryConditionType = BoundaryConditionType.FREE_EDGE
    load_type: LoadType = LoadType.PATCH
    n_vertical_interfaces: int = 0
    n_horizontal_interfaces: int = 0

    def validate(self) -> None:
        if self.case_id <= 0:
            raise ValueError("case_id must be positive.")
        if not self.name:
            raise ValueError("case name must be non-empty.")
        if self.n_vertical_interfaces < 0:
            raise ValueError("n_vertical_interfaces must be non-negative.")
        if self.n_horizontal_interfaces < 0:
            raise ValueError("n_horizontal_interfaces must be non-negative.")
        for parameter_range in self.parameter_ranges:
            parameter_range.validate()
