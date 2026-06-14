"""Core data structures for parameterized plate models."""

from paramplate.core.boundary_conditions import BoundaryConditionType
from paramplate.core.cases import CaseDefinition
from paramplate.core.geometry import PlateGeometry
from paramplate.core.loads import LinearLoad, LoadType, PatchLoad
from paramplate.core.materials import OrthotropicPlateRigidity, ThermalExpansion, WinklerFoundation
from paramplate.core.parameters import ParameterRange, ParameterRole, ParameterScale

__all__ = [
    "BoundaryConditionType",
    "CaseDefinition",
    "LinearLoad",
    "LoadType",
    "OrthotropicPlateRigidity",
    "ParameterRange",
    "ParameterRole",
    "ParameterScale",
    "PatchLoad",
    "PlateGeometry",
    "ThermalExpansion",
    "WinklerFoundation",
]
