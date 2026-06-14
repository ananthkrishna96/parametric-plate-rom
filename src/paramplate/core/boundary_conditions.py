"""Boundary-condition identifiers for plate models."""

from __future__ import annotations

from enum import Enum


class BoundaryConditionType(str, Enum):
    """Supported boundary-condition families."""

    SIMPLY_SUPPORTED = "simply_supported"
    FREE_EDGE = "free_edge"
