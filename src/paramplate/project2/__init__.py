"""Project-2 thermomechanical compatibility layer.

This package is the first non-lossy migration layer from the notebook-converted
``THERMOMECHANICAL_FOM_ROM.py`` source into the professional ``paramplate``
repository. The exact source file is preserved under ``legacy/project2``.
The importable definitions-only layer is exposed here for gradual refactoring.
"""

from paramplate.project2.solver import GeneralMultiphysicsSolver

__all__ = ["GeneralMultiphysicsSolver"]
