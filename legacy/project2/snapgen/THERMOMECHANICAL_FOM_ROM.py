#!/usr/bin/env python3
"""Compatibility bridge for legacy Project-2 SNAPGEN scripts.

The original SNAPGEN scripts expect a sibling THERMOMECHANICAL_FOM_ROM.py
defining GeneralMultiphysicsSolver. In the professional repo, the full
legacy solver is preserved in paramplate.project2.legacy_definitions.
This bridge keeps the old SNAPGEN scripts runnable without duplicating
the solver file in this folder.
"""

from paramplate.project2.legacy_definitions import *  # noqa: F401,F403
