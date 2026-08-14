#!/usr/bin/env python3
"""Evaluate the lightweight simply supported Navier reference."""

from __future__ import annotations

from paramplate.core.parameters import OrthotropicRigidity
from paramplate.mechanical.navier import solve_ssss_navier


def main() -> int:
    solution = solve_ssss_navier(
        OrthotropicRigidity(9500.0, 9500.0, 2850.0, 3320.0),
        foundation_stiffness=1000.0,
        load_amplitude=-1000.0,
        n_terms=21,
        nx=41,
        ny=21,
    )
    print(f"maximum |w| = {solution.maximum_absolute_displacement:.6e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
