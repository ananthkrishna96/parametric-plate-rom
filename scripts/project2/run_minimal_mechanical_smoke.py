#!/usr/bin/env python3
"""Minimal Project-2 mechanical smoke run.

Run from repository root after installing the package in editable mode:

    python scripts/project2/run_minimal_mechanical_smoke.py

This is intentionally small. It verifies that the migrated compatibility layer
can import the legacy solver and run a very small mechanical solve in a FEniCS
environment.
"""

from __future__ import annotations

from paramplate.project2 import GeneralMultiphysicsSolver


def main() -> None:
    solver = GeneralMultiphysicsSolver(study_case=1)

    # Very small smoke-test controls. Increase only after the import works.
    solver.size = 8
    solver.degree = 2
    solver.bc_type = "simply_supported"
    solver.load_type = "patch"

    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)

    mu = [9.5e3, 9.5e3, 2.85e3, 3.32e3, 1.0e8, -7000.0]
    solution = solver.offline_solve_kirchhoff_problem(mu, return_mode="global", thermal_on=False)

    print("Smoke run completed.")
    print("Solution space dimension:", solution.function_space().dim())


if __name__ == "__main__":
    main()
