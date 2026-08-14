# Reproducibility levels

## Level A: quick software-path checks

Level A uses the small configurations in each study directory and the deterministic synthetic arrays in `data/sample/`. It checks configuration parsing, solver entry points, snapshot schemas, data splitting, POD, prediction, reconstruction, and plotting at modest cost. The static Navier reference, centered thermal-driver reduction, algebraic Newmark integrator, and ROM examples run without FEniCS.

A Level A result is evidence that the software path is functioning. It is not a reproduction of the numerical values, timing tables, trained models, or error distributions in the dissertation.

## Level B: thesis workflow

Level B uses the `thesis_*.yml` configurations and, where needed, external snapshot or trajectory archives. The configurations preserve the reported spatial degree, mesh resolution, heat discretization, time grid, stored-state cadence, data roles, seeds, and reduced dimensions. Full campaigns can require hundreds of nonlinear or coupled solves and large field arrays.

The repository therefore distinguishes three cases:

1. a full-order path that can regenerate data when the legacy FEniCS environment and computing resources are available;
2. a documented external-data path that validates and consumes existing snapshot archives;
3. a dependency-light ROM path that can be tested independently of FEniCS.

Published numerical tables should be compared only after confirming the exact case configuration, external data provenance, finite-element metric, accepted-state filtering, and machine-specific timing protocol.
