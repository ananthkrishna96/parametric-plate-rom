# Contributing

Changes should preserve the scientific distinction between the full-order state and the reduced output. In particular, the thermomechanical FOM remains coupled although its predictive ROMs are field-specific, and the transient predictors are time-conditioned field surrogates rather than reduced time integrators.

Before submitting a change:

1. add or update a focused test;
2. run `python -m compileall -q src scripts examples` and `pytest -q`;
3. keep large snapshots, meshes, checkpoints, and generated result folders outside Git;
4. document any FEniCS/DOLFIN test that could not be executed;
5. avoid changing sign conventions, units, split roles, or method names without updating the corresponding workflow documentation and configurations.

The repository license is not yet finalized. External contributions should not be merged until that status is resolved.
