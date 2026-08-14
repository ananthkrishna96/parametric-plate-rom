# Contributing

Issue reports and reproducibility questions are welcome. Pull requests should be discussed with the repository maintainer before substantial work begins because this public source release does not grant an open-source license.

Changes must preserve the distinction between the full-order state and the reduced output. In particular, the thermomechanical FOM remains coupled although its predictive ROMs are field-specific, and the transient predictors are time-conditioned field surrogates rather than reduced time integrators.

Before proposing a change:

1. add or update a focused test;
2. run `python -m compileall -q src scripts examples` and `pytest -q -m 'not fenics'`;
3. keep large snapshots, meshes, checkpoints, and generated results outside Git;
4. state explicitly which FEniCS/DOLFIN checks were not executed;
5. update the corresponding configurations and workflow documentation whenever sign conventions, units, split roles, or method names change.

See [`LICENSE`](LICENSE) for the current reuse status.
