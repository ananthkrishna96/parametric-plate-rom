# Reproducibility levels

## Level A: software-path checks

Level A uses small study configurations and deterministic synthetic arrays in `data/sample/`. It checks configuration parsing, operational sampling ranges, solver entry points, archive schemas, data splitting, POD, prediction, reconstruction, and plotting at modest cost. The static Navier reference, centered thermal-driver reduction, algebraic Newmark integrator, and ROM examples run without FEniCS.

Level A demonstrates that the software path is internally executable. It does not reproduce the thesis meshes, databases, trained models, wall times, tables, or publication figures.

## Level B: thesis workflows

Level B uses the thesis and verification configurations together with a compatible legacy DOLFIN/PETSc environment and, where required, external research artifacts. It preserves the reported discretization, signs, coupling settings, data partitions, output spaces, POD ranks, method inventories, and timing definitions.

The full campaigns are computationally expensive:

- static mechanics requires repeated nonlinear plate solves and, for some comparisons, independent external references;
- steady thermomechanics requires repeated three-dimensional heat and nonlinear plate solves within a partitioned iteration;
- transient dynamics requires one complete Newmark trajectory per physical query before trajectory-level splitting.

A Level-B configuration records the intended workflow but is not evidence that the complete campaign was rerun on the current machine. The repository does not bundle the heavy fine-mesh references, full snapshot databases, complete thesis trajectory archives, or trained model checkpoints.

## Reproduction record

For a full study, retain at least:

- the composed configuration and package revision;
- software and solver versions;
- hardware and thread/MPI settings;
- random seeds and accepted/rejected sample identifiers;
- archive manifests and hashes;
- split assignments;
- POD dimensions and preprocessing fitted only on training data;
- solver convergence metadata;
- output arrays used for post-processing.
