# Transient dynamics workflow

## Full-order trajectory

The transient study advances an aligned-orthotropic Kirchhoff--Love plate with transverse translational inertia, consistent mass, time-dependent loading, and optional algebraic Rayleigh damping. One compression, uplift, or no-foundation branch is fixed for the complete trajectory. The model does not switch contact state during time integration.

`TransientPlateFOM` reuses the spatial `C0-IPG` and seam formulation, assembles the consistent mass matrix, removes constrained and zero-mass mixed-space entries through a common active-degree restriction, and advances the active system with average-acceleration Newmark. The thesis campaigns are undamped, start from zero displacement and velocity, use `dt = 1e-3` s on `[0, 0.20]` s, and store every fifth state, including both endpoints, for 41 states per trajectory.

The current sparse-matrix conversion and Newmark path is implemented for serial DOLFIN execution. It rejects a distributed PETSc matrix rather than silently assembling an invalid local SciPy system.

The implemented verification utilities cover the analytical simply supported frequency expression, modal extraction, direct time integration, and energy drift. Full panel-to-monolithic and analytical-field comparisons require a compatible DOLFIN environment and the thesis reference configurations.

## Campaigns and split protocol

| Campaign | Physical parameters | Trajectories | Split | POD rank |
|---|---|---:|---:|---:|
| Case 2, monolithic | load amplitude | 200 | 160 / 20 / 20 | 7 |
| Case 2, one interface | load amplitude | 250 | 200 / 25 / 25 | 8 |
| Case 3, monolithic | `Dx`, load amplitude | 250 | 200 / 25 / 25 | 8 |

The sampled load amplitude is positive in the database and is applied as the downward pressure `q = -Aq`. Complete trajectory IDs are divided into training, validation, and test subsets before the state rows are flattened. Only training trajectories enter the SVD, finite-element reorthonormalization, input and coordinate preprocessing, and model fitting.

## Predictive ROM

The reduced output is displacement. PODI--RBF, PODI--Linear, POD--GPR, POD--NN, and POD--DL-ROM map physical parameters and query time to displacement coordinates. Time is an independent query coordinate, so these models are non-causal, time-conditioned field predictors. The active code enforces the known zero initial coordinates at `time = 0` after predictive coordinate evaluation. Direct DL-ROM is not part of the transient workflow.

Reported transient timing must preserve the operation being compared: one complete FOM trajectory versus one reconstructed ROM state. A lightweight timing of one vectorized prediction is not interchangeable with that ratio.

```bash
python scripts/dynamics/run_workflow.py \
  --config configs/dynamics/thesis_case2_monolithic.yml --dry-run
python scripts/dynamics/run_workflow.py \
  --config configs/dynamics/smoke.yml \
  --action snapshots --output results/dynamics_smoke.npz
paramplate-validate-archive /external/path/case2_monolithic.npz --kind dynamics
python scripts/common/run_rom.py /external/path/case2_monolithic.npz \
  --study dynamics --method pod-dl-rom --rank 7 \
  --options configs/rom/transient_thesis.json
```
