# Transient dynamics workflow

## Full-order trajectory

The transient study advances an aligned-orthotropic Kirchhoff–Love plate with transverse translational inertia, consistent mass, time-dependent loading, and optional algebraic Rayleigh damping. One compression, uplift, or no-foundation branch is fixed for the complete trajectory. The model does not switch contact state during time integration.

`TransientPlateFOM` reuses the spatial `C0-IPG` and seam formulation, assembles the consistent mass matrix, removes constrained and zero-mass mixed-space entries through a common active-degree restriction, and advances the active system with average-acceleration Newmark. The thesis ROM campaigns are undamped, start from zero displacement and velocity, use `dt = 1e-3` s on `[0, 0.20]` s, and store every fifth state, including both endpoints, for 41 states per trajectory.

The PETSc-to-SciPy matrix conversion and Newmark path currently require serial DOLFIN execution. A distributed PETSc matrix is rejected rather than treated as a complete local matrix.

## Verification

`configs/dynamics/verification_base.yml` records the Chapter 10 simply supported benchmark with mesh resolution 32, quadratic elements, the isotropic specialization of the aligned-orthotropic rigidities, density `1200 kg/m^3`, and the no-foundation, compression, and regularized-uplift regimes.

The verification driver provides:

- ordered finite-element and Navier natural frequencies;
- direct Newmark response against a spatially analytical, time-discrete first-mode reference;
- direct versus finite-element modal Newmark integration;
- undamped free-vibration energy history;
- an eight-mode harmonic-response diagnostic;
- direct versus twelve-mode response under a localized patch load;
- two-panel versus monolithic probe histories.

```bash
python scripts/dynamics/run_verification.py \
  --config configs/dynamics/verification_base.yml \
  --mode frequencies --dry-run

python scripts/dynamics/run_verification.py \
  --config configs/dynamics/verification_panel_2.yml \
  --reference-config configs/dynamics/verification_base.yml \
  --mode panel --dry-run
```

Removing `--dry-run` requires DOLFIN 2019.1 and `mshr`. The driver writes numerical arrays and compact JSON summaries; it does not embed the published thesis values as expected answers.

## Campaigns and trajectory split

| Campaign | Physical parameters | Trajectories | Split | POD rank |
|---|---|---:|---:|---:|
| Case 2, monolithic | load amplitude | 200 | 160 / 20 / 20 | 7 |
| Case 2, one interface | load amplitude | 250 | 200 / 25 / 25 | 8 |
| Case 3, monolithic | `Dx`, load amplitude | 250 | 200 / 25 / 25 | 8 |

The sampled load amplitude is positive in the database and is applied as the downward pressure `q = -Aq`. Complete trajectory IDs are divided into training, validation, and test subsets before state rows are flattened. Only training trajectories enter SVD, finite-element reorthonormalization, input and coordinate preprocessing, and model fitting.

```bash
python scripts/dynamics/run_workflow.py \
  --config configs/dynamics/thesis_case2_monolithic.yml --dry-run
python scripts/dynamics/run_workflow.py \
  --config configs/dynamics/smoke.yml \
  --action snapshots --output results/dynamics_smoke.npz
```

## Predictive ROM

The reduced output is displacement. PODI–RBF, PODI–Linear, POD–GPR, POD–NN, and POD–DL-ROM map physical parameters and query time to displacement coordinates. Time is an independent query coordinate, so the predictors are non-causal and time-conditioned. The code enforces the known zero initial coordinates at `time = 0` after predictive coordinate evaluation. Direct DL-ROM is not part of the transient workflow.

Reported timing preserves the operation being compared: one complete FOM trajectory versus one reconstructed ROM state. A vectorized batch prediction is not interchangeable with that ratio.
