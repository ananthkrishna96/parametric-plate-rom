# Capability matrix

The matrix distinguishes dependency-light verification, DOLFIN-dependent implementation, external-data workflows, and deliberately unsupported methods. “Verified” means exercised by automated tests in this release. “Environment-dependent” means the code path is present but was not executed without DOLFIN 2019.1 and `mshr`. “External data” means the generation, loading, and validation path is maintained while the full research archive is intentionally not committed.

## Static mechanics

| Capability | Status | Implementation |
|---|---|---|
| FOM | ENVIRONMENT-DEPENDENT | `paramplate.mechanical.fem.MechanicalPlateFOM` |
| Verification/smoke path | ENVIRONMENT-DEPENDENT | Navier, external-grid, and panel-to-monolithic drivers in `paramplate.mechanical.verification` and `scripts/mechanical/run_verification.py` |
| Snapshot/data pipeline | EXTERNAL DATA | `paramplate.mechanical.snapshots` |
| POD | VERIFIED | `paramplate.rom.pod` |
| Intrusive POD–Galerkin | ENVIRONMENT-DEPENDENT | `paramplate.mechanical.intrusive` |
| POD–Proj diagnostic | VERIFIED | `paramplate.rom.pod` |
| PODI–RBF | VERIFIED | `paramplate.rom.predictors` |
| PODI–Linear | VERIFIED | `paramplate.rom.predictors` |
| POD–GPR | VERIFIED | `paramplate.rom.predictors` |
| POD–NN | VERIFIED | `paramplate.rom.neural` |
| Post-processing | VERIFIED | `paramplate.postprocessing` |
| Configuration | VERIFIED | `configs/mechanical` |
| Documentation | VERIFIED | `docs/workflows/mechanical.md` |
| Tests | VERIFIED | `tests` |

## Steady thermomechanics

| Capability | Status | Implementation |
|---|---|---|
| Coupled FOM | ENVIRONMENT-DEPENDENT | `paramplate.thermomechanical.fem.ThermomechanicalPlateFOM` |
| Thermal-to-plate reduction | ENVIRONMENT-DEPENDENT | FEniCS reduction path in `paramplate.thermomechanical.heat`; analytical first-moment checks are verified |
| Partitioned solver | VERIFIED | `paramplate.thermomechanical.coupling` |
| Verification/smoke path | ENVIRONMENT-DEPENDENT | Fixed-driver auxiliary Navier and external-grid comparisons in `paramplate.thermomechanical.verification` and `scripts/thermomechanical/run_verification.py` |
| Snapshot/data pipeline | EXTERNAL DATA | `paramplate.thermomechanical.snapshots` |
| Displacement POD | VERIFIED | `paramplate.rom.pod` |
| Thermal-driver POD | VERIFIED | `paramplate.rom.pod` |
| Hybrid mechanical POD–Galerkin | ENVIRONMENT-DEPENDENT | `paramplate.thermomechanical.hybrid` |
| POD–Proj diagnostic | VERIFIED | `paramplate.rom.pod` |
| PODI–RBF | VERIFIED | `paramplate.rom.predictors` |
| PODI–Linear | VERIFIED | `paramplate.rom.predictors` |
| POD–GPR | VERIFIED | `paramplate.rom.predictors` |
| POD–NN | VERIFIED | `paramplate.rom.neural` |
| POD–DL-ROM | VERIFIED | `paramplate.rom.deep.PODDLROM` |
| DL-ROM | VERIFIED | `paramplate.rom.deep.DirectDLROM` |
| Post-processing | VERIFIED | `paramplate.postprocessing` |
| Configuration | VERIFIED | `configs/thermomechanical` |
| Documentation | VERIFIED | `docs/workflows/thermomechanical.md` |
| Tests | VERIFIED | `tests` |

## Transient dynamics

| Capability | Status | Implementation |
|---|---|---|
| FOM | ENVIRONMENT-DEPENDENT | `paramplate.dynamics.fem.TransientPlateFOM` |
| Active dynamic DOFs | VERIFIED | `paramplate.dynamics.active_dofs` |
| Newmark integration | VERIFIED | `paramplate.dynamics.newmark` |
| Verification/smoke path | ENVIRONMENT-DEPENDENT | Frequency, single-mode, finite-element modal, energy, harmonic, patch, and panel drivers in `paramplate.dynamics.verification` and `scripts/dynamics/run_verification.py` |
| Complete-trajectory snapshot pipeline | EXTERNAL DATA | `paramplate.dynamics.trajectories` |
| Trajectory-level split | VERIFIED | `paramplate.rom.splits` |
| POD | VERIFIED | `paramplate.rom.pod` |
| POD–Proj diagnostic | VERIFIED | `paramplate.rom.pod` |
| PODI–RBF | VERIFIED | `paramplate.rom.predictors` |
| PODI–Linear | VERIFIED | `paramplate.rom.predictors` |
| POD–GPR | VERIFIED | `paramplate.rom.predictors` |
| POD–NN | VERIFIED | `paramplate.rom.neural` |
| POD–DL-ROM | VERIFIED | `paramplate.rom.deep.TransientPODDLROM` |
| Parameter–time prediction | VERIFIED | `paramplate.rom.features` |
| Post-processing | VERIFIED | `paramplate.postprocessing` |
| Configuration | VERIFIED | `configs/dynamics` |
| Documentation | VERIFIED | `docs/workflows/transient.md` |
| Tests | VERIFIED | `tests` |
| Direct transient DL-ROM | NOT SUPPORTED | Intentionally rejected by the method inventory |
