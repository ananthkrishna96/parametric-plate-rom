# Capability matrix

Statuses use the repository acceptance categories. “Verified” means exercised by dependency-light tests in this reconstruction. DOLFIN-dependent code is present but was not executed in the preparation environment. “External data” means the generation or loading path is documented and implemented while the full archive is intentionally not committed.

## Static mechanics

| Capability | Status | Implementation |
|---|---|---|
| FOM | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | `paramplate.mechanical.fem.MechanicalPlateFOM` |
| Verification/smoke path | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | FOM comparisons require DOLFIN; Navier/reference utilities are tested |
| Snapshot/data pipeline | DOCUMENTED EXTERNAL-DATA WORKFLOW | `paramplate.mechanical.snapshots` |
| POD | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.pod` |
| Intrusive POD--Galerkin | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | `paramplate.mechanical.intrusive` |
| POD--Proj | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.pod` |
| PODI--RBF | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.predictors` |
| PODI--Linear | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.predictors` |
| POD--GPR | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.predictors` |
| POD--NN | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.neural` |
| Post-processing, configuration, documentation, tests | IMPLEMENTED AND VERIFIED IN THIS REBUILD | package, `configs/mechanical`, `docs`, `tests` |

## Steady thermomechanics

| Capability | Status | Implementation |
|---|---|---|
| Coupled FOM | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | `paramplate.thermomechanical.fem.ThermomechanicalPlateFOM` |
| Thermal-to-plate reduction | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | FEniCS reduction path present; analytical first-moment checks are tested |
| Partitioned solver | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.thermomechanical.coupling` |
| Snapshot/data pipeline | DOCUMENTED EXTERNAL-DATA WORKFLOW | `paramplate.thermomechanical.snapshots` |
| Displacement POD | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.pod` |
| Thermal-driver POD | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.pod` |
| Hybrid mechanical POD--Galerkin | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | `paramplate.thermomechanical.hybrid` |
| POD--Proj, PODI--RBF, PODI--Linear, POD--GPR, POD--NN | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom` |
| POD--DL-ROM | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.deep.PODDLROM` |
| DL-ROM | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.deep.DirectDLROM` |
| Post-processing, configuration, documentation, tests | IMPLEMENTED AND VERIFIED IN THIS REBUILD | package, `configs/thermomechanical`, `docs`, `tests` |

## Transient dynamics

| Capability | Status | Implementation |
|---|---|---|
| FOM | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | `paramplate.dynamics.fem.TransientPlateFOM` |
| Active dynamic DOFs | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.dynamics.active_dofs` |
| Newmark integration | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.dynamics.newmark` |
| Verification/smoke path | IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN | FOM/modal paths require DOLFIN; analytical and algebraic checks are tested |
| Complete-trajectory data pipeline | DOCUMENTED EXTERNAL-DATA WORKFLOW | `paramplate.dynamics.trajectories` |
| Trajectory-level split | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.splits` |
| POD and POD--Proj | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.pod` |
| PODI--RBF, PODI--Linear, POD--GPR, POD--NN | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom` |
| POD--DL-ROM | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.deep.TransientPODDLROM` |
| Parameter--time prediction | IMPLEMENTED AND VERIFIED IN THIS REBUILD | `paramplate.rom.features` |
| Direct DL-ROM | NOT SUPPORTED BY CURRENT AUTHORITATIVE FILES | intentionally rejected for transient data |
| Post-processing, configuration, documentation, tests | IMPLEMENTED AND VERIFIED IN THIS REBUILD | package, `configs/dynamics`, `docs`, `tests` |
