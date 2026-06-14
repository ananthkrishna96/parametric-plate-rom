# Project-2 ROM suite architecture

This stage reorganizes the workflow into a suite-based structure rather than
adding one ROM method at a time.

## Common foundation

Already available:

1. `scripts/project2/check_rom_dataset.py`
2. `scripts/project2/build_pod_dataset.py`

This foundation is shared by both intrusive and non-intrusive ROMs.

## Intrusive ROM suite

Current entries:

- `pod-projected`: available now. It projects FOM snapshots onto the POD basis
  and reports train/test projection errors.
- `pod-galerkin`: capability entry, disabled by default. It should only be
  enabled after the legacy FEniCS residual/Jacobian and thermomechanical case
  reconstruction are wrapped safely.

## Non-intrusive ROM suite

Current entries:

- `podi-rbf`
- `podi-linear`
- `pod-gpr`
- `pod-nn`
- `pod-ae`

The `pod-ae` entry is an initial lightweight latent baseline: POD coefficients
are compressed to a bottleneck using a linear autoencoding/PCA step and a neural
regressor maps parameters to the latent space. A nonlinear dense autoencoder can
later replace this backend behind the same API.

## Unified command

List the registry:

```bash
python scripts/project2/run_rom_suite.py --list --suite all
```

Run the intrusive suite:

```bash
python scripts/project2/run_rom_suite.py   --suite intrusive   --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH
```

Run the non-intrusive suite:

```bash
python scripts/project2/run_rom_suite.py   --suite nonintrusive   --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH
```

Write outputs outside Git:

```bash
python scripts/project2/run_rom_suite.py   --suite all   --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH   --write   --overwrite
```

Output root:

```text
parametric-plate-rom-data/paper2_thermomechanical/rom_training/suites/
```
