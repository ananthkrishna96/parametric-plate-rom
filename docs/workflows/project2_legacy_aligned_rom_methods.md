# Project-2 legacy-aligned non-intrusive ROM methods

This patch intentionally aligns the repository non-intrusive suite with the
method definitions used in the original `THERMOMECHANICAL_FOM_ROM.py` notebook.

It replaces the earlier lightweight suite backends with legacy-style defaults:

| Method | Legacy alignment |
|---|---|
| PODI-RBF | `RBFInterpolator`, `kernel="thin_plate_spline"`, `smoothing=0.0`; legacy `Rbf` fallback |
| PODI-linear | `LinearNDInterpolator` with nearest-neighbour fallback |
| POD-GPR | One `GaussianProcessRegressor` per POD coefficient; standardized parameters and coefficients; RBF/Matern kernel with `ConstantKernel`; optional `WhiteKernel`; `alpha=1e-9`; `n_restarts_optimizer=10`; `random_state=100` |
| POD-NN | PyTorch MLP with hidden layers `(128, 96, 64)`, ELU activation, dropout `0.0`, no layer norm, AdamW, coefficient stage and field-aware fine-tuning |
| POD-AE | Dense coefficient autoencoder plus parameter-to-latent MLP, both with hidden layers `(128, 96, 64)`, ELU activation, staged training, and optional end-to-end field-aware path |

## Important design choice

The original notebook classes fit their own `StandardScaler` objects internally.
Therefore `scripts/project2/run_rom_suite.py` now passes **raw parameters** from
`parameters_scaled.npz["raw_parameters"]` to the non-intrusive methods, not the
already-scaled common-preprocessing parameters.

The common foundation remains:

```text
data loader + POD preprocessing
```

The method layer now follows the notebook:

```text
intrusive:
  POD-projected
  POD-Galerkin capable but disabled by default

non-intrusive:
  PODI-RBF
  PODI-linear
  POD-GPR
  POD-NN
  POD-AE
```

## Run

```bash
python scripts/project2/run_rom_suite.py \
  --suite nonintrusive \
  --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH
```

To write artifacts outside Git:

```bash
python scripts/project2/run_rom_suite.py \
  --suite nonintrusive \
  --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH \
  --write \
  --overwrite
```

## Runtime note

The legacy defaults are intentionally not "fast":

- POD-NN coefficient stage: 2500 epochs
- POD-NN field stage: 275 epochs
- POD-AE coefficient AE stage: 2500 epochs
- POD-AE latent-map stage: 2500 epochs
- POD-GPR optimizer restarts: 10

This is expected because the goal is reproduction of the paper-quality notebook
configuration, not a short smoke-test setting.
