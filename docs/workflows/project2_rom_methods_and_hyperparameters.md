# Project-2 ROM methods and hyperparameters

This file records the current Project-2 ROM method configuration after the legacy-aligned migration. The non-intrusive methods intentionally mirror the `THERMOMECHANICAL_FOM_ROM.py` notebook defaults. They are not independently retuned for the repository.

## POD preprocessing

Command:

```bash
python scripts/project2/build_pod_dataset.py --case "$CASE" --write --overwrite
```

Default preprocessing settings:

| setting | default |
|---|---:|
| train fraction | `0.8` |
| test fraction | `0.2` |
| split seed | `42` |
| POD energy tolerance | `0.9999` |
| centering | enabled |
| displacement basis file | `pod_w.npz` |
| thermal-driver basis file | `pod_theta.npz` |

For the current Case-2 benchmark, the clean fresh run gives:

| field | rank | retained energy | train projection error | test projection error |
|---|---:|---:|---:|---:|
| `w` | 3 | `0.999980833902` | `2.633392e-03` | `2.282533e-03` |
| `theta` | 1 | `0.999990329083` | `1.212565e-03` | `9.530887e-04` |

## Basis-selection policy

The basis dimension should be chosen using both POD spectrum information and projection-error validation:

1. start from a strict cumulative energy threshold;
2. compute the POD projection train/test error;
3. check whether the projection error has reached a plateau;
4. use the smallest rank that reaches the plateau;
5. only increase rank if it improves the projection lower bound or downstream ROM accuracy in a meaningful way.

The projection result is the lower bound for non-intrusive methods that use the same POD space.

## Intrusive suite

| method | status | role |
|---|---|---|
| `pod-projected` | available | projection/truncation lower bound |
| `pod-galerkin` | capable-disabled | true online nonlinear intrusive ROM, not yet migrated |

`pod-galerkin` is intentionally disabled. Enabling it requires case-specific FEniCS state reconstruction, residual/Jacobian projection, boundary-condition handling, interface penalties, contact/foundation nonlinearities, and thermomechanical coupling consistency.

## PODI-RBF

Purpose: simple interpolation baseline from parameters to POD coefficients.

| setting | value |
|---|---|
| method | `RBFInterpolator` with fallback support |
| kernel | `thin_plate_spline` |
| smoothing | `0.0` |
| nearest fallback | enabled |
| seed | `100` |
| internal split defaults | train/val/test `0.80/0.10/0.10` |

## PODI-linear

Purpose: linear interpolation baseline.

| setting | value |
|---|---|
| method | `LinearNDInterpolator` |
| fallback | `NearestNDInterpolator` for out-of-hull/NaN cases |
| seed | `100` |
| internal split defaults | train/val/test `0.80/0.10/0.10` |

## POD-GPR

Purpose: probabilistic coefficient regression, one Gaussian process per POD coefficient.

| setting | value |
|---|---|
| coefficient scaling | `StandardScaler` |
| kernel type | RBF |
| constant kernel initial value | `1.0` |
| constant bounds | `(1e-3, 1e3)` |
| length-scale bounds | `(1e-2, 1e2)` |
| white kernel | enabled |
| white-noise level | `1e-7` |
| white-noise bounds | `(1e-10, 1e-3)` |
| alpha | `1e-9` |
| optimizer restarts | `10` |
| random state | `100` |
| normalize_y | `False` |
| fallback alpha | `1e-7` |
| fallback restarts | `5` |

## POD-NN

Purpose: neural parameter-to-POD-coefficient regression with optional field-aware fine-tuning.

Architecture:

```text
input parameters -> 128 -> 96 -> 64 -> n_basis
```

| setting | value |
|---|---|
| activation | ELU |
| dropout | `0.0` |
| layer norm | disabled |
| optimizer | AdamW |
| device preference | CPU |
| PyTorch threads | 1 |
| seed | `100` |
| coefficient epochs | `2500` |
| coefficient learning rate | `5e-4` |
| coefficient batch size | `256` |
| coefficient weight decay | `2e-6` |
| coefficient scheduler patience | `120` |
| coefficient early-stop patience | `450` |
| coefficient min delta | `5e-9` |
| coefficient gradient clip | `1.0` |
| field fine-tuning | enabled |
| field epochs | `275` |
| field learning rate | `3e-5` |
| field batch size | `256` |
| field weight decay | `5e-7` |
| field scheduler patience | `35` |
| field early-stop patience | `80` |
| field min delta | `5e-10` |
| field gradient clip | `1.0` |
| field loss weight | `1.0` |
| coefficient anchor weight | `0.01` |

Important implementation note: the field fine-tuning loss reconstructs centered POD fields as

```text
u_pred = C_pred @ basis.T + mean
```

This is required because the repository POD preprocessing uses centered SVD artifacts.

## POD-AE

Purpose: dense coefficient autoencoder plus parameter-to-latent neural map.

Coefficient autoencoder:

```text
coefficients -> 128 -> 96 -> 64 -> latent -> 64 -> 96 -> 128 -> coefficients
```

Parameter-to-latent network:

```text
input parameters -> 128 -> 96 -> 64 -> latent
```

| setting | value |
|---|---|
| latent dimension | `basis` by default |
| activation | ELU |
| dropout | `0.0` |
| layer norm | disabled |
| device preference | CPU |
| seed | `100` |
| AE epochs | `2500` |
| AE learning rate | `5e-4` |
| AE batch size | `256` |
| AE weight decay | `2e-6` |
| AE scheduler patience | `120` |
| AE early-stop patience | `450` |
| AE min delta | `5e-9` |
| latent-map epochs | `2500` |
| latent-map learning rate | `5e-4` |
| latent-map batch size | `256` |
| latent-map weight decay | `2e-6` |
| latent-map scheduler patience | `120` |
| latent-map early-stop patience | `450` |
| end-to-end fine-tuning | enabled |
| e2e epochs | `500` |
| e2e learning rate | `3e-5` |
| e2e batch size | `256` |
| e2e train decoder | `True` |
| e2e train encoder | `False` |
| e2e weight decay | `5e-7` |
| e2e scheduler patience | `50` |
| e2e patience | `120` |
| e2e loss | `field_relative_plus_coeff` |
| e2e field weight | `1.0` |
| e2e coefficient weight | `0.01` |

## Current Case-2 result interpretation

For the clean `v0.8` run, the expected ranking is:

1. `pod-projected`: POD projection lower bound.
2. `pod-gpr`: best current non-intrusive method.
3. `pod-nn`: best current neural method.
4. `pod-ae`: useful latent baseline.
5. `podi-rbf`: simple interpolation baseline.
6. `podi-linear`: weakest for this case.

These rankings should be rechecked for every new case/campaign. Do not assume that a method ranking from one parameterization transfers automatically to another.
