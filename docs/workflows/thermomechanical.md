# Steady thermomechanical workflow

## Coupled full-order problem

The thermomechanical workflow combines a steady three-dimensional heat problem with the nonlinear plate solver. The top face includes convection, long-wave radiation, and absorbed solar flux. The lower-face conductance varies smoothly between contact and gap values according to the current upward-positive displacement. The temperature increment is reduced to the centered through-thickness driver `theta` in K/m; `theta` is not temperature. Orthotropic thermal moments constructed from this driver enter mechanical equilibrium.

`ThermomechanicalPlateFOM` executes a relaxed partitioned iteration:

1. solve the three-dimensional heat problem at the current displacement;
2. reduce the temperature increment to `theta` on the plate mesh;
3. solve the nonlinear mechanical problem with fixed thermal bending moments;
4. relax displacement and test the finite-element updates of both displacement and `theta`.

The accepted record follows the executed research workflow: it stores the last relaxed displacement and the `theta` field computed at the start of that same iteration. It does not add an unreported post-convergence thermal correction.

The final ROM databases use plate resolution 64 with quadratic Lagrange functions, a `64 x 32 x 16` linear thermal mesh, a linear `theta` space, 20 through-thickness quadrature points, relaxation `0.7`, tolerance `1e-5`, and at most 25 coupling iterations. Chapter 7 verification configurations use separate, tighter settings.

## Verification

The centered first-moment check is dependency-light:

```bash
python scripts/thermomechanical/run_verification.py \
  --config configs/thermomechanical/verification_auxiliary_navier.yml \
  --mode thermal-driver
```

The auxiliary Navier driver solves the coupled FOM, freezes the converged thermal driver, projects the mechanical load and `theta` onto a double-sine basis, and compares the finite-element displacement with the finite constant-foundation series. The full configuration records the Chapter 7 geometry, material data, `401 x 401` projection grid, modes `1 <= m,n <= 159`, thermal discretization, and coupling tolerances.

```bash
python scripts/thermomechanical/run_verification.py \
  --config configs/thermomechanical/verification_auxiliary_navier.yml \
  --mode navier --dry-run
```

The reference is intentionally limited: it is a single-panel, simply supported, constant-foundation problem using a fixed reconstructed thermal driver. It is not a solution of the displacement-dependent fixed point. External regular-grid displacement and thermal-driver references can be supplied with `--mode external --reference-archive ...`; loading remains pickle-disabled.

The plate-only and coupled refinement tables require the fine-mesh reference states and the legacy DOLFIN environment. Those heavy fields are not bundled, but their comparison route, units, and expected regular-grid archive format are documented.

## Snapshot cases and output separation

| Case | Parameters | Accepted samples | Split | POD ranks `(w, theta)` |
|---|---|---:|---:|---:|
| 2 | mechanical load, absorbed solar flux | 450 | 360 / 45 / 45 | 6 / 4 |
| 3 | foundation stiffness, ambient temperature, substrate temperature | 378 | 302 / 38 / 38 | 6 / 4 |
| 5 | foundation stiffness, load, solar flux, contact conductance, gap conductance | 500 | 400 / 50 / 50 | 6 / 4 |

One nonconverged sample is omitted from the requested 379 Case-3 runs. Displacement and `theta` use matched sample assignments but separate arrays, spaces, metrics, bases, scalers, models, units, and errors. They are not a coupled ROM target.

```bash
python scripts/thermomechanical/run_workflow.py \
  --config configs/thermomechanical/thesis_case2.yml --dry-run
python scripts/thermomechanical/run_workflow.py \
  --config configs/thermomechanical/smoke.yml \
  --action snapshots --output results/thermomechanical_smoke.npz
```

## Reduced-order workflows

The hybrid workflow keeps the heat solve, thermal reduction, and full-order mechanical assembly while solving only the mechanical correction in a POD space. Each one-field predictive workflow supports POD–Proj, PODI–RBF, PODI–Linear, POD–GPR, POD–NN, POD–DL-ROM, and field-specific direct DL-ROM.

```python
import numpy as np

from paramplate.io.configuration import load_config
from paramplate.thermomechanical.hybrid import HybridMechanicalProjection
from paramplate.workflows import thermomechanical_config

raw = load_config(
    "configs/thermomechanical/thesis_case2.yml",
    expected_study="thermomechanical",
)
native_basis = np.load(
    "/external/path/thermomechanical_native_mechanical_basis.npy",
    allow_pickle=False,
)
result = HybridMechanicalProjection(
    thermomechanical_config(raw),
    native_basis,
).solve()
```

The native mechanical basis must match the plate mesh and native degree-of-freedom ordering used by the coupled configuration.
