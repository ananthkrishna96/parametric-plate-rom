# Steady thermomechanical workflow

## Coupled full-order problem

The thermomechanical workflow combines a steady three-dimensional heat problem with the nonlinear plate solver. The top face includes convection, long-wave radiation, and absorbed solar flux. The lower-face conductance varies smoothly between contact and gap values according to the current upward-positive displacement. The temperature increment is reduced to the centered through-thickness driver `theta` in K/m, and the corresponding orthotropic thermal moments enter mechanical equilibrium.

`ThermomechanicalPlateFOM` executes a relaxed partitioned iteration:

1. solve the three-dimensional heat problem at the current displacement;
2. reduce the temperature increment to `theta` on the plate mesh;
3. solve the nonlinear mechanical problem with thermal bending moments;
4. relax displacement and test both displacement and `theta` updates.

The accepted record follows the executed notebook protocol: it stores the last relaxed displacement and the `theta` field computed at the start of that same iteration. It does not add an unreported post-convergence correction.

The thesis discretization uses a plate resolution of 64 with quadratic Lagrange functions, a `64 x 32 x 16` linear thermal mesh, a linear `theta` space, 20 through-thickness quadrature points, relaxation `0.7`, tolerance `1e-5`, and at most 25 coupling iterations.

## Snapshot cases and output separation

The active cases are recorded in the case configurations:

| Case | Parameters | Accepted samples | Split | POD ranks `(w, theta)` |
|---|---|---:|---:|---:|
| 2 | mechanical load, absorbed solar flux | 450 | 360 / 45 / 45 | 6 / 4 |
| 3 | foundation stiffness, ambient temperature, substrate temperature | 378 | 302 / 38 / 38 | 6 / 4 |
| 5 | foundation stiffness, load, solar flux, contact conductance, gap conductance | 500 | 400 / 50 / 50 | 6 / 4 |

One nonconverged sample is omitted from the requested 379 Case-3 runs. Displacement and `theta` use matched sample assignments but separate arrays, spaces, metrics, bases, scalers, models, units, and errors. They are not a coupled ROM target.

## ROMs

The hybrid workflow keeps the heat solve, thermal reduction, and full-order mechanical assembly, while solving only the mechanical correction in a POD space. The one-field predictive inventory is POD--Proj, PODI--RBF, PODI--Linear, POD--GPR, POD--NN, POD--DL-ROM, and field-specific direct DL-ROM.

As in the static intrusive method, the hybrid solver requires a mechanical POD basis in the native finite-element space:

```python
import numpy as np

from paramplate.io.configuration import load_config
from paramplate.thermomechanical.hybrid import HybridMechanicalProjection
from paramplate.workflows import thermomechanical_config

raw = load_config(
    "configs/thermomechanical/thesis_case2.yml",
    expected_study="thermomechanical",
)
native_basis = np.load("/external/path/thermomechanical_native_mechanical_basis.npy")
result = HybridMechanicalProjection(
    thermomechanical_config(raw),
    native_basis,
).solve()
```

The basis must match the plate mesh and native mechanical degree-of-freedom ordering used by the coupled configuration.

```bash
python scripts/thermomechanical/run_workflow.py \
  --config configs/thermomechanical/thesis_case2.yml --dry-run
python scripts/thermomechanical/run_workflow.py \
  --config configs/thermomechanical/smoke.yml \
  --action snapshots --output results/thermomechanical_smoke.npz
paramplate-validate-archive /external/path/case2.npz --kind thermomechanical
python scripts/common/run_rom.py /external/path/case2.npz \
  --study thermomechanical --output-field thermal_driver \
  --method pod-gpr --rank 4
```

The smoke configuration is smaller only in mesh and sample counts; it preserves the same coupling sequence and field definitions.
