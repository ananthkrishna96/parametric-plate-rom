# Static mechanical workflow

## Model and full-order path

The static study solves the small-deflection aligned-orthotropic Kirchhoff--Love plate problem with parameterized bending rigidities, load, foundation stiffness, and optional panel seams. The foundation is regularized rather than removed in uplift. `MechanicalPlateFOM` assembles the panel-interior symmetric `C0-IPG` terms, physical seam penalties, and the nonlinear foundation residual in DOLFIN. A manual active-set/Newton path applies an algebraic residual stopping criterion.

Monolithic states are represented directly in the plate space. Panelized states use independent panel components and are transferred to a fixed continuous displacement output for the non-intrusive ROM and field-error workflows. Native panel states remain available for the intrusive method.

## Verification and data generation

`paramplate.mechanical.navier` provides the truncated double-sine reference for simply supported rectangular plates. The thesis also compares the finite-element solution with an independent three-dimensional solid model, examines spatial refinement, and assesses two-, three-, four-, and six-panel layouts. The repository includes the Navier evaluator and comparison utilities; reproducing the full three-dimensional comparison requires the external reference arrays used in the thesis.

```bash
python scripts/mechanical/run_workflow.py \
  --config configs/mechanical/smoke.yml --action navier
```

The six-parameter direct campaign records `Dx`, `Dy`, `Dxy`, `Ds`, `ks`, and the signed load amplitude. The thesis configuration uses 1000 Latin-hypercube snapshots and a separate predictive assessment. Panel campaign data remain external because of their size.

## ROMs

The active static methods are:

- non-hyper-reduced POD--Galerkin in the native finite-element space;
- POD--Proj;
- PODI--RBF and PODI--Linear;
- POD--GPR;
- coefficient-only POD--NN with `128/96/64` ELU hidden layers and a linear output.

The intrusive implementation retains full residual and Jacobian assembly. Its online acceleration is therefore limited by full-order assembly, exactly as stated in the thesis; no hyper-reduction claim is made. POD--DL-ROM and direct DL-ROM are not exposed as static thesis workflows.

The intrusive entry point is programmatic because its basis must be assembled in the native finite-element space rather than the transferred output space:

```python
import numpy as np

from paramplate.io.configuration import load_config
from paramplate.mechanical.fem import MechanicalPlateFOM
from paramplate.mechanical.intrusive import MechanicalPODGalerkin
from paramplate.workflows import mechanical_config

raw = load_config("configs/mechanical/thesis_direct.yml", expected_study="mechanical")
fom = MechanicalPlateFOM(mechanical_config(raw))
native_basis = np.load("/external/path/mechanical_native_basis.npy")
result = MechanicalPODGalerkin(fom, native_basis).solve()
```

The native basis and FOM must use the same mesh, mixed-space ordering, and boundary treatment. A transferred displacement basis is not interchangeable with the native basis required by this solver.

## Configurations

- `configs/mechanical/smoke.yml`: small simply supported dependency check;
- `configs/mechanical/thesis_direct.yml`: six-parameter monolithic campaign;
- `configs/mechanical/thesis_panel_smoke.yml`: panel-aligned full-order path and external-data setup.

A DOLFIN run requires the legacy FEniCS environment. Configuration composition and the Navier reference can be checked without it.
