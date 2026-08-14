# Static mechanical workflow

## Full-order problem

The static study solves the small-deflection aligned-orthotropic Kirchhoff–Love plate problem with parameterized bending rigidities, load, foundation stiffness, and optional panel seams. Displacement and transverse loading are upward-positive; downward pressure is therefore negative. The foundation uses the compression stiffness where `w < 0` and the regularized uplift stiffness elsewhere.

`MechanicalPlateFOM` assembles the panel-interior symmetric `C0-IPG` terms, physical seam penalties, and the nonlinear foundation residual in DOLFIN. Numerical interior facets and physical panel interfaces are treated separately. The active-set/Newton iteration stops on the configured algebraic residual threshold.

A panelized solve stores a native broken mixed-space field and transfers the physical panel displacement to a fixed continuous output space. The transferred field is used by the non-intrusive ROMs and field-error calculations. A native basis is required by intrusive POD–Galerkin.

## Verification

`configs/mechanical/verification_monolithic.yml` records the simply supported monolithic benchmark used by the Navier comparison. The driver can also compare a FOM field with a trusted regular-grid archive or compare a finite-penalty panel model with a monolithic FOM.

```bash
python scripts/mechanical/run_verification.py \
  --config configs/mechanical/verification_monolithic.yml \
  --mode navier --dry-run

python scripts/mechanical/run_verification.py \
  --config configs/mechanical/verification_panel_2.yml \
  --reference-config configs/mechanical/verification_panel_monolithic.yml \
  --mode panel --dry-run
```

The panel configurations cover two-, three-, four-, and six-panel layouts. Removing `--dry-run` requires DOLFIN 2019.1 and `mshr`. The independent three-dimensional solid and refined unilateral reference fields are external research artifacts; they may be supplied through the regular-grid comparison path after conversion to the documented numeric schema.

## Snapshot generation

The monolithic direct campaign uses the ordered parameters

```text
Dx, Dy, Dxy, Ds, ks, q
```

and an upward-positive load convention. Ranges declared in YAML are passed to the Latin-hypercube sampler rather than used as display-only metadata.

```bash
python scripts/mechanical/run_workflow.py \
  --config configs/mechanical/thesis_direct.yml --dry-run
python scripts/mechanical/run_workflow.py \
  --config configs/mechanical/smoke.yml \
  --action snapshots --output results/mechanical_smoke.npz
```

The archive stores the transferred displacement, optional native state, parameter order, Newton iterations, residuals, output unit, and a companion manifest. Full thesis snapshots remain external.

## Reduced-order workflows

The active static inventory is:

- non-hyper-reduced intrusive POD–Galerkin in the native finite-element space;
- POD–Proj as a compression diagnostic;
- PODI–RBF and PODI–Linear;
- POD–GPR;
- coefficient-only POD–NN with `128/96/64` ELU hidden layers and a linear output.

The intrusive implementation retains full residual and Jacobian assembly. Its online acceleration is therefore limited by full-order assembly; no hyper-reduction claim is made. POD–DL-ROM and direct DL-ROM are not exposed as static thesis workflows.

```python
import numpy as np

from paramplate.io.configuration import load_config
from paramplate.mechanical.fem import MechanicalPlateFOM
from paramplate.mechanical.intrusive import MechanicalPODGalerkin
from paramplate.workflows import mechanical_config

raw = load_config("configs/mechanical/thesis_direct.yml", expected_study="mechanical")
fom = MechanicalPlateFOM(mechanical_config(raw))
native_basis = np.load("/external/path/mechanical_native_basis.npy", allow_pickle=False)
result = MechanicalPODGalerkin(fom, native_basis).solve()
```

The native basis and FOM must use the same mesh, mixed-space ordering, boundary treatment, and constrained degrees of freedom.
