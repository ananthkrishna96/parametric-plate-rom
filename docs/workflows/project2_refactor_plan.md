# Project-2 thermomechanical refactor plan

This migration kit is a **non-lossy first-stage refactor**.

## Principle

The exact original notebook-converted file is preserved in:

```text
legacy/project2/THERMOMECHANICAL_FOM_ROM_legacy.py
```

An importable definitions-only compatibility layer is provided in:

```text
src/paramplate/project2/legacy_definitions.py
```

This avoids importing the full notebook-export file with all top-level study
execution cells, while keeping all top-level function and class definitions
available for gradual refactoring.

## Migration order

1. Preserve legacy source.
2. Add pure-Python core definitions.
3. Keep Project-2 solver available through `paramplate.project2`.
4. Gradually move implementation from `legacy_definitions.py` into:
   - `fem/`
   - `thermal/`
   - `coupling/`
   - `snapshots/`
   - `rom/`
   - `postprocessing/`
   - `io/`
5. Replace wrapper imports with direct implementations after tests are added.

## Target future imports

```python
from paramplate.project2 import GeneralMultiphysicsSolver
from paramplate.thermal.project2_thermal import solve_heat3d
from paramplate.coupling.project2_coupling import solve_coupled_thermomechanical
from paramplate.rom.project2_models import PODNNReducedOrderModel
```
