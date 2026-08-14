# Source traceability

The maintained implementation is organized by scientific study rather than by paper number or notebook cell order.

| Maintained package area | Scientific source represented |
|---|---|
| `paramplate.mechanical` | Final thesis Chapters 3–5 and the active static-mechanics implementation |
| `paramplate.thermomechanical` | Final thesis Chapters 6–8, the active coupled implementation, and compatible snapshot-generation routines |
| `paramplate.dynamics` | Final thesis Chapters 9–11, the active transient implementation, and compatible trajectory-generation routines |
| `paramplate.rom` | Appendix A data roles, POD conventions, active non-intrusive models, and neural architectures |

The research notebooks are not distributed as primary implementations. Their final algorithms, parameter overrides, split rules, feature choices, and accepted-state logic are expressed through maintained modules, scripts, and configurations. Dormant branches, superseded study material, machine-local paths, cached model files, and large snapshot archives are excluded.

The dissertation is the scientific reference for terminology and reported configurations. Repository documentation describes only code paths contained here and distinguishes DOLFIN-dependent execution from dependency-light checks.
