# Source traceability

The public implementation is organized by scientific study rather than by paper number or notebook cell order.

| Maintained package area | Scientific source represented |
|---|---|
| `paramplate.mechanical` | final thesis Chapters 3--5 and the active static mechanics notebook path |
| `paramplate.thermomechanical` | final thesis Chapters 6--8, the active coupled notebook path, and compatible snapshot-generation routines |
| `paramplate.dynamics` | final thesis Chapters 9--11, the active transient notebook path, and compatible trajectory-generation routines |
| `paramplate.rom` | Appendix A data roles, POD conventions, active non-intrusive models, and neural architectures |

The original notebooks are not shipped as primary implementations. Their final executed classes, parameter overrides, split rules, active feature flags, and accepted-state logic were separated into modules, scripts, and configurations. Dormant branches, superseded study material, local paths, cached model files, and heavy snapshot archives are not part of the maintained tree.

The dissertation remains the scientific contract for terminology and reported configurations. The repository documentation describes only code paths present in this working tree and marks DOLFIN-dependent execution separately from dependency-light checks.
