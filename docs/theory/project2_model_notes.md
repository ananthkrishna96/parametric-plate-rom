# Project-2 thermomechanical model notes

The uploaded legacy source describes a thermomechanical orthotropic
Kirchhoff--Love plate framework with:

- unilateral Winkler foundation contact,
- penalty-based panel/domain coupling,
- C0 interior-penalty finite element discretization,
- 3D steady heat conduction,
- heat-to-plate through-thickness reduction,
- one-way and fixed-point staggered thermomechanical coupling,
- intrusive and non-intrusive POD-based ROM methods.

During refactoring, mathematical formulas and solver behavior should remain
traceable to the preserved legacy file.
