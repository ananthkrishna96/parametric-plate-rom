# Changelog

## 0.2.0

- Reorganized the package around static mechanics, steady thermomechanics, and transient dynamics.
- Migrated the active notebook algorithms into reusable modules, command-line drivers, configurations, and tests.
- Added metric-aware POD, trajectory-safe splitting, interpolation, Gaussian-process, neural, POD–DL-ROM, and direct DL-ROM components in the studies that support them.
- Added FEniCS/DOLFIN code paths for the three full-order workflows while keeping ROM-only imports independent of DOLFIN.
- Added external-data manifests and archive validation for static snapshots, thermomechanical accepted states, and complete transient trajectories.
- Added study-specific verification drivers, configuration-controlled sampling, restricted archive/checkpoint loading, continuous integration, and release-tree checks.
- Removed the former single-study framing, obsolete method aliases, machine-specific paths, generated caches, and internal review material.

The version number identifies the repository state described here; no DOI, archived software record, or remote release is asserted.

## 0.1.0

Initial thermomechanical repository structure.
