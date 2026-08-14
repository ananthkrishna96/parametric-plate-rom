# Changelog

## 0.2.0 (unreleased working tree)

- Reorganized the package around static mechanics, steady thermomechanics, and transient dynamics.
- Migrated the active notebook algorithms into reusable modules, command-line drivers, configurations, and tests.
- Added metric-aware POD, trajectory-safe splitting, interpolation, Gaussian-process, neural, POD--DL-ROM, and direct DL-ROM components.
- Added real FEniCS/DOLFIN code paths for the three full-order workflows, with imports kept optional for ROM-only use.
- Added external-data manifests and archive validation for thermomechanical snapshots and transient trajectories.
- Removed the former single-study public framing, obsolete method aliases, local machine paths, generated caches, and internal review material.

This working-tree version does not assert a Git tag or a published remote release.

## 0.1.0

Initial thermomechanical repository structure.
