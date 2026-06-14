# Changelog

## [0.1.0] - Unreleased

### Added

- Initial professional repository skeleton.
- Package layout for mechanical, thermomechanical, dynamic, snapshot-generation, and ROM extensions.
- Project-2 legacy-safe solver preservation milestone (`v0.1-project2-legacy-safe`).
- Project-2 snapshot-generation migration milestone (`v0.2-project2-snapgen-safe`).
- External ROM data loader and snapshot archive discovery (`v0.3-rom-data-loader-safe`).
- Project-2 POD preprocessing workflow (`v0.4-project2-pod-preprocessing-safe`).
- Intrusive/non-intrusive ROM suite architecture (`v0.5-project2-rom-suite-architecture-safe`).
- Legacy-aligned non-intrusive ROM methods: PODI-RBF, PODI-linear, POD-GPR, POD-NN, and POD-AE (`v0.6-project2-legacy-aligned-rom-methods-safe`).
- Neural ROM field-loss alignment and POD-AE end-to-end fine-tuning (`v0.7-project2-neural-alignment-safe`).
- Project-2 ROM reporting and post-processing workflow with CSV, Markdown, LaTeX, JSON, PNG, and PDF outputs (`v0.8-project2-rom-reporting-safe`).
- Professional repository README and Project-2 usage documentation.

### Notes

- True nonlinear intrusive POD-Galerkin is intentionally kept capability-disabled until the full FEniCS residual/Jacobian reduced solve is migrated and validated.
- Large snapshot archives and generated ROM outputs are expected to live outside the repository in `parametric-plate-rom-data/`.
