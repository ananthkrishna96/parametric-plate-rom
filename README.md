# Parametric Plate ROM

A research software framework for high-fidelity and reduced-order modelling of parameterized orthotropic Kirchhoff--Love plate systems, including elastic foundation interaction, panel-interface coupling, thermomechanical loading, snapshot generation, and projection/data-driven reduced-order models.

## Scope

This repository is intended to support long-term development of finite-element and reduced-order methods for parameterized plate systems, including:

- C0 interior-penalty finite element approximation of Kirchhoff--Love plates
- Orthotropic plates on Winkler-type elastic foundations
- Weak coupling across panel interfaces
- Mechanical, thermomechanical, dynamic, and transient plate extensions
- One-way and partitioned staggered thermomechanical coupling
- Snapshot generation for many-query studies
- POD-based intrusive and non-intrusive reduced-order models
- Data-driven ROMs including interpolation, neural-network, Gaussian-process, and autoencoder-based variants

## Repository status

This repository is currently under active research development. The first public release will correspond to the thesis/paper reproducibility version.

## Repository structure

- `src/paramplate/` — main Python package
- `configs/` — YAML configuration files for models and campaigns
- `scripts/` — command-line entry points
- `examples/` — lightweight reproducible examples
- `notebooks/` — curated notebooks, not primary source code
- `docs/` — theory and workflow documentation
- `tests/` — unit and smoke tests
- `data/sample/` — small sample data only
- `results/` — placeholder for generated results

## Installation

A complete installation guide will be provided with the first reproducibility release.

For development, the expected stack includes Python, NumPy, SciPy, Matplotlib, scikit-learn, PyTorch, FEniCS/dolfin, and ROM-related scientific-computing tools.

## Citation

If you use this repository, please cite the thesis/release associated with the corresponding version. A `CITATION.cff` file is included for citation metadata.

## License

The license will be finalized before public release.
