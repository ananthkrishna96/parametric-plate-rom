# Repository polish checklist

Use this checklist before sharing the repository with a supervisor, collaborator, or reviewer.

## README and About panel

The README should make the following clear within the first screen:

- what the repository does;
- current stable milestone/tag;
- main Project-2 workflow;
- where the data live;
- how to reproduce the Case-2 ROM benchmark;
- what methods are implemented;
- what is intentionally not yet implemented.

Suggested GitHub About description:

```text
Research software for high-fidelity and reduced-order modelling of parametrized orthotropic Kirchhoff--Love plates with thermomechanical coupling, snapshot generation, POD projection, PODI, POD-GPR, POD-NN, POD-AE, and report-ready ROM benchmarking.
```

Suggested topics:

```text
finite-element-method
reduced-order-modeling
pod
thermomechanics
kirchhoff-love-plates
orthotropic-plates
gaussian-process-regression
neural-networks
autoencoder
scientific-computing
```

## What should be committed

Commit source, configuration templates, small documentation figures, tests, and documentation:

```text
README.md
CHANGELOG.md
CITATION.cff
configs/*.template.yml
docs/**
scripts/**
src/**
tests/**
```

## What should not be committed

Do not commit local/private or large generated data:

```text
configs/data_locations.local.yml
parametric-plate-rom-data/
*.npz snapshot archives
large generated suite outputs
machine-specific paths
__pycache__/
.pytest_cache/
*.egg-info/
```

A small representative figure under `docs/figures/` is acceptable when it improves the README.

## Final command sequence

```bash
git switch main
git pull origin main
git status

python -m pip install -e .
python -m pytest tests/test_snapshot_archive_loading.py
python -m pytest tests/test_pod_preprocessing.py
python -m pytest tests/test_rom_suite_architecture.py
python -m pytest tests/test_legacy_aligned_nonintrusive.py
python -m pytest tests/test_torch_runtime_safety.py
python -m pytest tests/test_centered_pod_field_loss_alignment.py
python -m pytest tests/test_run_rom_suite_cli.py
python -m pytest tests/test_rom_suite_reporting.py
```

Then run one end-to-end case following `project2_full_reproducibility_run.md`.

## Repository state expected after `v0.8`

```text
main up to date with origin/main
working tree clean
v0.8-project2-rom-reporting-safe present
25 workflow tests passing
Case-2 report generated with 12 rows
```
