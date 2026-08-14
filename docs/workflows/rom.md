# POD and predictive ROM workflow

## Data roles

The package applies the statistical split before any fitted operation. For steady datasets, sample indices are split with training seed 100 and holdout seed 101. For transient data, unique trajectory IDs are split first and then expanded to state-row indices. `validate_no_trajectory_leakage` checks that every trajectory belongs to exactly one subset.

The training subset alone determines:

- POD modes and retained-coordinate targets;
- input and coordinate standardizers;
- interpolation or regression fits;
- neural checkpoints and validation-based stopping.

The held-out test subset is evaluated only after model selection.

## POD paths

`compute_metric_pod` implements uncentered or centered POD in a declared coefficient metric. Static and thermomechanical workflow configurations use uncentered bases. `compute_transient_pod` follows the transient campaign path: training-only Euclidean SVD followed by reorthonormalization of the retained modes in the finite-element displacement metric.

POD–Proj reconstructs an available full-order field and reports compression error. It is never presented as a predictive model.

## Predictive methods

- **PODI–RBF:** thin-plate-spline coordinate interpolation; the transient setup uses local neighborhoods of 96 and smoothing `1e-10`.
- **PODI–Linear:** Delaunay/linear interpolation with nearest-neighbor fallback outside the convex hull; the transient fit can use a maximin trajectory subset of at most 2500 state queries.
- **POD–GPR:** one Gaussian process per retained coordinate; the transient setup uses a Matérn-5/2 kernel and a maximin limit of 1200 training queries.
- **POD–NN:** coefficient-only `128/96/64` ELU network with linear output.
- **POD–DL-ROM:** a coefficient autoencoder and parameter-to-latent map for steady fields; the transient version uses the executed raw, normalized, logarithmic, polynomial-time, and harmonic feature map with a `192/160/128` SiLU network, layer normalization, and dropout `0.02`.
- **DL-ROM:** field-specific direct autoencoder and parameter-to-latent map for steady thermomechanical outputs only.

Neural artifacts are generated outputs and are excluded from Git by default.

## Command line

```bash
python scripts/common/run_rom.py ARCHIVE \
  --study mechanical \
  --output-field displacement \
  --method podi-rbf \
  --rank 4 \
  --options configs/rom/mechanical_smoke.json \
  --report results/mechanical/rbf_report.json
```

A finite-element metric may be supplied as a NumPy matrix with `--metric`. When no metric is supplied, the dependency-light examples use the Euclidean metric and must not be compared directly with thesis finite-element error values.
