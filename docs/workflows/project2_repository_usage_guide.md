# Project-2 repository usage guide

This guide explains how to use the repository after the `v0.8-project2-rom-reporting-safe` milestone. It assumes that Project-2 snapshot archives already exist as `.npz` files in the external data directory.

## 1. What the workflow does

The Project-2 ROM workflow is split into four stages:

```text
1. Load snapshot archive
2. Build POD preprocessing artifacts
3. Run intrusive and non-intrusive ROM methods
4. Generate report-ready tables and plots
```

The high-fidelity thermomechanical solver and snapshot generator are preserved in the repository/legacy path, but the recommended reproducibility workflow starts from saved snapshot archives. This keeps the ROM comparison independent of long FEniCS solves.

## 2. Required local files

You need two things outside the repository:

1. an external data root, for example:

   ```text
   /Users/<user>/Documents/PHD_WORKS/parametric-plate-rom-data
   ```

2. one or more Project-2 snapshot archives under:

   ```text
   paper2_thermomechanical/campaigns/baseline/<CASE>/SNAPSHOTS__<CASE>.npz
   ```

The archive must contain at least:

```text
parameters      # parameter samples, shape n_samples x n_parameters
snapshots       # displacement snapshots, shape n_samples x n_w_dofs
theta_snapshots # thermal-driver snapshots, shape n_samples x n_theta_dofs, when available
```

Some archives may also contain metadata, case descriptors, raw mechanical/thermal parameters, and provenance information.

## 3. Configure data locations

Copy the template:

```bash
cp configs/data_locations.template.yml configs/data_locations.local.yml
```

Edit `configs/data_locations.local.yml` and set:

```yaml
data_root: "/absolute/path/to/parametric-plate-rom-data"
```

The local YAML file should remain untracked. The code searches in this order:

1. explicit command-line `--data-root`, when available;
2. local data-location YAML;
3. environment variable support, where implemented;
4. template/default paths.

## 4. Inspect available datasets

Run:

```bash
python scripts/project2/check_rom_dataset.py
```

or inspect one case by substring using the workflow scripts:

```bash
python scripts/project2/build_pod_dataset.py --case CASE_02 --no-theta
```

Use exact case names when preparing final reproducibility results.

## 5. Build POD artifacts

For the Case-2 thermomechanical benchmark:

```bash
CASE="CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"

python scripts/project2/build_pod_dataset.py \
  --case "$CASE" \
  --write \
  --overwrite
```

This writes:

```text
<DATA_ROOT>/paper2_thermomechanical/rom_training/pod_bases/<CASE>/
├── pod_w.npz
├── pod_theta.npz
├── parameters_scaled.npz
├── parameter_scaler.json
├── train_test_split.json
├── pod_summary.json
├── energy_w.csv
└── energy_theta.csv
```

## 6. Run the ROM suite

Run all currently enabled methods:

```bash
python scripts/project2/run_rom_suite.py \
  --suite all \
  --case "$CASE" \
  --write \
  --overwrite
```

This runs:

```text
intrusive:
  pod-projected

nonintrusive:
  podi-rbf
  podi-linear
  pod-gpr
  pod-nn
  pod-ae
```

`pod-galerkin` is listed as capability-disabled because the true online FEniCS residual/Jacobian path has not yet been migrated.

## 7. Generate reports

```bash
python scripts/project2/report_rom_suite.py --case "$CASE"
```

Outputs are written to:

```text
<DATA_ROOT>/paper2_thermomechanical/rom_training/suites/reports/<CASE>/
```

The expected files are:

```text
rom_suite_comparison.csv
rom_suite_comparison.md
rom_suite_comparison_table.tex
rom_suite_report.json
rom_suite_w_test_error.png
rom_suite_w_test_error.pdf
rom_suite_theta_test_error.png
rom_suite_theta_test_error.pdf
```

## 8. Recommended final check

After a clean run:

```bash
REPORT_DIR="<DATA_ROOT>/paper2_thermomechanical/rom_training/suites/reports/$CASE"
ls -lh "$REPORT_DIR"
cat "$REPORT_DIR/rom_suite_comparison.md"
git status
```

Expected Git status:

```text
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

Generated reports live in the external data root and should not be committed unless a small curated example is intentionally added.
