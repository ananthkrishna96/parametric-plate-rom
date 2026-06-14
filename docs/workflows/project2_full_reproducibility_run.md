# Project-2 clean-cache reproducibility run

This page gives a complete fresh run for the current Case-2 ROM benchmark. It is intended as the final local validation command sequence after cloning/pulling the repository and preparing the external data root.

## 1. Start from clean `main`

```bash
cd /path/to/parametric-plate-rom

git switch main
git pull origin main
git status
```

Expected:

```text
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

## 2. Remove Python/repository caches

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf build dist .mypy_cache .ruff_cache
```

This does not delete external snapshot archives or ROM outputs.

## 3. Reinstall editable package

```bash
python -m pip install -e .
```

## 4. Run workflow tests

```bash
python -m pytest tests/test_snapshot_archive_loading.py
python -m pytest tests/test_pod_preprocessing.py
python -m pytest tests/test_rom_suite_architecture.py
python -m pytest tests/test_legacy_aligned_nonintrusive.py
python -m pytest tests/test_torch_runtime_safety.py
python -m pytest tests/test_centered_pod_field_loss_alignment.py
python -m pytest tests/test_run_rom_suite_cli.py
python -m pytest tests/test_rom_suite_reporting.py
```

Expected current result: all tests pass.

## 5. Reset generated ROM outputs for one case

```bash
CASE="CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"
DATA="/absolute/path/to/parametric-plate-rom-data/paper2_thermomechanical/rom_training"

rm -rf "$DATA/pod_bases/$CASE"
rm -rf "$DATA/suites/intrusive/$CASE"
rm -rf "$DATA/suites/nonintrusive/$CASE"
rm -rf "$DATA/suites/reports/$CASE"
```

This removes generated POD/ROM/report artifacts only. It does not remove the source snapshot archive under `paper2_thermomechanical/campaigns/`.

## 6. Rebuild POD artifacts

```bash
python scripts/project2/build_pod_dataset.py \
  --case "$CASE" \
  --write \
  --overwrite
```

Expected Case-2 POD summary:

```text
parameters: 150x2
snapshots : 150x20503
theta     : 150x5194
train     : 120
test      : 30
w rank    : 3
theta rank: 1
```

## 7. Run all ROM methods

```bash
python scripts/project2/run_rom_suite.py \
  --suite all \
  --case "$CASE" \
  --write \
  --overwrite
```

Expected enabled methods:

```text
pod-projected
podi-rbf
podi-linear
pod-gpr
pod-nn
pod-ae
```

`pod-galerkin` remains capability-disabled.

## 8. Generate report artifacts

```bash
python scripts/project2/report_rom_suite.py \
  --case "$CASE"
```

Expected output:

```text
collected rows: 12
wrote: .../rom_suite_comparison.csv
wrote: .../rom_suite_comparison.md
wrote: .../rom_suite_comparison_table.tex
wrote: .../rom_suite_report.json
```

Plot outputs:

```text
rom_suite_w_test_error.png
rom_suite_w_test_error.pdf
rom_suite_theta_test_error.png
rom_suite_theta_test_error.pdf
```

## 9. Final verification

```bash
REPORT_DIR="$DATA/suites/reports/$CASE"

ls -lh "$REPORT_DIR"
cat "$REPORT_DIR/rom_suite_comparison.md"
git status
```

The repository should remain clean because generated artifacts are outside Git.
