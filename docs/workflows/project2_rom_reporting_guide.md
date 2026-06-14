# Project-2 ROM suite reporting workflow

This reporting layer reads already-generated Project-2 ROM suite summaries and writes publication-ready artifacts. It does not retrain any model.

Supported production layout:

```text
<rom_root>/suites/intrusive/<CASE>/<method>/intrusive_projection_summary.json
<rom_root>/suites/nonintrusive/<CASE>/<method>/nonintrusive_summary.json
```

Run:

```bash
CASE="CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"
python scripts/project2/report_rom_suite.py --case "$CASE"
```

Outputs are written to:

```text
<rom_root>/suites/reports/<CASE>/
```

Expected report files:

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

The report ranks rows by field, with `pod-projected` acting as the POD projection/truncation lower bound and non-intrusive methods ranked by field test error.

## Nested Project-2 summary compatibility

The reporting script supports the actual `run_rom_suite.py` method-summary format, where each JSON file contains parent metadata (`suite`, `method`, `case_name`) and nested field metrics under `w` and `theta`. This is the production format written by the intrusive and non-intrusive suite runners.
