# Project-2 data layout and snapshot storage

The repository intentionally keeps large generated data outside Git. This file documents the expected external folder structure and where snapshot archives should be placed.

## Recommended external root

Use a sibling data directory such as:

```text
/Users/<user>/Documents/PHD_WORKS/parametric-plate-rom-data
```

Configure it in:

```text
configs/data_locations.local.yml
```

Create the local config from the template:

```bash
cp configs/data_locations.template.yml configs/data_locations.local.yml
```

## Folder tree

Recommended tree:

```text
parametric-plate-rom-data/
├── paper1_mechanical/
│   ├── snapshots/
│   ├── rom_training/
│   ├── results/
│   └── logs/
└── paper2_thermomechanical/
    ├── snapshots/
    ├── campaigns/
    │   ├── baseline/
    │   ├── addon_gapfill/
    │   └── merged/
    ├── rom_training/
    │   ├── pod_bases/
    │   └── suites/
    │       ├── intrusive/
    │       ├── nonintrusive/
    │       └── reports/
    ├── convergence/
    ├── results/
    └── logs/
```

The repository code writes ROM artifacts under `paper2_thermomechanical/rom_training/`.

## Snapshot archive placement

For a Project-2 case named `<CASE>`, place the baseline archive as:

```text
paper2_thermomechanical/campaigns/baseline/<CASE>/SNAPSHOTS__<CASE>.npz
```

For the current Case-2 tutorial:

```text
paper2_thermomechanical/campaigns/baseline/
└── CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH/
    └── SNAPSHOTS__CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH.npz
```

## Archive contents

The loader expects a NumPy `.npz` snapshot archive with the following arrays:

| key | meaning |
|---|---|
| `parameters` | parameter samples used for ROM input |
| `snapshots` | displacement/solution snapshots, one row per parameter sample |
| `theta_snapshots` | thermal-driver/`T1` snapshots, when available |

Additional metadata keys are allowed and are preserved or ignored depending on the loader path.

## Generated ROM artifacts

After running `build_pod_dataset.py`, the POD artifacts are written to:

```text
paper2_thermomechanical/rom_training/pod_bases/<CASE>/
```

After running `run_rom_suite.py`, method summaries are written to:

```text
paper2_thermomechanical/rom_training/suites/intrusive/<CASE>/
paper2_thermomechanical/rom_training/suites/nonintrusive/<CASE>/
```

After running `report_rom_suite.py`, reports are written to:

```text
paper2_thermomechanical/rom_training/suites/reports/<CASE>/
```

## Safe clean-up commands

To rerun one case from a clean ROM state without deleting original snapshots:

```bash
CASE="CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"
DATA="/absolute/path/to/parametric-plate-rom-data/paper2_thermomechanical/rom_training"

rm -rf "$DATA/pod_bases/$CASE"
rm -rf "$DATA/suites/intrusive/$CASE"
rm -rf "$DATA/suites/nonintrusive/$CASE"
rm -rf "$DATA/suites/reports/$CASE"
```

Do not delete `paper2_thermomechanical/campaigns/` unless you intentionally want to remove the source snapshot archives.
