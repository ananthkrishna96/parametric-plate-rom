# Project-2 ROM preprocessing guide

This guide describes the first ROM-preprocessing layer after the snapshot
data loader milestone.

## Goal

The first target is to build POD artifacts from one thermomechanical
Project-2 archive, without calling FEniCS and without writing large binary
files into the Git repository.

Recommended first case:

```text
CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH
```

This case has 150 samples, 2 input parameters, a monolithic geometry, and
fixed snapshot dimensions.

## Data flow

```text
external SNAPSHOTS__*.npz
  -> ROMDataset
  -> train/test split
  -> parameter scaling
  -> POD basis for displacement snapshots
  -> POD basis for theta snapshots
  -> saved POD artifacts
```

## Command

```bash
python scripts/project2/build_pod_dataset.py \
  --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH \
  --train-fraction 0.8 \
  --energy-tol 0.9999 \
  --write
```

## Output location

POD artifacts are written outside the repository:

```text
/Users/ananth/Documents/PHD_WORKS/parametric-plate-rom-data/
└── paper2_thermomechanical/
    └── rom_training/
        └── pod_bases/
            └── CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH/
```

## Generated files

```text
pod_w.npz
pod_theta.npz
parameters_scaled.npz
parameter_scaler.json
train_test_split.json
pod_summary.json
energy_w.csv
energy_theta.csv
```

## Development rule

Do not commit generated POD artifacts.  They are reproducible external
data products and belong in `parametric-plate-rom-data`, not in Git.
