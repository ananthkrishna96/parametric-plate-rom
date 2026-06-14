# Project-2 Thermomechanical Snapshot Generation and Data Guide

## Purpose

This guide explains how Project-2 thermomechanical snapshot generation should be handled in the professional `parametric-plate-rom` repository.

The repository stores source code, configurations, documentation, tests, and small samples. Full generated snapshots, logs, ROM training outputs, and large result files must live outside Git.

## Current safe code layout

The preserved non-lossy SNAPGEN source files are kept under:

```text
legacy/project2/snapgen/
```

The clean user-facing wrapper is:

```bash
python scripts/project2/snapgen.py
```

The wrapper delegates to the preserved scripts, so old functionality remains available while the repo gets a clean interface.

## External data root

Recommended Mac data root:

```text
/Users/ananth/Documents/PHD_WORKS/parametric-plate-rom-data
```

Recommended workstation data root:

```text
/u/a/aorunnuk/parametric-plate-rom-data
```

## Create the data layout

From the repo root:

```bash
python scripts/project2/snapgen.py setup-data-layout --link-legacy
```

This creates the external Paper-1/Paper-2 data folders and creates local legacy symlinks so old SNAPGEN scripts can still use names like `THERMO_SNAPSHOTS` without writing data into Git.

## Existing Paper-1 snapshots

Your current Paper-1 location is:

```text
~/Documents/PAPER_1/SNAPSHOTS
```

Recommended target:

```text
~/Documents/PHD_WORKS/parametric-plate-rom-data/paper1_mechanical/snapshots/archives
```

Use `rsync` instead of `mv` until everything is verified:

```bash
rsync -avh --progress ~/Documents/PAPER_1/SNAPSHOTS/ \
  ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper1_mechanical/snapshots/archives/
```

## Existing Paper-2 snapshots

Your current Paper-2 baseline location is:

```text
~/Documents/PAPER_2/SNAPGEN/THERMO_SNAPSHOTS
```

Recommended target:

```text
~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/campaigns/baseline
```

Copy first:

```bash
rsync -avh --progress ~/Documents/PAPER_2/SNAPGEN/THERMO_SNAPSHOTS/ \
  ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/campaigns/baseline/
```

Also preserve logs:

```bash
rsync -avh --progress ~/Documents/PAPER_2/SNAPGEN/RUN_LOGS/ \
  ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/logs/RUN_LOGS/
```

Do not delete the old folders until ROM loading from the new data root is verified.

## Generate a manifest

After copying data:

```bash
python scripts/project2/snapgen.py manifest
```

This creates:

```text
parametric-plate-rom-data/shared/manifests/snapshot_manifest.csv
parametric-plate-rom-data/shared/manifests/snapshot_manifest.json
```

The manifest records archive paths, keys, sizes, modification time, and common array shapes.

## Run one case

```bash
python scripts/project2/snapgen.py run-case \
  --case 1 \
  --n-snapshots 2 \
  --panel-type monolithic \
  --bc-type free_edge \
  --load-type patch \
  --solve-mode coupled \
  --overwrite \
  --make-plots
```

## Run official monolithic baseline campaign

```bash
nohup python -u scripts/project2/snapgen.py run-campaign \
  --cases "1 2 3 4 5 6 7 8 9 10" \
  --panel-type monolithic \
  --bc-type free_edge \
  --load-type patch \
  --overwrite \
  > ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/logs/NOHUP_MONOLITHIC_BASELINE.out 2>&1 &
```

Resume:

```bash
nohup python -u scripts/project2/snapgen.py run-campaign \
  --cases "1 2 3 4 5 6 7 8 9 10" \
  --panel-type monolithic \
  --bc-type free_edge \
  --load-type patch \
  --resume \
  > ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/logs/NOHUP_MONOLITHIC_BASELINE_RESUME.out 2>&1 &
```

## Check status

```bash
python scripts/project2/snapgen.py status
```

## Addon/gap-fill snapshots

Plan:

```bash
python scripts/project2/snapgen.py addon --mode plan
```

Run addon:

```bash
nohup python -u scripts/project2/snapgen.py addon \
  --mode run-addon \
  --overwrite \
  > ~/Documents/PHD_WORKS/parametric-plate-rom-data/paper2_thermomechanical/logs/NOHUP_ADDON_GAPFILL.out 2>&1 &
```

Merge:

```bash
python scripts/project2/snapgen.py addon --mode merge
```

## What should be used for ROM?

For ROM training, prefer final consolidated archives:

```text
SNAPSHOTS__*.npz
```

Each case folder also contains compatibility names such as `snapshots.npz`, metadata, `parameter_matrix.csv`, `parameter_matrix.npy`, `summary.csv`, and diagnostic figures. The final `SNAPSHOTS__*.npz` archive should be the primary source for training; metadata and parameter matrices should be used to interpret the snapshot columns/rows.

## Keep out of Git

Do not commit:

```text
THERMO_SNAPSHOTS/
RUN_LOGS/
output/
tmp/
sample_*.npz
SNAPSHOTS__*.npz
trained model checkpoints
full ROM output folders
```

Keep only code, configs, documentation, tests, small sample data, and manifests.
