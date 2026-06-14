# ROM data loading guide

This guide explains the clean data-loading layer for Paper 1 and Project 2.

## Data rule

The Git repository stores code, documentation, tests, and small samples only. Full snapshots and trained ROM artifacts stay outside Git in:

```text
/Users/ananth/Documents/PHD_WORKS/parametric-plate-rom-data
```

## Main command

From the repository root:

```bash
python scripts/project2/check_rom_dataset.py --project paper2 --write-manifest
```

This scans the Project-2 final snapshot archives and prints each archive's parameter, displacement, and theta snapshot shapes.

For both Paper 1 and Project 2:

```bash
python scripts/project2/check_rom_dataset.py --project all --write-manifest
```

## Python API

```python
from paramplate.io.data_locations import load_data_locations
from paramplate.io.snapshot_manifest import project2_final_archives
from paramplate.io.snapshot_archives import summarize_snapshot_archive
from paramplate.rom.datasets import load_rom_dataset

loc = load_data_locations()
archives = project2_final_archives(loc)

for path in archives:
    info = summarize_snapshot_archive(path)
    print(info.path.name, info.parameter_shape, info.snapshot_shape, info.theta_shape)

# Load one archive fully into memory for ROM training.
ds = load_rom_dataset(archives[0])
print(ds.parameters.shape, ds.snapshots.shape, ds.theta_snapshots.shape)
```

## Project-2 archive convention

Project-2 SNAPGEN final archives should contain:

```text
parameters
snapshots
theta_snapshots
mech_parameters
thermal_parameters
```

`parameters` is the ROM input matrix. `snapshots` is the displacement/FOM solution snapshot matrix. `theta_snapshots` is the thermal reduced field used for thermomechanical ROM workflows.

## Paper-1 archive convention

Paper-1 archives are detected using historical keys such as:

```text
mus
fom_snapshots
training_parameters
nonlinear_snapshots
```

The loader maps these into the same `ROMDataset` structure.
