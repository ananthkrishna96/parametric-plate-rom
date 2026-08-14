# External data and archive schemas

Large snapshots, complete trajectories, raw meshes, trained neural checkpoints, and full result folders are not stored in Git. Local paths belong in the ignored `configs/data_locations.local.yml` or in `PARAMPLATE_DATA_ROOT`.

Normal NPZ loading uses `allow_pickle=False`. Shared archives must therefore contain numeric arrays and fixed-width Unicode or byte strings. Pickle-backed object arrays are not part of the repository schema. An explicit `allow_unsafe_pickle=True` option exists only for trusted legacy conversion work and should not be used on downloaded or unverified files.

## Mechanical archive

Required arrays:

- `parameters`: `(n_samples, n_parameters)`;
- `snapshots`: `(n_samples, n_output_dofs)`.

The direct campaign uses parameter names `Dx`, `Dy`, `Dxy`, `Ds`, `ks`, and `q`. An optional `native_snapshots` array may accompany the transferred output fields for intrusive POD–Galerkin. Native snapshots are mesh- and ordering-specific.

## Thermomechanical archive

Required arrays:

- `parameters`;
- `displacement_snapshots`;
- `thermal_driver_snapshots`.

The two field arrays must have the same sample count but may have different output dimensions. `theta` has unit K/m and must not be interpreted as temperature. The loader recognizes the active notebook-export aliases documented in `paramplate.rom.datasets`, but it always returns one selected field rather than concatenating the two outputs.

For external verification on one regular grid, use:

- `x`: strictly increasing coordinates;
- `y`: strictly increasing coordinates;
- `displacement`: shape `(len(y), len(x))`, in meters;
- `thermal_driver`: shape `(len(y), len(x))`, in K/m.

## Transient archive

Required arrays:

- `query_parameters`: flattened physical-parameter and time rows;
- `snapshots`: flattened displacement states;
- `trajectory_ids`: one complete-trajectory identifier per state row;
- `times`: the common stored-time vector.

Every trajectory ID must occur exactly `len(times)` times. Splits are made on unique trajectory IDs, not on flattened rows.

## Validation

```bash
paramplate-validate-archive data/sample/mechanical_smoke.npz
paramplate-validate-archive data/sample/thermomechanical_smoke.npz
paramplate-validate-archive data/sample/dynamics_smoke.npz
```

JSON manifests record the selected snapshot and parameter keys, parameter order, output name and unit, split level, expected shape where useful, metric description, and provenance notes. A manifest may also store a SHA-256 digest for an externally distributed immutable dataset.

## Neural checkpoints

POD–NN checkpoints are loaded with PyTorch's restricted weight-only loader and must match the repository payload schema. This reduces the deserialization surface but does not make an arbitrary checkpoint trustworthy. Only load checkpoints obtained from a trusted source. Generated `*.pt`, `*.pth`, `*.ckpt`, and model directories are ignored by Git.
