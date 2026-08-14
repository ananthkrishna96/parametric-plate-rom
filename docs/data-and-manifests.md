# External data and archive schemas

Large snapshots, complete trajectories, raw meshes, trained neural checkpoints, and full result folders are not stored in Git. Local paths belong in the ignored `configs/data_locations.local.yml` or in `PARAMPLATE_DATA_ROOT`.

## Mechanical archive

Required arrays:

- `parameters`: `(n_samples, n_parameters)`;
- `snapshots`: `(n_samples, n_output_dofs)`.

The migrated direct campaign uses parameter names `Dx`, `Dy`, `Dxy`, `Ds`, `ks`, and `q`. An optional `native_snapshots` array may accompany the transferred output fields for intrusive POD--Galerkin.

## Thermomechanical archive

Required arrays:

- `parameters`;
- `displacement_snapshots`;
- `thermal_driver_snapshots`.

The two field arrays must have the same sample count but may have different numbers of output degrees of freedom. `theta` has unit K/m and must not be interpreted as temperature. The loader also recognizes the active notebook-export aliases documented in `paramplate.rom.datasets`.

## Transient archive

Required arrays:

- `query_parameters`: flattened physical-parameter and time rows;
- `snapshots`: flattened displacement states;
- `trajectory_ids`: one complete-trajectory identifier per state row;
- `times`: the common stored-time vector.

Every trajectory ID must occur exactly `len(times)` times. Splits are made on unique trajectory IDs, not on rows.

## Validation

```bash
paramplate-validate-archive data/sample/mechanical_smoke.npz
paramplate-validate-archive data/sample/thermomechanical_smoke.npz
paramplate-validate-archive data/sample/dynamics_smoke.npz
```

JSON manifests record the selected snapshot and parameter keys, parameter order, output name and unit, split level, expected shape where useful, metric description, and provenance notes. A manifest may also store a SHA-256 digest for externally distributed immutable data.
