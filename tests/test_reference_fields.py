from __future__ import annotations

from pathlib import Path

import numpy as np

from paramplate.io.reference_fields import (
    interpolate_reference,
    load_reference_field,
    load_regular_grid_field,
)


def test_regular_grid_reference_load_and_interpolation(tmp_path: Path) -> None:
    x = np.asarray([0.0, 1.0])
    y = np.asarray([0.0, 1.0])
    values = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    archive = tmp_path / "reference.npz"
    np.savez_compressed(archive, x=x, y=y, displacement=values)

    regular = load_regular_grid_field(archive, key="displacement", unit="m")
    np.testing.assert_allclose(regular.values, values)

    field = load_reference_field(archive, value_key="displacement", unit="m")
    result = interpolate_reference(field, np.asarray([[0.25, 0.50], [0.75, 0.25]]))
    np.testing.assert_allclose(result, [0.75, 1.0], atol=1.0e-12)


def test_scattered_reference_uses_nearest_fallback_outside_hull(tmp_path: Path) -> None:
    archive = tmp_path / "scattered.npz"
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.asarray([0.0, 1.0, 1.0])
    np.savez_compressed(archive, coordinates=coordinates, displacement=values)
    field = load_reference_field(archive, value_key="displacement")
    result = interpolate_reference(field, np.asarray([[0.25, 0.25], [2.0, 0.0]]))
    np.testing.assert_allclose(result, [0.5, 1.0], atol=1.0e-12)


def test_field_specific_coordinates_support_distinct_output_spaces(tmp_path: Path) -> None:
    archive = tmp_path / "two_fields.npz"
    displacement_coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    thermal_driver_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    np.savez_compressed(
        archive,
        displacement_coordinates=displacement_coordinates,
        displacement=np.asarray([0.0, 1.0, 1.0]),
        thermal_driver_coordinates=thermal_driver_coordinates,
        thermal_driver=np.asarray([0.0, 2.0, 2.0, 4.0]),
    )
    displacement = load_reference_field(archive, value_key="displacement", unit="m")
    thermal_driver = load_reference_field(archive, value_key="thermal_driver", unit="K/m")
    assert displacement.coordinates.shape == (3, 2)
    assert thermal_driver.coordinates.shape == (4, 2)
