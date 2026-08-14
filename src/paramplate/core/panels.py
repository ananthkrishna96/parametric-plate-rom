"""Pure geometric bookkeeping for rectangular panel partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import PlateGeometry


@dataclass(frozen=True)
class PanelBox:
    panel_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float, *, atol: float = 1.0e-12) -> bool:
        return (
            self.x_min - atol <= x <= self.x_max + atol
            and self.y_min - atol <= y <= self.y_max + atol
        )


def panel_boxes(geometry: PlateGeometry) -> tuple[PanelBox, ...]:
    xs = np.linspace(0.0, geometry.length, geometry.n_vertical_interfaces + 2)
    ys = np.linspace(0.0, geometry.width, geometry.n_horizontal_interfaces + 2)
    boxes: list[PanelBox] = []
    panel_id = 1
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            boxes.append(PanelBox(panel_id, xs[i], xs[i + 1], ys[j], ys[j + 1]))
            panel_id += 1
    return tuple(boxes)


def interface_coordinates(geometry: PlateGeometry) -> tuple[tuple[str, float, int], ...]:
    records: list[tuple[str, float, int]] = []
    marker = 5
    for x in np.linspace(0.0, geometry.length, geometry.n_vertical_interfaces + 2)[1:-1]:
        records.append(("vertical", float(x), marker))
        marker += 1
    for y in np.linspace(0.0, geometry.width, geometry.n_horizontal_interfaces + 2)[1:-1]:
        records.append(("horizontal", float(y), marker))
        marker += 1
    return tuple(records)
