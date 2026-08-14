from __future__ import annotations

from pathlib import Path

import numpy as np

from paramplate.postprocessing import (
    save_energy_history,
    save_error_histogram,
    save_field_image,
    save_pod_spectrum,
    save_probe_history,
)


def test_plotting_functions_create_nonempty_files(tmp_path: Path) -> None:
    x = np.linspace(0.0, 2.0, 12)
    y = np.linspace(0.0, 1.0, 8)
    field = np.outer(np.sin(np.pi * y), np.sin(np.pi * x / 2.0))
    outputs = [
        save_field_image(field, x, y, tmp_path / "field.png", label="w [m]"),
        save_error_histogram(np.linspace(0.0, 0.1, 20), tmp_path / "errors.png"),
        save_pod_spectrum(np.geomspace(1.0, 1.0e-6, 8), tmp_path / "spectrum.png"),
        save_probe_history(np.linspace(0.0, 0.2, 20), np.sin(np.linspace(0.0, 1.0, 20)), tmp_path / "probe.png"),
        save_energy_history(np.linspace(0.0, 0.2, 20), np.ones(20), tmp_path / "energy.png"),
    ]
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 1000
