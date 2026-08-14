#!/usr/bin/env python3
"""Integrate one undamped oscillator with average-acceleration Newmark."""

from __future__ import annotations

import numpy as np

from paramplate.dynamics.newmark import average_acceleration_newmark
from paramplate.dynamics.verification import relative_energy_drift


def main() -> int:
    times = np.linspace(0.0, 2.0, 2001)
    result = average_acceleration_newmark(
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[16.0]]),
        times,
        lambda _time: np.zeros(1),
        initial_displacement=np.array([1.0]),
    )
    print(f"relative energy drift = {relative_energy_drift(result.mechanical_energy):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
