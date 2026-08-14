#!/usr/bin/env python3
"""Check the centered first-moment reduction on a linear temperature profile."""

from __future__ import annotations

from paramplate.thermomechanical.thermal_driver import centered_first_moment


def main() -> int:
    imposed_gradient = 18.0
    theta = centered_first_moment(lambda z: imposed_gradient * z, 0.05)
    print(f"theta = {theta:.12g} K/m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
