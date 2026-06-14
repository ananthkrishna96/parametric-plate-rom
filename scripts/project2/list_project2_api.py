#!/usr/bin/env python3
"""List migrated Project-2 API symbols without running a solve."""

from __future__ import annotations

import inspect

from paramplate.project2 import legacy_definitions as legacy


def main() -> None:
    exported = [
        name for name, value in vars(legacy).items()
        if not name.startswith("_") and (inspect.isclass(value) or inspect.isfunction(value))
    ]

    print(f"Project-2 importable definitions: {len(exported)}")
    for name in sorted(exported):
        print(" -", name)


if __name__ == "__main__":
    main()
