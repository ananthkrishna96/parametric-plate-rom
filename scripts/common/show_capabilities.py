#!/usr/bin/env python3
"""Print the thesis-method capability matrix recorded by the package."""

from __future__ import annotations

import argparse
from collections import defaultdict

from paramplate.capabilities import CAPABILITIES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["all", "mechanical", "thermomechanical", "dynamics"], default="all")
    args = parser.parse_args()
    aliases = {
        "mechanical": "static mechanics",
        "thermomechanical": "steady thermomechanics",
        "dynamics": "transient dynamics",
    }
    requested = aliases.get(args.study)
    grouped = defaultdict(list)
    for item in CAPABILITIES:
        if requested is None or item.study == requested:
            grouped[item.study].append(item)
    for study, entries in grouped.items():
        print(f"\n{study.upper()}")
        for entry in entries:
            print(f"- {entry.item}: {entry.status.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
