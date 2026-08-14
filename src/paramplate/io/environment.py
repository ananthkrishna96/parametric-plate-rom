"""Report the software environment relevant to reproducibility records."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from typing import Iterable

from paramplate.core.fenics import fenics_available


def environment_record() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "torch", "yaml"):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "dolfin_available": fenics_available(require_mshr=False),
        "mshr_available": fenics_available(require_mshr=True),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a paramplate software-environment record.")
    parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(environment_record(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
