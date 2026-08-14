"""Lazy access to the legacy FEniCS/DOLFIN high-fidelity environment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


class FenicsUnavailableError(RuntimeError):
    """Raised when a high-fidelity workflow is used without FEniCS 2019."""


@dataclass(frozen=True)
class FenicsModules:
    dolfin: Any
    mshr: Any


def require_fenics(*, require_mshr: bool = True) -> FenicsModules:
    """Import DOLFIN and, when requested, mshr.

    ROM, archive, and post-processing modules do not call this function and can
    therefore be used in a normal modern Python environment.
    """

    try:
        dolfin = import_module("dolfin")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FenicsUnavailableError(
            "This operation requires FEniCS/DOLFIN 2019.1.0. "
            "Use the recorded legacy environment; `pip install fenics` does not "
            "provide the complete DOLFIN runtime."
        ) from exc

    if require_mshr:
        try:
            mshr = import_module("mshr")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise FenicsUnavailableError(
                "This mesh workflow also requires mshr compatible with DOLFIN 2019.1.0."
            ) from exc
    else:
        mshr = None
    return FenicsModules(dolfin=dolfin, mshr=mshr)


def fenics_available(*, require_mshr: bool = False) -> bool:
    """Return whether the requested FEniCS components can be imported."""

    try:
        require_fenics(require_mshr=require_mshr)
    except FenicsUnavailableError:
        return False
    return True
