"""ROM suite registry for Project-2 workflows.

The project now separates the ROM workflow into:

* common foundation: snapshot data loading + POD preprocessing;
* intrusive suite: POD-projected and POD-Galerkin-capable workflows;
* non-intrusive suite: coefficient-learning/surrogate methods.

This module is intentionally lightweight and has no FEniCS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ROMMethodSpec:
    """Description of one ROM method in the project suite registry."""

    name: str
    suite: str
    family: str
    status: str
    description: str
    default_enabled: bool = True


COMMON_METHODS: tuple[ROMMethodSpec, ...] = (
    ROMMethodSpec(
        name="data-loader",
        suite="common",
        family="foundation",
        status="available",
        description="Load Paper-1 and Project-2 snapshot archives into ROMDataset objects.",
    ),
    ROMMethodSpec(
        name="pod-preprocessing",
        suite="common",
        family="foundation",
        status="available",
        description="Build POD bases, coefficient matrices, train/test splits, and scalers.",
    ),
)

INTRUSIVE_METHODS: tuple[ROMMethodSpec, ...] = (
    ROMMethodSpec(
        name="pod-projected",
        suite="intrusive",
        family="projection",
        status="available",
        description="Project FOM snapshots onto an existing POD basis and evaluate projection error.",
    ),
    ROMMethodSpec(
        name="pod-galerkin",
        suite="intrusive",
        family="galerkin",
        status="capable-disabled",
        description=(
            "POD-Galerkin reduced solve. Kept disabled by default because it needs the "
            "legacy FEniCS residual/Jacobian path and case-specific state reconstruction."
        ),
        default_enabled=False,
    ),
)

NONINTRUSIVE_METHODS: tuple[ROMMethodSpec, ...] = (
    ROMMethodSpec(
        name="podi-rbf",
        suite="nonintrusive",
        family="interpolation",
        status="available",
        description="RBF interpolation from scaled parameters to POD coefficients.",
    ),
    ROMMethodSpec(
        name="podi-linear",
        suite="nonintrusive",
        family="interpolation",
        status="available",
        description="Linear interpolation from scaled parameters to POD coefficients with nearest fallback.",
    ),
    ROMMethodSpec(
        name="pod-gpr",
        suite="nonintrusive",
        family="probabilistic-regression",
        status="available",
        description="Gaussian-process regression from scaled parameters to POD coefficients.",
    ),
    ROMMethodSpec(
        name="pod-nn",
        suite="nonintrusive",
        family="neural-regression",
        status="available",
        description="Feed-forward neural network regressor from scaled parameters to POD coefficients.",
    ),
    ROMMethodSpec(
        name="pod-ae",
        suite="nonintrusive",
        family="latent-regression",
        status="available-initial",
        description=(
            "Initial POD-AE-style latent baseline: linear bottleneck autoencoding of POD "
            "coefficients plus parameter-to-latent neural regression. The dense nonlinear "
            "autoencoder can replace this backend later without changing the suite API."
        ),
    ),
)

ALL_METHODS: tuple[ROMMethodSpec, ...] = COMMON_METHODS + INTRUSIVE_METHODS + NONINTRUSIVE_METHODS


def methods_for_suite(suite: str, *, include_disabled: bool = True) -> tuple[ROMMethodSpec, ...]:
    """Return method specifications for one suite."""

    suite_key = suite.lower().strip()
    if suite_key == "all":
        methods = ALL_METHODS
    elif suite_key == "common":
        methods = COMMON_METHODS
    elif suite_key == "intrusive":
        methods = INTRUSIVE_METHODS
    elif suite_key in {"nonintrusive", "non-intrusive"}:
        methods = NONINTRUSIVE_METHODS
    else:
        raise ValueError(f"Unknown ROM suite {suite!r}. Use common, intrusive, nonintrusive, or all.")

    if not include_disabled:
        methods = tuple(m for m in methods if m.default_enabled)
    return tuple(methods)


def method_names_for_suite(suite: str, *, include_disabled: bool = False) -> tuple[str, ...]:
    """Return method names for one suite."""

    return tuple(m.name for m in methods_for_suite(suite, include_disabled=include_disabled))


def get_method_spec(name: str) -> ROMMethodSpec:
    """Return the method spec matching ``name``."""

    key = name.lower().strip()
    aliases = {
        "podi_linear": "podi-linear",
        "podi_rbf": "podi-rbf",
        "pod_gpr": "pod-gpr",
        "pod_nn": "pod-nn",
        "pod_ae": "pod-ae",
        "pod_projected": "pod-projected",
        "pod_galerkin": "pod-galerkin",
    }
    key = aliases.get(key, key)
    for spec in ALL_METHODS:
        if spec.name == key:
            return spec
    raise KeyError(f"Unknown ROM method {name!r}.")


def validate_method_list(methods: Iterable[str], *, suite: str | None = None) -> tuple[str, ...]:
    """Normalize and validate a user-provided method list."""

    normalized = tuple(get_method_spec(m).name for m in methods)
    if suite is not None and suite.lower() not in {"all", ""}:
        allowed = set(method_names_for_suite(suite, include_disabled=True))
        bad = [m for m in normalized if m not in allowed]
        if bad:
            raise ValueError(f"Methods {bad} do not belong to suite {suite!r}.")
    return normalized
