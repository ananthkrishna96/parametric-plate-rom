"""Deterministic parameter designs for snapshot campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterRange:
    name: str
    lower: float
    upper: float
    unit: str = ""
    scale: str = "linear"
    role: str = "parameter"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParameterRange.name cannot be empty.")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.upper <= self.lower:
            raise ValueError(f"Invalid interval for {self.name}: [{self.lower}, {self.upper}].")
        if self.scale not in {"linear", "log"}:
            raise ValueError("scale must be 'linear' or 'log'.")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError("Logarithmic ranges require a positive lower bound.")

    def decode(self, u: np.ndarray) -> np.ndarray:
        a = np.asarray(u, dtype=float)
        if self.scale == "linear":
            return self.lower + (self.upper - self.lower) * a
        lo, hi = np.log(self.lower), np.log(self.upper)
        return np.exp(lo + (hi - lo) * a)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_RANGE_FIELDS = {"lower", "upper", "scale", "unit", "role", "description"}
_SAMPLING_METADATA = {
    "design",
    "n_samples",
    "n_trajectories",
    "requested_samples",
    "accepted_samples",
    "rejected_nonconverged",
    "seed",
    "optimization",
}


def _parse_range(default: ParameterRange, raw: Any) -> ParameterRange:
    if isinstance(raw, Mapping):
        unknown = set(raw) - _RANGE_FIELDS
        if unknown:
            raise ValueError(f"Unknown range fields for {default.name}: {sorted(unknown)}")
        if "lower" not in raw or "upper" not in raw:
            raise ValueError(
                f"Configured range {default.name!r} requires lower and upper values."
            )
        return replace(
            default,
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            scale=str(raw.get("scale", default.scale)).lower(),
            unit=str(raw.get("unit", default.unit)),
            role=str(raw.get("role", default.role)),
            description=str(raw.get("description", default.description)),
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) not in {2, 3}:
            raise ValueError(
                f"Configured range {default.name!r} must have two values and an optional scale."
            )
        return replace(
            default,
            lower=float(values[0]),
            upper=float(values[1]),
            scale=(default.scale if len(values) == 2 else str(values[2]).lower()),
        )
    raise ValueError(f"Configured range {default.name!r} must be a sequence or mapping.")


def configured_parameter_ranges(
    sampling: Mapping[str, Any] | None,
    defaults: Sequence[ParameterRange],
    *,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> tuple[ParameterRange, ...]:
    """Return parameter ranges with configuration values taking precedence.

    ``aliases`` maps each canonical range name to one or more public configuration
    aliases. Ranges may appear under ``sampling.ranges`` or directly in the
    ``sampling`` mapping for compatibility with the transient campaign files.
    Unknown range declarations and duplicate aliases are rejected rather than
    silently ignored.
    """

    declared = tuple(defaults)
    if not declared:
        raise ValueError("At least one default parameter range is required.")
    raw_sampling = {} if sampling is None else dict(sampling)
    nested = raw_sampling.get("ranges")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError("sampling.ranges must be a mapping.")
    source = dict(nested) if nested is not None else raw_sampling

    canonical = {item.name: item for item in declared}
    accepted: dict[str, str] = {name: name for name in canonical}
    for name, raw_aliases in dict(aliases or {}).items():
        if name not in canonical:
            raise ValueError(f"Alias target {name!r} is not a declared parameter range.")
        names = (raw_aliases,) if isinstance(raw_aliases, str) else tuple(raw_aliases)
        for alias in names:
            key = str(alias)
            existing = accepted.get(key)
            if existing is not None and existing != name:
                raise ValueError(f"Sampling alias {key!r} maps to more than one range.")
            accepted[key] = name

    normalized: dict[str, Any] = {}
    for raw_name, value in source.items():
        key = str(raw_name)
        if nested is None and key in _SAMPLING_METADATA:
            continue
        name = accepted.get(key)
        if name is None:
            raise ValueError(
                f"Unknown sampling range {raw_name!r}; expected {sorted(accepted)}."
            )
        if name in normalized:
            raise ValueError(f"Sampling range {name!r} was declared more than once.")
        normalized[name] = value

    return tuple(
        _parse_range(item, normalized[item.name]) if item.name in normalized else item
        for item in declared
    )


def parameter_ranges_from_config(
    sampling: Mapping[str, Any] | None,
    defaults: Sequence[ParameterRange],
    *,
    aliases: Mapping[str, str] | None = None,
) -> tuple[ParameterRange, ...]:
    """Compatibility wrapper for the earlier alias-to-canonical API."""

    canonical_aliases: dict[str, list[str]] = {}
    for alias, canonical in dict(aliases or {}).items():
        canonical_aliases.setdefault(str(canonical), []).append(str(alias))
    return configured_parameter_ranges(
        sampling,
        defaults,
        aliases={name: tuple(values) for name, values in canonical_aliases.items()},
    )


def latin_hypercube(
    ranges: Sequence[ParameterRange],
    n_samples: int,
    *,
    seed: int = 100,
    optimization: str | None = None,
) -> np.ndarray:
    """Sample a declared physical parameter design.

    Log-scaled variables are stratified in logarithmic coordinates. The return
    shape is ``(n_samples, n_parameters)`` and column order follows ``ranges``.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not ranges:
        raise ValueError("At least one parameter range is required.")
    sampler = qmc.LatinHypercube(d=len(ranges), seed=int(seed), optimization=optimization)
    unit = sampler.random(n=int(n_samples))
    return np.column_stack([spec.decode(unit[:, j]) for j, spec in enumerate(ranges)])


def parameter_records(ranges: Iterable[ParameterRange], samples: np.ndarray) -> list[dict[str, float]]:
    specs = list(ranges)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(specs):
        raise ValueError("samples do not match the declared parameter ranges.")
    return [dict(zip((s.name for s in specs), row, strict=True)) for row in values]
