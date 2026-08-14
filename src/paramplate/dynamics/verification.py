"""Analytical, modal, and algebraic checks for transient plate dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse

from paramplate.core.parameters import OrthotropicRigidity

from .newmark import NewmarkResult, average_acceleration_newmark


@dataclass(frozen=True)
class FrequencyReference:
    frequencies_hz: np.ndarray
    angular_frequencies: np.ndarray
    mode_pairs: np.ndarray


@dataclass(frozen=True)
class FrequencyComparison:
    finite_element_hz: np.ndarray
    reference_hz: np.ndarray
    mode_pairs: np.ndarray
    relative_errors: np.ndarray

    @property
    def maximum_relative_error(self) -> float:
        return float(np.max(self.relative_errors))

    @property
    def mean_relative_error(self) -> float:
        return float(np.mean(self.relative_errors))

    def to_dict(self) -> dict[str, object]:
        return {
            "finite_element_hz": self.finite_element_hz.tolist(),
            "reference_hz": self.reference_hz.tolist(),
            "mode_pairs": self.mode_pairs.tolist(),
            "relative_errors": self.relative_errors.tolist(),
            "maximum_relative_error": self.maximum_relative_error,
            "mean_relative_error": self.mean_relative_error,
        }


@dataclass(frozen=True)
class HistoryComparison:
    relative_l2: float
    relative_linf: float
    reference_maximum: float
    comparison_maximum: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SingleModeReference:
    result: NewmarkResult
    probe_history: np.ndarray
    angular_frequency: float
    modal_mass: float
    modal_stiffness: float
    modal_load: float


def navier_angular_frequency(
    m: int,
    n: int,
    *,
    length: float,
    width: float,
    thickness: float,
    density: float,
    rigidity: OrthotropicRigidity,
    foundation_stiffness: float = 0.0,
) -> float:
    """Natural angular frequency of an SSSS aligned-orthotropic mode."""

    if m < 1 or n < 1:
        raise ValueError("Mode indices must be positive.")
    if length <= 0.0 or width <= 0.0 or thickness <= 0.0 or density <= 0.0:
        raise ValueError("Geometry, thickness, and density must be positive.")
    ax = m * np.pi / float(length)
    ay = n * np.pi / float(width)
    numerator = (
        rigidity.Dx * ax**4
        + 2.0 * (rigidity.Dxy + 2.0 * rigidity.Ds) * ax**2 * ay**2
        + rigidity.Dy * ay**4
        + float(foundation_stiffness)
    )
    if numerator < 0.0:
        raise ValueError("The modal stiffness must be nonnegative.")
    return float(np.sqrt(numerator / (float(density) * float(thickness))))


def ordered_navier_frequencies(
    n_modes: int,
    *,
    length: float,
    width: float,
    thickness: float,
    density: float,
    rigidity: OrthotropicRigidity,
    foundation_stiffness: float = 0.0,
    search_order: int | None = None,
) -> FrequencyReference:
    """Return the lowest ordered SSSS frequencies and their ``(m,n)`` pairs."""

    count = int(n_modes)
    if count < 1:
        raise ValueError("n_modes must be positive.")
    order = int(search_order or max(6, int(np.ceil(np.sqrt(4 * count))) + 2))
    if order < 1:
        raise ValueError("search_order must be positive.")
    records: list[tuple[float, int, int]] = []
    for m in range(1, order + 1):
        for n in range(1, order + 1):
            omega = navier_angular_frequency(
                m,
                n,
                length=length,
                width=width,
                thickness=thickness,
                density=density,
                rigidity=rigidity,
                foundation_stiffness=foundation_stiffness,
            )
            records.append((omega, m, n))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = records[:count]
    angular = np.asarray([item[0] for item in selected], dtype=float)
    pairs = np.asarray([[item[1], item[2]] for item in selected], dtype=np.int64)
    return FrequencyReference(
        frequencies_hz=angular / (2.0 * np.pi),
        angular_frequencies=angular,
        mode_pairs=pairs,
    )


def compare_frequencies(
    finite_element_hz: np.ndarray,
    reference: FrequencyReference,
) -> FrequencyComparison:
    fe = np.asarray(finite_element_hz, dtype=float).reshape(-1)
    ref = np.asarray(reference.frequencies_hz, dtype=float).reshape(-1)
    if fe.shape != ref.shape or fe.size == 0:
        raise ValueError("Finite-element and reference frequency arrays must share a nonempty shape.")
    if np.any(ref <= 0.0) or not np.all(np.isfinite(fe)) or not np.all(np.isfinite(ref)):
        raise ValueError("Frequency arrays must be finite and the reference strictly positive.")
    errors = np.abs(fe - ref) / ref
    return FrequencyComparison(fe, ref, reference.mode_pairs.copy(), errors)


def compare_histories(reference: np.ndarray, comparison: np.ndarray) -> HistoryComparison:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cmp = np.asarray(comparison, dtype=float).reshape(-1)
    if ref.shape != cmp.shape or ref.size == 0:
        raise ValueError("Histories must share a nonempty shape.")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cmp)):
        raise ValueError("Histories must contain finite values.")
    eps = np.finfo(float).eps
    return HistoryComparison(
        relative_l2=float(np.linalg.norm(cmp - ref) / max(np.linalg.norm(ref), eps)),
        relative_linf=float(np.max(np.abs(cmp - ref)) / max(np.max(np.abs(ref)), eps)),
        reference_maximum=float(np.max(np.abs(ref))),
        comparison_maximum=float(np.max(np.abs(cmp))),
    )


def single_mode_step_reference(
    times: np.ndarray,
    *,
    m: int,
    n: int,
    load_amplitude: float,
    probe: tuple[float, float],
    length: float,
    width: float,
    thickness: float,
    density: float,
    rigidity: OrthotropicRigidity,
    foundation_stiffness: float = 0.0,
) -> SingleModeReference:
    """Advance the spatially analytical, time-discrete Navier modal equation."""

    omega = navier_angular_frequency(
        m,
        n,
        length=length,
        width=width,
        thickness=thickness,
        density=density,
        rigidity=rigidity,
        foundation_stiffness=foundation_stiffness,
    )
    modal_mass = float(density * thickness * length * width / 4.0)
    modal_stiffness = float(modal_mass * omega**2)
    modal_load = float(load_amplitude * length * width / 4.0)
    result = average_acceleration_newmark(
        sparse.csr_matrix([[modal_mass]]),
        sparse.csr_matrix((1, 1)),
        sparse.csr_matrix([[modal_stiffness]]),
        np.asarray(times, dtype=float),
        lambda _time: np.asarray([modal_load], dtype=float),
    )
    spatial_value = np.sin(m * np.pi * float(probe[0]) / float(length)) * np.sin(
        n * np.pi * float(probe[1]) / float(width)
    )
    return SingleModeReference(
        result=result,
        probe_history=np.asarray(result.displacement[:, 0] * spatial_value, dtype=float),
        angular_frequency=omega,
        modal_mass=modal_mass,
        modal_stiffness=modal_stiffness,
        modal_load=modal_load,
    )


def modal_harmonic_response(
    angular_frequencies: np.ndarray,
    reduced_load: np.ndarray,
    probe_mode_values: np.ndarray,
    excitation_angular_frequencies: np.ndarray,
    *,
    damping_ratio: float = 0.02,
) -> np.ndarray:
    """Evaluate a mass-normalized modal frequency-response amplitude."""

    omega_n = np.asarray(angular_frequencies, dtype=float).reshape(-1)
    forcing = np.asarray(reduced_load, dtype=float).reshape(-1)
    probe = np.asarray(probe_mode_values, dtype=float).reshape(-1)
    excitation = np.asarray(excitation_angular_frequencies, dtype=float).reshape(-1)
    if omega_n.shape != forcing.shape or omega_n.shape != probe.shape:
        raise ValueError("Modal frequencies, loads, and probe values must share one shape.")
    if np.any(omega_n <= 0.0) or damping_ratio < 0.0 or excitation.size == 0:
        raise ValueError("Modal frequencies must be positive and damping nonnegative.")
    denominator = (
        omega_n[:, None] ** 2
        - excitation[None, :] ** 2
        + 2.0j * float(damping_ratio) * omega_n[:, None] * excitation[None, :]
    )
    response = np.sum((probe * forcing)[:, None] / denominator, axis=0)
    return np.abs(response)


def relative_energy_drift(mechanical_energy: np.ndarray) -> float:
    energy = np.asarray(mechanical_energy, dtype=float).reshape(-1)
    if energy.size == 0 or not np.all(np.isfinite(energy)):
        raise ValueError("mechanical_energy must be finite and nonempty.")
    reference = max(abs(float(energy[0])), np.finfo(float).eps)
    return float(np.max(np.abs(energy - energy[0])) / reference)


def relative_energy_range(mechanical_energy: np.ndarray) -> float:
    """Return ``(max(E)-min(E))/mean(abs(E))`` used in Chapter 10."""

    energy = np.asarray(mechanical_energy, dtype=float).reshape(-1)
    if energy.size == 0 or not np.all(np.isfinite(energy)):
        raise ValueError("mechanical_energy must be finite and nonempty.")
    denominator = max(float(np.mean(np.abs(energy))), np.finfo(float).eps)
    return float((np.max(energy) - np.min(energy)) / denominator)
