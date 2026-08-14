"""Figure generation from stored numerical arrays."""

from .fields import save_error_histogram, save_field_image, save_pod_spectrum
from .transient import save_energy_history, save_probe_history

__all__ = [
    "save_energy_history",
    "save_error_histogram",
    "save_field_image",
    "save_pod_spectrum",
    "save_probe_history",
]
