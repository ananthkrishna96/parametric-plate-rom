"""Input/output helpers for external research data."""

from .data_locations import DataLocations, load_data_locations
from .snapshot_archives import SnapshotArchiveInfo, summarize_snapshot_archive, detect_archive_kind
from .snapshot_manifest import build_snapshot_manifest, project2_final_archives, paper1_archives

__all__ = [
    "DataLocations",
    "load_data_locations",
    "SnapshotArchiveInfo",
    "summarize_snapshot_archive",
    "detect_archive_kind",
    "build_snapshot_manifest",
    "project2_final_archives",
    "paper1_archives",
]
