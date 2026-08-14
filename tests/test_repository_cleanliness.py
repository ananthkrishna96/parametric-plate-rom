from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
}
FORBIDDEN_TRACKED_NAMES = {".DS_Store", "__MACOSX", ".ipynb_checkpoints"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".pid", ".ckpt", ".pth", ".pt"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".cff", ".txt", ".json"}


def repository_files() -> tuple[Path, ...]:
    """Return committed files when Git is present, otherwise source-tree files.

    Test and compile commands create bytecode and cache directories. Those
    runtime artifacts must not make the repository test fail unless they are
    actually tracked or packaged.
    """

    if (ROOT / ".git").is_dir() and shutil.which("git"):
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        names = [name for name in completed.stdout.decode().split("\0") if name]
        return tuple(ROOT / name for name in names)

    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in GENERATED_DIRECTORY_NAMES]
        base = Path(directory)
        paths.extend(base / name for name in filenames)
    return tuple(paths)


def test_no_forbidden_generated_artifacts_or_heavy_files() -> None:
    problems: list[str] = []
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_TRACKED_NAMES or part.endswith(".egg-info") for part in relative.parts):
            problems.append(str(relative))
        if path.suffix in FORBIDDEN_SUFFIXES:
            problems.append(str(relative))
        if path.stat().st_size > 2_000_000:
            problems.append(f"heavy:{relative}")
    assert not problems, problems


def test_no_personal_absolute_paths_or_internal_generation_language() -> None:
    forbidden_fragments = [
        "/Users/",
        "~/Documents",
        "/home/ananth",
        "ChatGPT",
        "as an AI",
        "POD-AE",
        "capable-disabled",
        "Project-2-only",
    ]
    hits: list[str] = []
    for path in repository_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name == "data_locations.template.yml":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in forbidden_fragments:
            if fragment.lower() in text.lower():
                hits.append(f"{path.relative_to(ROOT)}: {fragment}")
    assert not hits, hits


def test_no_notebook_primary_implementations_or_local_config() -> None:
    files = repository_files()
    assert not [path for path in files if path.suffix == ".ipynb"]
    assert ROOT / "configs/data_locations.local.yml" not in files
    assert not [path for path in files if any(part.endswith(".egg-info") for part in path.parts)]


def test_markdown_relative_links_resolve() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    missing: list[str] = []
    for document in (path for path in repository_files() if path.suffix == ".md"):
        text = document.read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            target = raw.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, missing
