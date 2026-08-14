from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {
    ".git",
    ".DS_Store",
    "__MACOSX",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".pid", ".ckpt", ".pth", ".pt"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".cff", ".txt", ".json"}


def test_no_forbidden_generated_artifacts_or_heavy_files() -> None:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            problems.append(str(relative))
        if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            problems.append(str(relative))
        if path.is_file() and path.stat().st_size > 2_000_000:
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
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
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
    assert not list(ROOT.rglob("*.ipynb"))
    assert not (ROOT / "configs/data_locations.local.yml").exists()
    assert not list(ROOT.rglob("*.egg-info"))


def test_markdown_relative_links_resolve() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    missing: list[str] = []
    for document in ROOT.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            target = raw.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, missing
