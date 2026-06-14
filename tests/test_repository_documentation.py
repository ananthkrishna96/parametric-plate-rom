from pathlib import Path


def test_readme_mentions_current_project2_workflow():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    required = [
        "v0.8-project2-rom-reporting-safe",
        "PODI-RBF",
        "POD-GPR",
        "POD-NN",
        "POD-AE",
        "parametric-plate-rom-data",
        "report_rom_suite.py",
    ]
    for needle in required:
        assert needle in readme


def test_project2_documentation_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "docs" / "workflows" / "project2_repository_usage_guide.md",
        root / "docs" / "workflows" / "project2_data_layout_and_snapshots.md",
        root / "docs" / "workflows" / "project2_rom_methods_and_hyperparameters.md",
        root / "docs" / "workflows" / "project2_full_reproducibility_run.md",
        root / "docs" / "workflows" / "project2_repository_polish_checklist.md",
        root / "docs" / "figures" / "comparison_grid_sample_50.png",
    ]
    for path in required:
        assert path.exists(), path
