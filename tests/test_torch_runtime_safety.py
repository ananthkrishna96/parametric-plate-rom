"""Runtime-safety checks for legacy-aligned neural ROM execution."""

import os


def test_thread_environment_is_conservative_before_torch_import():
    import paramplate.rom.nonintrusive  # noqa: F401
    import torch

    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "1"
    assert torch.get_num_threads() == 1


def test_runner_exports_max_iter_flag():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/project2/run_rom_suite.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--max-iter" in proc.stdout
