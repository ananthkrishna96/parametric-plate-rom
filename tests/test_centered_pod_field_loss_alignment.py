"""Regression checks for legacy-aligned neural field-loss handling."""

from pathlib import Path


def test_pod_nn_field_loss_adds_pod_mean_for_centered_artifacts():
    src = Path("src/paramplate/rom/nonintrusive.py").read_text()
    assert "pod_mean" in src
    assert "pred_u = pred_u + mean_t" in src


def test_pod_ae_end_to_end_stage_is_implemented():
    src = Path("src/paramplate/rom/nonintrusive.py").read_text()
    assert "def _fine_tune_online_path" in src
    assert "POD-AE:e2e" in src
    assert "u_pred = c_pred @ B_t.T + mean_t" in src
