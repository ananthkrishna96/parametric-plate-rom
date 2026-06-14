import numpy as np

from paramplate.rom.intrusive import evaluate_pod_projection
from paramplate.rom.nonintrusive import (
    evaluate_coefficient_regressor,
    make_coefficient_regressor,
)
from paramplate.rom.pod import compute_pod, project_snapshots
from paramplate.rom.suites import (
    method_names_for_suite,
    methods_for_suite,
    validate_method_list,
)


def test_suite_registry_contains_requested_methods():
    assert "pod-projected" in method_names_for_suite("intrusive", include_disabled=True)
    assert "pod-galerkin" in method_names_for_suite("intrusive", include_disabled=True)
    non = set(method_names_for_suite("nonintrusive"))
    assert {"podi-rbf", "podi-linear", "pod-gpr", "pod-nn", "pod-ae"}.issubset(non)
    assert validate_method_list(["podi_rbf", "pod_gpr"], suite="nonintrusive") == ("podi-rbf", "pod-gpr")


def test_projected_evaluation_with_loadedpod_like_object():
    from paramplate.rom.intrusive import LoadedPOD

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 8))
    pod = compute_pod(X[:15], energy_tol=0.99, center=True)
    loaded = LoadedPOD(
        basis=pod.basis,
        mean=pod.mean,
        singular_values=pod.singular_values,
        energy=pod.energy,
        cumulative_energy=pod.cumulative_energy,
        train_coefficients=project_snapshots(X[:15], pod),
        test_coefficients=project_snapshots(X[15:], pod),
        train_indices=np.arange(15),
        test_indices=np.arange(15, 20),
        centered=True,
    )
    metrics = evaluate_pod_projection(X, loaded, field_name="w")
    assert metrics.field_name == "w"
    assert metrics.rank >= 1
    assert metrics.relative_error_train >= 0.0


def test_nonintrusive_linear_regressor_on_smooth_coefficients():
    from paramplate.rom.intrusive import LoadedPOD

    x = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
    coeff = np.column_stack([1.0 + 2.0 * x[:, 0], -0.5 + x[:, 0]])
    basis = np.eye(5, 2)
    mean = np.zeros(5)
    snapshots = coeff @ basis.T + mean
    pod = LoadedPOD(
        basis=basis,
        mean=mean,
        singular_values=np.ones(2),
        energy=np.array([0.5, 0.5]),
        cumulative_energy=np.array([0.5, 1.0]),
        train_coefficients=coeff[:20],
        test_coefficients=coeff[20:],
        train_indices=np.arange(20),
        test_indices=np.arange(20, 30),
        centered=True,
    )
    model = make_coefficient_regressor("podi-linear")
    metrics, ctrain_pred, ctest_pred = evaluate_coefficient_regressor(
        method="podi-linear",
        field_name="w",
        regressor=model,
        parameters_train=x[:20],
        parameters_test=x[20:],
        coefficients_train=coeff[:20],
        coefficients_test=coeff[20:],
        pod=pod,
        snapshots_train=snapshots[:20],
        snapshots_test=snapshots[20:],
    )
    assert ctrain_pred.shape == coeff[:20].shape
    assert ctest_pred.shape == coeff[20:].shape
    assert metrics.field_relative_error_train < 1e-10


def test_make_requested_nonintrusive_models():
    for name in ["podi-rbf", "podi-linear", "pod-gpr", "pod-nn", "pod-ae"]:
        model = make_coefficient_regressor(name, max_iter=10)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
