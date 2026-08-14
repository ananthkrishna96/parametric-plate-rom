"""Training and held-out evaluation of one field-specific ROM workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .datasets import SnapshotDataset
from .deep import DeepTrainingConfig, DirectDLROM, PODDLROM, TransientPODDLROM
from .evaluation import EvaluationSummary, summarize_errors
from .features import impose_zero_time
from .metrics import Metric, relative_metric_errors
from .neural import PODNNConfig, PODNNRegressor
from .pipeline import PreparedROMData, prepare_rom_data
from .predictors import PODGPR, PODILinear, PODIRBF
from .preprocessing import Standardizer


@dataclass(frozen=True)
class ROMRunResult:
    method: str
    study: str
    output: str
    rank: int
    n_train: int
    n_validation: int
    n_test: int
    errors: np.ndarray
    summary: EvaluationSummary
    predicted_fields: np.ndarray
    metadata: Mapping[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "study": self.study,
            "output": self.output,
            "rank": self.rank,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "error_summary": asdict(self.summary),
            "metadata": dict(self.metadata),
        }




_SUPPORTED_METHODS: dict[str, frozenset[str]] = {
    "mechanical": frozenset(
        {"POD--Proj", "PODI--RBF", "PODI--Linear", "POD--GPR", "POD--NN"}
    ),
    "thermomechanical": frozenset(
        {
            "POD--Proj",
            "PODI--RBF",
            "PODI--Linear",
            "POD--GPR",
            "POD--NN",
            "POD--DL-ROM",
            "DL-ROM",
        }
    ),
    "dynamics": frozenset(
        {"POD--Proj", "PODI--RBF", "PODI--Linear", "POD--GPR", "POD--NN", "POD--DL-ROM"}
    ),
    "transient": frozenset(
        {"POD--Proj", "PODI--RBF", "PODI--Linear", "POD--GPR", "POD--NN", "POD--DL-ROM"}
    ),
}


def _validate_study_method(study: str, method: str) -> None:
    key = study.lower().strip()
    if key == "thermo":
        key = "thermomechanical"
    elif key in {"static", "static mechanics"}:
        key = "mechanical"
    supported = _SUPPORTED_METHODS.get(key)
    if supported is None:
        raise ValueError(f"Unknown ROM study {study!r}.")
    if method not in supported:
        raise ValueError(f"{method} is not an active thesis method for study {study!r}.")


def _method_name(method: str) -> str:
    key = method.lower().replace("_", "-").strip()
    aliases = {
        "podi-rbf": "PODI--RBF",
        "rbf": "PODI--RBF",
        "podi-linear": "PODI--Linear",
        "linear": "PODI--Linear",
        "pod-gpr": "POD--GPR",
        "gpr": "POD--GPR",
        "pod-nn": "POD--NN",
        "nn": "POD--NN",
        "pod-dl-rom": "POD--DL-ROM",
        "pod-dlrom": "POD--DL-ROM",
        "dl-rom": "DL-ROM",
        "direct-dl-rom": "DL-ROM",
        "pod-proj": "POD--Proj",
        "projection": "POD--Proj",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported ROM method {method!r}.")
    return aliases[key]


def _reconstruct_scaled(
    prepared: PreparedROMData,
    scaled_coordinates: np.ndarray,
    *,
    impose_transient_initial_state: bool = True,
) -> np.ndarray:
    coordinates = prepared.coordinate_scaler.inverse_transform(scaled_coordinates)
    if impose_transient_initial_state and prepared.dataset.is_trajectory_dataset:
        coordinates = impose_zero_time(
            coordinates,
            prepared.dataset.parameters[prepared.test_rows],
            time_column=-1,
        )
    return prepared.pod.reconstruct(coordinates)


def run_rom(
    dataset: SnapshotDataset,
    *,
    method: str,
    rank: int,
    metric: Metric = None,
    metric_name: str = "Euclidean",
    options: Mapping[str, Any] | None = None,
) -> ROMRunResult:
    """Fit one active method and evaluate only the untouched held-out subset."""

    selected = _method_name(method)
    _validate_study_method(dataset.study, selected)
    opts = dict(options or {})
    prepared = prepare_rom_data(
        dataset,
        metric=metric,
        rank=int(rank),
        metric_name=metric_name,
        transient_protocol=dataset.is_trajectory_dataset,
    )
    metadata: dict[str, Any] = {}

    if selected == "POD--Proj":
        predicted = prepared.pod.reconstruct(
            prepared.pod.project(dataset.snapshots[prepared.test_rows], metric)
        )
    elif selected in {"PODI--RBF", "PODI--Linear", "POD--GPR", "POD--NN"}:
        if selected == "PODI--RBF":
            predictor = PODIRBF(
                smoothing=float(opts.get("smoothing", 0.0)),
                neighbors=opts.get("neighbors"),
                kernel=str(opts.get("kernel", "thin_plate_spline")),
                degree=int(opts.get("degree", 1)),
            )
        elif selected == "PODI--Linear":
            predictor = PODILinear(
                fit_limit=opts.get("linear_fit_limit", opts.get("fit_limit")),
                subset_seed=int(opts.get("seed", 100)),
            )
        elif selected == "POD--GPR":
            predictor = PODGPR(
                kernel=str(opts.get("kernel", "matern52" if dataset.is_trajectory_dataset else "rbf")),
                alpha=float(opts.get("alpha", 1.0e-9)),
                white_noise=float(opts.get("white_noise", 1.0e-8)),
                optimizer_restarts=int(opts.get("optimizer_restarts", 2)),
                fit_limit=opts.get("gpr_fit_limit", opts.get("fit_limit")),
                subset_seed=int(opts.get("seed", 100)),
            )
        else:
            config = PODNNConfig(
                batch_size=int(opts.get("batch_size", 256)),
                learning_rate=float(opts.get("learning_rate", 5.0e-4)),
                weight_decay=float(opts.get("weight_decay", 2.0e-6)),
                maximum_epochs=int(opts.get("maximum_epochs", 2500)),
                patience=int(opts.get("patience", 150)),
                seed=int(opts.get("seed", 100)),
                device=str(opts.get("device", "cpu")),
            )
            predictor = PODNNRegressor(
                prepared.inputs_train.shape[1],
                prepared.coordinates_train.shape[1],
                config,
            )
            predictor.fit(
                prepared.inputs_train,
                prepared.coordinates_train,
                prepared.inputs_validation,
                prepared.coordinates_validation,
            )
            metadata["best_epoch"] = predictor.history.best_epoch
        if selected != "POD--NN":
            predictor.fit(prepared.inputs_train, prepared.coordinates_train)
        scaled = predictor.predict(prepared.inputs_test)
        predicted = _reconstruct_scaled(prepared, scaled)
    elif selected == "POD--DL-ROM":
        deep_config = DeepTrainingConfig(
            maximum_epochs=int(opts.get("maximum_epochs", 1200)),
            patience=int(opts.get("patience", 100)),
            batch_size=int(opts.get("batch_size", 256)),
            learning_rate=float(opts.get("learning_rate", 5.0e-4)),
            weight_decay=float(opts.get("weight_decay", 2.0e-6)),
            seed=int(opts.get("seed", 100)),
            device=str(opts.get("device", "cpu")),
        )
        if dataset.is_trajectory_dataset:
            all_coordinates = prepared.pod.project(dataset.snapshots, metric)
            model = TransientPODDLROM(
                dataset.parameters.shape[1],
                int(rank),
                config=deep_config,
            )
            model.fit(
                dataset.parameters[prepared.train_rows],
                all_coordinates[prepared.train_rows],
                dataset.parameters[prepared.validation_rows],
                all_coordinates[prepared.validation_rows],
            )
            coordinates = model.predict(dataset.parameters[prepared.test_rows])
            predicted = prepared.pod.reconstruct(coordinates)
        else:
            model = PODDLROM(
                prepared.inputs_train.shape[1],
                int(rank),
                latent_dimension=int(opts.get("latent_dimension", rank)),
                config=deep_config,
            )
            model.fit(
                prepared.inputs_train,
                prepared.coordinates_train,
                prepared.inputs_validation,
                prepared.coordinates_validation,
                joint_refinement_epochs=opts.get("joint_refinement_epochs"),
            )
            predicted = _reconstruct_scaled(prepared, model.predict(prepared.inputs_test))
    elif selected == "DL-ROM":
        if dataset.is_trajectory_dataset:
            raise ValueError("Direct DL-ROM is not an active transient thesis method.")
        state_scaler = Standardizer.fit(dataset.snapshots[prepared.train_rows])
        train_states = state_scaler.transform(dataset.snapshots[prepared.train_rows])
        validation_states = state_scaler.transform(dataset.snapshots[prepared.validation_rows])
        deep_config = DeepTrainingConfig(
            maximum_epochs=int(opts.get("maximum_epochs", 1200)),
            patience=int(opts.get("patience", 100)),
            batch_size=int(opts.get("batch_size", 128)),
            learning_rate=float(opts.get("learning_rate", 5.0e-4)),
            weight_decay=float(opts.get("weight_decay", 2.0e-6)),
            seed=int(opts.get("seed", 100)),
            device=str(opts.get("device", "cpu")),
        )
        model = DirectDLROM(
            prepared.inputs_train.shape[1],
            dataset.n_dofs,
            field=dataset.output_name,
            encoder_widths=opts.get("encoder_widths"),
            latent_dimension=opts.get("latent_dimension"),
            config=deep_config,
        )
        model.fit(
            prepared.inputs_train,
            train_states,
            prepared.inputs_validation,
            validation_states,
            joint_refinement_epochs=opts.get("joint_refinement_epochs"),
        )
        predicted = state_scaler.inverse_transform(model.predict(prepared.inputs_test))
    else:  # pragma: no cover - guarded by _method_name
        raise AssertionError(selected)

    errors = relative_metric_errors(
        dataset.snapshots[prepared.test_rows],
        predicted,
        metric,
        zero_policy="nan",
    )
    return ROMRunResult(
        method=selected,
        study=dataset.study,
        output=dataset.output_name,
        rank=int(rank),
        n_train=int(len(prepared.train_rows)),
        n_validation=int(len(prepared.validation_rows)),
        n_test=int(len(prepared.test_rows)),
        errors=errors,
        summary=summarize_errors(errors),
        predicted_fields=predicted,
        metadata=metadata,
    )
