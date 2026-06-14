"""Legacy-aligned non-intrusive ROM suite.

This module intentionally mirrors the method choices, defaults, architectures,
and training style from the original ``THERMOMECHANICAL_FOM_ROM.py`` notebook
while keeping a repository-friendly, NumPy/PyTorch API.

The goal here is not to introduce new "better" hyper-parameters.  The defaults
below are the legacy paper-quality defaults:

* PODI-RBF: ``RBFInterpolator`` with thin-plate spline, smoothing 0.0.
* PODI-linear: ``LinearNDInterpolator`` with nearest-neighbour fallback.
* POD-GPR: one GP per POD coefficient, standardized coefficients, RBF/Matern
  kernel with ConstantKernel and optional WhiteKernel, optimizer restarts.
* POD-NN: PyTorch MLP ``input -> 128 -> 96 -> 64 -> n_basis`` with ELU,
  AdamW, coefficient stage and optional field-aware fine-tuning.
* POD-AE: Dense coefficient autoencoder and parameter-to-latent MLP with
  ``(128, 96, 64)`` hidden layers, ELU activation, and optional end-to-end
  field-aware fine-tuning.

All regressors keep their own parameter and coefficient scalers, matching the
legacy notebook design; they do not assume that the input parameters are already
scaled by the common POD preprocessing stage.
"""

from __future__ import annotations

# Match the original notebook's conservative CPU/thread policy before NumPy,
# SciPy, scikit-learn, and PyTorch initialise their native backends.  This is
# intentionally a runtime-safety/alignment change only; it does not change any
# ROM architecture or hyper-parameter values.
import os

for _var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator, Rbf
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as GP_Const
from sklearn.gaussian_process.kernels import Matern as GP_Matern
from sklearn.gaussian_process.kernels import RBF as GP_RBF
from sklearn.gaussian_process.kernels import WhiteKernel as GP_WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Notebook parity: force single-thread PyTorch execution.  This avoids fragile
# OpenMP/Accelerate interactions in mixed FEniCS + SciPy + PyTorch conda envs,
# while preserving the exact legacy network architectures and optimizer values.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch allows this only before parallel work starts; ignore when already set.
    pass

from .intrusive import LoadedPOD
from .pod import reconstruct_snapshots, relative_frobenius_error


# =============================================================================
# Legacy defaults copied from THERMOMECHANICAL_FOM_ROM.py
# =============================================================================

LEGACY_PODI_CONFIG: dict[str, Any] = dict(
    seed=100,
    split_seed=100,
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,
    method_list=("rbf", "linear"),
    rbf_kwargs=dict(kernel="thin_plate_spline", smoothing=0.0),
    linear_kwargs=dict(),
    nearest_fallback=True,
    save_plots=True,
    save_csv=True,
    print_worst_samples=True,
    n_worst_samples=10,
    verbose=True,
)

LEGACY_PODGPR_CONFIG: dict[str, Any] = dict(
    seed=100,
    split_seed=100,
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,
    kernel_type="rbf",
    use_white_kernel=True,
    alpha=1e-9,
    n_restarts_optimizer=10,
    random_state=100,
    normalize_y=False,
    copy_X_train=True,
    const_value=1.0,
    const_bounds=(1e-3, 1e3),
    length_scale_bounds=(1e-2, 1e2),
    white_noise_level=1e-7,
    white_noise_bounds=(1e-10, 1e-3),
    fallback_alpha=1e-7,
    fallback_n_restarts_optimizer=5,
    save_plots=True,
    save_csv=True,
    print_worst_samples=True,
    n_worst_samples=10,
    verbose=True,
)

LEGACY_PODNN_CONFIG: dict[str, Any] = dict(
    seed=100,
    split_seed=100,
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,
    hidden_layers=(128, 96, 64),
    activation="elu",
    dropout=0.00,
    layer_norm=False,
    coeff_epochs=2500,
    coeff_lr=5e-4,
    coeff_batch_size=256,
    coeff_weight_decay=2e-6,
    coeff_scheduler_patience=120,
    coeff_early_stopping_patience=450,
    coeff_min_delta=5e-9,
    coeff_grad_clip=1.0,
    field_finetune=True,
    field_epochs=275,
    field_lr=3e-5,
    field_batch_size=256,
    field_weight_decay=5e-7,
    field_scheduler_patience=35,
    field_early_stopping_patience=80,
    field_min_delta=5e-10,
    field_grad_clip=1.0,
    field_loss_weight=1.0,
    coeff_anchor_weight=0.01,
    use_full_snapshot_target=True,
    device_preference="cpu",
    save_plots=True,
    save_csv=True,
    log_every=50,
    print_worst_samples=True,
    n_worst_samples=10,
)

LEGACY_PODAE_CONFIG: dict[str, Any] = dict(
    snapshot_archive=None,
    prefer_projected=True,
    snapshot_key=None,
    n_basis=None,
    latent_dim="basis",
    pod_energy_tol=0.999999,
    n_basis_max=40,
    seed=100,
    split_seed=100,
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,
    device_preference="cpu",
    ae_hidden=(128, 96, 64),
    latent_hidden=(128, 96, 64),
    activation="elu",
    dropout=0.00,
    layer_norm=False,
    ae_epochs=2500,
    lr_ae=5e-4,
    ae_batch_size=256,
    ae_weight_decay=2e-6,
    ae_scheduler_patience=120,
    ae_early_stopping_patience=450,
    ae_min_delta=5e-9,
    ae_grad_clip=1.0,
    latent_epochs=2500,
    lr_latent=5e-4,
    latent_batch_size=256,
    latent_weight_decay=2e-6,
    latent_scheduler_patience=120,
    latent_early_stopping_patience=450,
    latent_min_delta=5e-9,
    latent_grad_clip=1.0,
    end_to_end_finetune=True,
    e2e_epochs=500,
    e2e_lr=3e-5,
    e2e_batch_size=256,
    e2e_train_decoder=True,
    e2e_train_encoder=False,
    e2e_weight_decay=5e-7,
    e2e_patience=120,
    e2e_scheduler_patience=50,
    e2e_min_delta=5e-10,
    e2e_loss="field_relative_plus_coeff",
    e2e_use_full_snapshot_target=True,
    e2e_field_weight=1.0,
    e2e_coeff_weight=0.01,
    e2e_grad_clip=1.0,
    save_model=True,
    save_plots=True,
    save_csv=True,
    model_name="pod_ae_rom_robust",
    log_every=50,
    print_worst_samples=True,
    n_worst_samples=10,
)


@dataclass(frozen=True)
class NonIntrusiveMetrics:
    """Prediction metrics for one non-intrusive field surrogate."""

    field_name: str
    method: str
    rank: int
    coefficient_relative_error_train: float
    coefficient_relative_error_test: float
    field_relative_error_train: float
    field_relative_error_test: float

    def to_dict(self) -> dict[str, Any]:
        return dict(
            field_name=self.field_name,
            method=self.method,
            rank=self.rank,
            coefficient_relative_error_train=self.coefficient_relative_error_train,
            coefficient_relative_error_test=self.coefficient_relative_error_test,
            field_relative_error_train=self.field_relative_error_train,
            field_relative_error_test=self.field_relative_error_test,
        )


def _seed(seed: int = 100) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _device(preference: str = "cpu") -> torch.device:
    pref = str(preference).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        return torch.device("mps" if ok else "cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        return torch.device("mps" if ok else "cpu")
    raise ValueError("device_preference must be one of 'cpu', 'cuda', 'mps', or 'auto'.")


def _activation(name: str) -> type[nn.Module]:
    acts = dict(tanh=nn.Tanh, relu=nn.ReLU, elu=nn.ELU, gelu=nn.GELU, silu=nn.SiLU)
    key = str(name).lower()
    if key not in acts:
        raise ValueError(f"Unsupported activation {name!r}.")
    return acts[key]


def _mlp(dims: tuple[int, ...], *, activation: str = "elu", dropout: float = 0.0,
         layer_norm: bool = False, final_activation: bool = False) -> nn.Sequential:
    act = _activation(activation)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(int(a), int(b)))
        is_last = i == len(dims) - 2
        if (not is_last) or final_activation:
            if layer_norm:
                layers.append(nn.LayerNorm(int(b)))
            layers.append(act())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _as_2d(A: np.ndarray) -> np.ndarray:
    X = np.asarray(A, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Array contains NaN or infinite values.")
    return X


def _relative_matrix_error(reference: np.ndarray, approximation: np.ndarray, eps: float = 1e-14) -> float:
    A = _as_2d(reference)
    B = _as_2d(approximation)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}.")
    denom = max(float(np.linalg.norm(A, ord="fro")), float(eps))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


class CoefficientRegressor:
    """Base interface for legacy-aligned coefficient regressors."""

    method_name = "base"

    def fit(self, X: np.ndarray, Y: np.ndarray, **context: Any) -> "CoefficientRegressor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LegacyPODIRegressor(CoefficientRegressor):
    """Legacy PODI: RBFInterpolator/Rbf or LinearNDInterpolator + nearest fallback."""

    def __init__(self, *, method: str = "rbf", config: dict[str, Any] | None = None):
        self.method = method.lower()
        self.method_name = "podi-rbf" if self.method == "rbf" else "podi-linear"
        self.config = dict(LEGACY_PODI_CONFIG if config is None else config)
        self.param_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()
        self.interpolator = None
        self.nearest = None
        self._rbf_legacy_models = None
        self._x1 = None
        self._y1 = None
        self.fallback_count = 0

    def fit(self, X: np.ndarray, Y: np.ndarray, **context: Any) -> "LegacyPODIRegressor":
        _seed(self.config.get("seed", 100))
        X_raw = _as_2d(X)
        Y_raw = _as_2d(Y)
        Xs = self.param_scaler.fit_transform(X_raw)
        Ys = self.coeff_scaler.fit_transform(Y_raw)
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        if self.method == "rbf":
            kwargs = dict(self.config.get("rbf_kwargs", dict(kernel="thin_plate_spline", smoothing=0.0)))
            try:
                self.interpolator = RBFInterpolator(Xs, Ys, **kwargs)
                self._rbf_legacy_models = None
            except Exception:
                # Retain legacy scipy.interpolate.Rbf fallback, one coefficient at a time.
                kernel = kwargs.get("kernel", "thin_plate_spline")
                smoothing = float(kwargs.get("smoothing", 0.0))
                self._rbf_legacy_models = [
                    Rbf(*[Xs[:, k] for k in range(Xs.shape[1])], Ys[:, j], function=kernel, smooth=smoothing)
                    for j in range(Ys.shape[1])
                ]
        elif self.method == "linear":
            if Xs.shape[1] == 1:
                order = np.argsort(Xs[:, 0])
                self._x1 = Xs[order, 0]
                self._y1 = Ys[order]
            else:
                self.interpolator = LinearNDInterpolator(Xs, Ys, **dict(self.config.get("linear_kwargs", {})))
                if bool(self.config.get("nearest_fallback", True)):
                    self.nearest = NearestNDInterpolator(Xs, Ys)
        else:
            raise ValueError("LegacyPODIRegressor method must be 'rbf' or 'linear'.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.param_scaler.transform(_as_2d(X))
        if self.method == "rbf":
            if self._rbf_legacy_models is not None:
                Ys = np.column_stack([
                    m(*[Xs[:, k] for k in range(Xs.shape[1])])
                    for m in self._rbf_legacy_models
                ])
            else:
                Ys = np.asarray(self.interpolator(Xs), dtype=np.float64)
        else:
            if self._x1 is not None:
                Ys = np.vstack([np.interp(Xs[:, 0], self._x1, self._y1[:, j]) for j in range(self._y1.shape[1])]).T
            else:
                Ys = np.asarray(self.interpolator(Xs), dtype=np.float64)
                if Ys.ndim == 1:
                    Ys = Ys.reshape(-1, 1)
                missing = ~np.all(np.isfinite(Ys), axis=1)
                self.fallback_count = int(np.sum(missing))
                if np.any(missing):
                    if self.nearest is None:
                        raise RuntimeError("Linear PODI produced NaNs and nearest fallback is disabled.")
                    Ys[missing] = np.asarray(self.nearest(Xs[missing]), dtype=np.float64)
        return _as_2d(self.coeff_scaler.inverse_transform(_as_2d(Ys)))


def _gpr_kernel(cfg: dict[str, Any], n_features: int):
    ls0 = np.ones(int(n_features), dtype=float)
    bounds = tuple(cfg.get("length_scale_bounds", (1e-2, 1e2)))
    if str(cfg.get("kernel_type", "rbf")).lower() == "matern":
        spatial = GP_Matern(length_scale=ls0, length_scale_bounds=bounds, nu=2.5)
    else:
        spatial = GP_RBF(length_scale=ls0, length_scale_bounds=bounds)
    kernel = GP_Const(float(cfg.get("const_value", 1.0)), tuple(cfg.get("const_bounds", (1e-3, 1e3)))) * spatial
    if bool(cfg.get("use_white_kernel", True)):
        kernel = kernel + GP_WhiteKernel(
            noise_level=float(cfg.get("white_noise_level", 1e-7)),
            noise_level_bounds=tuple(cfg.get("white_noise_bounds", (1e-10, 1e-3))),
        )
    return kernel


class LegacyGPRRegressor(CoefficientRegressor):
    """Legacy POD-GPR: one GaussianProcessRegressor per standardized coefficient."""

    method_name = "pod-gpr"

    def __init__(self, *, config: dict[str, Any] | None = None, kernel_type: str | None = None):
        self.config = dict(LEGACY_PODGPR_CONFIG if config is None else config)
        if kernel_type is not None and kernel_type in {"rbf", "matern"}:
            self.config["kernel_type"] = kernel_type
        self.param_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()
        self.models: list[GaussianProcessRegressor] = []
        self.training_rows: list[dict[str, Any]] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, **context: Any) -> "LegacyGPRRegressor":
        cfg = self.config
        _seed(cfg.get("seed", 100))
        Xs = self.param_scaler.fit_transform(_as_2d(X))
        Ys = self.coeff_scaler.fit_transform(_as_2d(Y))
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0
        self.models = []
        self.training_rows = []

        for j in range(Ys.shape[1]):
            try:
                gp = GaussianProcessRegressor(
                    kernel=_gpr_kernel(cfg, Xs.shape[1]),
                    alpha=float(cfg.get("alpha", 1e-9)),
                    n_restarts_optimizer=int(cfg.get("n_restarts_optimizer", 10)),
                    random_state=int(cfg.get("random_state", 100)),
                    normalize_y=bool(cfg.get("normalize_y", False)),
                    copy_X_train=bool(cfg.get("copy_X_train", True)),
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.filterwarnings("always", category=ConvergenceWarning)
                    gp.fit(Xs, Ys[:, j])
                status = "ok"
                n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            except Exception as exc:
                gp = GaussianProcessRegressor(
                    kernel=GP_Const(1.0, (1e-3, 1e3)) * GP_RBF(np.ones(Xs.shape[1]), tuple(cfg.get("length_scale_bounds", (1e-2, 1e2)))) +
                    GP_WhiteKernel(noise_level=float(cfg.get("white_noise_level", 1e-7)),
                                   noise_level_bounds=tuple(cfg.get("white_noise_bounds", (1e-10, 1e-3)))),
                    alpha=float(cfg.get("fallback_alpha", 1e-7)),
                    random_state=int(cfg.get("random_state", 100)),
                    n_restarts_optimizer=int(cfg.get("fallback_n_restarts_optimizer", 5)),
                    normalize_y=False,
                )
                gp.fit(Xs, Ys[:, j])
                status = "fallback"
                n_warn = math.nan
                self.training_rows.append({"coeff_index": j, "status": status, "note": str(exc)[:120]})
            self.models.append(gp)
            self.training_rows.append({
                "coeff_index": int(j),
                "status": status,
                "kernel": str(gp.kernel_),
                "log_marginal_likelihood": float(gp.log_marginal_likelihood(gp.kernel_.theta)),
                "n_convergence_warnings": n_warn,
            })
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("LegacyGPRRegressor has not been fitted.")
        Xs = self.param_scaler.transform(_as_2d(X))
        Ys = np.column_stack([m.predict(Xs) for m in self.models])
        return _as_2d(self.coeff_scaler.inverse_transform(Ys))


class LegacyTorchNNRegressor(CoefficientRegressor):
    """Legacy POD-NN with PyTorch MLP and two-stage training."""

    method_name = "pod-nn"

    def __init__(self, *, n_basis: int | None = None, config: dict[str, Any] | None = None):
        self.config = dict(LEGACY_PODNN_CONFIG if config is None else config)
        self.n_basis = None if n_basis is None else int(n_basis)
        self.device = _device(self.config.get("device_preference", "cpu"))
        self.X_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.history: dict[str, Any] = {}

    def _split_train_val(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(int(n))
        if n < 3:
            return idx, idx
        train_fraction = float(self.config.get("train_fraction", 0.80))
        val_fraction = float(self.config.get("val_fraction", 0.10))
        # Split only the provided training set into inner train/validation while
        # preserving the legacy 80/10 proportion as closely as possible.
        inner_val = val_fraction / max(train_fraction + val_fraction, 1e-12)
        tr, va = train_test_split(idx, test_size=inner_val, random_state=int(self.config.get("split_seed", 100)) + 1, shuffle=True)
        return np.asarray(tr, dtype=int), np.asarray(va, dtype=int)

    def _loader(self, Xs, Ys, ids, batch_size, shuffle=False, S=None):
        ids = np.asarray(ids, dtype=int)
        if S is None:
            ds = TensorDataset(torch.tensor(Xs[ids], dtype=torch.float32), torch.tensor(Ys[ids], dtype=torch.float32))
        else:
            ds = TensorDataset(torch.tensor(Xs[ids], dtype=torch.float32), torch.tensor(Ys[ids], dtype=torch.float32), torch.tensor(S[ids], dtype=torch.float32))
        return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))

    def _build(self, input_dim: int, output_dim: int) -> None:
        cfg = self.config
        self.model = _mlp(
            (int(input_dim), *tuple(cfg.get("hidden_layers", (128, 96, 64))), int(output_dim)),
            activation=cfg.get("activation", "elu"),
            dropout=float(cfg.get("dropout", 0.0)),
            layer_norm=bool(cfg.get("layer_norm", False)),
        ).to(self.device)

    def _train_stage(self, train_loader, val_loader, *, epochs, lr, weight_decay, scheduler_patience,
                     early_stopping_patience, min_delta, grad_clip, use_field_loss=False,
                     basis=None, pod_mean=None, coeff_mean=None, coeff_std=None, field_weight=1.0, coeff_anchor=0.01,
                     label="coeff"):
        assert self.model is not None
        optimizer = optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(scheduler_patience))
        mse = nn.MSELoss()
        best_val = np.inf
        best_state = None
        wait = 0
        hist = {"train": [], "val": [], "lr": []}
        B_t = torch.tensor(basis, dtype=torch.float32, device=self.device) if basis is not None else None
        mean_t = torch.tensor(pod_mean, dtype=torch.float32, device=self.device) if pod_mean is not None else None
        cm_t = torch.tensor(coeff_mean, dtype=torch.float32, device=self.device) if coeff_mean is not None else None
        cs_t = torch.tensor(coeff_std, dtype=torch.float32, device=self.device) if coeff_std is not None else None

        def loss_for_batch(batch):
            xb = batch[0].to(self.device)
            yb = batch[1].to(self.device)
            pred = self.model(xb)
            coeff_loss = mse(pred, yb)
            if not use_field_loss or B_t is None or len(batch) < 3:
                return coeff_loss
            sb = batch[2].to(self.device)
            pred_c = pred * cs_t + cm_t
            pred_u = pred_c @ B_t.T
            if mean_t is not None:
                pred_u = pred_u + mean_t
            field_loss = torch.mean(torch.sum((pred_u - sb) ** 2, dim=1) / (torch.sum(sb ** 2, dim=1) + 1e-24))
            return float(field_weight) * field_loss + float(coeff_anchor) * coeff_loss

        log_every = int(self.config.get("log_every", 50))
        for ep in range(1, int(epochs) + 1):
            self.model.train()
            total = count = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = loss_for_batch(batch)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()
                bs = batch[0].shape[0]
                total += float(loss.item()) * bs
                count += bs
            tr = total / max(count, 1)

            self.model.eval()
            total = count = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss = loss_for_batch(batch)
                    bs = batch[0].shape[0]
                    total += float(loss.item()) * bs
                    count += bs
            va = total / max(count, 1)
            scheduler.step(va)
            hist["train"].append(tr)
            hist["val"].append(va)
            hist["lr"].append(float(optimizer.param_groups[0]["lr"]))
            if ep == 1 or ep % log_every == 0 or ep == int(epochs):
                print(f" [{label}] Epoch {ep:5d}/{int(epochs):5d} | train={tr:.4e} | val={va:.4e} | lr={optimizer.param_groups[0]['lr']:.2e}")
            if va < best_val - float(min_delta):
                best_val = va
                best_state = _clone_state(self.model)
                wait = 0
            else:
                wait += 1
            if wait >= int(early_stopping_patience):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        hist["best_val"] = float(best_val)
        return hist

    def fit(self, X: np.ndarray, Y: np.ndarray, **context: Any) -> "LegacyTorchNNRegressor":
        cfg = self.config
        _seed(cfg.get("seed", 100))
        self.device = _device(cfg.get("device_preference", "cpu"))
        X_raw = _as_2d(X)
        Y_raw = _as_2d(Y)
        self.n_basis = int(self.n_basis or Y_raw.shape[1])
        Xs = self.X_scaler.fit_transform(X_raw).astype(np.float32)
        Ys = self.coeff_scaler.fit_transform(Y_raw).astype(np.float32)
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0
        tr, va = self._split_train_val(X_raw.shape[0])
        self._build(Xs.shape[1], Ys.shape[1])
        bs1 = int(cfg.get("coeff_batch_size", 256))
        self.history["coeff"] = self._train_stage(
            self._loader(Xs, Ys, tr, bs1, shuffle=bs1 < len(tr)),
            self._loader(Xs, Ys, va, bs1, shuffle=False),
            epochs=cfg.get("coeff_epochs", 2500),
            lr=cfg.get("coeff_lr", 5e-4),
            weight_decay=cfg.get("coeff_weight_decay", 2e-6),
            scheduler_patience=cfg.get("coeff_scheduler_patience", 120),
            early_stopping_patience=cfg.get("coeff_early_stopping_patience", 450),
            min_delta=cfg.get("coeff_min_delta", 5e-9),
            grad_clip=cfg.get("coeff_grad_clip", 1.0),
            label="coeff",
        )

        S = context.get("snapshots_train")
        pod = context.get("pod")
        if bool(cfg.get("field_finetune", True)) and S is not None and pod is not None:
            S32 = np.asarray(S, dtype=np.float32)
            bs2 = int(cfg.get("field_batch_size", 256))
            self.history["field"] = self._train_stage(
                self._loader(Xs, Ys, tr, bs2, shuffle=bs2 < len(tr), S=S32),
                self._loader(Xs, Ys, va, bs2, shuffle=False, S=S32),
                epochs=cfg.get("field_epochs", 275),
                lr=cfg.get("field_lr", 3e-5),
                weight_decay=cfg.get("field_weight_decay", 5e-7),
                scheduler_patience=cfg.get("field_scheduler_patience", 35),
                early_stopping_patience=cfg.get("field_early_stopping_patience", 80),
                min_delta=cfg.get("field_min_delta", 5e-10),
                grad_clip=cfg.get("field_grad_clip", 1.0),
                use_field_loss=True,
                basis=pod.basis,
                pod_mean=getattr(pod, "mean", None),
                coeff_mean=self.coeff_scaler.mean_,
                coeff_std=self.coeff_scaler.scale_,
                field_weight=cfg.get("field_loss_weight", 1.0),
                coeff_anchor=cfg.get("coeff_anchor_weight", 0.01),
                label="field",
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LegacyTorchNNRegressor has not been fitted.")
        Xs = self.X_scaler.transform(_as_2d(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(Xs, dtype=torch.float32, device=self.device)).cpu().numpy()
        return _as_2d(self.coeff_scaler.inverse_transform(pred))


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(128, 96, 64), activation="elu",
                 dropout=0.0, layer_norm=False):
        super().__init__()
        self.encoder = _mlp((int(input_dim), *map(int, hidden), int(latent_dim)),
                            activation=activation, dropout=dropout, layer_norm=layer_norm)
        self.decoder = _mlp((int(latent_dim), *map(int, reversed(hidden)), int(input_dim)),
                            activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ParameterToLatentMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(128, 96, 64), activation="elu",
                 dropout=0.0, layer_norm=False):
        super().__init__()
        self.net = _mlp((int(input_dim), *map(int, hidden), int(latent_dim)),
                        activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, x):
        return self.net(x)


class LegacyPODAERegressor(CoefficientRegressor):
    """Legacy POD-AE: coefficient AE + parameter-to-latent MLP."""

    method_name = "pod-ae"

    def __init__(self, *, n_basis: int | None = None, latent_dim: int | str | None = None,
                 config: dict[str, Any] | None = None):
        self.config = dict(LEGACY_PODAE_CONFIG if config is None else config)
        if latent_dim is not None:
            self.config["latent_dim"] = latent_dim
        self.n_basis = None if n_basis is None else int(n_basis)
        self.latent_dim = latent_dim
        self.device = _device(self.config.get("device_preference", "cpu"))
        self.mu_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()
        self.latent_scaler = StandardScaler()
        self.autoencoder: DenseAutoencoder | None = None
        self.latent_map: ParameterToLatentMLP | None = None
        self.history: dict[str, Any] = {}

    def _resolve_latent_dim(self, n_basis: int, n_mu: int) -> int:
        val = self.config.get("latent_dim", "basis")
        if val in (None, "basis"):
            return int(n_basis)
        if val == "compressed":
            return max(1, min(int(n_basis), max(1, int(n_mu))))
        return max(1, min(int(n_basis), int(val)))

    def _loader(self, X, Y, batch_size, shuffle):
        ds = TensorDataset(torch.tensor(np.asarray(X, dtype=np.float32)), torch.tensor(np.asarray(Y, dtype=np.float32)))
        return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))

    def _train_model(self, model, train_loader, val_loader, *, epochs, lr, weight_decay, patience, scheduler_patience,
                     min_delta, grad_clip, label):
        model = model.to(self.device)
        opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=int(scheduler_patience))
        loss_fn = nn.MSELoss()
        best = np.inf
        wait = 0
        best_state = None
        hist = {"train": [], "val": [], "lr": []}
        log_every = int(self.config.get("log_every", 50))
        for ep in range(1, int(epochs) + 1):
            model.train()
            tr_sum = tr_n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                pred = pred[0] if isinstance(pred, tuple) else pred
                loss = loss_fn(pred, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()
                tr_sum += float(loss.item()) * xb.shape[0]
                tr_n += xb.shape[0]
            model.eval()
            va_sum = va_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    pred = pred[0] if isinstance(pred, tuple) else pred
                    loss = loss_fn(pred, yb)
                    va_sum += float(loss.item()) * xb.shape[0]
                    va_n += xb.shape[0]
            tr = tr_sum / max(tr_n, 1)
            va = va_sum / max(va_n, 1)
            sched.step(va)
            hist["train"].append(tr)
            hist["val"].append(va)
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
            if ep == 1 or ep % log_every == 0 or ep == int(epochs):
                print(f" [{label}] Epoch {ep:5d}/{int(epochs):5d} | train={tr:.4e} | val={va:.4e} | lr={opt.param_groups[0]['lr']:.2e}")
            if va < best - float(min_delta):
                best = va
                wait = 0
                best_state = _clone_state(model)
            else:
                wait += 1
            if wait >= int(patience):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        hist["best_val"] = float(best)
        return hist

    def fit(self, X: np.ndarray, Y: np.ndarray, **context: Any) -> "LegacyPODAERegressor":
        cfg = self.config
        _seed(cfg.get("seed", 100))
        self.device = _device(cfg.get("device_preference", "cpu"))
        X_raw = _as_2d(X)
        Y_raw = _as_2d(Y)
        n_mu = X_raw.shape[1]
        self.n_basis = int(self.n_basis or Y_raw.shape[1])
        self.latent_dim = self._resolve_latent_dim(self.n_basis, n_mu)

        # Inner train/validation split from supplied training set.
        idx = np.arange(X_raw.shape[0])
        if X_raw.shape[0] >= 3:
            train_idx, val_idx = train_test_split(
                idx,
                test_size=float(cfg.get("val_fraction", 0.10)) / max(float(cfg.get("train_fraction", 0.80)) + float(cfg.get("val_fraction", 0.10)), 1e-12),
                random_state=int(cfg.get("split_seed", 100)) + 1,
                shuffle=True,
            )
        else:
            train_idx = val_idx = idx

        Mu = self.mu_scaler.fit_transform(X_raw).astype(np.float32)
        C = self.coeff_scaler.fit_transform(Y_raw).astype(np.float32)
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        self.autoencoder = DenseAutoencoder(
            self.n_basis,
            self.latent_dim,
            hidden=cfg.get("ae_hidden", (128, 96, 64)),
            activation=cfg.get("activation", "elu"),
            dropout=float(cfg.get("dropout", 0.0)),
            layer_norm=bool(cfg.get("layer_norm", False)),
        ).to(self.device)

        self.history["autoencoder"] = self._train_model(
            self.autoencoder,
            self._loader(C[train_idx], C[train_idx], cfg.get("ae_batch_size", 256), True),
            self._loader(C[val_idx], C[val_idx], cfg.get("ae_batch_size", 256), False),
            epochs=cfg.get("ae_epochs", 2500),
            lr=cfg.get("lr_ae", 5e-4),
            weight_decay=cfg.get("ae_weight_decay", 2e-6),
            patience=cfg.get("ae_early_stopping_patience", 450),
            scheduler_patience=cfg.get("ae_scheduler_patience", 120),
            min_delta=cfg.get("ae_min_delta", 5e-9),
            grad_clip=cfg.get("ae_grad_clip", 1.0),
            label="AE",
        )

        self.autoencoder.eval()
        with torch.no_grad():
            Z = self.autoencoder.encoder(torch.tensor(C, dtype=torch.float32, device=self.device)).cpu().numpy()
        Zs = self.latent_scaler.fit_transform(Z).astype(np.float32)

        self.latent_map = ParameterToLatentMLP(
            n_mu,
            self.latent_dim,
            hidden=cfg.get("latent_hidden", (128, 96, 64)),
            activation=cfg.get("activation", "elu"),
            dropout=float(cfg.get("dropout", 0.0)),
            layer_norm=bool(cfg.get("layer_norm", False)),
        ).to(self.device)

        self.history["latent_map"] = self._train_model(
            self.latent_map,
            self._loader(Mu[train_idx], Zs[train_idx], cfg.get("latent_batch_size", 256), True),
            self._loader(Mu[val_idx], Zs[val_idx], cfg.get("latent_batch_size", 256), False),
            epochs=cfg.get("latent_epochs", 2500),
            lr=cfg.get("lr_latent", 5e-4),
            weight_decay=cfg.get("latent_weight_decay", 2e-6),
            patience=cfg.get("latent_early_stopping_patience", 450),
            scheduler_patience=cfg.get("latent_scheduler_patience", 120),
            min_delta=cfg.get("latent_min_delta", 5e-9),
            grad_clip=cfg.get("latent_grad_clip", 1.0),
            label="mu-to-latent",
        )
        self._fine_tune_online_path(
            Mu,
            C,
            train_idx,
            val_idx,
            snapshots=context.get("snapshots_train"),
            pod=context.get("pod"),
        )
        return self

    def _fine_tune_online_path(self, Mu, C, train_idx, val_idx, *, snapshots, pod) -> None:
        """Legacy POD-AE end-to-end online-path fine-tuning.

        This mirrors the notebook's ``finetune_pod_ae_end_to_end`` stage:
        parameter-to-latent MLP + decoder are optimized with a field-relative
        loss and a small coefficient anchor.  The encoder is normally frozen.
        For centered POD artifacts, the POD mean must be added back when the
        reconstructed field is formed; the original notebook POD basis is
        uncentered, so this term is zero there.
        """
        if self.autoencoder is None or self.latent_map is None:
            return
        if snapshots is None or pod is None:
            return

        cfg = self.config
        if not bool(cfg.get("end_to_end_finetune", True)):
            return

        epochs = int(cfg.get("e2e_epochs", 500))
        if epochs <= 0:
            return

        S = np.asarray(snapshots, dtype=np.float32)
        B_t = torch.tensor(np.asarray(pod.basis, dtype=np.float32), device=self.device)
        mean_t = torch.tensor(np.asarray(getattr(pod, "mean", np.zeros(B_t.shape[0])), dtype=np.float32), device=self.device)
        cmean_t = torch.tensor(self.coeff_scaler.mean_, dtype=torch.float32, device=self.device)
        cstd_t = torch.tensor(self.coeff_scaler.scale_, dtype=torch.float32, device=self.device)

        train_decoder = bool(cfg.get("e2e_train_decoder", True))
        train_encoder = bool(cfg.get("e2e_train_encoder", False))
        if not train_encoder:
            for par in self.autoencoder.encoder.parameters():
                par.requires_grad_(False)

        params = list(self.latent_map.parameters())
        if train_decoder:
            params += list(self.autoencoder.decoder.parameters())
        if train_encoder:
            params += list(self.autoencoder.encoder.parameters())
        if not params:
            return

        ds_train = TensorDataset(
            torch.tensor(Mu[train_idx], dtype=torch.float32),
            torch.tensor(C[train_idx], dtype=torch.float32),
            torch.tensor(S[train_idx], dtype=torch.float32),
        )
        ds_val = TensorDataset(
            torch.tensor(Mu[val_idx], dtype=torch.float32),
            torch.tensor(C[val_idx], dtype=torch.float32),
            torch.tensor(S[val_idx], dtype=torch.float32),
        )
        bs = int(cfg.get("e2e_batch_size", 256))
        train_loader = DataLoader(ds_train, batch_size=bs, shuffle=bs < len(train_idx))
        val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False)

        opt = optim.AdamW(params, lr=float(cfg.get("e2e_lr", 3e-5)), weight_decay=float(cfg.get("e2e_weight_decay", 5e-7)))
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=int(cfg.get("e2e_scheduler_patience", 50)))
        coeff_loss_fn = nn.MSELoss()
        field_weight = float(cfg.get("e2e_field_weight", 1.0))
        coeff_weight = float(cfg.get("e2e_coeff_weight", 0.01))
        grad_clip = cfg.get("e2e_grad_clip", 1.0)
        patience = int(cfg.get("e2e_patience", 120))
        min_delta = float(cfg.get("e2e_min_delta", 5e-10))
        log_every = int(cfg.get("log_every", 50))

        best = np.inf
        best_state = None
        best_epoch = 0
        wait = 0
        hist = {"train": [], "val": [], "lr": []}

        def loss_for_batch(batch):
            xb, yb, sb = (t.to(self.device) for t in batch)
            z_scaled = self.latent_map(xb)
            z_raw = torch.tensor(self.latent_scaler.mean_, dtype=torch.float32, device=self.device) + z_scaled * torch.tensor(self.latent_scaler.scale_, dtype=torch.float32, device=self.device)
            c_pred_s = self.autoencoder.decoder(z_raw)
            coeff_loss = coeff_loss_fn(c_pred_s, yb)
            c_pred = c_pred_s * cstd_t + cmean_t
            u_pred = c_pred @ B_t.T + mean_t
            field_loss = torch.mean(torch.sum((u_pred - sb) ** 2, dim=1) / (torch.sum(sb ** 2, dim=1) + 1e-24))
            return field_weight * field_loss + coeff_weight * coeff_loss

        print(" [POD-AE:e2e] Fine-tuning online path with field-relative loss "
              f"(epochs={epochs}, lr={float(cfg.get('e2e_lr', 3e-5)):.2e}, "
              f"field_weight={field_weight:g}, coeff_weight={coeff_weight:g})")

        for ep in range(1, epochs + 1):
            self.latent_map.train()
            self.autoencoder.decoder.train(train_decoder)
            self.autoencoder.encoder.train(train_encoder)
            total = count = 0.0
            for batch in train_loader:
                opt.zero_grad()
                loss = loss_for_batch(batch)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
                opt.step()
                total += float(loss.item()) * batch[0].shape[0]
                count += batch[0].shape[0]
            tr = total / max(count, 1)

            self.latent_map.eval()
            self.autoencoder.eval()
            total = count = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss = loss_for_batch(batch)
                    total += float(loss.item()) * batch[0].shape[0]
                    count += batch[0].shape[0]
            va = total / max(count, 1)
            sched.step(va)
            hist["train"].append(tr)
            hist["val"].append(va)
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                print(f" [POD-AE:e2e] Epoch {ep:5d}/{epochs:5d} | train={tr:.4e} | val={va:.4e} | lr={opt.param_groups[0]['lr']:.2e}")
            if va < best - min_delta:
                best = va
                best_epoch = ep
                wait = 0
                best_state = {
                    "latent_map": _clone_state(self.latent_map),
                    "decoder": _clone_state(self.autoencoder.decoder),
                    "encoder": _clone_state(self.autoencoder.encoder),
                }
            else:
                wait += 1
            if wait >= patience:
                break
        if best_state is not None:
            self.latent_map.load_state_dict(best_state["latent_map"])
            self.autoencoder.decoder.load_state_dict(best_state["decoder"])
            self.autoencoder.encoder.load_state_dict(best_state["encoder"])
        hist["best_val"] = float(best)
        hist["best_epoch"] = int(best_epoch)
        self.history["end_to_end"] = hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.autoencoder is None or self.latent_map is None:
            raise RuntimeError("LegacyPODAERegressor has not been fitted.")
        Mu = self.mu_scaler.transform(_as_2d(X)).astype(np.float32)
        self.autoencoder.eval()
        self.latent_map.eval()
        with torch.no_grad():
            Zs = self.latent_map(torch.tensor(Mu, dtype=torch.float32, device=self.device)).cpu().numpy()
            Z = self.latent_scaler.inverse_transform(Zs)
            C_scaled = self.autoencoder.decoder(torch.tensor(Z, dtype=torch.float32, device=self.device)).cpu().numpy()
        return _as_2d(self.coeff_scaler.inverse_transform(C_scaled))


def make_coefficient_regressor(method: str, **kwargs: Any) -> CoefficientRegressor:
    """Create a legacy-aligned coefficient regressor by suite method name."""

    key = method.lower().strip().replace("_", "-")
    if key == "podi-rbf":
        cfg = dict(LEGACY_PODI_CONFIG)
        cfg["rbf_kwargs"] = dict(cfg["rbf_kwargs"])
        if "kernel" in kwargs and kwargs["kernel"]:
            cfg["rbf_kwargs"]["kernel"] = kwargs["kernel"]
        if "config" in kwargs and kwargs["config"] is not None:
            cfg.update(kwargs["config"])
        return LegacyPODIRegressor(method="rbf", config=cfg)
    if key == "podi-linear":
        cfg = dict(LEGACY_PODI_CONFIG)
        if "config" in kwargs and kwargs["config"] is not None:
            cfg.update(kwargs["config"])
        return LegacyPODIRegressor(method="linear", config=cfg)
    if key == "pod-gpr":
        cfg = dict(LEGACY_PODGPR_CONFIG)
        if "gpr_kernel" in kwargs and kwargs["gpr_kernel"] in {"rbf", "matern"}:
            cfg["kernel_type"] = kwargs["gpr_kernel"]
        elif "kernel" in kwargs and kwargs["kernel"] in {"rbf", "matern"}:
            cfg["kernel_type"] = kwargs["kernel"]
        if "config" in kwargs and kwargs["config"] is not None:
            cfg.update(kwargs["config"])
        return LegacyGPRRegressor(config=cfg)
    if key == "pod-nn":
        cfg = dict(LEGACY_PODNN_CONFIG)
        if "config" in kwargs and kwargs["config"] is not None:
            cfg.update(kwargs["config"])
        return LegacyTorchNNRegressor(n_basis=kwargs.get("n_basis"), config=cfg)
    if key == "pod-ae":
        cfg = dict(LEGACY_PODAE_CONFIG)
        if "config" in kwargs and kwargs["config"] is not None:
            cfg.update(kwargs["config"])
        return LegacyPODAERegressor(n_basis=kwargs.get("n_basis"), latent_dim=kwargs.get("latent_dim"), config=cfg)
    raise ValueError(f"Unknown non-intrusive method {method!r}.")


def evaluate_coefficient_regressor(
    *,
    method: str,
    field_name: str,
    regressor: CoefficientRegressor,
    parameters_train: np.ndarray,
    parameters_test: np.ndarray,
    coefficients_train: np.ndarray,
    coefficients_test: np.ndarray,
    pod: LoadedPOD,
    snapshots_train: np.ndarray,
    snapshots_test: np.ndarray,
) -> tuple[NonIntrusiveMetrics, np.ndarray, np.ndarray]:
    """Fit/evaluate a coefficient regressor using legacy-aligned model internals."""

    Ctr = _as_2d(coefficients_train)
    Cte = _as_2d(coefficients_test)
    regressor.fit(
        parameters_train,
        Ctr,
        pod=pod,
        snapshots_train=snapshots_train,
        snapshots_test=snapshots_test,
        field_name=field_name,
    )
    Ctr_pred = _as_2d(regressor.predict(parameters_train))
    Cte_pred = _as_2d(regressor.predict(parameters_test))

    pod_result = pod.as_pod_result()
    Xtr_pred = reconstruct_snapshots(Ctr_pred, pod_result)
    Xte_pred = reconstruct_snapshots(Cte_pred, pod_result)

    metrics = NonIntrusiveMetrics(
        field_name=field_name,
        method=method,
        rank=pod.rank,
        coefficient_relative_error_train=_relative_matrix_error(Ctr, Ctr_pred),
        coefficient_relative_error_test=_relative_matrix_error(Cte, Cte_pred),
        field_relative_error_train=relative_frobenius_error(snapshots_train, Xtr_pred),
        field_relative_error_test=relative_frobenius_error(snapshots_test, Xte_pred),
    )
    return metrics, Ctr_pred, Cte_pred


def write_nonintrusive_result(
    output_dir: str | Path,
    *,
    case_name: str,
    method: str,
    source_archive: str | Path,
    w_metrics: NonIntrusiveMetrics,
    w_predictions: tuple[np.ndarray, np.ndarray],
    w_model: CoefficientRegressor,
    theta_metrics: NonIntrusiveMetrics | None = None,
    theta_predictions: tuple[np.ndarray, np.ndarray] | None = None,
    theta_model: CoefficientRegressor | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a non-intrusive method result to disk."""

    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Use overwrite=True.")
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "suite": "nonintrusive",
        "alignment": "legacy_THERMOMECHANICAL_FOM_ROM",
        "method": method,
        "case_name": case_name,
        "source_archive": str(source_archive),
        "w": w_metrics.to_dict(),
        "theta": theta_metrics.to_dict() if theta_metrics is not None else None,
    }
    (out / "nonintrusive_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez_compressed(out / "w_predictions.npz", train_coefficients_pred=w_predictions[0], test_coefficients_pred=w_predictions[1])
    joblib.dump(w_model, out / "w_model.joblib")

    if theta_metrics is not None and theta_predictions is not None and theta_model is not None:
        np.savez_compressed(out / "theta_predictions.npz", train_coefficients_pred=theta_predictions[0], test_coefficients_pred=theta_predictions[1])
        joblib.dump(theta_model, out / "theta_model.joblib")

    return out / "nonintrusive_summary.json"
