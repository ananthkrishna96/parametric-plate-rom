"""Coefficient-only POD--NN regression used across the three studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def require_torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("POD--NN requires the optional `torch` dependency.") from exc
    return torch, nn


@dataclass(frozen=True)
class PODNNConfig:
    hidden_widths: tuple[int, int, int] = (128, 96, 64)
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    weight_decay: float = 2.0e-6
    maximum_epochs: int = 2500
    patience: int = 150
    scheduler_patience: int = 35
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-7
    seed: int = 100
    device: str = "cpu"


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_validation_loss: float = float("inf")
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pod_nn(input_dimension: int, output_dimension: int, config: PODNNConfig | None = None):
    """Build the thesis POD--NN: 128/96/64 ELU and a linear output."""

    cfg = config or PODNNConfig()
    torch, nn = require_torch()
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(int(cfg.seed))
    layers: list[Any] = []
    previous = int(input_dimension)
    for width in cfg.hidden_widths:
        layers.extend([nn.Linear(previous, int(width)), nn.ELU()])
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dimension)))
    return nn.Sequential(*layers)


class PODNNRegressor:
    """Small sklearn-like wrapper around the executed POD--NN architecture."""

    def __init__(self, input_dimension: int, output_dimension: int, config: PODNNConfig | None = None):
        self.input_dimension = int(input_dimension)
        self.output_dimension = int(output_dimension)
        self.config = config or PODNNConfig()
        self.model = build_pod_nn(self.input_dimension, self.output_dimension, self.config)
        self.history = TrainingHistory()

    def fit(
        self,
        inputs_train: np.ndarray,
        coordinates_train: np.ndarray,
        inputs_validation: np.ndarray,
        coordinates_validation: np.ndarray,
    ) -> "PODNNRegressor":
        torch, nn = require_torch()
        Xtr = np.asarray(inputs_train, dtype=np.float32)
        Ytr = np.asarray(coordinates_train, dtype=np.float32)
        Xva = np.asarray(inputs_validation, dtype=np.float32)
        Yva = np.asarray(coordinates_validation, dtype=np.float32)
        if Xtr.ndim != 2 or Ytr.ndim != 2 or Xva.ndim != 2 or Yva.ndim != 2:
            raise ValueError("POD--NN arrays must be two-dimensional.")
        if len(Xtr) != len(Ytr) or len(Xva) != len(Yva):
            raise ValueError("POD--NN input and target row counts differ.")
        if Xtr.shape[1] != self.input_dimension or Ytr.shape[1] != self.output_dimension:
            raise ValueError("POD--NN dimensions do not match the configured network.")

        torch.manual_seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        device = torch.device(self.config.device)
        self.model.to(device)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
        generator = torch.Generator().manual_seed(int(self.config.seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(int(self.config.batch_size), len(dataset)),
            shuffle=True,
            generator=generator,
        )
        Xva_t = torch.from_numpy(Xva).to(device)
        Yva_t = torch.from_numpy(Yva).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.scheduler_factor),
            patience=int(self.config.scheduler_patience),
            min_lr=float(self.config.minimum_learning_rate),
        )
        criterion = nn.MSELoss()
        best_state = None
        stale = 0
        history = TrainingHistory()

        for epoch in range(int(self.config.maximum_epochs)):
            self.model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(xb)
                count += len(xb)
            train_loss = total / max(count, 1)

            self.model.eval()
            with torch.no_grad():
                validation_loss = float(criterion(self.model(Xva_t), Yva_t).detach().cpu())
            scheduler.step(validation_loss)
            history.train_loss.append(train_loss)
            history.validation_loss.append(validation_loss)

            if validation_loss < history.best_validation_loss - 1.0e-12:
                history.best_validation_loss = validation_loss
                history.best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(self.config.patience):
                history.stopped_early = True
                break

        if best_state is None:
            raise RuntimeError("POD--NN training did not produce a finite validation checkpoint.")
        self.model.load_state_dict(best_state)
        self.model.to(device)
        self.history = history
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        torch, _ = require_torch()
        X = np.asarray(inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.input_dimension:
            raise ValueError("POD--NN query shape is inconsistent with the fitted network.")
        device = torch.device(self.config.device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(X).to(device)).detach().cpu().numpy()
        return np.asarray(output, dtype=float)

    def save(self, path: str | Path) -> Path:
        torch, _ = require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dimension": self.input_dimension,
                "output_dimension": self.output_dimension,
                "config": asdict(self.config),
                "history": self.history.to_dict(),
                "state_dict": self.model.state_dict(),
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> "PODNNRegressor":
        torch, _ = require_torch()
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = PODNNConfig(**payload["config"])
        obj = cls(payload["input_dimension"], payload["output_dimension"], config)
        obj.model.load_state_dict(payload["state_dict"])
        obj.history = TrainingHistory(**payload["history"])
        return obj
