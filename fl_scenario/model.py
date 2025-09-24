""Modelo simples de regressão logística e utilitários de treino."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LogisticModel:
    """Modelo de regressão logística binária."""

    weights: np.ndarray

    @classmethod
    def initialize(cls, n_features: int) -> "LogisticModel":
        return cls(weights=np.zeros(n_features, dtype=np.float32))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights
        return 1.0 / (1.0 + np.exp(-logits))

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        proba = self.predict_proba(x)
        eps = 1e-8
        loss = -np.mean(y * np.log(proba + eps) + (1 - y) * np.log(1 - proba + eps))
        gradient = x.T @ (proba - y) / x.shape[0]
        return float(loss), gradient

    def fit_epoch(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1) -> float:
        loss, gradient = self.loss_and_grad(x, y)
        self.weights -= lr * gradient
        return loss

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        loss, _ = self.loss_and_grad(x, y)
        predictions = (self.predict_proba(x) >= 0.5).astype(np.float32)
        accuracy = float(np.mean(predictions == y))
        return loss, accuracy
