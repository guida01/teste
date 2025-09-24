""Cliiente Flower que assina atualizações do modelo."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import flwr as fl

from .digital_signatures import RSAKeyPair, sign_ndarrays
from .model import LogisticModel


class SignedClient(fl.client.NumPyClient):
    """Cliente Flower que envia uma assinatura digital com cada atualização."""

    def __init__(
        self,
        cid: str,
        key_pair: RSAKeyPair,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        local_epochs: int = 3,
        learning_rate: float = 0.1,
    ) -> None:
        self.cid = cid
        self.key_pair = key_pair
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data
        self.model = LogisticModel.initialize(self.train_x.shape[1])
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

    # Métodos obrigatórios pela API NumPyClient
    def get_parameters(self, config):  # type: ignore[override]
        return [self.model.weights.copy()]

    def fit(self, parameters, config):  # type: ignore[override]
        self.model.weights = parameters[0].copy()
        for _ in range(self.local_epochs):
            self.model.fit_epoch(self.train_x, self.train_y, lr=self.learning_rate)
        updated_parameters = [self.model.weights.copy()]
        signature = sign_ndarrays(updated_parameters, self.key_pair.private_key)
        metrics: Dict[str, str] = {
            "signature": signature.hex(),
            "cid": self.cid,
        }
        return updated_parameters, len(self.train_x), metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        self.model.weights = parameters[0].copy()
        loss, accuracy = self.model.evaluate(self.test_x, self.test_y)
        metrics = {"accuracy": accuracy}
        return loss, len(self.test_x), metrics
