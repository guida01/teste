"""Executa uma simulação de federated learning com assinaturas digitais."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import flwr as fl

from .client import AdversarialSignedClient, SignedClient
from .digital_signatures import RSAKeyPair, generate_rsa_key_pair
from .model import LogisticModel
from .server import SignatureAwareFedAvg


DEFAULT_NUM_CLIENTS = 2
DEFAULT_NUM_ROUNDS = 3
DEFAULT_SAMPLES_PER_CLIENT = 120
DEFAULT_RANDOM_SEED = 42
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.2


@dataclass
class ClientState:
    """Armazena os recursos necessários para inicializar um cliente."""

    key_pair: RSAKeyPair
    train_data: Tuple[np.ndarray, np.ndarray]
    test_data: Tuple[np.ndarray, np.ndarray]


def generate_client_data(
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Cria dados sintéticos com distribuição ligeiramente diferente por cliente."""

    rng = np.random.default_rng(seed)
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cid in range(num_clients):
        offset = (-1) ** cid * 1.5
        features = rng.normal(
            loc=offset, scale=1.0, size=(samples_per_client, 2)
        ).astype(np.float32)
        logits = (
            features[:, 0] * 0.8
            + features[:, 1] * (-0.6)
            + rng.normal(scale=0.5, size=samples_per_client)
        )
        labels = (logits > 0).astype(np.float32)
        datasets[str(cid)] = (features, labels)
    return datasets


def build_client_states(
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> Dict[str, ClientState]:
    datasets = generate_client_data(num_clients, samples_per_client, seed)
    key_pairs: Dict[str, RSAKeyPair] = {
        str(cid): generate_rsa_key_pair() for cid in range(num_clients)
    }
    states: Dict[str, ClientState] = {}
    for cid, (client_data, client_labels) in datasets.items():
        split = int(0.8 * len(client_data))
        train_x, test_x = client_data[:split], client_data[split:]
        train_y, test_y = client_labels[:split], client_labels[split:]
        states[cid] = ClientState(
            key_pair=key_pairs[cid],
            train_data=(train_x, train_y),
            test_data=(test_x, test_y),
        )
    return states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulação Flower com validação de assinaturas digitais."
    )
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument(
        "--samples-per-client", type=int, default=DEFAULT_SAMPLES_PER_CLIENT
    )
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--tamper-client",
        type=str,
        default=None,
        help="ID do cliente que adulterará os pesos (opcional).",
    )
    parser.add_argument(
        "--tamper-round",
        type=int,
        default=1,
        help="Round em que o cliente malicioso adulterará os pesos.",
    )
    parser.add_argument(
        "--tamper-magnitude",
        type=float,
        default=5.0,
        help="Magnitude adicionada aos pesos adulterados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    states = build_client_states(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
    )

    public_keys = {cid: state.key_pair.public_key for cid, state in states.items()}

    any_state = next(iter(states.values()))
    n_features = any_state.train_data[0].shape[1]
    initial_model = LogisticModel.initialize(n_features=n_features)
    initial_parameters = fl.common.ndarrays_to_parameters([initial_model.weights])

    def client_fn(cid: str) -> SignedClient:
        state = states[cid]
        base_kwargs = dict(
            cid=cid,
            key_pair=state.key_pair,
            train_data=state.train_data,
            test_data=state.test_data,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
        )
        if args.tamper_client is not None and cid == args.tamper_client:
            client: SignedClient = AdversarialSignedClient(
                tamper_round=args.tamper_round,
                tamper_magnitude=args.tamper_magnitude,
                **base_kwargs,
            )
        else:
            client = SignedClient(**base_kwargs)
        print(
            f"[Cliente {cid}] Iniciado com {len(state.train_data[0])} exemplos de treino e "
            f"{len(state.test_data[0])} exemplos de teste."
        )
        return client

    strategy = SignatureAwareFedAvg(
        client_public_keys=public_keys,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=initial_parameters,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print("\n[Métricas globais]")
    if history.metrics_distributed_fit:
        for rnd, metrics in history.metrics_distributed_fit:
            print(f"Rodada {rnd}: {metrics}")
    if history.metrics_distributed_eval:
        for rnd, metrics in history.metrics_distributed_eval:
            print(f"Avaliação distribuída rodada {rnd}: {metrics}")


if __name__ == "__main__":
    main()
