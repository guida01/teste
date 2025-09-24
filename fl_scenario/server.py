"""Estratégia Flower que valida assinaturas digitais."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
from cryptography.hazmat.primitives.asymmetric import rsa
from flwr.common import Metrics, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .digital_signatures import verify_signature


class SignatureAwareFedAvg(fl.server.strategy.FedAvg):
    """FedAvg que valida assinaturas antes de agregar."""

    def __init__(self, client_public_keys: Dict[str, rsa.RSAPublicKey], **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_public_keys = client_public_keys

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Metrics]]:
        verified_results: List[Tuple[ClientProxy, fl.common.FitRes]] = []
        missing_signature = 0
        unknown_key = 0
        invalid_signature = 0
        for client, fit_res in results:
            signature_hex = fit_res.metrics.get("signature") if fit_res.metrics else None
            if signature_hex is None:
                print(f"[Servidor] Assinatura ausente do cliente {client.cid}, ignorando atualização.")
                missing_signature += 1
                continue
            public_key = self.client_public_keys.get(client.cid)
            if public_key is None:
                print(f"[Servidor] Chave pública desconhecida para o cliente {client.cid}, ignorando.")
                unknown_key += 1
                continue
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            is_valid = verify_signature(ndarrays, bytes.fromhex(signature_hex), public_key)
            if not is_valid:
                print(f"[Servidor] Assinatura inválida recebida de {client.cid}, atualização descartada.")
                invalid_signature += 1
                continue
            verified_results.append((client, fit_res))
        if not verified_results:
            print("[Servidor] Nenhuma atualização válida foi recebida nesta rodada.")
            return None, {
                "num_verified_updates": 0,
                "num_missing_signatures": missing_signature,
                "num_unknown_public_keys": unknown_key,
                "num_invalid_signatures": invalid_signature,
            }
        print(f"[Servidor] {len(verified_results)} atualizações validadas com sucesso.")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, verified_results, failures
        )
        metrics = {
            "num_verified_updates": len(verified_results),
            "num_missing_signatures": missing_signature,
            "num_unknown_public_keys": unknown_key,
            "num_invalid_signatures": invalid_signature,
        }
        if aggregated_metrics:
            metrics.update(aggregated_metrics)
        return aggregated_parameters, metrics
