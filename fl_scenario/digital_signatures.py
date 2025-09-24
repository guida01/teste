""Utilitários para geração e validação de assinaturas digitais."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


RSA_KEY_SIZE = 2048
PUBLIC_EXPONENT = 65537


@dataclass
class RSAKeyPair:
    """Armazena um par de chaves RSA para um cliente federado."""

    private_key: rsa.RSAPrivateKey
    public_key: rsa.RSAPublicKey

    def private_bytes(self) -> bytes:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


def generate_rsa_key_pair() -> RSAKeyPair:
    """Gera um novo par de chaves RSA."""

    private_key = rsa.generate_private_key(
        public_exponent=PUBLIC_EXPONENT,
        key_size=RSA_KEY_SIZE,
    )
    public_key = private_key.public_key()
    return RSAKeyPair(private_key=private_key, public_key=public_key)


def serialize_ndarrays(parameters: Iterable[np.ndarray]) -> bytes:
    """Serializa um conjunto de `ndarrays` de forma determinística."""

    arrays: List[np.ndarray] = [np.asarray(arr) for arr in parameters]
    return pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL)


def sign_ndarrays(parameters: Iterable[np.ndarray], private_key: rsa.RSAPrivateKey) -> bytes:
    """Assina uma sequência de arrays NumPy usando RSA."""

    payload = serialize_ndarrays(parameters)
    signature = private_key.sign(
        payload,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return signature


def verify_signature(
    parameters: Iterable[np.ndarray],
    signature: bytes,
    public_key: rsa.RSAPublicKey,
) -> bool:
    """Valida uma assinatura digital para um conjunto de arrays."""

    payload = serialize_ndarrays(parameters)
    try:
        public_key.verify(
            signature,
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


def load_public_key(pem_bytes: bytes) -> rsa.RSAPublicKey:
    """Carrega uma chave pública em formato PEM."""

    return serialization.load_pem_public_key(pem_bytes)


def load_private_key(pem_bytes: bytes) -> rsa.RSAPrivateKey:
    """Carrega uma chave privada em formato PEM."""

    return serialization.load_pem_private_key(pem_bytes, password=None)
