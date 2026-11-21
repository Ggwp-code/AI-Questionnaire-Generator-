"""Embedders and the shared term tokenizer."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import regex

from tribunal.errors import ConfigurationError
from tribunal.runtime_config import EmbeddingConfig

_WORD = regex.compile(r"\w+")


def terms(text: str) -> set[str]:
    out = set()
    for tok in _WORD.findall(text.lower()):
        if len(tok) >= 3:
            out.add(tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok)
    return out


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.model_id = f"hashing-{dimensions}"

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in terms(text):
                d = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
                m[row, d % self.dimensions] += 1.0 if (d >> 63) & 1 else -1.0
        return _normalize(m)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                ["retrieval.embedding.provider 'sentence_transformers' needs the embeddings-local extra"]
            ) from exc
        self.model_id = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts), dtype=np.float32))


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "hashing":
        return HashingEmbedder(cfg.dimensions)
    if cfg.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg.model)
    raise ConfigurationError(["retrieval.embedding.provider 'openai' is not available in this build"])
