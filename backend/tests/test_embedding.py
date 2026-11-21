import numpy as np
import pytest

from tribunal.errors import ConfigurationError
from tribunal.retrieval.embedding import HashingEmbedder, SentenceTransformerEmbedder, make_embedder, terms
from tribunal.runtime_config import EmbeddingConfig


def test_terms():
    assert terms("Decision trees and a tree, CAFÉ") == {"decision", "tree", "and", "café"}


def test_hashing_deterministic():
    a = HashingEmbedder(64).embed(["decision tree", "entropy"])
    b = HashingEmbedder(64).embed(["decision tree", "entropy"])
    assert a.shape == (2, 64) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hashing_similarity():
    v = HashingEmbedder(256).embed(["decision tree split entropy", "entropy of a decision tree", "steam engine factory"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_zero_text():
    assert not HashingEmbedder(8).embed([""])[0].any()


def test_model_id_and_factory():
    cfg = EmbeddingConfig(provider="hashing", model="x", dimensions=16)
    assert make_embedder(cfg).model_id == "hashing-16"


def test_openai_embedding_unavailable():
    with pytest.raises(ConfigurationError):
        make_embedder(EmbeddingConfig(provider="openai", model="x", dimensions=16))


def test_sentence_transformers_missing():
    with pytest.raises(ConfigurationError, match="embeddings-local"):
        SentenceTransformerEmbedder("m")
