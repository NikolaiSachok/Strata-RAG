"""Embeddings — turning text into vectors that capture meaning.

WHY embeddings are the heart of RAG: a computer can't compare two passages by
"meaning" directly. An *embedding model* maps text into a high-dimensional vector
space where semantically similar texts land close together. "reset my password"
and "I forgot my login" produce nearby vectors even though they share no words.
That nearness is what makes similarity search — and therefore retrieval — possible.

CRITICAL RULE: you must embed your documents (at ingest time) and your queries (at
retrieve time) with the SAME model. Vectors from different models live in different,
incomparable spaces. That's why both ingest.py and retrieve.py go through this module.

Two backends, one interface:
  * local  — sentence-transformers, runs on your machine, NO API key, downloads a
             small model (~80MB) the first time. This is the default.
  * openai — OpenAI's embedding API, needs OPENAI_API_KEY. Useful if you want
             higher-quality embeddings or to avoid the local model download.
"""

from __future__ import annotations

from typing import Protocol

from .config import SETTINGS, Settings


class Embedder(Protocol):
    """Anything that turns a list of strings into a list of float vectors."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """sentence-transformers, fully local. The model is loaded lazily and cached on
    the instance so the (slow) load happens once per process, not per call."""

    def __init__(self, model_name: str):
        # Imported lazily so that importing this module (e.g. in a test that only
        # checks chunking) doesn't pay the heavy torch import cost.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # normalize_embeddings=True returns unit vectors, which makes cosine
        # similarity a simple dot product and keeps scores well-behaved.
        vectors = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()


class OpenAIEmbedder:
    """OpenAI embeddings via the official SDK. Needs OPENAI_API_KEY in the env."""

    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "RAGEVAL_EMBEDDINGS=openai requires the openai package. "
                'Install it with:  pip install -e ".[openai]"'
            ) from e
        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder(settings: Settings = SETTINGS) -> Embedder:
    """Factory: build the configured embedder. Call once and reuse — constructing a
    LocalEmbedder loads a model into memory."""
    if settings.embeddings == "openai":
        return OpenAIEmbedder(settings.embed_model)
    return LocalEmbedder(settings.embed_model)
