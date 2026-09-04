"""Sentence-transformer encoder with L2-normalized embeddings.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` (PRD §7) to produce 384-dim
dense vectors. The model is loaded once per process (singleton) and is never
fine-tuned.
"""

from __future__ import annotations

import logging

import numpy as np

from src.config import EMBEDDING_MODEL, LOCAL_MODEL_DIR

logger = logging.getLogger(__name__)

Embedding = np.ndarray


class Embedder:
    """Wrapper around a sentence-transformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to the PRD-constrained model.
    device:
        Optional device string (``"cpu"``, ``"mps"``, ``"cuda"``). If ``None``
        the library's default (best available) is used.
    batch_size:
        Encoding batch size (PRD §7 suggests 64).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        # Prefer the bundled, repo-local model copy so the app never needs to
        # download from HuggingFace (important on constrained hosts like Render).
        if model_name is None:
            model_name = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else EMBEDDING_MODEL
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device} if device else {}
        if LOCAL_MODEL_DIR.is_dir() and model_name == str(LOCAL_MODEL_DIR):
            # All files ship with the repo; never let transformers hit the hub.
            import os

            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._model = SentenceTransformer(model_name, **kwargs)

        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Loaded embedding model %s (dim=%d, device=%s)",
            model_name,
            self.dimension,
            self._model.device,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> Embedding:
        """Encode *texts* into a normalized ``(N, dim)`` float32 array.

        Parameters
        ----------
        texts:
            List of strings to embed.
        batch_size:
            Overrides the model default batch size if given.
        normalize:
            L2-normalize each vector so cosine similarity == dot product.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> Embedding:
        """Encode a single query string into a ``(1, dim)`` normalized vector.

        Uses the exact same model/normalization as chunks (guardrail: never mix).
        """
        vec = self.encode([query])
        return vec[0]
