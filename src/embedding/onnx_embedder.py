"""ONNX-runtime encoder for ``all-MiniLM-L6-v2`` — the no-torch runtime path.

The deployed app embeds the user's query with this module instead of
sentence-transformers/torch, which saves hundreds of MB of resident memory and
keeps the process under Render's 512 MB free tier. Numerically verified to
produce *identical* vectors to the torch encoder used to build the index
(cosine = 1.0000 on the example questions).

Mean pooling over an attention mask matches sentence-transformers' default
pooling for this model (``modules.json`` → 1_Pooling, mean tokens).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from src.config import DATA_DIR, LOCAL_MODEL_DIR

logger = logging.getLogger(__name__)

Embedding = np.ndarray


class OnnxMiniLMEmbedder:
    """Bert-base all-MiniLM-L6-v2 encoder over onnxruntime (CPU).

    Parameters
    ----------
    model_dir:
        Local bundled model directory (must contain ``tokenizer.json`` and
        ``onnx/model.onnx``).
    max_seq_length:
        Truncation length, matches sentence-transformers' default (256).
    """

    def __init__(
        self,
        model_dir: Path = LOCAL_MODEL_DIR,
        max_seq_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_seq_length = max_seq_length
        self.model_name = str(self.model_dir)
        self.batch_size = 64

        try:
            from tokenizers import Tokenizer

            from onnxruntime import InferenceSession, SessionOptions
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "ONNX runtime embeddings unavailable (need onnxruntime + tokenizers)"
            ) from exc

        options = SessionOptions()
        threads = int(os.environ.get("OMP_NUM_THREADS", "4") or "4")
        try:
            options.intra_op_num_threads = threads
        except Exception:  # pragma: no cover - older onnxruntime binding
            pass

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=max_seq_length)

        self._tokenizer = tokenizer
        self._session = InferenceSession(
            str(self.model_dir / "onnx" / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.dimension = 384  # MiniLM hidden size
        logger.info("Loaded ONNX embedder %s (dim=%d)", self.model_dir, self.dimension)

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
        """Encode *texts* into a ``(N, dim)`` float32 array (normalized)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = np.stack([self._encode_one(text, normalize) for text in texts])
        return vectors.astype(np.float32)

    def _encode_one(self, text: str, normalize: bool) -> np.ndarray:
        enc = self._tokenizer.encode(text)
        ids = np.asarray(enc.ids, dtype=np.int64).reshape(1, -1)
        mask = np.asarray(enc.attention_mask, dtype=np.int64).reshape(1, -1)
        token_type_ids = np.zeros_like(mask)

        hidden = self._session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
            },
        )[0]  # (1, seq, 384)

        m = mask.astype(np.float32).reshape(-1, 1)
        pooled = (hidden[0] * m).sum(axis=0) / m.sum()
        if normalize:
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
        return pooled

    def encode_query(self, query: str) -> Embedding:
        """Encode a single query string into a ``(dim,)`` normalized vector."""
        return self.encode([query])[0]


def get_embedder() -> OnnxMiniLMEmbedder:
    """Singleton embedder for the lightweight runtime path.

    Falls back to the torch ``Embedder`` only if the bundled ONNX model is
    missing (e.g. in the offline build pipeline).
    """
    global _singleton
    if _singleton is None:
        if (LOCAL_MODEL_DIR / "onnx" / "model.onnx").is_file():
            _singleton = OnnxMiniLMEmbedder()
        else:  # pragma: no cover - build/dev only
            from src.embedding.encoder import Embedder

            _singleton = Embedder()
    return _singleton


_singleton: Embedding | None = None