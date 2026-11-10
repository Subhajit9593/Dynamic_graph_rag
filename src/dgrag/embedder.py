"""Sentence-transformer embedder (bge-base-en-v1.5) with OOM-safe batching.

* fp16 on GPU: half the memory, ~2x faster, negligible quality change for retrieval.
* Vectors are L2-normalised, so inner product == cosine similarity (matches FAISS IndexFlatIP).
* bge v1.5 convention: passages are embedded as-is; *queries* get an instruction prefix.
* If CUDA runs out of memory, the batch size is halved and the batch retried (never crashes a long run).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64, max_seq_length: int = 512,
                 fp16: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available; embedding on CPU")
            device = "cpu"
        try:  # use the cached copy without network calls; fall back to downloading on first use
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if fp16 and device == "cuda":
            self.model.half()
        self.device = device
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode(self, texts: Sequence[str], is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        bs = self.batch_size
        while True:
            try:
                vecs = self.model.encode(list(texts), batch_size=bs, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=show_progress)
                return vecs.astype(np.float32, copy=False)
            except self.torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.torch.cuda.empty_cache()
                log.warning("CUDA OOM while embedding; retrying with batch_size=%d", bs)

    def close(self) -> None:
        """Free GPU memory (call before starting the LLM server)."""
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
