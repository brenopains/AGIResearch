"""
NovaMind — Semantic Encoder (real sentence embeddings)
======================================================

Replaces string-overlap heuristics (Jaccard / NCD) with genuine semantic
similarity from a small sentence-embedding model (all-MiniLM-L6-v2, 384-dim,
~80MB). Runs locally on CPU or the RTX 3050 (uses <120MB VRAM in fp32).

Why this matters: Jaccard/NCD only match shared *characters/words*. Two
sentences that mean the same thing but share no words ("What is my profession"
vs "I work as a software engineer") score ~0. MiniLM was trained on >1B
sentence pairs to place paraphrases close together in vector space, so cosine
similarity captures MEANING, not surface form.

No external API — the model weights are downloaded once to the HF cache and
all inference is local.
"""
from typing import List, Optional
import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticEncoder:
    """Lazy-loaded MiniLM encoder with an in-memory embedding cache."""

    _shared: Optional["SemanticEncoder"] = None

    def __init__(self, device: Optional[str] = None):
        self._tokenizer = None
        self._model = None
        self._device = device
        self._cache: dict = {}            # text -> np.ndarray (normalized)
        self._dim = 384

    @classmethod
    def shared(cls) -> "SemanticEncoder":
        """Process-wide singleton so we load the model only once."""
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModel.from_pretrained(_MODEL_NAME).to(self._device)
        self._model.eval()
        self._dim = self._model.config.hidden_size

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        self._ensure_loaded()
        import torch

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            key = t.strip().lower()
            cached = self._cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                todo.append(t)
                todo_idx.append(i)

        if todo:
            with torch.no_grad():
                enc = self._tokenizer(
                    todo, padding=True, truncation=True, max_length=64,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state
                pooled = self._mean_pool(hidden, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vecs = pooled.cpu().numpy().astype(np.float32)
            for j, i in enumerate(todo_idx):
                out[i] = vecs[j]
                self._cache[todo[j].strip().lower()] = vecs[j]

        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for already-normalized vectors == dot product."""
        return float(np.dot(a, b))


def attach_encoder(memory_system, device: Optional[str] = None):
    """Attach the shared encoder to a MemorySystem instance.

    After this, MemorySystem.recall_episodic / recall_semantic use cosine
    similarity over MiniLM embeddings instead of Jaccard/NCD.
    """
    memory_system.encoder = SemanticEncoder.shared() if device is None else SemanticEncoder(device)
    return memory_system.encoder
