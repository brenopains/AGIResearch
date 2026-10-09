"""
BGE Encoder — Substitui spaCy 300D por BGE 768D + JL → 10000D

Diferencas:
  - spaCy 300D: embeddings de palavras, ruins para similaridade semantica
  - BGE 768D: embeddings de frases, SOTA em similaridade semantica

O encoder e CONGELADO — nao raciocina, nao gera respostas, so projeta.
"""
import torch
import numpy as np
from typing import List, Optional

DIM = 10000

class BGEEncoder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = "cuda"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print(f"[BGE] Carregando {model_name} em {self.device}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.input_dim = self.model.get_sentence_embedding_dimension()
        print(f"[BGE] {self.input_dim}D -> {DIM}D (JL projection)")

        # JL projection matrix: 768D -> 10000D
        seed = 42
        rng = np.random.RandomState(seed)
        self.jl = rng.randn(self.input_dim, DIM).astype(np.float32)
        self.jl /= np.sqrt(self.input_dim)  # scaling

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """BGE 768D -> JL -> 10000D float32."""
        embs = self.model.encode(texts, normalize_embeddings=normalize)
        embs = np.asarray(embs, dtype=np.float32)
        hdv = embs @ self.jl  # (batch, 10000)
        return hdv

    def encode_tensor(self, texts: List[str]) -> torch.Tensor:
        """Retorna tensor 10000D."""
        hdv = self.encode(texts)
        return torch.from_numpy(hdv)

    def encode_bipolar(self, texts: List[str]) -> torch.Tensor:
        """Retorna tensor bipolar ±1."""
        hdv = self.encode(texts)
        return torch.sign(torch.from_numpy(hdv))

    def project(self, vec_768d: torch.Tensor) -> torch.Tensor:
        """Projeta um vetor 768D para 10000D bipolar."""
        if vec_768d.dim() == 1:
            vec_768d = vec_768d.unsqueeze(0)
        jl_t = torch.from_numpy(self.jl).to(vec_768d.device, dtype=vec_768d.dtype)
        hdv = vec_768d @ jl_t
        return torch.sign(hdv).squeeze(0)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity entre duas strings."""
        emb_a = self.model.encode([a], normalize_embeddings=True)[0]
        emb_b = self.model.encode([b], normalize_embeddings=True)[0]
        return float(np.dot(emb_a, emb_b))


if __name__ == "__main__":
    enc = BGEEncoder()
    # Teste de similaridade semantica
    pairs = [
        ("dog", "cat"),
        ("dog", "fireplace"),
        ("gravity", "gravitational force"),
        ("gravity", "quantum physics"),
        ("river", "bank"),
        ("evolution", "natural selection"),
    ]
    print("\nSimilaridades BGE 768D:")
    for a, b in pairs:
        sim = enc.similarity(a, b)
        print(f"  {a:20s} vs {b:20s}: {sim:.4f}")

    # Teste de projecao
    print("\nProjecao 768D -> 10000D bipolar:")
    hdv = enc.encode_bipolar(["hello world"])
    print(f"  Shape: {hdv.shape}")
    print(f"  Values: {hdv[:10].tolist()}")
    print(f"  Mean: {hdv.mean().item():.4f} (deve ser ~0 para bipolar)")
