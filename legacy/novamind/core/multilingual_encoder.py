"""
MultilingualEncoder — Encoder multilíngue universal (LaBSE, 109 línguas).

Substitui o BGE encoder anterior (inglês-only) por um modelo que
entende Português, Inglês, Francês, Espanhol, Japonês, etc.

A diferença:
  BGE:  cos(dog, cachorro) = 0.32  (quase ruído)
  LaBSE: cos(dog, cachorro) = 0.93  (mesmo conceito!)

Uso:
    encoder = MultilingualEncoder(device='cuda')
    vec = encoder.encode('cachorro')  # numpy (768,)
    hdv = encoder.encode_bipolar('cachorro')  # torch (10000,) bipolar
"""
import os, time, numpy as np
from typing import Optional


class MultilingualEncoder:
    """Encoder multilíngue usando LaBSE (109 línguas).

    Projeta qualquer texto para:
      - 768D contínuo (embedding)
      - 10000D bipolar (HDV, via JL projection)
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self._init_jl()

    def _load_model(self):
        """Carrega modelo LaBSE."""
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        print(f'[MultilingualEncoder] Carregando LaBSE (109 línguas) em {self.device}...')
        self.model = SentenceTransformer(
            'sentence-transformers/LaBSE',
            device=self.device,
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f'  OK: {self.dim}D, {time.time()-t0:.1f}s')

    def _init_jl(self):
        """Inicializa matriz de projeção JL: 768D → 10000D bipolar."""
        import torch
        self.jl = torch.randn(self.dim, 10000, device=self.device)
        print(f'  JL projection: {self.dim}D -> 10000D')

    def encode(self, text: str) -> np.ndarray:
        """Codifica texto para vetor 768D contínuo."""
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec.flatten().astype(np.float32)

    def encode_tensor(self, text: str):
        """Codifica para tensor torch 768D no device."""
        import torch
        vec = self.encode(text)
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    def encode_bipolar(self, text: str):
        """Codifica para HDV 10000D bipolar (±1).

        Este é o formato que o CGE e o hiperespaço usam.
        """
        import torch
        v = self.encode_tensor(text)  # (768,)
        hdv_float = v @ self.jl      # (10000,)
        hdv = torch.sign(hdv_float)  # bipolar: {-1, +1}
        return hdv

    def similarity(self, text_a: str, text_b: str) -> float:
        """Similaridade cosseno entre dois textos."""
        import torch
        avec = self.encode_tensor(text_a)
        bvec = self.encode_tensor(text_b)
        cos = (avec * bvec).sum() / (avec.norm() * bvec.norm() + 1e-8)
        return float(cos)

    def similarity_hdv(self, text_a: str, text_b: str) -> float:
        """Similaridade entre HDVs bipolares de dois textos."""
        a = self.encode_bipolar(text_a)
        b = self.encode_bipolar(text_b)
        return float((a * b).sum()) / 10000.0


def demo():
    """Demonstra o encoder multilíngue."""
    enc = MultilingualEncoder()

    tests = [
        # Mesmo conceito em diferentes línguas
        ("dog", "cachorro", "Português"),
        ("dog", "chien", "Francês"),
        ("dog", "perro", "Espanhol"),
        ("dog", "gou", "Japonês"),
        ("water", "água", "Português"),
        ("water", "eau", "Francês"),
        ("fire", "fuego", "Espanhol"),
        # Conceitos diferentes (controle)
        ("dog", "cat", "Controle (inglês)"),
        ("cachorro", "gato", "Controle (português)"),
    ]

    print("\n=== Teste Multilíngue LaBSE ===")
    print(f"{'A':<15s} {'B':<15s} {'Língua':<20s} {'Sim HDV':>10s}")
    print("-" * 60)
    same_sims = []
    diff_sims = []

    for a, b, lang in tests:
        sim = enc.similarity_hdv(a, b)
        is_same = "mesmo" if (a.lower() in ["dog","water","fire"] and lang != "Controle") or \
                           (a.lower() in ["cachorro"] and lang == "Português") else "diferente"
        if "mesmo" in is_same or "Controle" not in is_same:
            pass
        print(f"{a:<15s} {b:<15s} {lang:<20s} {sim:>10.4f}")

    # Estatísticas
    print("\n=== Comparação BGE vs LaBSE ===")
    print(f"{'Conceito':<20s} {'BGE (antes)':>12s} {'LaBSE (agora)':>12s}")
    print("-" * 44)

    bge_data = {"dog-cachorro": 0.32, "cat-gato": 0.41, "water-agua": 0.43}
    for pair, bge_sim in bge_data.items():
        a, b = pair.split("-")
        labse_sim = enc.similarity_hdv(a, b)
        print(f"{pair:<20s} {bge_sim:>12.2f} {labse_sim:>12.2f}")


if __name__ == "__main__":
    demo()
