"""
VisualMemory — Memoria visuo-espacial que alimenta o ciclo cognitivo.

Nao contem transformacoes hardcoded. Apenas:
  1. DINOv2 processa imagem -> regioes 3x3 com posicao
  2. Cada regiao vira HDV (bind com vetor de posicao)
  3. Memoria armazena experiencias visuais
  4. Comparacao por similaridade entre HDVs de regioes
  5. Resultados alimentam o CognitiveCycle -> CGE -> Forja

A Forja descobre padroes visuais naturalmente, sem operadores pre-definidos.
"""
import numpy as np
import time
from typing import Dict, List, Optional

DIM = 10000

# ─── 9 regioes espaciais (3x3) ──────────────────────────────
REGIONS_3X3 = [
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]


class VisualMemory:
    """Memoria de experiencias visuais com HDVs por regiao.

    Cada entrada: { "id", "timestamp", "regions": {regiao: hdv}, "metadata": {} }
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.entries: list = []  # mais recente primeiro

    def store(self, regions: Dict[str, np.ndarray], metadata: dict = None) -> str:
        entry_id = f"vis_{int(time.time()*1000)}_{len(self.entries)}"
        self.entries.insert(0, {
            "id": entry_id, "timestamp": time.time(),
            "regions": {k: v.copy() for k, v in regions.items()},
            "metadata": metadata or {},
        })
        if len(self.entries) > self.capacity:
            self.entries = self.entries[:self.capacity]
        return entry_id

    def get_recent(self, n: int = 10) -> list:
        return self.entries[:n]

    def search(self, query_regions: Dict, top_k: int = 5) -> List[Dict]:
        """Busca experiencias visuais similares por regiao.

        100% HDV, 0% regras. A Forja descobre os padroes.
        """
        scored = []
        for entry in self.entries:
            similarities = []
            for region in REGIONS_3X3:
                if region in query_regions and region in entry["regions"]:
                    cos = float(np.dot(query_regions[region], entry["regions"][region]) / DIM)
                    similarities.append(cos)
            overall = float(np.mean(similarities)) if similarities else 0.0

            # Deltas entre regioes (para deteccao de mudanca)
            deltas = {}
            for region in REGIONS_3X3:
                if region in query_regions and region in entry["regions"]:
                    deltas[region] = 1.0 - float(
                        np.dot(query_regions[region], entry["regions"][region]) / DIM
                    )

            scored.append({
                "similarity": overall,
                "entry": entry,
                "deltas": deltas,
                "n_changed": sum(1 for d in deltas.values() if d > 0.5),
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def __len__(self):
        return len(self.entries)
