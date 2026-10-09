"""
EmergentContextBinder — Domínios do Conhecimento que Emergem Sozinhos
======================================================================
A AGI NÃO tem categorias fixas. Elas emergem da topologia do hiperespaço.

Mecanismo:
  1. Quando um novo conceito chega, mede similaridade com contextos conhecidos
  2. Se similaridade > limiar (0.45), associa ao contexto mais próximo
  3. Se similaridade < limiar, CRIA UM NOVO CONTEXTO (neurogênese de domínio)
  4. O conceito é salvo em DUAS cópias:
     a) Vinculada ao contexto (para foco atencional)
     b) Pura (para analogia livre / cruzamento de domínios)

O Maestro decide quando usar contexto:
  - MODO FOCO: query * context_vec → busca só naquele domínio
  - MODO GÊNIO: query pura → busca no hiperespaço inteiro (cruzamentos inéditos)

Referência: "Vou ser direto e limpar a bagunça a.txt" + sent.txt
"""

import os
import json
import time
import torch
from typing import Optional, Dict, List, Tuple

DIM = 10000


class EmergentContextBinder:
    """
    Categorização autônoma de domínios do conhecimento.

    A AGI começa com ZERO contextos conhecidos.
    Cada novo conceito é comparado com os existentes.
    Se não encaixar em nenhum, CRIA UM NOVO DOMÍNIO.
    """

    def __init__(self, engine, path: str = "D:/temp/contexts.json",
                 novelty_threshold: float = 0.15, device: str = "cuda"):
        self.engine = engine
        self.path = path
        self.novelty_threshold = novelty_threshold
        self.device = torch.device(device)

        self.contexts: Dict[str, torch.Tensor] = {}
        self.context_metadata: Dict[str, dict] = {}
        self._load()

        print(f"[CTX] Binder de Contextos Emergentes ativo")
        print(f"[CTX] Contextos conhecidos: {len(self.contexts)}")

    def find_or_create(self, concept_vec: torch.Tensor,
                       label: str = "") -> Tuple[str, torch.Tensor]:
        """
        Categoriza um conceito: acha o melhor contexto ou CRIA um novo.

        O vetor do contexto é o CENTRÓIDE dos conceitos atribuídos a ele
        (running average bipolar), NÃO um vetor aleatório fixo. Isso permite
        que contextos ajam como "ilhas semânticas" que se movem conforme
        novos conceitos são assimilados.
        """
        cv = concept_vec.flatten().to(self.device)

        best_name = None
        best_sim = 0.0

        for name, ctx_vec in self.contexts.items():
            sim = float(torch.cosine_similarity(
                cv.unsqueeze(0), ctx_vec.flatten().unsqueeze(0)))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name is None or best_sim < self.novelty_threshold:
            new_vec = cv.clone()
            new_name = f"ctx_{len(self.contexts)}"
            if label:
                clean = "".join(c for c in label if c.isalnum() or c == "_")[:20]
                new_name = f"ctx_{clean}_{len(self.contexts)}"
            self.contexts[new_name] = new_vec
            self.context_metadata[new_name] = {
                "created_at": time.time(),
                "seed_label": label,
                "centroid_n": 1,
                "birth_similarity": best_sim if best_name else 0.0,
                "n_concepts": 1,
            }
            self._save()
            return new_name, new_vec

        self.context_metadata[best_name]["n_concepts"] = \
            self.context_metadata[best_name].get("n_concepts", 0) + 1
        n = self.context_metadata[best_name].get("centroid_n", 1)
        merged = torch.sign(self.contexts[best_name] * n + cv)
        self.contexts[best_name] = merged
        self.context_metadata[best_name]["centroid_n"] = n + 1
        return best_name, self.contexts[best_name]

    def bind_concept(self, concept_vec: torch.Tensor,
                     label: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vincula um conceito ao seu contexto.

        Returns:
            (bound_vector, pure_vector)
            bound_vector: conceito * contexto (para foco)
            pure_vector: conceito puro (para analogia livre)
        """
        ctx_name, ctx_vec = self.find_or_create(concept_vec, label)
        bound = torch.sign(concept_vec.flatten().to(self.device)
                           * ctx_vec.flatten().to(self.device))
        pure = concept_vec.flatten().to(self.device)
        return bound, pure

    def apply_query_focus(self, query_vec: torch.Tensor,
                          context_name: str) -> torch.Tensor:
        """
        Modo FOCO: filtra a busca para um domínio específico.
        query * context_vec → só acha conceitos daquele domínio.
        """
        ctx = self.contexts.get(context_name)
        if ctx is None:
            return query_vec
        return torch.sign(query_vec.flatten().to(self.device)
                          * ctx.flatten().to(self.device))

    def get_contexts_summary(self) -> List[dict]:
        return [
            {"name": name, "n_concepts": meta.get("n_concepts", 0),
             "seed": meta.get("seed_label", ""),
             "created": meta.get("created_at", 0)}
            for name, meta in self.context_metadata.items()
        ]

    def _save(self):
        data = {
            "version": 2,
            "contexts": {
                name: {
                    "vector": vec.cpu().tolist(),
                    "metadata": self.context_metadata.get(name, {}),
                }
                for name, vec in self.contexts.items()
            }
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for name, cdata in data.get("contexts", {}).items():
                self.contexts[name] = torch.tensor(
                    cdata["vector"], device=self.device)
                self.context_metadata[name] = cdata.get("metadata", {})
        except Exception as e:
            print(f"[CTX] Erro carregando contextos: {e}")
