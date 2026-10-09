"""
HDCDecoder — Decodificador de invenções HDC para texto
======================================================
Traduz bind(A, B, C) → ranking de conceitos similares no vocabulário.

O decoder:
  1. Mantém uma matriz de todos os conceitos do CGE como HDVs 10000D
  2. Dado um vetor invenção, computa similaridade com TODO o vocabulário
  3. Retorna top-K conceitos relacionados (excluindo os componentes originais)
  4. Usa estes como "descrição textual" da invenção

Isto NÃO é um decoder generativo neural — é um nearest-neighbor no espaço HDV
sobre um vocabulário completo. A "geração" é a seleção e ranking dos vizinhos
mais próximos, que tendem a ser conceitos relacionados aos componentes.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


class HDCDecoder:
    """
    Decodificador HDC: invenção HDV → ranking de conceitos.

    Uso:
        decoder = HDCDecoder(bge, cge, device='cuda')
        decoder.build_vocabulary()  # indexa todo CGE + vocabulário extra
        invention = decoder.create_invention(['wheel', 'lever', 'gear'])
        top_k = decoder.decode(invention, exclude=['wheel', 'lever', 'gear'])
        # top_k = [('crank', 0.45), ('mechanism', 0.42), ('compound', 0.39), ...]
    """

    def __init__(self, bge, cge, device=None):
        self.bge = bge
        self.cge = cge
        self.device = device or cge.device

        # Vocabulário indexado
        self.vocab_names: List[str] = []
        self.vocab_vectors: Optional[torch.Tensor] = None  # (N, 10000) bipolar

    def build_vocabulary(self,
                         extra_words: Optional[List[str]] = None,
                         max_vocab: int = 5000):
        """
        Constrói vocabulário a partir de:
          1. Nós do CGE (já embebidos)
          2. BGE encode de palavras extras
          3. Até max_vocab entradas
        """
        names_set = set()

        # 1. CGE nodes
        for node_id, node_data in self.cge.nodes.items():
            if isinstance(node_id, str) and len(node_id) > 1 and len(node_id) < 40:
                if not node_id.startswith('not_'):
                    names_set.add(node_id)

        # 2. Extra words
        if extra_words:
            for w in extra_words:
                if isinstance(w, str) and len(w) > 1:
                    names_set.add(w)

        # 3. Limit
        names = list(names_set)[:max_vocab]

        if not names:
            self.vocab_names = []
            self.vocab_vectors = None
            print("[DECODER] Warning: empty vocabulary")
            return

        # Build vectors — prefer CGE cached vectors, fallback to BGE
        vectors = []
        valid_names = []
        for name in names:
            vec = None
            # Try CGE cache
            if name in self.cge.nodes and 'vector' in self.cge.nodes[name]:
                vec = self.cge.nodes[name]['vector'].flatten().to(self.device)
            # Try BGE
            if vec is None and self.bge:
                try:
                    vec = self.bge.encode_bipolar([name])[0].to(self.device)
                except:
                    pass
            if vec is not None:
                vectors.append(vec)
                valid_names.append(name)

        if vectors:
            self.vocab_vectors = torch.stack(vectors)
            self.vocab_names = valid_names
            print(f"[DECODER] Vocabulary: {len(self.vocab_names)} concepts")
        else:
            self.vocab_names = []
            self.vocab_vectors = None
            print("[DECODER] Warning: no vectors could be built")

    def create_invention(self, components: List[str]) -> Optional[torch.Tensor]:
        """Cria invenção via bind(A, B, C, ...) usando BGE."""
        if not self.bge:
            return None
        vec = None
        for comp in components:
            try:
                v = self.bge.encode_bipolar([comp])[0].to(self.device)
                if vec is None:
                    vec = v
                else:
                    vec = torch.sign(vec * v)
            except:
                return None
        return vec

    def decode(self, invention_vec: torch.Tensor,
               exclude: Optional[List[str]] = None,
               top_k: int = 20,
               min_sim: float = 0.0) -> List[Tuple[str, float]]:
        """
        Decodifica invenção: encontra conceitos mais similares no vocabulário.
        Exclui componentes originais (se fornecidos).
        """
        if self.vocab_vectors is None or len(self.vocab_names) == 0:
            return []

        exclude_set = set()
        if exclude:
            exclude_set = set(e.lower() for e in exclude)

        # Compute similarities: invention (10000D) x vocabulary (N, 10000D)
        inv = invention_vec.flatten().unsqueeze(0)  # (1, 10000)
        sims = torch.cosine_similarity(inv, self.vocab_vectors)  # (N,)

        # Get top-k
        top_sims, top_indices = sims.topk(min(top_k * 3, len(self.vocab_names)))

        # Build results (excluding components)
        results = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            name = self.vocab_names[idx]
            if name.lower() in exclude_set:
                continue
            if sim < min_sim:
                continue
            results.append((name, sim))
            if len(results) >= top_k:
                break

        return results

    def component_recovery(self, invention_vec: torch.Tensor,
                           components: List[str],
                           top_n_candidates: int = 20) -> Dict:
        """
        Teste de recuperação de componentes.
        Retorna quantos dos componentes estão no top-N (excluindo os
        próprios componentes da lista, que naturalmente são candidatos).
        """
        decoded = self.decode(invention_vec, exclude=components, top_k=top_n_candidates)
        decoded_names = [d[0].lower() for d in decoded]

        # How many components appear in decoded?
        found = 0
        for comp in components:
            if comp.lower() in decoded_names:
                found += 1
            # Check if component appears as substring of any decoded name
            elif any(comp.lower() in dn for dn in decoded_names):
                found += 1

        # Also check: do the decoded concepts have BGE similarity to components?
        # (semantic relation, not exact match)
        semantic_scores = []
        for comp in components:
            comp_vec = self.bge.encode_bipolar([comp])[0].to(self.device)
            for dname, dsim in decoded[:10]:
                if dname != comp:
                    dvec = self.bge.encode_bipolar([dname])[0].to(self.device)
                    rel_sim = torch.cosine_similarity(
                        comp_vec.unsqueeze(0), dvec.unsqueeze(0)
                    ).item()
                    semantic_scores.append((comp, dname, rel_sim))

        return {
            "n_components_found": found,
            "n_components": len(components),
            "recovery_rate": found / len(components) if components else 0,
            "top_decoded": decoded[:10],
            "semantic_relations": sorted(semantic_scores, key=lambda x: -x[2])[:10],
        }
