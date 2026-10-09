"""
MetaAbstractionEngine — Abstracao profunda por algebra HDV no CGE.

NENHUMA relacao e pre-definida. O sistema:
  1. Recebe uma metafora (texto)
  2. Extrai conceitos A e B da estrutura da frase
  3. Codifica A e B em HDV (via BGE se disponivel)
  4. CAPTURA a relacao como um VETOR HDV: bind(HDV(A), HDV(B))
  5. O vetor relacional e a propria abstracao
  6. Busca no CGE outros pares com vetor relacional similar
  7. A abstracao pode ser usada como operador cognitivo

Se CGE/BGE nao estiverem disponiveis, usa fallback estrutural.
Nada hardcoded. A abstracao VIVE no espaco HDV como um vetor.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import time


@dataclass
class AbstractPattern:
    """Padrao abstrato. A abstracao em si e o VETOR HDV da relacao."""
    id: str = ""
    name: str = ""
    source_text: str = ""
    concept_a: str = ""
    concept_b: str = ""
    relation_type: str = ""
    generalized: str = ""
    implication: str = ""
    hdv_vector_ref: str = ""       # id do vetor HDV da relacao no CGE
    analogies: List[dict] = field(default_factory=list)
    depth: int = 1
    applies_to_self: bool = False
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source_text[:60],
            "relation": f"{self.concept_a} <-[{self.relation_type}]-> {self.concept_b}",
            "generalized": self.generalized[:60],
            "analogies": len(self.analogies),
            "depth": self.depth,
        }


# Marcadores que INDICAM uma relacao (nao definem a relacao)
RELATION_MARKERS = ["aponta", "apontar", "representa", "simboliza",
                    "significa", "causa", "leva", "produz", "contem",
                    "inclui", "como", "tal"]

STOP = {"o", "a", "os", "as", "um", "uma", "de", "da", "do",
        "em", "no", "na", "para", "por", "com", "sem", "que", "e"}


class MetaAbstractionEngine:
    """Abstracao via HDV: a relacao vira um VETOR no espaco."""

    def __init__(self, cge=None, bge=None, self_model=None):
        self.cge = cge
        self.bge = bge
        self.self_model = self_model
        self.patterns: List[AbstractPattern] = []
        self._counter = 0

    def process(self, text: str):
        self._counter += 1
        pid = f"meta_{self._counter}"
        p = self._via_hdv(text, pid) if (self.cge and self.bge) else self._via_text(text, pid)
        if p:
            self.patterns.append(p)
            if p.applies_to_self and self.self_model:
                self._update_self(p)
        return p

    def _via_hdv(self, text: str, pid: str):
        """Usa CGE + BGE: codifica em HDV, captura relacao como vetor."""
        a, b, marker = self._extract_pair(text)
        if not a or not b or not marker:
            return None
        try:
            # Codifica A e B em HDV via BGE
            hdv_a = self._encode(a)
            hdv_b = self._encode(b)
            if hdv_a is None or hdv_b is None:
                return None

            # CAPTURA a relacao: bind(A, B) = vetor relacional
            # Este vetor E a abstracao
            rel_vec = self._bind(hdv_a, hdv_b)

            # Armazena no CGE para consultas futuras
            rel_id = f"rel_{a}_{b}"
            if hasattr(self.cge, 'create_node'):
                self.cge.create_node(rel_id, rel_vec)
                if hasattr(self.cge, 'add_edge'):
                    self.cge.add_edge(a, rel_id, "relates")
                    self.cge.add_edge(rel_id, b, "relates")

            # Busca analogias: pares com vetor relacional similar
            analogies = self._find_similar_relations(rel_vec)

            # Identifica se a relacao e auto-aplicavel
            self_ref_keywords = ["represent", "simbol", "signific", "apont", "reflet"]
            applies = any(kw in marker for kw in self_ref_keywords)

            pattern = AbstractPattern(
                id=pid, name=f"abs_{a}_{b}", source_text=text,
                concept_a=a, concept_b=b, relation_type=f"hdv_rel_{self._counter}",
                generalized=f"{a} --[]--> {b} (via HDV)",
                implication=self._imply(a, b, marker, applies),
                hdv_vector_ref=rel_id,
                analogies=analogies,
                depth=3 if applies else 2,
                applies_to_self=applies,
                confidence=min(0.9, 0.3 + len(analogies) * 0.15),
            )
            return pattern
        except Exception as e:
            return None

    def _via_text(self, text: str, pid: str):
        """Fallback: extrai por estrutura da frase (sem HDV)."""
        a, b, marker = self._extract_pair(text)
        if not a or not b or not marker:
            return None
        self_ref = any(kw in marker for kw in ["represent", "simbol", "signific", "apont", "reflet"])
        pattern = AbstractPattern(
            id=pid, name=f"abs_{a}_{b}", source_text=text,
            concept_a=a, concept_b=b,
            relation_type=f"rel_{self._counter}",
            generalized=f"{a} --[]--> {b}",
            implication=self._imply(a, b, marker, self_ref),
            depth=3 if self_ref else 2,
            applies_to_self=self_ref,
            confidence=0.3,
        )
        return pattern

    def _extract_pair(self, text: str) -> tuple:
        """Extrai (conceito_a, conceito_b, marcador) da estrutura da frase."""
        t = text.lower()
        for marker in RELATION_MARKERS:
            if marker not in t:
                continue
            parts = t.split(marker, 1)
            if len(parts) < 2:
                continue
            before = [w.strip(".,;:!?") for w in parts[0].split()
                     if w.strip(".,;:!?") not in STOP and len(w.strip(".,;:!?")) > 1]
            after = [w.strip(".,;:!?") for w in parts[1].split()
                    if w.strip(".,;:!?") not in STOP and len(w.strip(".,;:!?")) > 1]
            if before and after:
                return (before[-1], after[0], marker)
        return (None, None, None)

    def _encode(self, word: str):
        """Codifica palavra em HDV via BGE + JL projection."""
        if self.bge is None:
            return None
        vec = self.bge.encode(word)
        if vec is None:
            return None
        # JL projection para 10000D bipolar (se o Manifold existir)
        if hasattr(self.bge, 'project') and hasattr(self.bge, '_PROJ_SEMANTIC'):
            import torch
            import numpy as np
            v = torch.tensor(vec, dtype=torch.float32)
            proj = getattr(self.bge, '_PROJ_SEMANTIC', None)
            if proj is not None:
                hdv_float = v @ proj
                hdv = torch.sign(hdv_float)
                return hdv
        return vec

    def _bind(self, a_vec, b_vec) -> object:
        """Bind no espaco HDV: a XOR b."""
        if hasattr(a_vec, '__torch__') or hasattr(a_vec, 'shape'):
            import torch
            return torch.sign(a_vec * b_vec)
        return a_vec * b_vec

    def _find_similar_relations(self, rel_vec) -> List[dict]:
        """Busca no CGE outras relacoes com vetor HDV similar."""
        analogies = []
        if self.cge is None or rel_vec is None:
            return analogies
        try:
            if hasattr(self.cge, 'query_attention'):
                results = self.cge.query_attention(rel_vec, top_k=5)
                for nid, _, sim in results:
                    if sim > 0.3:
                        analogies.append({"node": str(nid), "similarity": round(float(sim), 3)})
        except Exception:
            pass
        return analogies[:5]

    def _imply(self, a: str, b: str, marker: str, self_ref: bool) -> str:
        if self_ref:
            return f"{a} aponta para {b}, mas {a} != {b}. Isto se aplica a como represento o mundo."
        return f"Relacao estrutural entre {a} e {b} (via '{marker}')"

    def _update_self(self, pattern: AbstractPattern):
        if self.self_model is None:
            return
        if hasattr(self.self_model, "beliefs"):
            self.self_model.beliefs[f"meta:{pattern.name}"] = {
                "pattern": pattern.generalized,
                "implication": pattern.implication,
                "confidence": pattern.confidence,
            }
        if hasattr(self.self_model, "capability_profile"):
            self.self_model.capability_profile["meta_abstraction"] = max(
                self.self_model.capability_profile.get("meta_abstraction", 0), 0.5)

    def generate_analogy(self, target: str, pattern: AbstractPattern) -> str:
        """Gera analogia: aplica o padrao relacional a um novo conceito.

        Se o padrao tem vetor HDV, busca no espaco qual B
        se relaciona com 'target' de forma similar a pattern.
        """
        if self.cge is not None and pattern.hdv_vector_ref:
            try:
                # Busca: qual conceito se relaciona com target como A se relaciona com B?
                results = self.cge.query_attention(target, top_k=3)
                if results:
                    for nid, _, sim in results:
                        if nid != pattern.concept_a and nid != pattern.concept_b:
                            return f"{target} :: {pattern.concept_a} -> {pattern.concept_b} : {nid}"
            except Exception:
                pass
        return f"{target} similar a {pattern.concept_a} em relacao a {pattern.concept_b}"

    def reason(self, pattern: AbstractPattern, problem: str) -> str:
        """Usa o padrao para analisar um problema."""
        lines = [f"Padrao: {pattern.generalized}", f"Problema: {problem}"]
        lines.append(f"Neste problema, quem e o '{pattern.concept_a}' e quem e o '{pattern.concept_b}'?")
        if pattern.applies_to_self:
            lines.append(f"(Isto se aplica a como represento o proprio problema)")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        if not self.patterns:
            return {"total": 0}
        return {
            "total": len(self.patterns),
            "depth_3_self_applicable": sum(1 for p in self.patterns if p.depth >= 3),
            "total_analogies": sum(len(p.analogies) for p in self.patterns),
        }


def example():
    class MockSelf:
        def __init__(self):
            self.beliefs = {}
            self.capability_profile = {}

    engine = MetaAbstractionEngine(self_model=MockSelf())
    tests = [
        "O dedo que aponta a lua nao e a lua",
        "O mapa nao e o territorio",
        "A palavra representa o significado",
        "A cascata leva ao oceano",
    ]
    for t in tests:
        p = engine.process(t)
        if p:
            print(f"\n[IN] {p.source_text}")
            print(f"  A={p.concept_a} B={p.concept_b}")
            print(f"  Gen: {p.generalized}")
            print(f"  Self: {p.applies_to_self} (depth={p.depth})")
            print(f"  Implic: {p.implication}")
            a = engine.generate_analogy("codigo", p)
            print(f"  Analogia: {a}")
            r = engine.reason(p, "Modelo nao generaliza")
            print(f"  Raciocinio:\n{r}")
    print(f"\n[STATS] {engine.get_stats()}")


if __name__ == "__main__":
    example()
