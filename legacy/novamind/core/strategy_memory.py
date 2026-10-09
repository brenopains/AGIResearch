"""
StrategyMemory — Memoria de planos que funcionaram.

Quando o sistema resolve uma tarefa, o plano (sequencia de primitives)
e armazenado indexado por:
  - Assinatura do problema: (n_objetos_input, n_objetos_output,
    cores_input, cores_output, shape_input, shape_output, tipo_gap)
  - Contexto: dominio, complexidade

Quando um novo problema chega:
  1. Extrai assinatura do problema
  2. Busca planos similares na memoria
  3. Adapta o plano mais similar (ajusta parametros se necessario)
  4. Tenta executar o plano adaptado

A memoria NAO e um cache. Ela GENERALIZA: se dois problemas
tem assinaturas similares, planos similares devem funcionar.
"""
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import time
import json


@dataclass
class ProblemSignature:
    """Assinatura unica de um problema, usada para indexar planos."""
    n_input_objects: int = 0
    n_output_objects: int = 0
    input_colors: Tuple[int, ...] = ()
    output_colors: Tuple[int, ...] = ()
    input_shape: Tuple[int, int] = (0, 0)
    output_shape: Tuple[int, int] = (0, 0)
    gap_type: str = ""
    domain: str = ""

    @classmethod
    def from_grids(cls, inp: np.ndarray, out: np.ndarray,
                   gap_type: str = "", domain: str = "") -> 'ProblemSignature':
        """Extrai assinatura de grids input/output."""
        in_colors = tuple(sorted(int(c) for c in np.unique(inp) if c != 0))
        out_colors = tuple(sorted(int(c) for c in np.unique(out) if c != 0))
        return cls(
            n_input_objects=int((inp > 0).sum()),
            n_output_objects=int((out > 0).sum()),
            input_colors=in_colors,
            output_colors=out_colors,
            input_shape=inp.shape,
            output_shape=out.shape,
            gap_type=gap_type,
            domain=domain,
        )

    def similarity(self, other: 'ProblemSignature') -> float:
        """Similaridade entre duas assinaturas (0-1)."""
        score = 0.0
        # Objetos
        if self.n_input_objects == other.n_input_objects:
            score += 0.15
        if self.n_output_objects == other.n_output_objects:
            score += 0.15
        # Cores
        common_in = set(self.input_colors) & set(other.input_colors)
        common_out = set(self.output_colors) & set(other.output_colors)
        if self.input_colors:
            score += 0.15 * len(common_in) / max(1, len(self.input_colors))
        if self.output_colors:
            score += 0.15 * len(common_out) / max(1, len(self.output_colors))
        # Shape
        if self.input_shape == other.input_shape:
            score += 0.1
        if self.output_shape == other.output_shape:
            score += 0.1
        # Gap type
        if self.gap_type and self.gap_type == other.gap_type:
            score += 0.2
        return score

    def to_dict(self) -> dict:
        return {
            "n_in": self.n_input_objects,
            "n_out": self.n_output_objects,
            "colors_in": list(self.input_colors),
            "colors_out": list(self.output_colors),
            "shape_in": list(self.input_shape),
            "shape_out": list(self.output_shape),
            "gap_type": self.gap_type,
        }


@dataclass
class StoredPlan:
    """Um plano armazenado na memoria de estrategias."""
    signature: ProblemSignature
    primitive_sequence: List[str]        # ["recolor", "move", ...]
    params_sequence: List[Dict]          # [{"from_color":2}, {"dr":1}, ...]
    score: float = 0.0
    n_success: int = 1
    n_attempts: int = 1
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    task_id: str = ""

    @property
    def success_rate(self) -> float:
        return self.n_success / max(1, self.n_attempts)

    def to_dict(self) -> dict:
        return {
            "signature": self.signature.to_dict(),
            "plan": " -> ".join(self.primitive_sequence),
            "score": round(self.score, 3),
            "success_rate": round(self.success_rate, 3),
            "n_success": self.n_success,
        }


class StrategyMemory:
    """Memoria que armazena e recupera planos por similaridade de problema.

    Uso:
        memory = StrategyMemory()
        memory.store(signature, ["recolor", "move"], [{"from":2},{"dr":1}], score=0.9)
        best = memory.retrieve(new_signature)
        if best:
            plan = memory.adapt(best, new_signature)
            execute(plan)
    """

    def __init__(self, max_size: int = 1000):
        self.plans: List[StoredPlan] = []
        self.max_size = max_size

    def store(self, signature: ProblemSignature,
              primitive_sequence: List[str],
              params_sequence: List[Dict],
              score: float = 0.0,
              task_id: str = ""):
        """Armazena um plano que funcionou."""
        # Deduplica: mesmo signature + mesma sequencia?
        for existing in self.plans:
            if (existing.signature.similarity(signature) > 0.9 and
                    existing.primitive_sequence == primitive_sequence):
                existing.n_success += 1
                existing.n_attempts += 1
                existing.last_used = time.time()
                existing.score = max(existing.score, score)
                return

        plan = StoredPlan(
            signature=signature,
            primitive_sequence=primitive_sequence,
            params_sequence=params_sequence,
            score=score,
            task_id=task_id,
        )
        self.plans.append(plan)

        # Poda se exceder tamanho maximo
        if len(self.plans) > self.max_size:
            self.plans.sort(key=lambda p: p.score * p.success_rate)
            self.plans = self.plans[-self.max_size:]

    def retrieve(self, signature: ProblemSignature,
                 min_similarity: float = 0.4) -> Optional[StoredPlan]:
        """Recupera o plano mais similar para esta assinatura."""
        best = None
        best_sim = 0

        for plan in self.plans:
            sim = signature.similarity(plan.signature)
            if sim > best_sim and sim >= min_similarity:
                best_sim = sim
                best = plan

        return best

    def retrieve_top_k(self, signature: ProblemSignature,
                        k: int = 3) -> List[StoredPlan]:
        """Recupera os K planos mais similares."""
        scored = [(signature.similarity(p.signature), p) for p in self.plans]
        scored.sort(key=lambda x: -x[0])
        return [p for s, p in scored[:k] if s > 0.3]

    def adapt(self, plan: StoredPlan,
              target_sig: ProblemSignature) -> Tuple[List[str], List[Dict]]:
        """Adapta um plano para uma nova assinatura.

        Ajusta parametros (cores, posicoes) para o novo contexto.
        """
        seq = list(plan.primitive_sequence)
        params = [dict(p) for p in plan.params_sequence]

        # Ajusta parametros de cor
        for i, (op, param) in enumerate(zip(seq, params)):
            if op == "recolor":
                # Tenta mapear from_color para cor correspondente
                old_from = param.get("from_color")
                old_to = param.get("to_color")
                if old_from is not None and old_from in target_sig.input_colors:
                    # Cor de origem ainda existe no novo problema
                    pass
                elif target_sig.input_colors:
                    # Usa primeira cor disponivel
                    params[i]["from_color"] = target_sig.input_colors[0]

        return seq, params

    def get_stats(self) -> dict:
        if not self.plans:
            return {"total": 0}
        scores = [p.score for p in self.plans]
        return {
            "total": len(self.plans),
            "avg_score": round(np.mean(scores), 3),
            "max_score": round(max(scores), 3),
            "unique_gap_types": len(set(p.signature.gap_type for p in self.plans)),
            "total_attempts": sum(p.n_attempts for p in self.plans),
            "total_successes": sum(p.n_success for p in self.plans),
        }

    def get_plans_by_gap(self, gap_type: str) -> List[StoredPlan]:
        return [p for p in self.plans if p.signature.gap_type == gap_type]

    def clear(self):
        self.plans.clear()
