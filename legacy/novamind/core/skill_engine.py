"""
SkillEngine — Habilidade abstrata definida por INTENCAO, nao por sequencia fixa.

Diferenca entre Macro e Skill:

  Macro: rotate -> move (sequencia FIXA)
  Skill: normalize_object_orientation (INTENCAO)
         implementavel por: rotate->move OU flip->move OU transpose->move

A Skill descobre QUAL sequencia usar baseado no contexto atual.
"""
from typing import List, Optional, Dict, Tuple, Set, Callable
from dataclasses import dataclass, field
import numpy as np
import time
from collections import defaultdict


# ─── Skill — intencao abstrata ─────────────────────────────

@dataclass
class Skill:
    """Habilidade abstrata: O QUE fazer, nao COMO fazer.

    Attributes:
        name: "normalize_orientation"
        intent: descricao do objetivo ("alinhar objetos a referencia")
        implementations: sequencias que atingem o intento
                          [("rotate", "move"), ("flip", "move"), ...]
        confidence: quao confiavel e a skill
        frequency: quantas vezes foi usada
    """
    name: str
    intent: str
    description: str = ""
    implementations: List[Tuple[str, ...]] = field(default_factory=list)
    confidence: float = 0.0
    frequency: int = 0
    n_success: int = 0
    n_attempts: int = 0
    domains: Set[str] = field(default_factory=set)
    discovered_at: float = field(default_factory=time.time)

    @property
    def best_implementation(self) -> Optional[Tuple[str, ...]]:
        """Melhor implementacao (mais curta, maior taxa de sucesso)."""
        if not self.implementations:
            return None
        return min(self.implementations, key=len)

    def execute(self, grid: np.ndarray, registry) -> np.ndarray:
        """Executa a skill usando a melhor implementacao para este grid."""
        impl = self.choose_implementation(grid, registry)
        if not impl:
            return grid.copy()
        result = grid.copy()
        for pname in impl:
            prim = registry.get(pname)
            if prim:
                result = prim.apply(result, {})
        return result

    def choose_implementation(self, grid: np.ndarray, registry) -> Optional[Tuple[str, ...]]:
        """Escolhe a melhor implementacao para um grid especifico.

        Tenta cada implementacao e retorna a que funciona melhor.
        """
        if not self.implementations:
            return None

        best = None
        best_score = -1

        for impl in self.implementations:
            try:
                result = grid.copy()
                for pname in impl:
                    prim = registry.get(pname)
                    if prim:
                        result = prim.apply(result, {})
                # Score: quantos objetos nao-zero foram preservados
                score = (result > 0).sum()
                if score > best_score:
                    best_score = score
                    best = impl
            except Exception:
                pass

        return best or self.implementations[0]

    def add_implementation(self, seq: Tuple[str, ...], success: bool = True):
        if seq not in self.implementations:
            self.implementations.append(seq)
        self.n_attempts += 1
        if success:
            self.n_success += 1
        self.confidence = self.n_success / max(1, self.n_attempts)

    def to_dict(self) -> dict:
        return {
            "skill": self.name,
            "intent": self.intent,
            "implementations": len(self.implementations),
            "n_attempts": self.n_attempts,
            "confidence": round(self.confidence, 3),
            "domains": list(self.domains),
        }


# ─── SkillRegistry ─────────────────────────────────────────

class SkillRegistry:
    """Registry de skills (intencoes abstratas)."""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        if skill.name in self.skills:
            existing = self.skills[skill.name]
            for imp in skill.implementations:
                if imp not in existing.implementations:
                    existing.implementations.append(imp)
            existing.confidence = max(existing.confidence, skill.confidence)
            existing.domains |= skill.domains
        else:
            self.skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def get_best(self, n: int = 10, min_confidence: float = 0.0) -> List[Skill]:
        candidates = [s for s in self.skills.values() if s.confidence >= min_confidence]
        candidates.sort(key=lambda s: -s.confidence * s.frequency)
        return candidates[:n]

    def get_stats(self) -> dict:
        if not self.skills:
            return {"total": 0}
        confs = [s.confidence for s in self.skills.values()]
        impls = [len(s.implementations) for s in self.skills.values()]
        return {
            "total": len(self.skills),
            "avg_confidence": round(np.mean(confs), 3),
            "avg_implementations": round(np.mean(impls), 1),
            "total_attempts": sum(s.n_attempts for s in self.skills.values()),
        }


# ─── SkillDiscovery — descobre skills de clusters de estado ─

class SkillDiscovery:
    """Descobre skills a partir de operacoes que produzem o mesmo estado intermediario.

    Fluxo:
      1. Recebe um conjunto de macros (sequencias frequentes)
      2. Para cada macro, computa o estado apos o primeiro passo
      3. Agrupa macros cujo primeiro passo produz o mesmo estado
      4. Cada grupo vira uma Skill: o INTENT e o estado resultante,
         as implementacoes sao as sequencias completas
    """

    def __init__(self, registry=None):
        self.registry = registry
        self.skill_registry = SkillRegistry()

    def discover(self, macro_sequences: List[Tuple[str, ...]],
                 source_tasks: Dict[str, Set[str]] = None,
                 domain: str = "") -> SkillRegistry:
        """Descobre skills de uma lista de sequencias de macros.

        Args:
            macro_sequences: lista de tuplas de nomes de primitives
            source_tasks: mapeamento macro_name -> set de task_ids
            domain: dominio (ex: "ARC")

        Returns: SkillRegistry populado.
        """
        # Para cada sequencia, extrai o primeiro passo (operacao inicial)
        # e ve quais outras sequencias comecam com a mesma operacao
        # produzindo o mesmo efeito.

        # Agrupa por OPERACAO INICIAL
        by_first_op = defaultdict(list)
        for seq in macro_sequences:
            if len(seq) >= 2:
                by_first_op[seq[0]].append(seq)

        # Para cada grupo, cria uma skill
        for first_op, sequences in by_first_op.items():
            if len(sequences) < 2:
                continue

            # Infere o intent baseado na operacao inicial
            intent = infer_intent(first_op)

            # Extrai o SEGUNDO passo como variacao de implementacao
            variations = set()
            for seq in sequences:
                if len(seq) >= 2:
                    variations.add(tuple(seq[1:]))

            if not variations:
                continue

            # Cria a skill com todas as implementacoes (combinacoes)
            skill_name = f"Skill_{first_op}"
            skill = Skill(
                name=skill_name,
                intent=intent,
                description=f"{intent}: via {first_op} + variacoes",
            )
            # Cada implementacao e first_op + variacao
            for var in variations:
                full_seq = (first_op,) + var
                skill.add_implementation(full_seq, success=True)

            skill.domains.add(domain)
            self.skill_registry.register(skill)

        return self.skill_registry

    def get_stats(self) -> dict:
        return self.skill_registry.get_stats()


def infer_intent(first_op: str) -> str:
    """Infere o proposito (intent) baseado na primeira operacao."""
    intent_map = {
        "rotate_90": "reorient_objects",
        "rotate_180": "reorient_objects",
        "rotate_270": "reorient_objects",
        "flip_horizontal": "mirror_objects",
        "flip_vertical": "mirror_objects",
        "transpose": "swap_axes",
        "move": "relocate_objects",
        "recolor": "recolor_objects",
        "remove": "filter_objects",
        "add": "add_objects",
        "transform": "resize_objects",
    }
    return intent_map.get(first_op, f"apply_{first_op}")
