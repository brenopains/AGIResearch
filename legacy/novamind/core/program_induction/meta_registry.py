"""
MetaPrimitiveRegistry — Primitives como organismos vivos com ciclo de vida.

Cada primitive agora tem:
  - utilidade: quao util e em tarefas reais (0-1)
  - frequencia: quantas vezes foi usada
  - dominios: em que contextos funciona
  - custo: custo computacional medio
  - composicoes_conhecidas: com quais outras primitives costuma compor
  - historico_sucesso: timeline de sucessos
  - historico_falha: timeline de falhas
  - idade: desde quando existe
  - pre_condicoes: o que precisa ser verdade no grid para funcionar
  - efeitos: o que muda no grid apos aplicacao

O registry gerencia o ciclo de vida:
  - promove: primitives com alta utilidade ganham prioridade
  - arquiva: primitives com baixa utilidade sao movidas para arquivo
  - funde: primitives que sempre sao usadas juntas viram uma so
  - monitora: metricas de saude de cada primitiva
"""
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class PrimitiveHealth:
    """Metricas de saude de uma primitiva."""
    utility: float = 0.0       # utilidade media em tarefas (0-1)
    frequency: int = 0          # vezes que foi usada
    domains: Set[str] = field(default_factory=set)  # contextos de uso
    avg_cost: float = 0.0       # custo computacional medio
    success_rate: float = 0.0   # taxa de sucesso historica
    age: float = 0.0            # horas desde descoberta
    last_used: float = 0.0      # timestamp do ultimo uso
    compositions: Dict[str, int] = field(default_factory=dict)  # A->B: n vezes
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    n_success: int = 0
    n_fail: int = 0
    discovery_time: float = field(default_factory=time.time)

    @property
    def viability(self) -> float:
        """Viabilidade geral: utilidade * taxa_sucesso * (1 - custo/10)."""
        return self.utility * self.success_rate * max(0.1, 1.0 - self.avg_cost / 5.0)


class MetaPrimitiveRegistry:
    """Registry com ciclo de vida.

    Uso:
        meta = MetaPrimitiveRegistry(registry)
        meta.record_use("recolor", success=True, cost=0.1, domain="ARC")
        meta.record_composition("recolor", "move", success=True)
        meta.prune()  # arquiva inuteis, funde frequentes
    """

    def __init__(self, registry=None):
        self._registry = registry  # PrimitiveRegistry externo (opcional)
        self.health: Dict[str, PrimitiveHealth] = {}
        self.archived: Dict[str, PrimitiveHealth] = {}
        self._total_uses = 0

    def ensure(self, name: str):
        """Garante que a primitiva tem registro de saude."""
        if name not in self.health:
            self.health[name] = PrimitiveHealth()

    def record_use(self, name: str, success: bool = True,
                   cost: float = 0.1, domain: str = ""):
        """Registra um uso da primitiva."""
        self.ensure(name)
        h = self.health[name]
        h.frequency += 1
        h.last_used = time.time()
        h.age = (time.time() - h.discovery_time) / 3600
        if domain:
            h.domains.add(domain)
        if success:
            h.n_success += 1
        else:
            h.n_fail += 1
        total = h.n_success + h.n_fail
        h.success_rate = h.n_success / max(1, total)
        h.avg_cost = (h.avg_cost * (total - 1) + cost) / max(1, total)
        # Utilidade exponencial: usos recentes pesam mais
        h.utility = h.success_rate * (1 - 1 / (1 + h.frequency * 0.1))
        self._total_uses += 1

    def record_composition(self, a: str, b: str, success: bool = True):
        """Registra que a primitive 'a' foi composta com 'b'."""
        self.ensure(a)
        self.ensure(b)
        if success:
            self.health[a].compositions[b] = self.health[a].compositions.get(b, 0) + 1

    def get_best(self, n: int = 5, domain: str = "") -> List[Tuple[str, PrimitiveHealth]]:
        """Retorna as N primitives com maior viabilidade."""
        candidates = []
        for name, h in self.health.items():
            if domain and domain not in h.domains:
                continue
            if h.frequency < 1:
                continue
            candidates.append((name, h))
        candidates.sort(key=lambda x: -x[1].viability)
        return candidates[:n]

    def prune(self, min_viability: float = 0.05,
              max_age_hours: float = 72):
        """Arquiva primitives inuteis e funde composicoes frequentes."""
        now = time.time()
        changes = {"archived": 0, "merged": 0}

        # 1. Arquivar: baixa viabilidade e pouca frequencia
        to_archive = []
        for name, h in self.health.items():
            age_hours = (now - h.discovery_time) / 3600
            if h.viability < min_viability and h.frequency < 3 and age_hours > 1:
                to_archive.append(name)

        for name in to_archive:
            self.archived[name] = self.health.pop(name)
            changes["archived"] += 1
            if self._registry and name in [p.name for p in self._registry.get_all()]:
                pass  # Opcional: remover do registry

        # 2. Fundir: se A e B sempre sao usados juntos, propoe fusao
        merged_names = set()
        for name, h in self.health.items():
            if name in merged_names:
                continue
            for comp_name, count in h.compositions.items():
                if comp_name in merged_names:
                    continue
                # Se compuseram juntos mais de 5 vezes e mutualmente
                mutual = self.health.get(comp_name)
                if mutual and mutual.compositions.get(name, 0) >= 5 and count >= 5:
                    merged_names.add(name)
                    merged_names.add(comp_name)
                    changes["merged"] += 1

        return changes

    def get_merged_candidates(self) -> List[Tuple[str, str, int]]:
        """Retorna pares que sao candidatos a fusao."""
        candidates = []
        for name, h in self.health.items():
            for comp_name, count in h.compositions.items():
                mutual = self.health.get(comp_name)
                if mutual and mutual.compositions.get(name, 0) >= 3 and count >= 3:
                    candidates.append((name, comp_name, count + mutual.compositions.get(name, 0)))
        candidates.sort(key=lambda x: -x[2])
        return candidates

    def get_stats(self) -> dict:
        """Metricas de saude do ecossistema de primitives."""
        if not self.health:
            return {"total": 0}
        utils = [h.utility for h in self.health.values()]
        freqs = [h.frequency for h in self.health.values()]
        return {
            "total": len(self.health),
            "archived": len(self.archived),
            "total_uses": self._total_uses,
            "avg_utility": round(np.mean(utils), 4) if utils else 0,
            "max_utility": round(max(utils), 4) if utils else 0,
            "avg_frequency": round(np.mean(freqs), 1) if freqs else 0,
            "avg_viability": round(np.mean([h.viability for h in self.health.values()]), 4),
            "n_ready_to_merge": len(self.get_merged_candidates()),
            "total_compositions": sum(
                len(h.compositions) for h in self.health.values()
            ),
        }

    def get_all_health(self) -> dict:
        return {
            name: {
                "utility": round(h.utility, 3),
                "frequency": h.frequency,
                "success_rate": round(h.success_rate, 3),
                "viability": round(h.viability, 3),
                "domains": list(h.domains),
                "compositions": dict(sorted(h.compositions.items(),
                                            key=lambda x: -x[1])[:5]),
            }
            for name, h in sorted(self.health.items(), key=lambda x: -x[1].frequency)
        }
