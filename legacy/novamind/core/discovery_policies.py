"""
DiscoveryPolicies — Estilos de Descoberta Acima de Estrategias

Enquanto estrategias resolvem problemas,
Discovery Policies descobrem CONHECIMENTO.

Exemplos:
  "Para leis fisicas, analogias funcionam"
  "Para linguagem, analogias pioram tudo"
  "Em matematica, decomposicao antes de atacar"

Cada policy e um CognitiveStep que modifica como a Forja,
o Maestro e o SimEngine operam em um dominio.
"""
import time, uuid, hashlib, random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DiscoveryPolicy:
    id: str = ""
    principle: str = ""
    domain: str = "general"
    approach: str = ""  # "analogy", "decomposition", "simulation", "abstraction", "crossover"
    confidence: float = 0.3
    n_applications: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "principle": self.principle[:80],
                "domain": self.domain, "approach": self.approach,
                "confidence": round(self.confidence, 2)}


PRIMORDIAL_POLICIES = [
    DiscoveryPolicy(principle="Para dominios fisicos, simulacao antes de abstracao",
                    domain="physics", approach="simulation", confidence=0.8),
    DiscoveryPolicy(principle="Para dominios formais, decomposicao antes de atacar",
                    domain="mathematics", approach="decomposition", confidence=0.75),
    DiscoveryPolicy(principle="Para dominios biologicos, analogia com sistemas conhecidos",
                    domain="biology", approach="analogy", confidence=0.7),
    DiscoveryPolicy(principle="Para dominios sociais, observacao antes de teoria",
                    domain="social", approach="abstraction", confidence=0.65),
    DiscoveryPolicy(principle="Para dominios novos, exploracao ampla antes de foco",
                    domain="general", approach="crossover", confidence=0.6),
]


class PolicyLibrary:
    def __init__(self, cge=None):
        self.cge = cge
        self.policies: Dict[str, DiscoveryPolicy] = {}
        for p in PRIMORDIAL_POLICIES:
            pid = f"policy_{hashlib.md5(p.principle.encode()).hexdigest()[:8]}"
            p.id = pid
            self.policies[pid] = p

    def get_for_domain(self, domain: str) -> List[DiscoveryPolicy]:
        candidates = [p for p in self.policies.values() if p.domain == domain or p.domain == "general"]
        candidates.sort(key=lambda p: -p.confidence)
        return candidates[:3]

    def discover_from_traces(self, traces: List[Any]) -> List[DiscoveryPolicy]:
        new = []
        domain_success = defaultdict(list)
        for t in traces:
            td = t.to_dict() if hasattr(t, 'to_dict') else (t if isinstance(t, dict) else {})
            steps = td.get("steps", [])
            for s in steps:
                if isinstance(s, dict):
                    mode = s.get("mode", "")
                    reward = s.get("reward", 0)
                    domain_success[mode].append(reward)
        for domain, rewards in domain_success.items():
            if len(rewards) >= 5 and sum(rewards)/len(rewards) > 0.6:
                existing = [p for p in self.policies.values() if p.domain == domain]
                if any(p.confidence > 0.6 for p in existing):
                    continue
                p = DiscoveryPolicy(
                    principle=f"No dominio {domain}, modo {domain.upper()} gera alta recompensa",
                    domain=domain, approach="specialized",
                    confidence=min(0.6, sum(rewards)/len(rewards) * 0.6),
                    n_applications=len(rewards),
                    success_rate=sum(rewards)/len(rewards),
                )
                p.id = f"policy_{hashlib.md5(p.principle.encode()).hexdigest()[:8]}"
                self.policies[p.id] = p
                new.append(p)
        return new

    def get_state(self) -> Dict:
        return {"n_policies": len(self.policies),
                "by_domain": {d: sum(1 for p in self.policies.values() if p.domain == d)
                             for d in set(p.domain for p in self.policies.values())}}


class UnknownSpace:
    """Mapa do Desconhecido.

    Nao e "o que sei" (CGE).
    E "o que SEI que nao sei" — territorios inexplorados.

    Cada no: um vazio entre conceitos conhecidos.
    A Forja navega esses vazios para novas descobertas.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.unknown_regions: Dict[str, Dict] = {}

    def scan(self):
        """Escaneia o CGE para identificar vazios (conceitos desconhecidos)."""
        if self.cge is None:
            return
        # Conceitos conhecidos
        known = set(self.cge.nodes.keys())
        # Vazios: pares de conceitos conhecidos sem aresta direta
        for a in list(known)[:50]:
            for b in list(known)[:50]:
                if a < b:
                    direct_edge = any(
                        (e["from"] == a and e["to"] == b) or
                        (e["from"] == b and e["to"] == a)
                        for e in self.cge.edges
                    )
                    # Se nao ha aresta, mas ha potencial
                    if not direct_edge:
                        region_key = f"{a}<->{b}"
                        if region_key not in self.unknown_regions:
                            self.unknown_regions[region_key] = {
                                "concept_a": a, "concept_b": b,
                                "discovered_at": time.time(),
                                "explored": False,
                                "potential": random.uniform(0.1, 0.9),
                            }

    def get_frontier(self, top_k: int = 5) -> List[Dict]:
        """Retorna as regioes inexploradas mais promissoras."""
        unexplored = [r for r in self.unknown_regions.values() if not r["explored"]]
        unexplored.sort(key=lambda r: -r["potential"])
        return unexplored[:top_k]

    def explore(self, region_key: str) -> Optional[Dict]:
        """Marca uma regiao como explorada e retorna seus conceitos."""
        region = self.unknown_regions.get(region_key)
        if region:
            region["explored"] = True
            region["explored_at"] = time.time()
        return region

    def get_state(self) -> Dict:
        return {
            "n_regions": len(self.unknown_regions),
            "explored": sum(1 for r in self.unknown_regions.values() if r["explored"]),
            "frontier": len(self.get_frontier(10)),
        }
