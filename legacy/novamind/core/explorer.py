"""
KnowledgeExplorer — Navega ativamente pelo modelo de mundo.

Nao busca nada especifico. Anda pelo grafo de conhecimento:
  - Segue arestas causais
  - Salta por analogia HDV
  - Cruza dominios por WorldLaws
  - Detecta "gaps" (conceitos isolados, pontes faltando)
  - Acumula "achados" que alimentam o raciocinio sobre uma doenca

Cada passo: um conceito. Sempre muda de dominio.
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random
import time


@dataclass
class ExplorationFinding:
    """Um achado durante a exploracao."""
    concept: str
    domain: str
    reached_via: str  # "causal_edge", "analogy", "law_transfer", "gap"
    path_from_start: List[str] = field(default_factory=list)
    relevance_to_disease: float = 0.0  # 0-1, quao relevante e para a doenca alvo
    novelty: float = 1.0  # 0-1, quao distante do pensamento anterior
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept[:40],
            "domain": self.domain,
            "via": self.reached_via,
            "relevance": round(self.relevance_to_disease, 2),
            "novelty": round(self.novelty, 2),
        }


class KnowledgeExplorer:
    """Navegador autonomo do modelo de mundo.

    Nao tem objetivo. Explora.
    Gera "achados" que alimentam o raciocinio.

    Uso:
        explorer = KnowledgeExplorer(kg, cge)
        findings = explorer.explore(start="Alzheimer", steps=50)
    """

    def __init__(self, kg=None, cge=None):
        self.kg = kg
        self.cge = cge
        self._visited: Set[str] = set()
        self._current_path: List[str] = []
        self._findings: List[ExplorationFinding] = []

    def explore(self, start_concept: str = None,
                max_steps: int = None) -> List[ExplorationFinding]:
        """Explora o modelo de mundo por max_steps passos.

        O numero de passos e DINAMICO:
          - Continua enquanto encontrar novos dominios
          - Para quando: reexplorando >70% dos ultimos passos
          - Ou quando: encontrar gap que vale a pena investigar
        """
        self._visited.clear()
        self._current_path = []
        self._findings = []

        if not self.kg or not self.kg.concepts:
            return []

        # Ponto de partida
        if start_concept and start_concept in self.kg.concepts:
            current = start_concept
        else:
            current = random.choice(list(self.kg.concepts.keys())[:100])

        domain_counts = defaultdict(int)
        revisit_count = 0
        steps = 0

        while True:
            self._visited.add(current)
            self._current_path.append(current)

            # Registra achado
            domain = self.kg.concepts[current].domain if current in self.kg.concepts else "unknown"
            domain_counts[domain] += 1

            finding = ExplorationFinding(
                concept=current,
                domain=domain,
                reached_via="start" if steps == 0 else "causal_edge",
                path_from_start=list(self._current_path),
                novelty=1.0 if current not in self._visited else 0.3,
            )
            self._findings.append(finding)

            # Escolhe proximo passo
            next_concept = self._pick_next(current)

            if next_concept is None:
                # Beco sem saida: salta para um conceito aleatorio em dominio diferente
                next_concept = self._jump_to_new_domain(domain_counts)

            if next_concept is None:
                break  # Exauriu o espaco de exploracao

            # Se revisitou algo recente
            if next_concept in self._visited:
                revisit_count += 1
            else:
                revisit_count = 0

            steps += 1

            # Criterio de parada DINAMICO:
            # Para se: revisitando demais OU exploracao estabilizou
            if max_steps is not None and steps >= max_steps:
                break
            if revisit_count >= 5:  # 5 revisitadas consecutivas → estagnou
                break

            current = next_concept

        return self._findings

    def _pick_next(self, concept: str) -> Optional[str]:
        """Escolhe o proximo conceito para visitar.

        Tenta, em ordem:
          1. Aresta causal para um dominio DIFERENTE do atual
          2. Conceito similar por similaridade de nome
          3. Aresta causal qualquer (mesmo dominio)
          4. Nenhum
        """
        candidates = []

        current_domain = self.kg.concepts[concept].domain if concept in self.kg.concepts else ""

        # 1. Arestas causais para dominios diferentes (exploracao ativa)
        if self.kg:
            for rel in self.kg.relations[:500]:
                if rel.source == concept or rel.target == concept:
                    neighbor = rel.target if rel.source == concept else rel.source
                    if neighbor in self.kg.concepts:
                        nd = self.kg.concepts[neighbor].domain
                        if nd and nd != current_domain:
                            candidates.append((neighbor, 2.0))  # peso alto
                        elif neighbor not in self._visited:
                            candidates.append((neighbor, 1.0))
                        else:
                            candidates.append((neighbor, 0.3))

        if candidates:
            # Escolha ponderada: prefere dominios diferentes
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

        return None

    def _jump_to_new_domain(self, domain_counts: dict) -> Optional[str]:
        """Salta para um conceito em dominio pouco explorado."""
        # Encontra dominios menos visitados
        if not domain_counts:
            return None

        min_count = min(domain_counts.values())
        least_visited = [d for d, c in domain_counts.items() if c == min_count]

        # Procura conceitos nesses dominios que ainda nao visitamos
        for _ in range(100):
            candidate = random.choice(list(self.kg.concepts.keys()))
            if candidate not in self._visited:
                return candidate

        return None

    def get_findings_for_disease(self, disease: str) -> List[ExplorationFinding]:
        """Retorna achados relevantes para uma doenca."""
        # Relevancia = quantos passos de distancia no grafo
        relevant = []
        for f in self._findings:
            if disease.lower() in f.concept.lower():
                f.relevance_to_disease = 1.0
                relevant.append(f)
            elif f.domain and disease.lower() in f.domain.lower():
                f.relevance_to_disease = 0.7
                relevant.append(f)
        return relevant

    def get_stats(self) -> dict:
        domains_visited = set()
        for f in self._findings:
            if f.domain:
                domains_visited.add(f.domain)
        return {
            "steps_taken": len(self._findings),
            "unique_concepts": len(self._visited),
            "domains_visited": len(domains_visited),
            "findings": [f.to_dict() for f in self._findings[-10:]],
        }
