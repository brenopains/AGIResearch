"""
HypothesisCompetition V2 — Evidencias tipadas por categoria + hierarquia de confianca.

Cada hipotese recebe evidencias separadas por tipo:
  - Taxonomica (WordNet, ontologias) — maior peso
  - Causal (WorldModel, Science Curriculum)
  - Episodica (EpisodeStore, memorias)
  - Perceptual (DINOv2, Similaridade Visual)

Competicoes separadas por categoria:
  - IS_A: "whale is mammal"
  - HAS_PROPERTY: "mammal breathes air"
  - RELATED: "whale lives in ocean"
  - SIMILAR: "whale looks like dolphin"

Justification: cadeia causal que explica a conclusao.
"""
import time
import json
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ─── Pesos por tipo de evidencia (hierarquia de confianca) ──
EVIDENCE_WEIGHTS = {
    "taxonomic":   1.0,   # WordNet, ontologias
    "logical":     0.95,  # Regras logicas, inferencia
    "causal":      0.85,  # WorldModel, Science Curriculum
    "mental_model": 0.75, # Modelos mentais consolidados
    "episodic":    0.40,  # EpisodeStore, memorias passadas
    "perceptual":  0.35,  # DINOv2, similaridade visual
    "similarity":  0.20,  # Similaridade superficial
}


class EvidenceItem:
    def __init__(self, etype: str, source: str, strength: float, text: str = "",
                 favorable: bool = True):
        self.etype = etype  # taxonomic, causal, episodic, perceptual
        self.source = source
        self.strength = strength
        self.text = text[:100]
        self.favorable = favorable
        self.weight = EVIDENCE_WEIGHTS.get(etype, 0.3)

    def weighted_strength(self) -> float:
        return self.strength * self.weight

    def to_dict(self) -> dict:
        return {
            "type": self.etype, "source": self.source,
            "strength": round(self.strength, 3),
            "weight": round(self.weight, 2),
            "weighted": round(self.weighted_strength(), 3),
            "text": self.text, "favorable": self.favorable,
        }


class Hypothesis:
    def __init__(self, name: str, category: str = "related", description: str = ""):
        self.name = name
        self.category = category  # is_a, has_property, related, similar
        self.description = description
        self.evidence: List[EvidenceItem] = []
        self.score = 0.0
        self.created_at = time.time()

    def add_evidence(self, etype: str, source: str, strength: float,
                     text: str = "", favorable: bool = True):
        self.evidence.append(EvidenceItem(etype, source, strength, text, favorable))
        self._recompute()

    def _recompute(self):
        if not self.evidence:
            self.score = 0.0
            return
        # Score = media ponderada das evidencias
        total_weight = 0
        total_score = 0
        for ev in self.evidence:
            if ev.favorable:
                total_score += ev.weighted_strength()
            else:
                total_score -= ev.weighted_strength() * 0.7
            total_weight += ev.weight
        self.score = max(0.0, total_score / max(0.001, total_weight))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category,
            "description": self.description[:80],
            "score": round(self.score, 3),
            "evidence_summary": self._summary_by_type(),
            "evidence_details": [e.to_dict() for e in self.evidence[-5:]],
        }

    def _summary_by_type(self) -> dict:
        by_type = defaultdict(list)
        for e in self.evidence:
            by_type[e.etype].append(e.weighted_strength())
        return {k: round(sum(v)/len(v), 3) for k, v in by_type.items()}


class Justification:
    """Cadeia de justificacao: como a hipotese foi construida."""
    def __init__(self):
        self.steps: List[Dict] = []
    
    def add_step(self, premise: str, conclusion: str, rule: str = "", confidence: float = 1.0):
        self.steps.append({
            "premise": premise, "conclusion": conclusion,
            "rule": rule, "confidence": round(confidence, 3),
        })
    
    def to_text(self) -> str:
        lines = []
        for s in self.steps:
            if s["rule"]:
                lines.append(f"{s['premise']} --[{s['rule']}]--> {s['conclusion']}")
            else:
                lines.append(f"{s['conclusion']}")
        return " → ".join(lines) if lines else ""

    def to_dict(self) -> list:
        return self.steps[-8:]


class HypothesisCompetition:
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)  # cat -> [hyp_names]
        self.conflicts: List[Dict] = []

    def add_hypothesis(self, name: str, category: str = "related",
                       description: str = "") -> Hypothesis:
        if name not in self.hypotheses:
            self.hypotheses[name] = Hypothesis(name, category, description)
            self.categories[category].append(name)
        return self.hypotheses[name]

    def add_evidence(self, hyp_name: str, etype: str, source: str, strength: float,
                     text: str = "", favorable: bool = True):
        hyp = self.add_hypothesis(hyp_name)
        hyp.add_evidence(etype, source, strength, text, favorable)

    # ─── Geracao a partir do CGE ───────────────────────────

    def generate_from_cge(self, seed: str, cge, store=None, max_per_category: int = 5):
        """Gera hipoteses separadas por categoria."""
        if cge is None:
            return

        # Mapeia nos ativos para categorias
        active_nodes = []
        for nid, node in cge.nodes.items():
            state = node.get("state", "")
            if state in ("visible", "active"):
                act = node.get("activation_level", 0.0)
                conf = node.get("confidence", 0.5)
                src = node.get("activation_source", "cge")
                if nid != seed and act > 0.1:
                    active_nodes.append((nid, act, conf, src))

        active_nodes.sort(key=lambda x: x[1] * x[2], reverse=True)

        for nid, act, conf, src in active_nodes:
            # Determina categoria baseada no tipo de relacao
            cat = self._infer_category(seed, nid)
            hyp_name = f"{seed}→{nid}"
            self.add_hypothesis(hyp_name, cat, f"{seed} relates to {nid}")

            # Evidencia perceptual (CGE activation)
            self.add_evidence(hyp_name, "perceptual", src,
                            float(act), f"CGE activation: {act:.3f}")

            # Evidencia taxonomica (se conexao direta no CGE)
            if cge and self._has_taxonomic_edge(cge, seed, nid):
                self.add_evidence(hyp_name, "taxonomic", "CGE_edge",
                                0.9 * float(conf), f"direct CGE edge {seed}→{nid}")

            # Evidencia episodica
            if store and hasattr(store, "query_text"):
                try:
                    results = store.query_text(f"{seed} {nid}", top_k=1)
                    if results and results[0].get("score", 0) > 0.3:
                        self.add_evidence(hyp_name, "episodic", "EpisodeStore",
                                        results[0]["score"],
                                        f"episode matches: {results[0].get('content',{}).get('input','')[:50]}")
                except:
                    pass

        return seed

    def _infer_category(self, a: str, b: str) -> str:
        """Infere a categoria de relacao entre dois conceitos."""
        # Relacoes taxonomicas
        if b in ("animal", "mammal", "reptile", "bird", "fish", "plant",
                 "vertebrate", "invertebrate", "primate", "canine", "feline",
                 "rodent", "cetacean", "amphibian"):
            return "is_a"
        # Propriedades
        if b in ("breathe", "fly", "swim", "walk", "run", "eat", "drink",
                 "sleep", "grow", "die", "live", "move", "see", "hear",
                 "warm_blooded", "cold_blooded", "has_fur", "has_feathers",
                 "has_gills", "has_wings", "has_fins", "has_legs",
                 "lay_eggs", "give_birth", "nurse_young"):
            return "has_property"
        # Similaridade (mesma categoria)
        if a and b and self._same_category(a, b):
            return "similar"
        return "related"

    def _same_category(self, a: str, b: str) -> bool:
        """Verifica se dois conceitos sao da mesma categoria."""
        animals = {"cat","dog","bat","whale","dolphin","horse","cow","pig",
                   "elephant","giraffe","zebra","lion","tiger","bear","fox",
                   "wolf","deer","rabbit","mouse","rat","monkey","ape",
                   "seal","walrus","otter","beaver","squirrel","hedgehog",
                   "shark","tuna","salmon","trout","cod","eel",
                   "eagle","hawk","falcon","sparrow","robin","crow","raven",
                   "parrot","penguin","ostrich","duck","goose","swan",
                   "frog","toad","salamander","newt",
                   "snake","lizard","turtle","crocodile","alligator",
                   "ant","bee","butterfly","moth","beetle","spider","scorpion"}
        return a in animals and b in animals

    def _has_taxonomic_edge(self, cge, a: str, b: str) -> bool:
        """Verifica se existe aresta taxonomica direta entre dois nos."""
        if not cge:
            return False
        for e in cge.edges:
            if e.get("from") == a and e.get("to") == b:
                return True
            if e.get("from") == b and e.get("to") == a:
                return True
        return False

    # ─── Competicao ─────────────────────────────────────────

    def elect_winner(self) -> Optional[Tuple[str, Hypothesis]]:
        """Elege a vencedora GLOBAL (melhor score entre todas)."""
        if not self.hypotheses:
            return None
        best_name = max(self.hypotheses, key=lambda n: self.hypotheses[n].score)
        return (best_name, self.hypotheses[best_name])

    def elect_winner_by_category(self, category: str) -> Optional[Tuple[str, Hypothesis]]:
        """Elege a melhor hipotese de uma categoria especifica."""
        names = self.categories.get(category, [])
        if not names:
            return None
        best_name = max(names, key=lambda n: self.hypotheses[n].score)
        return (best_name, self.hypotheses[best_name])

    def get_ranked(self, category: str = None) -> List[Tuple[str, Hypothesis]]:
        """Retorna hipoteses ordenadas (filtradas por categoria se especificado)."""
        items = self.hypotheses.items()
        if category:
            items = [(n, h) for n, h in items if h.category == category]
        return sorted(items, key=lambda x: x[1].score, reverse=True)

    def compute_entropy(self) -> float:
        scores = [h.score for h in self.hypotheses.values()]
        if not scores:
            return 0.0
        total = sum(scores)
        if total == 0:
            return 0.0
        probs = [s / total for s in scores]
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
        max_entropy = math.log2(len(scores))
        return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    # ─── Justificacao ──────────────────────────────────────

    def build_justification(self, winner_name: str) -> Justification:
        """Constroi cadeia de justificacao para a hipotese vencedora."""
        j = Justification()
        if winner_name not in self.hypotheses:
            return j

        winner = self.hypotheses[winner_name]
        parts = winner_name.split("→")
        seed = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else ""

        j.add_step(seed, target, winner.category, winner.score)

        # Evidencias mais fortes viram etapas da justificacao
        sorted_ev = sorted(winner.evidence, key=lambda e: e.weighted_strength(), reverse=True)
        for ev in sorted_ev[:3]:
            j.add_step(ev.source, ev.text[:60], ev.etype, ev.weighted_strength())

        return j

    # ─── Export ─────────────────────────────────────────────

    def to_audit_dict(self) -> dict:
        ranked_by_cat = {}
        for cat in ("is_a", "has_property", "related", "similar"):
            winner = self.elect_winner_by_category(cat)
            ranked_by_cat[cat] = {
                "winner": winner[0] if winner else None,
                "winner_score": winner[1].score if winner else 0,
                "hypotheses": [{"name": n, "score": h.score}
                             for n, h in self.get_ranked(cat)[:5]],
            }

        global_winner = self.elect_winner()
        justification = self.build_justification(global_winner[0]) if global_winner else Justification()

        # Confidence decomposition para a vencedora global
        confidence_decomp = {}
        if global_winner:
            gw = global_winner[1]
            by_type = defaultdict(list)
            for e in gw.evidence:
                by_type[e.etype].append(e.weighted_strength())
            confidence_decomp = {
                k: round(sum(v)/len(v), 3) for k, v in by_type.items()
            }

        return {
            "n_hypotheses": len(self.hypotheses),
            "entropy": self.compute_entropy(),
            "by_category": ranked_by_cat,
            "global_winner": global_winner[0] if global_winner else None,
            "global_score": global_winner[1].score if global_winner else 0,
            "confidence_decomposition": confidence_decomp,
            "justification": justification.to_text(),
        }
