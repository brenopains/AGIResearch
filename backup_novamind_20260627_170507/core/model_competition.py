"""
ModelCompetition — Competicao de Modelos com Colapso Epistemico

Multiplos modelos competem para explicar observacoes.
Cada modelo tem:
  - descricao / prediccoes / confianca
  - threshold de sobrevivencia
  - ao ser refutado, suas conexoes sao enfraquecidas globalmente
  - o modelo vencedor e consolidado (Forja)

O ciclo de competicao e um programa cognitivo em si.
"""
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Model:
    """Um modelo candidato — pode ser crenca, hipotese, teoria."""
    id: str
    description: str = ""
    predictions: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    survival_threshold: float = 0.1
    explanatory_power: float = 0.0  # 0..1, quanto explica
    parsimony: float = 1.0  # 0..1, simplicidade (navalha de Occam)
    age: int = 0
    last_updated: float = field(default_factory=time.time)
    refuted: bool = False
    causal_graph: Dict[str, float] = field(default_factory=dict)
    domain: str = "general"

    @property
    def viability(self) -> float:
        """Viabilidade geral = poder explicativo * parsimonia * confianca."""
        return self.explanatory_power * self.parsimony * self.confidence

    def compute_prediction_error(self, observation: Dict,
                                 predictor_fn: Optional[Callable] = None) -> float:
        """Erro medio entre predicao do modelo e observacao."""
        if not predictor_fn:
            return 0.5
        try:
            pred = predictor_fn(observation)
            # So comparar chaves que o modelo preve
            shared = set(pred.keys()) & set(observation.keys())
            if not shared:
                return 0.5
            error = sum(
                abs(pred[k] - observation.get(k, 0))
                for k in shared
            ) / len(shared)
            return min(1.0, error)
        except Exception:
            return 1.0

    def update(self, prediction_error: float, lr: float = 0.1):
        """Atualiza confianca e poder explicativo baseado no erro."""
        self.age += 1
        # Quanto menor o erro, mais a confianca aumenta
        accuracy = 1.0 - prediction_error
        self.confidence = self.confidence * (1 - lr) + accuracy * lr
        # Poder explicativo = media movel da acuracia
        self.explanatory_power = self.explanatory_power * 0.9 + accuracy * 0.1
        self.last_updated = time.time()

        # Se confianca cai abaixo do threshold, modelo e refutado
        if self.confidence < self.survival_threshold:
            self.refuted = True

    def collapse(self, cge=None):
        """Colapso epistemico: enfraquece todas as conexoes do modelo.

        Quando um modelo e refutado, suas edges no CGE sao enfraquecidas.
        Isso e analogo ao que acontece no cerebro quando uma crenca
        e abandonada — as conexoes que a sustentavam se enfraquecem.
        """
        self.refuted = True
        self.confidence *= 0.1
        if cge and self.id in cge.nodes:
            # Enfraquecer edges ligadas a este modelo
            edges_to_remove = []
            for i, edge in enumerate(cge.edges):
                if self.id in (edge["from"], edge["to"]):
                    edges_to_remove.append(i)
            # Remover edges (do final para o inicio para manter indices)
            for i in reversed(edges_to_remove):
                cge.edges.pop(i)

    def reconsolidate(self, cge=None):
        """Consolidacao: reforca o modelo vencedor."""
        self.confidence = min(1.0, self.confidence * 1.5)
        if cge and self.id in cge.nodes:
            # Reforcar edges entre o modelo e conceitos relacionados
            for edge in cge.edges:
                if self.id in (edge["from"], edge["to"]):
                    edge["weight"] = min(1.0, edge.get("weight", 0.5) * 1.3)


@dataclass
class ModelEcosystem:
    """Ecosistema de modelos competindo.

    Mantem N modelos ativos, controla nascimento/morte,
    e promove o(s) vencedor(es).
    """
    models: Dict[str, Model] = field(default_factory=dict)
    max_models: int = 16
    competition_history: List[Dict] = field(default_factory=list)

    def add_model(self, model: Model) -> str:
        if len(self.models) >= self.max_models:
            self._cull()
        self.models[model.id] = model
        return model.id

    def get_best_model(self, domain: str = "general") -> Optional[Model]:
        """Melhor modelo viavel em um dominio."""
        viable = [m for m in self.models.values()
                 if not m.refuted and m.domain == domain]
        if not viable:
            return None
        return max(viable, key=lambda m: m.viability)

    def compete(self, observation: Dict,
                predictor_fn: Optional[Callable] = None) -> Dict:
        """Um ciclo de competicao: todos os modelos predizem, depois atualizam.

        Retorna: {
            "winner": modelo vencedor,
            "losers": modelos refutados,
            "avg_error": erro medio,
        }
        """
        results = {
            "winner": None,
            "losers": [],
            "avg_error": 0.0,
            "n_survivors": 0,
        }

        active = [m for m in self.models.values() if not m.refuted]
        if not active:
            return results

        total_error = 0.0
        for model in active:
            error = model.compute_prediction_error(observation, predictor_fn)
            model.update(error)
            total_error += error

            if model.refuted:
                results["losers"].append(model.id)

        results["avg_error"] = total_error / len(active)
        results["n_survivors"] = len([m for m in active if not m.refuted])
        results["winner"] = self.get_best_model()

        self.competition_history.append({
            "time": time.time(),
            "n_models": len(active),
            "refuted": len(results["losers"]),
            "avg_error": results["avg_error"],
            "winner_id": results["winner"].id if results["winner"] else None,
        })

        return results

    def generate_alternative(self, base_model: Model, mutation: float = 0.3) -> Model:
        """Gera um modelo alternativo (mutacao) a partir de um existente."""
        new_id = f"alt_{base_model.id}_{int(time.time())}"
        alt = Model(
            id=new_id,
            description=f"alternative to {base_model.id}",
            confidence=base_model.confidence * (1 - mutation + random.random() * mutation * 2),
            parsimony=max(0.1, base_model.parsimony + random.uniform(-0.2, 0.2)),
            survival_threshold=base_model.survival_threshold * random.uniform(0.5, 1.5),
            domain=base_model.domain,
        )
        return alt

    def _cull(self):
        """Remove modelos com baixa viabilidade."""
        scored = [(m.viability, m_id) for m_id, m in self.models.items()
                 if not m.refuted]
        scored.sort()
        n_remove = max(1, len(scored) // 4)
        for _, m_id in scored[:n_remove]:
            self.models[m_id].refuted = True

    # ─── MERCADO DE IDEIAS (extensao) ──────────────────────────

    def merge(self, model_a_id: str, model_b_id: str,
              merge_name: str = "") -> Optional[Model]:
        """Funde dois modelos em um novo (heranca de componentes).

        O filho herda:
          - Poder explicativo do melhor pai
          - Parsimonia media
          - Confiança ponderada
          - Dominio combinado
        """
        a = self.models.get(model_a_id)
        b = self.models.get(model_b_id)
        if not a or not b or a.refuted or b.refuted:
            return None
        child = Model(
            id=f"merged_{a.id[:4]}_{b.id[:4]}_{int(time.time())}",
            description=merge_name or f"merge({a.id},{b.id})",
            confidence=(a.confidence + b.confidence) / 2,
            explanatory_power=max(a.explanatory_power, b.explanatory_power),
            parsimony=(a.parsimony + b.parsimony) / 2,
            domain=f"{a.domain}+{b.domain}" if a.domain != b.domain else a.domain,
        )
        child.predictions = (a.predictions + b.predictions)[:20]
        self.add_model(child)
        return child

    def cooperate(self, problem: str, model_ids: List[str]) -> List[Model]:
        """Coopera: divide problema em sub-problemas, cada modelo pega um.

        Retorna submodeos especializados.
        """
        specialists = []
        for i, mid in enumerate(model_ids[:4]):
            base = self.models.get(mid)
            if not base or base.refuted:
                continue
            sub = Model(
                id=f"sub_{mid[:4]}_{i}_{int(time.time())}",
                description=f"({base.id}) especializado em {problem[:20]}[{i}]",
                confidence=base.confidence * random.uniform(0.8, 1.2),
                parsimony=base.parsimony,
                domain=f"{base.domain}/sub_{i}",
            )
            sub.causal_graph = dict(base.causal_graph)
            self.add_model(sub)
            specialists.append(sub)
        return specialists

    def specialize(self, model_id: str, niche: str) -> Optional[Model]:
        """Cria uma variante de modelo especializada em um nicho."""
        base = self.models.get(model_id)
        if not base or base.refuted:
            return None
        variant = Model(
            id=f"spec_{base.id[:4]}_{int(time.time())}",
            description=f"{base.id} especializado em {niche}",
            confidence=base.confidence * 0.8,
            explanatory_power=base.explanatory_power * 0.7,
            parsimony=base.parsimony,
            domain=niche,
        )
        variant.predictions = list(base.predictions)
        self.add_model(variant)
        return variant

    def get_niche_winners(self) -> Dict[str, Model]:
        """Retorna o melhor modelo por dominio (multiplos vencedores)."""
        winners = {}
        for model in self.models.values():
            if model.refuted:
                continue
            d = model.domain
            if d not in winners or model.viability > winners[d].viability:
                winners[d] = model
        return winners

    def get_state(self) -> Dict:
        active = [m for m in self.models.values() if not m.refuted]
        niche_winners = self.get_niche_winners()
        return {
            "total": len(self.models),
            "active": len(active),
            "refuted": sum(1 for m in self.models.values() if m.refuted),
            "best": self.get_best_model().id if self.get_best_model() else None,
            "niche_winners": {d: m.id for d, m in niche_winners.items()},
            "avg_confidence": sum(m.confidence for m in active) / max(1, len(active)),
        }
