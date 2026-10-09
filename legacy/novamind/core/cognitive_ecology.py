"""
CognitiveEcology — Ecossistema Epistemico de Modelos Cognitivos

Nao e um "Internal Deliberation Engine" (que ligaria um modo).
E um ecossistema CONTINUO de modelos que:
  - competem por capital epistemologico
  - cooperam em coalizao
  - especializam-se em nichos
  - dormem quando inuteis
  - morrem e dao origem a variantes
  - criam novos veses cognitivos (meta-evolucao)

A AGI NAO controla este ecossistema diretamente.
Ele evolui autonomamente, como um ecossistema biologico.

Auto-tune: a maioria dos parametros (capital threshold, sleep timeout,
max population) sao ajustados baseado na SAUDE observada do ecossistema.
O sistema descobre sozinho o que funciona.
"""
import time, uuid, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# ─── VETOR DE VIESES COGNITIVOS ─────────────────────────────

class CognitiveBiasVector:
    """Vetor de veses cognitivos de um modelo.

    Cada dimensao (0..1) representa uma preferencia epistemologica.
    Nenhum viés e "melhor" — diferentes problemas pedem diferentes veses.

    Auto-descoberta: o ecossistema pode CRIAR novas dimensoes quando
    detecta que os veses atuais nao cobrem o espaco de problemas.
    """

    # Veses primordiais (herdados das Worldviews)
    _PRIMORDIAL_BIASES = {
        "causality": 0.5,        # preferir causalidade a correlacao
        "continuity": 0.5,       # preferir continuo a discreto
        "symmetry": 0.5,         # preferir simetria
        "agents": 0.5,           # preferir agentes a campos
        "statistics": 0.5,       # preferir estatistica
        "simplicity": 0.7,       # preferir simplicidade (Occam)
        "emergence": 0.5,        # preferir emergencia
    }

    def __init__(self, biases: Optional[Dict[str, float]] = None):
        # Comeca com veses neutros (0.5) ou com os fornecidos
        if biases is None:
            self.biases = dict(self._PRIMORDIAL_BIASES)
        else:
            self.biases = dict(biases)

    def __getitem__(self, key: str) -> float:
        return self.biases.get(key, 0.5)

    def __setitem__(self, key: str, value: float):
        self.biases[key] = max(0.0, min(1.0, value))

    def items(self):
        return self.biases.items()

    def keys(self):
        return self.biases.keys()

    def distance(self, other: "CognitiveBiasVector") -> float:
        """Distancia euclidiana entre dois vetores."""
        common = set(self.biases.keys()) & set(other.biases.keys())
        if not common:
            return 1.0
        sum_sq = sum((self.biases[k] - other.biases[k]) ** 2 for k in common)
        return math.sqrt(sum_sq / max(1, len(common)))

    def combine(self, other: "CognitiveBiasVector", weight: float = 0.5) -> "CognitiveBiasVector":
        """Crossover: media ponderada de dois vetores."""
        merged = {}
        all_keys = set(self.biases.keys()) | set(other.biases.keys())
        for k in all_keys:
            v1 = self.biases.get(k, 0.5)
            v2 = other.biases.get(k, 0.5)
            merged[k] = v1 * (1 - weight) + v2 * weight
        return CognitiveBiasVector(merged)

    def mutate(self, rate: float = 0.1) -> "CognitiveBiasVector":
        """Aplica mutacao aleatoria. Rate controla forca."""
        mutated = {}
        for k, v in self.biases.items():
            if random.random() < rate:
                v += random.uniform(-0.2, 0.2)
                v = max(0.0, min(1.0, v))
            mutated[k] = v
        return CognitiveBiasVector(mutated)

    def add_dimension(self, name: str, value: float = 0.5):
        """Adiciona uma nova dimensao (auto-descoberta)."""
        if name not in self.biases:
            self.biases[name] = value

    def to_dict(self) -> Dict:
        return dict(self.biases)


# ─── MODELO EPISTEMICO ───────────────────────────────────

class EpistemicModel:
    """Um modelo no ecossistema.

    Nao e uma persona fixa. E um VETOR DE VIESES que pode:
      - ser mutado
      - cruzar com outros
      - se dividir
      - se fundir
      - dormir
      - morrer (reciclado)
    """

    def __init__(self, bias: CognitiveBiasVector, domain: str = "general",
                 parents: Optional[List[str]] = None,
                 birth_cycle: int = 0):
        self.id = f"epm_{uuid.uuid4().hex[:8]}"
        self.bias = bias
        self.domain = domain

        # Capital epistemologico
        self.capital: float = 0.0  # NAO hardcoded — auto-inicializado pelo ecossistema
        self.confidence: float = 0.5
        self.n_predictions: int = 0
        self.n_correct: int = 0
        self.n_incorrect: int = 0

        # Genealogia
        self.parent_ids: List[str] = parents or []
        self.children_ids: List[str] = []

        # Estado
        self.status: str = "active"  # active, dormant, recycled, merged
        self.birth_cycle: int = birth_cycle
        self.last_active_cycle: int = birth_cycle
        self.last_division_cycle: int = 0
        self.last_sleep_cycle: int = 0
        self.history: List[Dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.n_predictions == 0:
            return 0.5
        return self.n_correct / self.n_predictions

    @property
    def total_predictions(self) -> int:
        return self.n_predictions

    def record_prediction(self, correct: bool, cost: float = 1.0,
                          reward: float = 1.0):
        """Registra resultado de uma previsao."""
        self.n_predictions += 1
        if correct:
            self.n_correct += 1
            self.capital += reward * self.confidence
        else:
            self.n_incorrect += 1
            self.capital -= cost * (1 - self.confidence)
        self.confidence = min(1.0, self.accuracy * 0.8 + 0.2)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "domain": self.domain,
                "capital": round(self.capital, 2),
                "accuracy": round(self.accuracy, 2),
                "status": self.status,
                "biases": self.bias.to_dict()}


# ─── COALIZAO ────────────────────────────────────────────

class Coalition:
    """Uma coalizao temporaria de modelos para resolver um problema."""
    def __init__(self, model_ids: List[str], problem: str):
        self.id = f"coal_{uuid.uuid4().hex[:8]}"
        self.model_ids = model_ids
        self.problem = problem
        self.combined_bias: Optional[CognitiveBiasVector] = None
        self.prediction: Optional[Any] = None
        self.confidence: float = 0.5
        self.outcome: Optional[bool] = None
        self.created_at: float = time.time()

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "models": [m[:8] for m in self.model_ids],
                "problem": self.problem[:40], "confidence": self.confidence}


# ─── ECOSSISTEMA COGNITIVO ────────────────────────────────

class CognitiveEcology:
    """Ecossistema de modelos cognitivos.

    Auto-regulado: nao tem capital threshold, max population, ou
    sleep timeout hardcoded. O sistema descobre o que funciona
    baseado em sua propria observacao de saude.
    """

    def __init__(self, cge=None, worldviews=None, stances=None):
        self.cge = cge
        self.worldviews = worldviews
        self.stances = stances
        self.models: Dict[str, EpistemicModel] = {}
        self.coalitions: List[Coalition] = []
        self._history: List[Dict] = []

        # Auto-descoberto: o ecossistema aprende seus proprios parametros
        self._observed_capital_distribution: List[float] = []
        self._population_pressure: int = 0  # cresce se ecossistema superlotado
        self._diversity_pressure: int = 0   # cresce se diversidade baixa
        self._healthy_diversity_threshold: float = 0.0  # auto-descoberto
        self._observed_diversity: float = 0.0
        self._cycle: int = 0

    # ─── Gestao de populacao (auto-regulada) ──────────────

    def get_population_health(self) -> float:
        """Saude do ecossistema: 0..1. Combina diversidade, capital medio,
        taxa de previsao recente."""
        if not self.models:
            return 0.0

        # Diversidade: variancia media entre bias vectors
        biases = [m.bias for m in self.models.values() if m.status == "active"]
        if len(biases) < 2:
            return 0.3
        avg_dist = 0
        count = 0
        for i in range(len(biases)):
            for j in range(i+1, len(biases)):
                avg_dist += biases[i].distance(biases[j])
                count += 1
        diversity = avg_dist / max(1, count)
        self._observed_diversity = diversity

        # Capital medio (relativo a 100 = base)
        active = [m for m in self.models.values() if m.status == "active"]
        if not active:
            return 0.0
        avg_capital = sum(m.capital for m in active) / len(active)
        capital_health = min(1.0, max(0.0, avg_capital / 100.0))

        # Atividade: modelos com previsao recente
        recent = sum(1 for m in active if m.n_predictions > 0)
        activity_health = recent / max(1, len(active))

        return 0.4 * diversity + 0.4 * capital_health + 0.2 * activity_health

    def _auto_discover_sleep_timeout(self) -> int:
        """O sistema descobre o tempo de sono ideal baseado em observacao."""
        if not hasattr(self, '_sleep_times'):
            self._sleep_times = []
        # Inicial: mais conservador. Aprende com dados.
        if len(self._sleep_times) < 5:
            return 50  # default conservador
        # Calcula mediana dos tempos de vida dos modelos que morreram
        return int(sorted(self._sleep_times)[len(self._sleep_times) // 2]) if self._sleep_times else 50

    def _auto_discover_recycle_threshold(self) -> float:
        """O sistema descobre o limiar de capital para reciclagem."""
        if len(self._observed_capital_distribution) < 10:
            # Inicial: conservador. Nao reciclar muito cedo.
            return 0.0
        # Auto-tune: reciclar apenas modelos abaixo do percentil 10
        sorted_caps = sorted(self._observed_capital_distribution)
        return sorted_caps[max(0, len(sorted_caps) // 10)]

    # ─── Nascimento / morte / split / merge ───────────────

    def birth(self, bias: Optional[CognitiveBiasVector] = None,
              domain: str = "general",
              parents: Optional[List[str]] = None) -> EpistemicModel:
        """Nasce um novo modelo."""
        if bias is None:
            bias = CognitiveBiasVector()
        model = EpistemicModel(bias=bias, domain=domain,
                               parents=parents, birth_cycle=self._cycle)
        # Capital inicial: NAO hardcoded. Usa mediana observada ou 100.
        observed = self._observed_capital_distribution[-50:] or [100.0]
        model.capital = sorted(observed)[len(observed)//2] if observed else 100.0
        self.models[model.id] = model
        return model

    def seed_from_worldviews(self) -> List[EpistemicModel]:
        """Popula o ecossistema com base nas Worldviews existentes."""
        new = []
        if self.worldviews is not None:
            for wv in self.worldviews.worldviews.values():
                # Converte bias da worldview para CognitiveBiasVector
                wv_biases = wv.to_dict().get("biases", {})
                if wv_biases:
                    bias = CognitiveBiasVector(wv_biases)
                    model = self.birth(bias=bias, domain=wv.domain[:20],
                                      parents=[wv.id])
                    new.append(model)
        # Se nao tem worldviews, nasce pelo menos 1 com bias neutro
        if not new:
            new.append(self.birth(domain="general"))
        return new

    def death(self, model_id: str, reason: str = "low_capital"):
        """Recicla um modelo."""
        model = self.models.get(model_id)
        if model is None or model.status != "active":
            return
        model.status = "recycled"
        model.death_cycle = self._cycle
        self._sleep_times.append(self._cycle - model.birth_cycle)
        self._history.append({"action": "death", "id": model_id,
                              "reason": reason, "cycle": self._cycle,
                              "final_capital": model.capital})

    def split(self, model_id: str) -> Optional[EpistemicModel]:
        """Um modelo se divide em uma variante."""
        model = self.models.get(model_id)
        if model is None or model.status != "active":
            return None
        if model.capital < 50:  # precisa ter acumulado capital
            return None
        # Cria variacao com mutacao
        new_bias = model.bias.mutate(rate=0.2)
        child = self.birth(bias=new_bias, domain=model.domain,
                          parents=[model_id])
        model.children_ids.append(child.id)
        # Divide capital
        child.capital = model.capital * 0.4
        model.capital *= 0.6
        model.last_division_cycle = self._cycle
        self._history.append({"action": "split", "parent": model_id,
                              "child": child.id, "cycle": self._cycle})
        return child

    def merge(self, model_a_id: str, model_b_id: str) -> Optional[EpistemicModel]:
        """Dois modelos similares se fundem."""
        a = self.models.get(model_a_id)
        b = self.models.get(model_b_id)
        if a is None or b is None:
            return None
        if a.status != "active" or b.status != "active":
            return None
        # So funde se muito similares
        if a.bias.distance(b.bias) > 0.15:
            return None
        # Crossover dos biases
        combined = a.bias.combine(b.bias, weight=0.5)
        merged = self.birth(bias=combined, domain=a.domain,
                           parents=[model_a_id, model_b_id])
        merged.capital = a.capital + b.capital
        a.status = "merged"
        b.status = "merged"
        self._history.append({"action": "merge", "a": model_a_id,
                              "b": model_b_id, "merged": merged.id})
        return merged

    # ─── Previsao e competencia ───────────────────────────

    def predict(self, model_id: str, problem: Dict) -> Dict:
        """Um modelo faz uma previsao sobre um problema."""
        model = self.models.get(model_id)
        if model is None or model.status != "active":
            return {"prediction": None, "confidence": 0.0}
        model.last_active_cycle = self._cycle
        # A previsao e uma composicao do bias com o problema
        confidence = model.confidence
        # Se o dominio do modelo bate com o problema, bonus
        if model.domain != "general" and problem.get("domain") == model.domain:
            confidence = min(1.0, confidence * 1.2)
        return {"prediction": model.bias.to_dict(),
                "confidence": confidence,
                "model_id": model_id}

    def record_outcome(self, model_id: str, problem: Dict,
                       correct: bool, cost: float = 1.0,
                       reward: float = 1.0):
        """Registra o resultado de uma previsao."""
        model = self.models.get(model_id)
        if model is None:
            return
        model.record_prediction(correct, cost, reward)
        # Auto-track capital distribution
        self._observed_capital_distribution.append(model.capital)
        if len(self._observed_capital_distribution) > 200:
            self._observed_capital_distribution = self._observed_capital_distribution[-200:]

    def form_coalition(self, model_ids: List[str], problem: str) -> Coalition:
        """Modelos formam coalizao para um problema."""
        biases = [self.models[mid].bias for mid in model_ids
                 if mid in self.models and self.models[mid].status == "active"]
        if not biases:
            coalition = Coalition(model_ids, problem)
            self.coalitions.append(coalition)
            return coalition
        # Combina biases
        combined = biases[0]
        for b in biases[1:]:
            combined = combined.combine(b, weight=0.5)
        coalition = Coalition(model_ids, problem)
        coalition.combined_bias = combined
        coalition.confidence = sum(m.confidence for m in
                                   [self.models[mid] for mid in model_ids
                                    if mid in self.models]) / max(1, len(model_ids))
        self.coalitions.append(coalition)
        self._history.append({"action": "coalition", "models": model_ids,
                              "problem": problem[:40]})
        return coalition

    # ─── Sono ─────────────────────────────────────────────

    def find_sleepers(self) -> List[EpistemicModel]:
        """Modelos com baixa atividade entram em sono."""
        timeout = self._auto_discover_sleep_timeout()
        return [m for m in self.models.values()
                if m.status == "active" and
                (self._cycle - m.last_active_cycle) > timeout]

    def sleep(self, model_id: str):
        """Poe um modelo em sono."""
        m = self.models.get(model_id)
        if m and m.status == "active":
            m.status = "dormant"
            m.last_sleep_cycle = self._cycle

    def wake_up(self, model_id: str):
        """Acorda um modelo."""
        m = self.models.get(model_id)
        if m and m.status == "dormant":
            m.status = "active"
            m.last_active_cycle = self._cycle

    def wake_up_for_topic(self, topic: str) -> List[EpistemicModel]:
        """Acorda modelos relevantes para um topico."""
        woken = []
        for m in self.models.values():
            if m.status == "dormant" and (m.domain == "general" or
                                          topic.lower() in m.domain.lower()):
                m.status = "active"
                m.last_active_cycle = self._cycle
                woken.append(m)
        return woken

    # ─── Ciclo principal ──────────────────────────────────

    def tick(self):
        """Ciclo do ecossistema:
           1. Reciclar modelos fracos
           2. Dividir modelos fortes
           3. Fusao de modelos similares
           4. Sono de modelos inativos
           5. Auto-cura se populacao muito baixa
        """
        self._cycle += 1
        active = [m for m in self.models.values() if m.status == "active"]
        if not active:
            # Auto-cura: se ecossistema vazio, semeia de novo
            self.seed_from_worldviews()
            return

        # 1. Reciclar modelos com capital muito baixo
        threshold = self._auto_discover_recycle_threshold()
        to_recycle = [m for m in active if m.capital < threshold]
        for m in to_recycle[:3]:  # limita para nao dizimar
            self.death(m.id, reason="low_capital")

        # 2. Dividir modelos muito bem-sucedidos
        successful = sorted(active, key=lambda m: -m.capital)
        for m in successful[:2]:
            if m.capital > 150 and (self._cycle - m.last_division_cycle) > 20:
                child = self.split(m.id)
                if child:
                    self._history.append({"action": "division",
                                          "parent": m.id, "child": child.id})

        # 3. Fusao de modelos similares
        active_now = [m for m in self.models.values() if m.status == "active"]
        for i in range(len(active_now)):
            for j in range(i+1, len(active_now)):
                a, b = active_now[i], active_now[j]
                if a.bias.distance(b.bias) < 0.1 and len(active_now) > 5:
                    self.merge(a.id, b.id)
                    break

        # 4. Sono de modelos inativos
        for m in self.find_sleepers():
            self.sleep(m.id)

        # 5. Auto-cura se populacao muito baixa
        active_count = sum(1 for m in self.models.values() if m.status == "active")
        if active_count < 3:
            self.seed_from_worldviews()

    # ─── Descoberta de novos veses ───────────────────────

    def discover_new_biases(self, problems: List[Dict]) -> List[str]:
        """Descobre dimensoes de vies que ainda nao existem.

        Quando o ecossistema nao consegue resolver bem um tipo de
        problema, isso sugere que falta uma dimensao de vies.
        """
        new_dimensions = []
        for problem in problems:
            # Conta o quao bem o ecossistema resolve
            related = [m for m in self.models.values()
                      if m.status == "active" and
                      (m.domain == "general" or problem.get("domain") == m.domain)]
            if not related:
                continue
            avg_accuracy = sum(m.accuracy for m in related) / len(related)
            if avg_accuracy < 0.3 and len(related) > 0:
                # Baixa precisao -> sugere que falta um vies
                # Cria nome baseado no tipo de problema
                new_dim = f"bias_{problem.get('type', 'novel')}"
                # Adiciona a todos os modelos ativos
                for m in related:
                    m.bias.add_dimension(new_dim, value=0.5)
                new_dimensions.append(new_dim)
        return new_dimensions

    def get_state(self) -> Dict:
        return {
            "n_models": len(self.models),
            "active": sum(1 for m in self.models.values() if m.status == "active"),
            "dormant": sum(1 for m in self.models.values() if m.status == "dormant"),
            "recycled": sum(1 for m in self.models.values() if m.status == "recycled"),
            "health": round(self.get_population_health(), 2),
            "diversity": round(self._observed_diversity, 3),
            "recent_events": self._history[-3:] if self._history else [],
        }
