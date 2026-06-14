"""
MAESTRO — Arquitetura baseada em Global Workspace Theory + Predictive Processing
================================================================================

Implementa a arquitetura do "maestro" que o usuário descreveu:
- DMN (Default Mode Network) gera hipóteses paralelas
- Salience Network seleciona a mais saliente
- Global Workspace mantém UMA hipótese ativa
- Executive Control monitora e decide quando parar
- Predictive Processing: cada ator prevê, erro de predição dirige atenção

Referências 2026:
- Baars (1988), Dehaene (2001), Shanahan (2006) — Global Workspace Theory
- Friston — Free Energy Principle, Active Inference
- Seeley, Menon — Salience Network, Triple Network Model
- Smallwood et al. (2025 Nature Neuro) — DMN cytoarchitecture
- VERSES.ai (2026) — Active Inference agents
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Hypothesis:
    """Uma hipótese de regra no Global Workspace."""
    rules: List[Dict]
    grids: List[np.ndarray]
    error: float
    surprise: float = 0.0  # prediction error
    salience: float = 0.0
    depth: int = 0
    parent: Optional['Hypothesis'] = None
    actor_source: str = ""
    timestamp: float = 0.0

    def copy_with(self, **kwargs) -> 'Hypothesis':
        h = Hypothesis(
            rules=self.rules + ([kwargs['rule']] if 'rule' in kwargs else []),
            grids=kwargs.get('grids', self.grids),
            error=kwargs.get('error', self.error),
            surprise=kwargs.get('surprise', self.surprise),
            salience=kwargs.get('salience', self.salience),
            depth=self.depth + (1 if 'rule' in kwargs else 0),
            parent=self,
            actor_source=kwargs.get('actor_source', self.actor_source),
            timestamp=time.time(),
        )
        return h


class Actor:
    """Um 'ator' paralelo — módulo especializado que gera hipóteses.

    Inspirado no GWT: cada processador inconsciente (percepção, simetria, etc.)
    compete por entrar no Global Workspace.
    """
    def __init__(self, name: str, fn: Callable, weight: float = 1.0):
        self.name = name
        self.fn = fn
        self.weight = weight  # viés de saliência deste ator
        self.calls = 0
        self.useful_calls = 0  # quantas vezes produziu hipótese que reduziu erro

    def process(self, current_grids, train, target_grids) -> List[Hypothesis]:
        """Processa o estado atual e retorna lista de hipóteses candidatas.

        Cada hipótese inclui:
        - rules (opcional)
        - new_grids
        - error (quão longe do target)
        - surprise (quanto mudou do input — "novidade")
        - actor_source (qual ator gerou)
        """
        self.calls += 1
        try:
            results = self.fn(current_grids, train, target_grids)
            hyps = []
            for r in results:
                h = Hypothesis(
                    rules=[],
                    grids=r['grids'],
                    error=r['error'],
                    surprise=r.get('surprise', 0.0),
                    salience=r.get('salience', 0.0) * self.weight,
                    actor_source=self.name,
                    timestamp=time.time(),
                )
                hyps.append(h)
            return hyps
        except Exception as e:
            return []


class SalienceNetwork:
    """Salience Network (Menon, Seeley) — decide o que entra no Global Workspace.

    Implementação:
    - Surprise (prediction error) → alta saliência
    - Novelty (diferença do que já foi tentado) → bônus
    - Goal-relevance (redução de erro) → bônus
    - Actor weight → viés do módulo

    No cérebro: anterior insula + dACC fazem isso.
    """
    def __init__(self, novelty_window: int = 50):
        self.history = deque(maxlen=1000)
        self.novelty_window = deque(maxlen=novelty_window)

    def compute_salience(self, hyp: Hypothesis, goal_error: float = 0.0) -> float:
        """Computa saliência de uma hipótese. Quanto maior, mais provável de entrar no Workspace."""
        # 1. Erro de predição (Friston) — surpresa
        surprise = hyp.surprise if hyp.surprise > 0 else hyp.error

        # 2. Redução do erro (relevância ao objetivo)
        if goal_error > 0:
            improvement = max(0, (goal_error - hyp.error) / goal_error)
        else:
            improvement = 0

        # 3. Novidade — quão diferente das hipóteses recentes
        novelty = self._novelty(hyp)

        # 4. Combina
        salience = (1.0 * surprise + 2.0 * improvement + 0.5 * novelty) * (1 + hyp.salience)

        return salience

    def _novelty(self, hyp: Hypothesis) -> float:
        """Quão diferente esta hipótese é das recentes."""
        if not self.novelty_window:
            return 1.0
        # novelty = 1 - (max_similarity_to_history)
        # similarity = 1 / (1 + abs(err_diff))
        max_sim = 0
        for past in self.novelty_window:
            sim = 1.0 / (1.0 + abs(past.error - hyp.error))
            if sim > max_sim:
                max_sim = sim
        return 1.0 - max_sim

    def update(self, hyp: Hypothesis):
        """Registra hipótese no histórico."""
        self.history.append(hyp)
        self.novelty_window.append(hyp)


class GlobalWorkspace:
    """Global Workspace (Baars, Dehaene) — APENAS UMA hipótese ativa por vez.

    No cérebro: 'palco' iluminado pelo spotlight da consciência.
    Aqui: o 'winner' do broadcasting entre atores paralelos.
    """
    def __init__(self, salience: SalienceNetwork):
        self.salience = salience
        self.current: Optional[Hypothesis] = None
        self.broadcast_count = 0
        self.max_broadcasts = 0  # 0 = sem limite
        self.ignition_threshold = 0.1  # quão menor a saliência precisa ser para "ignite"

    def compete(self, candidates: List[Hypothesis]) -> Optional[Hypothesis]:
        """Atores competem. A hipótese mais saliente 'ignite' no Workspace."""
        if not candidates:
            return self.current
        # Computa saliência para cada
        goal = self.current.error if self.current else 0.0
        scored = []
        for c in candidates:
            s = self.salience.compute_salience(c, goal_error=goal)
            c.salience = s
            scored.append((s, c))
        # Ordena por saliência
        scored.sort(key=lambda x: -x[0])
        winner_salience, winner = scored[0]
        # "Ignite" se saliência acima do threshold OU se é estritamente melhor que atual
        if self.current is None or winner.error < self.current.error or winner_salience > self.ignition_threshold:
            self.current = winner
            self.salience.update(winner)
            self.broadcast_count += 1
            return winner
        return self.current

    def is_solved(self) -> bool:
        return self.current is not None and self.current.error == 0


class ExecutiveControl:
    """Executive Control Network — o 'maestro' que monitora e decide parar.

    No cérebro: córtex pré-frontal dorsolateral, ACC.
    Funções:
    - Define objetivo (resolver task com error=0)
    - Decide estratégia (paralelo, sequencial, recursivo)
    - Monitora progresso (history.error_trend)
    - Decide parar (quando certainty > threshold OU budget exausto)
    """
    def __init__(self, max_iterations: int = 100, max_time_s: float = 60.0,
                 confidence_threshold: float = 0.99):
        self.goal_error = 0.0
        self.max_iterations = max_iterations
        self.max_time_s = max_time_s
        self.confidence_threshold = confidence_threshold
        self.iteration = 0
        self.start_time = 0
        self.error_history = deque(maxlen=20)
        self.useful_iterations = 0  # quantas iter. reduziram erro

    def start(self):
        self.start_time = time.time()
        self.iteration = 0
        self.error_history.clear()
        self.useful_iterations = 0

    def should_continue(self) -> bool:
        if self.iteration >= self.max_iterations:
            return False
        if time.time() - self.start_time > self.max_time_s:
            return False
        return True

    def observe(self, hyp: Optional[Hypothesis]):
        """Observa o estado atual e atualiza histórico."""
        self.iteration += 1
        if hyp is not None:
            if self.error_history and hyp.error < self.error_history[-1]:
                self.useful_iterations += 1
            self.error_history.append(hyp.error)

    def confidence(self) -> float:
        """Quão confiante estamos de que o current está bom o suficiente?"""
        if not self.error_history:
            return 0.0
        # confidence = 1 se erro=0; decai com erro
        last_err = self.error_history[-1]
        if last_err == 0:
            return 1.0
        # penaliza estagnação
        recent = list(self.error_history)[-5:]
        if len(recent) >= 2 and max(recent) - min(recent) < 0.001:
            # estagnado
            return min(0.5, 1.0 / (1.0 + last_err))
        return 1.0 / (1.0 + last_err)

    def is_satisfied(self) -> bool:
        return self.confidence() >= self.confidence_threshold


class DefaultModeNetwork:
    """Default Mode Network — gerador de contrafactuais e hipóteses alternativas.

    No cérebro: 30-50% do tempo de vigília em self-generated thought.
    Funções:
    - Imaginar alternativas ('e se X fosse diferente?')
    - Recuperação de memórias de situações similares
    - Simulação de futuros (NO_FUTURO)
    """
    def __init__(self, max_alternatives: int = 20):
        self.max_alternatives = max_alternatives
        self.memory = deque(maxlen=200)  # episodic memory of past hypotheses
        self.novelty_bonus = 1.5

    def generate_alternatives(self, current: Optional[Hypothesis],
                              actors: List[Actor]) -> List[Hypothesis]:
        """Gera hipóteses alternativas — 'e se fosse diferente?'"""
        alternatives = []
        # 1. Recupera memórias similares (episodic)
        for mem in list(self.memory)[-10:]:
            # Varia a memória (mutação leve)
            mutated = self._mutate(mem)
            alternatives.append(mutated)
        # 2. Atores paralelos (default mode 'brooding')
        # (já é feito pelos actors — aqui só anotamos)
        return alternatives[:self.max_alternatives]

    def _mutate(self, hyp: Hypothesis) -> Hypothesis:
        """Muta levemente a hipótese para criar alternativa."""
        import copy
        new_h = copy.deepcopy(hyp)
        if new_h.grids:
            # troca cor de uma célula aleatória
            import random
            r = random.randint(0, new_h.grids[0].shape[0] - 1)
            c = random.randint(0, new_h.grids[0].shape[1] - 1)
            current = int(new_h.grids[0][r, c])
            new_color = (current + random.choice([-1, 1, 2])) % 10
            new_h.grids[0][r, c] = new_color
        new_h.salience += self.novelty_bonus  # bônus por novidade
        return new_h

    def remember(self, hyp: Hypothesis):
        self.memory.append(hyp)


class PredictiveModel:
    """Predictive Processing (Friston) — modelo gerador + prediction error.

    No cérebro: cada região gera previsões top-down; erro de predição bottom-up
    dirige atenção (Friston's Free Energy Principle).
    """
    def __init__(self):
        self.predictions = {}  # cache

    def predict(self, actor_name: str, grid: np.ndarray) -> np.ndarray:
        """Prediz a saída esperada dado um ator."""
        return self.predictions.get((actor_name, hash(grid.tobytes())), grid)

    def compute_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Prediction error — surprise (Friston)."""
        if predicted.shape != actual.shape:
            return float('inf')
        return float(np.sum(predicted != actual))


class Maestro:
    """O MAESTRO — orquestra todos os módulos.

    Inspirado no exemplo da água do usuário:
    - DMN gerando paralelamente (cheiro, queda, barulho, contaminação)
    - Salience decidindo o que é relevante
    - Global Workspace focado em uma coisa
    - ECN monitorando e decidindo quando parar
    - Tudo flexível, context-dependent
    """
    def __init__(self, max_iterations: int = 100, max_time_s: float = 60.0):
        self.executive = ExecutiveControl(max_iterations, max_time_s)
        self.salience = SalienceNetwork()
        self.workspace = GlobalWorkspace(self.salience)
        self.dmn = DefaultModeNetwork()
        self.predictor = PredictiveModel()
        self.actors: List[Actor] = []
        self.iteration_log = []

    def add_actor(self, actor: Actor):
        self.actors.append(actor)

    def solve(self, train: List[Tuple[np.ndarray, np.ndarray]],
              test_in: np.ndarray,
              initial_actor_fn: Optional[Callable] = None) -> Optional[List[Dict]]:
        """Resolve uma task ARC usando a arquitetura do maestro.

        Fluxo:
        1. Inicializa Workspace com hipótese nula (input=output?)
        2. Para cada iteração:
           a. Todos os atores processam em paralelo (DMN)
           b. Salience ranqueia
           c. Winner entra no Workspace (broadcast)
           d. ECN observa e decide parar
        3. Retorna a hipótese vencedora
        """
        self.executive.start()
        initial_grids = [gi.copy() for gi, go in train]
        initial_err = sum(self._err(gi, go) for gi, go in train)

        # Inicializa workspace com hipótese nula
        null_hyp = Hypothesis(
            rules=[], grids=initial_grids, error=initial_err,
            surprise=0, salience=0, actor_source="null", timestamp=time.time()
        )
        self.workspace.current = null_hyp
        self.executive.observe(null_hyp)

        while self.executive.should_continue():
            iteration_start = time.time()

            # 1. DMN + atores paralelos geram hipóteses
            candidates = []
            for actor in self.actors:
                hyps = actor.process(
                    self.workspace.current.grids if self.workspace.current else initial_grids,
                    train,
                    [go for gi, go in train]
                )
                candidates.extend(hyps)

            # 2. Adiciona alternativas do DMN (contrafactuais)
            if self.workspace.current is not None:
                alternatives = self.dmn.generate_alternatives(self.workspace.current, self.actors)
                candidates.extend(alternatives)

            # 3. Salience ranqueia; winner entra no Workspace
            if candidates:
                winner = self.workspace.compete(candidates)
                if winner is not None and winner.error < (self.workspace.current.error if self.workspace.current else float('inf')):
                    self.dmn.remember(winner)
            else:
                winner = self.workspace.current

            # 4. ECN observa
            self.executive.observe(self.workspace.current)

            # 5. Verifica solução
            if self.workspace.is_solved():
                self.iteration_log.append({
                    'iter': self.executive.iteration,
                    'solved': True,
                    'rules': len(self.workspace.current.rules) if self.workspace.current.rules else 0,
                })
                return self.workspace.current.rules

            # 6. Verifica confiança
            if self.executive.is_satisfied():
                break

        # Retorna o melhor encontrado
        if self.workspace.current and self.workspace.current.rules:
            return self.workspace.current.rules
        return None

    @staticmethod
    def _err(pred, target):
        if pred.shape != target.shape:
            return 999999.0
        return float(np.sum(pred != target))


# ============================================================================
# FACTORY: cria maestro com atores padrão
# ============================================================================
def make_maestro_with_actors(actors_dict: Dict[str, Callable],
                              max_iterations: int = 100,
                              max_time_s: float = 60.0) -> Maestro:
    """Cria um Maestro com atores pré-configurados."""
    m = Maestro(max_iterations, max_time_s)
    for name, fn in actors_dict.items():
        m.add_actor(Actor(name, fn))
    return m


def count_modules():
    return {
        'actors_in_maestro': 'variable',
        'salience_network': 1,
        'global_workspace': 1,
        'executive_control': 1,
        'default_mode_network': 1,
        'predictive_model': 1,
    }
