"""
Goal System — Metas Persistentes com Error-Driven Learning

Fecho o loop AGI:
  1. GOAL: objetivo auto-gerado ou externo
  2. DECOMPOSE: quebra em submetas atingiveis
  3. ACT: Maestro executa acao via CGE
  4. OBSERVE: Grounding compara resultado com criterio de sucesso
  5. LEARN: Erro ajusta arestas do CGE (reforco/fraqueza)
  6. REPEAT: ate sucesso ou abandono
"""
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any


class Goal:
    def __init__(self, description: str, success_criteria: Dict = None,
                 parent: Optional[str] = None, priority: int = 5):
        self.id = str(uuid.uuid4())[:8]
        self.description = description
        self.state = "active"  # active | completed | failed | abandoned
        self.parent = parent
        self.subgoals: List[str] = []
        self.priority = priority  # 1-10, 10 = mais importante
        self.progress = 0.0
        self.created_at = time.time()
        self.completed_at: Optional[float] = None
        self.fail_count = 0
        self.attempts: List[Dict] = []
        self.success_criteria = success_criteria or {}
        # Evidencia de sucesso: observacoes que confirmam a meta
        self.supporting_evidence: List[str] = []
        self.contradicting_evidence: List[str] = []


class GoalManager:
    """
    Gerencia metas persistentes com ciclo completo de aprendizado por erro.
    """

    def __init__(self, cge=None, maestro=None):
        self.goals: Dict[str, Goal] = {}
        self.active_goal_id: Optional[str] = None
        self.completed_goals: List[str] = []
        self.cge = cge
        self.maestro = maestro
        self._history: List[Dict] = []

    def create_goal(self, description: str, success_criteria: Dict = None,
                    parent: Optional[str] = None, priority: int = 5) -> str:
        """Cria uma nova meta e retorna seu ID."""
        goal = Goal(description, success_criteria, parent, priority)
        self.goals[goal.id] = goal
        if parent and parent in self.goals:
            self.goals[parent].subgoals.append(goal.id)
        self._history.append({
            "event": "created", "goal_id": goal.id,
            "description": description, "time": time.time()
        })
        return goal.id

    def activate_next(self) -> Optional[str]:
        """Ativa a proxima meta viavel (prioridade + prontidao)."""
        candidates = [
            g for g in self.goals.values()
            if g.state == "active" and g.progress < 1.0
        ]
        if not candidates:
            # Se nao ha metas ativas, gerar meta de exploracao
            return self._generate_curiosity_goal()

        # Ordenar por prioridade (mais alta primeiro), depois por progresso
        candidates.sort(key=lambda g: (-g.priority, g.progress))
        self.active_goal_id = candidates[0].id
        return self.active_goal_id

    def _generate_curiosity_goal(self) -> str:
        """Gera meta de curiosidade: entender algo desconhecido no CGE."""
        if self.cge is not None:
            # Encontrar no no CGE com menos conexoes (mais isolado = mais curioso)
            nodes = list(self.cge.nodes.keys())
            if nodes:
                # No com menos vizinhos
                scores = []
                for nid in nodes:
                    n_edges = sum(1 for e in self.cge.edges
                                  if e["from"] == nid or e["to"] == nid)
                    scores.append((n_edges, nid))
                scores.sort()
                most_isolated = scores[0][1]
                gid = self.create_goal(
                    f"explore_{most_isolated}",
                    {"min_connections": 3},
                    priority=3
                )
                return gid
        return self.create_goal("explore_environment", priority=2)

    def evaluate(self, goal_id: str, observation: Dict) -> float:
        """
        Avalia progresso de uma meta baseado em observacao.
        Retorna score 0.0 (falha) a 1.0 (sucesso).

        success_criteria define o que conta como sucesso.
        Ex: {"min_similarity": 0.7, "target_node": "dog"}
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return 0.0

        criteria = goal.success_criteria
        if not criteria:
            # Sem criterio explicito: usar prediction error do Maestro
            if self.maestro:
                pe = self.maestro.neuromodulator.as_dict().get("NA", 0)
                score = 1.0 - min(1.0, pe)
            else:
                score = 0.5
        else:
            score = self._compute_score(criteria, observation)

        # Registrar tentativa
        goal.attempts.append({
            "score": score,
            "observation": observation,
            "time": time.time()
        })

        # Atualizar estado
        if score >= 0.9:
            goal.state = "completed"
            goal.completed_at = time.time()
            goal.progress = 1.0
            self.completed_goals.append(goal_id)
        elif score < 0.2:
            goal.fail_count += 1
            if goal.fail_count >= 3:
                goal.state = "failed"
        else:
            goal.progress = max(goal.progress, score)

        self._history.append({
            "event": "evaluate", "goal_id": goal_id,
            "score": score, "state": goal.state, "time": time.time()
        })
        return score

    def _compute_score(self, criteria: Dict, observation: Dict) -> float:
        """Computa score baseado em criterios.

        Cada criterio pode ser:
          - valor numerico: tratado como minimo (observed >= target → 1.0,
            senao 0.0)
          - dict com chave 'min': threshold minimo
          - dict com chave 'exact': valor exato (erro relativo)
          - string: match exato
        """
        score = 0.0
        n_criteria = 0
        for key, target in criteria.items():
            observed = observation.get(key)
            if observed is None:
                continue
            n_criteria += 1

            # Dict criteria
            if isinstance(target, dict):
                if "min" in target:
                    score += 1.0 if observed >= target["min"] else 0.0
                elif "exact" in target:
                    t = target["exact"]
                    if isinstance(t, (int, float)) and isinstance(observed, (int, float)):
                        if t != 0:
                            score += 1.0 - min(1.0, abs(observed - t) / abs(t))
                        else:
                            score += 1.0 if abs(observed) < 0.01 else 0.0
                    else:
                        score += 1.0 if observed == t else 0.0
                else:
                    # Se dict mas sem chave conhecida, tratar valor unico
                    val = list(target.values())[0]
                    if isinstance(val, (int, float)):
                        score += 1.0 if observed >= val else 0.0
                    else:
                        score += 1.0 if observed == val else 0.0

            elif isinstance(target, (int, float)) and isinstance(observed, (int, float)):
                # Por padrao, tratar como threshold minimo
                score += 1.0 if observed >= target else 0.0

            else:
                # Match exato para strings
                score += 1.0 if observed == target else 0.0

        return score / max(1, n_criteria)

    def apply_error_signal(self, goal_id: str, score: float,
                           path_taken: List[Tuple[str, str]] = None):
        """
        ERROR-DRIVEN LEARNING: fortalece edges que levaram ao sucesso,
        enfraquece edges que levaram ao fracasso.

        E o analogo ao sinal de dopamina no cerebro: sucesso → fortalece
        sinapses que levaram a ele; fracasso → enfraquece.
        """
        if self.cge is None or not path_taken:
            return

        # Dopamine: learning rate modulado por quao inesperado foi o resultado
        dopamine = score - 0.5  # -0.5 (fracasso) a +0.5 (sucesso)
        lr = 0.1 * abs(dopamine)  # Mais aprendizado quando surpreendente

        for from_id, to_id in path_taken:
            for edge in self.cge.edges:
                if edge["from"] == from_id and edge["to"] == to_id:
                    old_conf = edge.get("confidence", 1.0)
                    # Atualizacao hebbiana com sinal de erro
                    adjustment = dopamine * lr
                    new_conf = max(0.05, min(1.0, old_conf + adjustment))
                    edge["confidence"] = new_conf

                    # Atualizar confianca dos nos tambem
                    if from_id in self.cge.nodes:
                        nc = self.cge.nodes[from_id].get("confidence", 1.0)
                        self.cge.nodes[from_id]["confidence"] = max(
                            0.05, min(1.0, nc + adjustment * 0.5))
                    if to_id in self.cge.nodes:
                        nc = self.cge.nodes[to_id].get("confidence", 1.0)
                        self.cge.nodes[to_id]["confidence"] = max(
                            0.05, min(1.0, nc + adjustment * 0.5))

    def get_active_goal(self) -> Optional[Goal]:
        """Retorna a meta ativa atual."""
        if self.active_goal_id and self.active_goal_id in self.goals:
            return self.goals[self.active_goal_id]
        return None

    def get_state(self) -> Dict:
        """Estado para debug/UI."""
        return {
            "total_goals": len(self.goals),
            "active": self.active_goal_id,
            "completed": len(self.completed_goals),
            "failed": sum(1 for g in self.goals.values() if g.state == "failed"),
            "active_goal_desc": self.get_active_goal().description[:60]
            if self.get_active_goal() else None,
            "goals": {gid: {"desc": g.description[:40], "state": g.state,
                            "progress": round(g.progress, 2), "fails": g.fail_count}
                      for gid, g in self.goals.items()},
        }
