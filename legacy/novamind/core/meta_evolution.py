"""
MetaEvolution — Auto-mutacao Controlada da Arquitetura

O sistema trata seus proprios mecanismos cognitivos como objetos
de estudo, propoe modificacoes localizadas, testa-as em ambiente
controlado, mede ganhos e incorpora apenas mudancas com beneficio.

Ciclo:
  1. DESCREVE: estado atual de todos os modulos
  2. MUTA: propoe alteracao em 1-2 parametros
  3. TESTA: executa N ciclos com a mutacao
  4. AVALIA: mediu melhora?
  5. COMMIT: se sim, torna permanente
  6. APRENDE: que tipos de mutacao funcionam
"""
import time, random, math, copy, hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MutationProposal:
    """Uma mutacao proposta para a arquitetura."""
    id: str = ""
    target_module: str = ""     # "meta_controller", "workspace", "stance", "attention"
    target_param: str = ""       # "search_radius", "exploration_rate", etc.
    current_value: float = 0.0
    proposed_value: float = 0.0
    mutation_type: str = ""      # "tweak", "toggle", "promote", "create"
    description: str = ""
    created_at: float = field(default_factory=time.time)

    def apply(self, meta, workspace, stances) -> bool:
        """Aplica a mutacao. Retorna True se bem-sucedido."""
        try:
            if self.target_module == "meta_controller" and meta:
                if hasattr(meta, self.target_param):
                    setattr(meta, self.target_param, self.proposed_value)
                    return True
            elif self.target_module == "workspace" and workspace:
                if self.target_param == "capacity":
                    workspace.capacity = int(self.proposed_value)
                    return True
            elif self.target_module == "stances" and stances:
                if self.target_param == "exploration_rate":
                    for s in stances.stances.values():
                        s.exploration_rate = self.proposed_value
                    return True
            return False
        except:
            return False

    def rollback(self, meta, workspace, stances) -> bool:
        """Reverte a mutacao."""
        try:
            if self.target_module == "meta_controller" and meta:
                if hasattr(meta, self.target_param):
                    setattr(meta, self.target_param, self.current_value)
                    return True
            elif self.target_module == "workspace" and workspace:
                if self.target_param == "capacity":
                    workspace.capacity = int(self.current_value)
                    return True
            elif self.target_module == "stances" and stances:
                if self.target_param == "exploration_rate":
                    for s in stances.stances.values():
                        s.exploration_rate = self.current_value
                    return True
            return False
        except:
            return False


@dataclass
class MutationResult:
    """Resultado de testar uma mutacao."""
    proposal_id: str = ""
    baseline_score: float = 0.0
    mutated_score: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    test_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ValueSystem:
    """Sistema de Valor: aprende o que vale a pena estudar/simular/esquecer.

    Nao e apenas reward de RL. E um vetor de importancia que reorganiza
    atencao, memoria, workspace e forja baseado em:
      - utilidade esperada
      - custo computacional
      - potencial de generalizacao
      - raridade
      - alinhamento com metas
    """

    def __init__(self, cge=None):
        self.cge = cge
        # Valor aprendido por conceito/dominio
        self._value_map: Dict[str, float] = {}
        self._cost_map: Dict[str, float] = {}

    def estimate_value(self, concept: str, domain: str = "general") -> float:
        """Estima o valor de estudar um conceito."""
        key = f"{domain}:{concept}"
        base = self._value_map.get(key, 0.3)
        # Bonus para conceitos que conectam dominios diferentes
        if self.cge and concept in self.cge.nodes:
            connection_count = sum(1 for e in self.cge.edges
                                 if concept in (e["from"], e["to"]))
            base += min(0.4, connection_count * 0.02)
        return min(1.0, base)

    def record_outcome(self, concept: str, domain: str, reward: float,
                       cost: float = 0.5):
        """Atualiza valor baseado em resultado real."""
        key = f"{domain}:{concept}"
        old = self._value_map.get(key, 0.3)
        self._value_map[key] = old * 0.9 + reward * 0.1
        old_cost = self._cost_map.get(key, 0.5)
        self._cost_map[key] = old_cost * 0.9 + cost * 0.1

    def get_priorities(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retorna os topicos de maior valor."""
        scored = sorted(self._value_map.items(),
                       key=lambda x: -x[1] / max(0.1, self._cost_map.get(x[0], 0.5)))
        return [(k[:40], v) for k, v in scored[:top_k]]


class MetaEvolution:
    """Controla a evolucao da arquitetura.

    Ciclo:
      1. snapshot: captura estado atual dos modulos
      2. propose: gera 1-2 mutacoes
      3. test: aplica e mede impacto
      4. commit/reject: baseado em melhora
      5. learn: atualiza modelo de que mutacoes funcionam
    """

    def __init__(self, meta_controller=None, workspace=None,
                 stance_library=None, cge=None, value_system=None):
        self.meta = meta_controller
        self.workspace = workspace
        self.stances = stance_library
        self.cge = cge
        self.values = value_system or ValueSystem(cge)

        # Historico de mutacoes
        self.mutation_history: List[MutationResult] = []
        self._active_proposal: Optional[MutationProposal] = None
        self._test_mode: bool = False
        self._test_cycles: int = 0
        self._test_duration: int = 5  # ciclos de teste

        # Modelo de aprendizado: que tipo de mutacao funciona
        self._mutation_model: Dict[str, List[float]] = defaultdict(list)
        # (tipo, modulo) -> [improvements]

        self._last_evolution_time: float = 0
        self._evolution_interval: float = 60.0  # segundos

    def tick(self, cycle: int, recent_rewards: List[float]) -> Optional[str]:
        """Um ciclo do MetaEvolution.

        A cada N segundos, propoe uma mutacao, testa e commita.
        """
        now = time.time()
        if now - self._last_evolution_time < self._evolution_interval:
            return None
        self._last_evolution_time = now

        # Se esta testando uma mutacao
        if self._test_mode and self._active_proposal:
            self._test_cycles += 1
            if self._test_cycles >= self._test_duration:
                return self._finish_test(recent_rewards)
            return f"testando: {self._active_proposal.description[:40]}..."

        # Propor nova mutacao
        proposal = self._propose_mutation()
        if proposal is None:
            return None

        # Aplicar e iniciar teste
        self._active_proposal = proposal
        self._test_mode = True
        self._test_cycles = 0
        self._baseline_rewards = list(recent_rewards)

        success = proposal.apply(self.meta, self.workspace, self.stances)
        if not success:
            self._test_mode = False
            self._active_proposal = None
            return None

        return f"mutacao: {proposal.description[:50]}..."

    def _propose_mutation(self) -> Optional[MutationProposal]:
        """Propoe uma mutacao para a arquitetura."""
        if self.meta is None:
            return None

        # Parametros mutaveis do MetaController
        tunable = [
            ("meta_controller", "search_radius", self.meta.search_radius, 0.5,
             random.choice([0.1, 0.3, 0.7, 0.9])),
            ("meta_controller", "exploration_rate", self.meta.exploration_rate, 0.15,
             random.uniform(0.05, 0.6)),
            ("meta_controller", "min_mode_time", self.meta.min_mode_time, 1.0,
             random.uniform(0.3, 3.0)),
            ("meta_controller", "impasse_threshold", float(self.meta.impasse_threshold), 4.0,
             float(random.choice([2, 3, 5, 6, 8]))),
        ]

        # Escolher parametro baseado em aprendizado
        # (favorecer tipos de mutacao que ja funcionaram)
        weights = []
        for mod, param, cur, default, prop in tunable:
            key = f"{mod}:{param}"
            hist = self._mutation_model.get(key, [])
            w = 1.0
            if hist:
                avg_imp = sum(hist) / len(hist)
                w = 0.5 + avg_imp * 2.0  # mais peso se historico positivo
            weights.append(max(0.1, w))

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        idx = random.choices(range(len(tunable)), weights=weights)[0]
        mod, param, cur, default, prop = tunable[idx]

        # Nao propor se for igual ao atual
        if abs(prop - cur) < 0.01:
            prop = default

        mutation_type = "tweak"
        desc = f"{mod}.{param}: {cur:.2f} -> {prop:.2f}"

        return MutationProposal(
            id=f"mut_{hashlib.md5(desc.encode()).hexdigest()[:8]}",
            target_module=mod, target_param=param,
            current_value=cur, proposed_value=prop,
            mutation_type=mutation_type, description=desc,
        )

    def _finish_test(self, recent_rewards: List[float]) -> str:
        """Finaliza o teste de uma mutacao."""
        if not self._active_proposal:
            return "sem mutacao ativa"

        proposal = self._active_proposal
        baseline = sum(self._baseline_rewards) / max(1, len(self._baseline_rewards))
        mutated = sum(recent_rewards) / max(1, len(recent_rewards))
        improvement = mutated - baseline

        result = MutationResult(
            proposal_id=proposal.id,
            baseline_score=baseline,
            mutated_score=mutated,
            improvement=improvement,
            accepted=improvement > 0.05,
            test_duration=self._test_duration,
        )
        self.mutation_history.append(result)

        # Registrar no modelo de aprendizado
        key = f"{proposal.target_module}:{proposal.target_param}"
        self._mutation_model[key].append(improvement)
        if len(self._mutation_model[key]) > 20:
            self._mutation_model[key] = self._mutation_model[key][-20:]

        if result.accepted:
            # Mutacao aceita: manter
            msg = (f"[OK] MUTACAO ACEITA: {proposal.description} "
                   f"(melhoria={improvement:.3f})")
            # Armazenar no CGE
            if self.cge:
                nid = f"_evolution_{proposal.id}"
                import torch
                g = torch.Generator(); g.manual_seed(42)
                self.cge.create_node(nid, torch.sign(torch.randn(10000, generator=g)))
                if nid in self.cge.nodes:
                    self.cge.nodes[nid]["type"] = "evolution"
                    self.cge.nodes[nid]["evolution_data"] = {
                        "proposal": proposal.description,
                        "improvement": improvement,
                        "accepted": True,
                    }
        else:
            # Mutacao rejeitada: reverter
            proposal.rollback(self.meta, self.workspace, self.stances)
            msg = (f"[--] MUTACAO REJEITADA: {proposal.description} "
                   f"(delta={improvement:.3f})")

        self._test_mode = False
        self._active_proposal = None
        return msg

    def get_state(self) -> Dict:
        return {
            "n_mutations": len(self.mutation_history),
            "acceptance_rate": sum(1 for m in self.mutation_history if m.accepted)
                               / max(1, len(self.mutation_history)),
            "avg_improvement": round(sum(m.improvement for m in self.mutation_history)
                                     / max(1, len(self.mutation_history)), 4),
            "testing": self._test_mode,
            "mutation_model": {k: round(sum(v)/len(v), 3) for k, v
                              in self._mutation_model.items() if v},
        }
