"""
ForjaEvolution — Minera CognitiveTraces para descobrir algoritmos,
estrategias, teorias, heuristicas, stances e METARREGRAS DE CRIACAO.

Hierarquia de descoberta:
  HDVs → Conceitos → Operadores
    → ALGORITMOS → ESTRATEGIAS → TEORIAS
    → HEURISTICAS → STANCES
    → METAFORJA (regras de criacao)
      → Auto-evolucao da propria Forja
"""
import time, uuid, hashlib, math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class DiscoveredAlgorithm:
    """Um algoritmo completo descoberto de traces.

    Avaliacao MULTI-DIMENSIONAL:
      - avg_reward: recompensa media
      - avg_cost: custo computacional (inverso da eficiencia)
      - generalization: quao bem generaliza para outros dominios
      - robustness: resiliencia a variacao
      - rarity: raridade (algoritmos raros podem ser valiosos em dominios unicos)
    """
    id: str = ""
    name: str = ""
    mode_sequence: List[str] = field(default_factory=list)
    operator_sequence: List[str] = field(default_factory=list)
    avg_reward: float = 0.0
    avg_cost: float = 0.5      # 0=eficiente, 1=caro
    generalization: float = 0.3  # 0=especifico, 1=universal
    robustness: float = 0.5     # 0=fragil, 1=robusto
    rarity: float = 0.5        # 0=comum, 1=raro
    complexity: float = 0.5    # 0=simples, 1=complexo (Navalha de Occam)

    n_occurrences: int = 0
    domains: List[str] = field(default_factory=list)
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)

    @property
    def multi_dim_value(self) -> float:
        """Valor multi-dimensional: reward * generalizacao * robustez * simplicidade / custo."""
        simplicity = 1.0 - self.complexity
        return (self.avg_reward * self.generalization * self.robustness * simplicity *
                (1.0 / max(0.1, self.avg_cost)))

    def to_program_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name,
            "trigger": {"mode": self.mode_sequence[0] if self.mode_sequence else ""},
            "steps": self.mode_sequence,
            "avg_reward": self.avg_reward,
            "confidence": self.confidence,
            "domains": self.domains,
        }


@dataclass
class DiscoveredParadigm:
    """Paradigma: nivel acima de teorias.

    Um paradigma muda COMO as teorias sao construidas.
    Ex: "paradigma de campos", "paradigma de agentes", "paradigma estatistico"

    Enquanto uma teoria explica um dominio, um paradigma define
    a abordagem para construir teorias.
    """
    id: str = ""
    name: str = ""
    description: str = ""
    theory_ids: List[str] = field(default_factory=list)
    core_approach: str = ""  # "field", "agent", "statistical", "symbolic", "hybrid"
    confidence: float = 0.3
    n_theories: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name,
                "approach": self.core_approach,
                "confidence": self.confidence,
                "n_theories": self.n_theories}


@dataclass
class DiscoveredHeuristic:
    """Heuristica condicional do tipo 'se X entao Y'."""
    id: str = ""
    condition: str = ""
    action: str = ""
    confidence: float = 0.3
    n_observations: int = 0
    success_rate: float = 0.0
    domain: str = "general"

    def to_insight(self) -> str:
        return f"Se {self.condition}, entao {self.action} (conf={self.confidence:.2f})"


@dataclass
class DiscoveredStrategy:
    """Estrategia: conjunto de algoritmos que co-ocorrem com sucesso."""
    id: str = ""
    name: str = ""
    algorithm_ids: List[str] = field(default_factory=list)
    domain: str = "general"
    success_rate: float = 0.0
    confidence: float = 0.3


@dataclass
class DiscoveredTheory:
    """Teoria: estrategia que domina um dominio inteiro."""
    id: str = ""
    name: str = ""
    domain: str = ""
    strategy_ids: List[str] = field(default_factory=list)
    concepts_explained: List[str] = field(default_factory=list)
    confidence: float = 0.2
    competing_theories: List[str] = field(default_factory=list)


class ForjaEvolution:
    """Forja que minera CognitiveTraces para descobrir algoritmos,
    estrategias, teorias e heuristicas.

    Pipeline de mineracao:
      1. mine_algorithms(traces) → descobre sequencias de sucesso
      2. mine_heuristics(traces) → descobre regras condicionais
      3. build_strategies(algorithms, heuristics) → agrupa em estrategias
      4. build_theories(strategies, cge) → constroi teorias de dominio
    """

    def __init__(self, cge=None, insight_library=None,
                 loop_evaluator=None, operator_forge=None):
        self.cge = cge
        self.insight_library = insight_library
        self.loop_evaluator = loop_evaluator
        self.operator_forge = operator_forge

        # Descobertas acumuladas
        self.algorithms: Dict[str, DiscoveredAlgorithm] = {}
        self.heuristics: Dict[str, DiscoveredHeuristic] = {}
        self.strategies: Dict[str, DiscoveredStrategy] = {}
        self.theories: Dict[str, DiscoveredTheory] = {}
        self.paradigms: Dict[str, DiscoveredParadigm] = {}

        self.discovery_log: List[Dict] = []
        self._last_mine_time: float = 0
        self._mine_interval: float = 30.0  # segundos entre mineracoes

    def mine_traces(self, traces: List[Any], force: bool = False) -> Dict:
        """Ponto de entrada principal: minera todos os traces disponiveis.

        Args:
            traces: lista de CognitiveTrace ou dicts
            force: se True, minera mesmo sem intervalo

        Returns:
            dict com contagem de descobertas
        """
        now = time.time()
        if not force and now - self._last_mine_time < self._mine_interval:
            return {"mined": False, "reason": "interval"}
        if len(traces) < 3:
            return {"mined": False, "reason": "few_traces"}

        self._last_mine_time = now
        results = {"algorithms": 0, "heuristics": 0,
                   "strategies": 0, "theories": 0, "paradigms": 0}

        # 1. Extrair dados estruturados dos traces
        trace_data = self._extract_trace_data(traces)

        # 2. Minerar algoritmos (sequencias de modo/operador)
        new_algs = self._mine_algorithms(trace_data)
        results["algorithms"] = len(new_algs)

        # 3. Minerar heuristicas (regras condicionais)
        new_heurs = self._mine_heuristics(trace_data)
        results["heuristics"] = len(new_heurs)

        # 4. Construir estrategias
        if len(self.algorithms) >= 2:
            new_strats = self._build_strategies()
            results["strategies"] = len(new_strats)

        # 5. Construir teorias
        if len(self.strategies) >= 2 and self.cge:
            new_theories = self._build_theories()
            results["theories"] = len(new_theories)

        # 6. Construir paradigmas (acima de teorias)
        if len(self.theories) >= 2:
            new_pars = self._build_paradigms()
            results["paradigms"] = len(new_pars)

        return results

    def _extract_trace_data(self, traces: List[Any]) -> Dict:
        """Extrai dados estruturados de CognitiveTraces."""
        sequences = []  # [(mode_seq, reward, domains)]
        mode_rewards = defaultdict(list)
        context_modes = defaultdict(list)
        transition_success = defaultdict(list)

        for t in traces:
            if hasattr(t, 'to_dict'):
                td = t.to_dict()
            else:
                td = t if isinstance(t, dict) else {}

            steps = td.get("steps", [])
            if not steps:
                continue

            mode_seq = [s.get("mode", "") for s in steps if isinstance(s, dict)]
            rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]
            novelty = [s.get("novelty", 0) for s in steps if isinstance(s, dict)]
            avg_r = sum(rewards) / max(1, len(rewards))
            domains = [td.get("problem", "general")[:20]]

            if mode_seq:
                sequences.append((mode_seq, avg_r, domains, rewards, novelty))

            # Agregar por modo
            for i, (mode, r) in enumerate(zip(mode_seq, rewards)):
                mode_rewards[mode].append(r)
                if i > 0:
                    transition_success[f"{mode_seq[i-1]}->{mode}"].append(r)

            # Contextos onde cada modo apareceu
            first_mode = mode_seq[0] if mode_seq else ""
            if first_mode:
                context_modes[first_mode].append({
                    "avg_reward": avg_r,
                    "novelty": sum(novelty) / max(1, len(novelty)) if novelty else 0,
                })

        return {
            "sequences": sequences,
            "mode_rewards": dict(mode_rewards),
            "context_modes": dict(context_modes),
            "transition_success": dict(transition_success),
        }

    def _mine_algorithms(self, td: Dict) -> List[str]:
        """Minera traces para descobrir algoritmos (sequencias de sucesso)."""
        new_ids = []
        sequences = td.get("sequences", [])

        # Contar bigramas e trigramas de modo com recompensa
        seq_patterns = defaultdict(list)
        for mode_seq, avg_r, domains, rewards, _ in sequences:
            for n in [2, 3, 4]:
                for i in range(len(mode_seq) - n + 1):
                    gram = tuple(mode_seq[i:i+n])
                    seq_patterns[gram].append({
                        "reward": avg_r,
                        "domains": domains,
                        "rewards": rewards[i:i+n] if len(rewards) >= i+n else [],
                    })

        # Filtrar padroes promissores
        for gram, instances in seq_patterns.items():
            if len(instances) < 2:
                continue
            avg_r = sum(i["reward"] for i in instances) / len(instances)
            if avg_r < 0.4:
                continue

            # Verificar se ja existe
            name = "_".join(gram)
            existing = [a for a in self.algorithms.values()
                       if a.mode_sequence == list(gram)]
            if existing:
                existing[0].n_occurrences += len(instances)
                existing[0].avg_reward = (existing[0].avg_reward + avg_r) / 2
                existing[0].confidence = min(1.0, existing[0].confidence + 0.05)
                continue

            domains = list(set(d for i in instances for d in i["domains"]))
            # Metricas multi-dimensionais
            n_domains = len(domains)
            reward_std = (max(i["reward"] for i in instances) - min(i["reward"] for i in instances)) if len(instances) > 1 else 0
            alg = DiscoveredAlgorithm(
                id=f"alg_{hashlib.md5(name.encode()).hexdigest()[:8]}",
                name=name,
                mode_sequence=list(gram),
                avg_reward=round(avg_r, 3),
                avg_cost=round(0.3 + len(gram) * 0.1, 3),  # sequencias longas custam mais
                generalization=round(min(1.0, n_domains * 0.2), 3),
                robustness=round(max(0.3, 1.0 - reward_std), 3),
                rarity=round(max(0.1, 1.0 - len(instances) * 0.05), 3),
                complexity=round(min(1.0, len(gram) * 0.15), 3),  # sequencias longas = complexas
                n_occurrences=len(instances),
                domains=domains,
                confidence=min(1.0, avg_r * 0.5 + len(instances) * 0.05),
            )
            self.algorithms[alg.id] = alg
            new_ids.append(alg.id)

            # Sincronizar com loop_evaluator como CognitiveProgram
            if self.loop_evaluator:
                from .cognitive_program import CognitiveProgram
                prog = CognitiveProgram(
                    id=alg.id,
                    trigger_context={"novelty": 0.5, "error": 1.0 - avg_r},
                    effect_profile={"exploration_weight": 0.5 if "explore" in gram else 0.3},
                    domain_tags=domains,
                    confidence=alg.confidence,
                )
                for _ in range(len(instances)):
                    prog.record_outcome(avg_r)
                self.loop_evaluator.register(prog)
                self.loop_evaluator.specialize(alg.id, domains[0]) if domains else None

            self.discovery_log.append({
                "time": time.time(), "type": "algorithm",
                "name": name, "reward": avg_r, "occurrences": len(instances),
            })

        return new_ids

    def _mine_heuristics(self, td: Dict) -> List[str]:
        """Minera traces para descobrir heuristicas condicionais."""
        new_ids = []
        transitions = td.get("transition_success", {})
        mode_rewards = td.get("mode_rewards", {})

        # Heuristica 1: transicoes com alta recompensa
        for transition, rewards in transitions.items():
            if len(rewards) < 2:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r < 0.5:
                continue
            parts = transition.split("->")
            if len(parts) != 2:
                continue
            h = DiscoveredHeuristic(
                id=f"heur_{hashlib.md5(transition.encode()).hexdigest()[:8]}",
                condition=f"modo {parts[0]} estagnou",
                action=f"mudar para {parts[1]}",
                confidence=min(1.0, avg_r * 0.6 + len(rewards) * 0.03),
                n_observations=len(rewards),
                success_rate=avg_r,
            )
            existing_id = f"heur_{hashlib.md5(transition.encode()).hexdigest()[:8]}"
            if existing_id not in self.heuristics:
                self.heuristics[existing_id] = h
                new_ids.append(existing_id)

                # Sincronizar com InsightLibrary
                if self.insight_library:
                    self.insight_library.add_insight(
                        principle=h.to_insight(),
                        confidence=h.confidence,
                        tags=["heuristic", "transition"] + parts,
                        domain="general",
                    )

        # Heuristica 2: modos que funcionam em contextos especificos
        context_modes = td.get("context_modes", {})
        for mode, contexts in context_modes.items():
            if len(contexts) < 3:
                continue
            avg_novelty = sum(c.get("novelty", 0) for c in contexts) / len(contexts)
            avg_r = sum(c.get("avg_reward", 0) for c in contexts) / len(contexts)
            if avg_r < 0.5:
                continue
            ctx_type = "alta novidade" if avg_novelty > 0.5 else "baixa novidade"
            h = DiscoveredHeuristic(
                id=f"heur_{hashlib.md5((mode+ctx_type).encode()).hexdigest()[:8]}",
                condition=f"contexto tem {ctx_type}",
                action=f"usar modo {mode.upper()}",
                confidence=min(1.0, avg_r * 0.5),
                n_observations=len(contexts),
                success_rate=avg_r,
            )
            hid = h.id
            if hid not in self.heuristics:
                self.heuristics[hid] = h
                new_ids.append(hid)
                if self.insight_library:
                    self.insight_library.add_insight(
                        principle=h.to_insight(),
                        confidence=h.confidence,
                        tags=["heuristic", "context", mode],
                    )

        return new_ids

    def _build_strategies(self) -> List[str]:
        """Agrupa algoritmos que co-ocorrem em estrategias."""
        new_ids = []
        # Agrupar por dominios compartilhados
        domain_algs = defaultdict(list)
        for alg in self.algorithms.values():
            for d in alg.domains:
                domain_algs[d].append(alg)

        for domain, algs in domain_algs.items():
            if len(algs) < 2:
                continue
            # So criar estrategia se os algoritmos sao complementares
            all_modes = set()
            for a in algs:
                all_modes.update(a.mode_sequence)
            if len(all_modes) < 3:
                continue
            sid = f"strat_{hashlib.md5(domain.encode()).hexdigest()[:8]}"
            if sid in self.strategies:
                continue
            avg_conf = sum(a.confidence for a in algs) / len(algs)
            strat = DiscoveredStrategy(
                id=sid,
                name=f"Estrategia para {domain[:20]}",
                algorithm_ids=[a.id for a in algs],
                domain=domain,
                success_rate=sum(a.avg_reward for a in algs) / len(algs),
                confidence=avg_conf,
            )
            self.strategies[sid] = strat
            new_ids.append(sid)
            if self.cge:
                self._store_in_cge(strat)

        return new_ids

    def _build_theories(self) -> List[str]:
        """Constroi teorias a partir de estrategias que dominam dominios."""
        new_ids = []
        domain_strats = defaultdict(list)
        for s in self.strategies.values():
            domain_strats[s.domain].append(s)

        for domain, strats in domain_strats.items():
            if len(strats) < 1:
                continue
            tid = f"theory_{hashlib.md5(domain.encode()).hexdigest()[:8]}"
            if tid in self.theories:
                continue
            # Encontrar conceitos no CGE deste dominio
            concepts = []
            if self.cge:
                for nid, node in self.cge.nodes.items():
                    if not nid.startswith("_"):
                        nd = node if isinstance(node, dict) else {}
                        if nd.get("domain", "") == domain or domain in str(nid):
                            concepts.append(nid)
            avg_conf = sum(s.confidence for s in strats) / len(strats)
            theory = DiscoveredTheory(
                id=tid,
                name=f"Teoria sobre {domain[:20]}",
                domain=domain,
                strategy_ids=[s.id for s in strats],
                concepts_explained=concepts[:20],
                confidence=avg_conf,
            )
            self.theories[tid] = theory
            new_ids.append(tid)
            if self.cge:
                self._store_in_cge(theory)

        return new_ids

    def _build_paradigms(self) -> List[str]:
        """Constroi paradigmas acima de teorias.

        Um paradigma e uma abordagem que gera multiplas teorias.
        Ex: "paradigma de campos" gera teorias sobre gravidade, eletromagnetismo.
        """
        new_ids = []
        if len(self.theories) < 2:
            return new_ids

        # Agrupar teorias por abordagem (core_approach)
        approach_groups = defaultdict(list)
        for t in self.theories.values():
            approach = t.domain[:15]  # simplificado: dominio como proxy de abordagem
            approach_groups[approach].append(t)

        for approach, theories in approach_groups.items():
            if len(theories) < 2:
                continue
            pid = f"paradigm_{hashlib.md5(approach.encode()).hexdigest()[:8]}"
            if pid in self.paradigms:
                continue
            success_rates = [t.confidence for t in theories]
            paradigm = DiscoveredParadigm(
                id=pid,
                name=f"Paradigma de {approach[:20]}",
                description=f"Abordagem explicativa para {len(theories)} teorias "
                           f"relacionadas a {approach[:30]}",
                theory_ids=[t.id for t in theories],
                core_approach=approach[:15],
                confidence=sum(success_rates) / len(success_rates),
                n_theories=len(theories),
                success_rate=sum(success_rates) / len(success_rates),
            )
            self.paradigms[pid] = paradigm
            new_ids.append(pid)
            if self.cge:
                nid = paradigm.id
                import torch
                g = torch.Generator()
                g.manual_seed(int(hashlib.md5(nid.encode()).hexdigest()[:8], 16))
                vec = torch.sign(torch.randn(10000, generator=g))
                self.cge.create_node(nid, vec)
                if nid in self.cge.nodes:
                    self.cge.nodes[nid]["type"] = "paradigm"
                    self.cge.nodes[nid]["paradigm_data"] = paradigm.to_dict()

        return new_ids

    def _store_in_cge(self, item):
        """Armazena descoberta no CGE."""
        if self.cge is None:
            return
        import torch
        nid = item.id
        g = torch.Generator()
        g.manual_seed(int(hashlib.md5(nid.encode()).hexdigest()[:8], 16))
        vec = torch.sign(torch.randn(10000, generator=g))
        self.cge.create_node(nid, vec)
        if nid in self.cge.nodes:
            nd = self.cge.nodes[nid]
            if isinstance(item, DiscoveredStrategy):
                nd["type"] = "strategy"
                nd["strategy_data"] = {"name": item.name, "domain": item.domain,
                                       "algorithms": item.algorithm_ids,
                                       "success_rate": item.success_rate}
            elif isinstance(item, DiscoveredTheory):
                nd["type"] = "theory"
                nd["theory_data"] = {"name": item.name, "domain": item.domain,
                                     "strategies": item.strategy_ids,
                                     "concepts": item.concepts_explained[:10]}

    def get_state(self) -> Dict:
        return {
            "algorithms": len(self.algorithms),
            "heuristics": len(self.heuristics),
            "strategies": len(self.strategies),
            "theories": len(self.theories),
            "paradigms": len(self.paradigms),
            "recent": self.discovery_log[-5:] if self.discovery_log else [],
            "last_mine": self._last_mine_time,
        }

    def get_best_algorithms(self, top_k: int = 5, by: str = "reward") -> List[Dict]:
        if by == "multi_dim":
            sorted_algs = sorted(self.algorithms.values(),
                                key=lambda a: -a.multi_dim_value)
        else:
            sorted_algs = sorted(self.algorithms.values(),
                                key=lambda a: -a.avg_reward * a.confidence)
        return [a.to_program_dict() for a in sorted_algs[:top_k]]

    def get_best_heuristics(self, top_k: int = 5) -> List[str]:
        sorted_h = sorted(self.heuristics.values(),
                         key=lambda h: -h.confidence)
        return [h.to_insight() for h in sorted_h[:top_k]]


# ═══════════════════════════════════════════════════════════════
# METAFORJA — Descobre Regras de Criacao
# ═══════════════════════════════════════════════════════════════
#
# Enquanto a Forja descobre operadores e a ForjaEvolution descobre
# algoritmos, a METAFORJA descobre PADROES sobre QUANDO criar.
#
# Exemplos de meta-regras que ela pode descobrir:
#   "Sempre que Observe+Compare+Predict aparece, crie anticipate"
#   "Quando novelty > 0.7, explorar operadores 'explore' rende mais"
#   "Toda vez que um padrao aparece 3x com reward > 0.6, crie operador"
#
# Essas regras sao armazenadas como CognitivePrograms e aplicadas
# automaticamente pelo OperatorForge.

@dataclass
class ForgeRule:
    """Uma regra de criacao descoberta pela MetaForja."""
    id: str = ""
    condition: str = ""        # descricao da condicao
    action: str = ""           # o que criar
    trigger_pattern: List[str] = field(default_factory=list)  # padrao que ativa
    target_type: str = "operator"  # "operator", "program", "stance"
    confidence: float = 0.3
    n_triggers: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id, "condition": self.condition,
                "action": self.action, "type": self.target_type,
                "confidence": self.confidence, "success_rate": self.success_rate}


class MetaForja:
    """Meta-Forja: descobre regras de criacao minerando:

    1. O discovery_log do OperatorForge (quando e por que operadores foram criados)
    2. CognitiveTraces (que contextos precedem criacoes bem-sucedidas)
    3. StanceLibrary (que stances favorecem criacao)

    As regras descobertas sao armazenadas como CognitivePrograms
    que o OperatorForge consulta para decidir se deve criar algo.
    """

    def __init__(self, operator_forge=None, loop_evaluator=None,
                 stance_library=None, cge=None, insight_library=None):
        self.operator_forge = operator_forge
        self.loop_evaluator = loop_evaluator
        self.stance_library = stance_library
        self.cge = cge
        self.insight_library = insight_library
        self.rules: Dict[str, ForgeRule] = {}
        self._last_mine = 0.0

    def mine(self, traces: List[Any], forge_log: List[Dict]) -> List[ForgeRule]:
        """Minera traces + log da forja para descobrir regras de criacao."""
        new_rules = []
        if len(forge_log) < 3:
            return new_rules

        # 1. Analisar o discovery_log do OperatorForge
        successful_creations = [e for e in forge_log if e.get("status") == "promoted"]
        if len(successful_creations) < 2:
            return new_rules

        # 2. Extrair padroes: que contexto precedeu criacoes bem-sucedidas
        creation_contexts = []
        for entry in successful_creations:
            seq = entry.get("sequence", [])
            if seq:
                creation_contexts.append(seq)

        # 3. Descobrir regras baseadas em sequencias recorrentes
        seq_counts = defaultdict(int)
        for seq in creation_contexts:
            key = "_".join(seq[:3])
            seq_counts[key] += 1

        for seq_key, count in seq_counts.items():
            ops = seq_key.split("_")
            if len(ops) < 2 or count < 2:
                continue
            rid = f"meta_{hashlib.md5(seq_key.encode()).hexdigest()[:8]}"
            if rid in self.rules:
                continue
            rule = ForgeRule(
                id=rid,
                condition=f"sequencia {seq_key} aparece {count}x com sucesso",
                action=f"criar operador composto: {seq_key}",
                trigger_pattern=ops,
                target_type="operator",
                confidence=min(0.7, count * 0.1),
                n_triggers=count,
                success_rate=0.5 + count * 0.05,
            )
            self.rules[rid] = rule
            new_rules.append(rule)

            # Sincronizar com InsightLibrary
            if self.insight_library:
                self.insight_library.add_insight(
                    principle=(
                        f"Quando uma sequencia de operadores ({seq_key}) "
                        f"aparece {count}x com sucesso, criar um operador composto"
                    ),
                    confidence=rule.confidence,
                    tags=["metaforja", "creation_rule"],
                )

        # 4. Descobrir regras de traces com alta recompensa
        if traces:
            high_reward_traces = []
            for t in traces:
                td = t.to_dict() if hasattr(t, 'to_dict') else (t if isinstance(t, dict) else {})
                steps = td.get("steps", [])
                if steps:
                    rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]
                    if rewards and sum(rewards) / len(rewards) > 0.6:
                        high_reward_traces.append(td)

            if len(high_reward_traces) >= 3:
                rid = f"meta_gen_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
                rule = ForgeRule(
                    id=rid,
                    condition=f"{len(high_reward_traces)} traces com reward > 0.6",
                    action="forjar novo programa cognitivo destes padroes",
                    trigger_pattern=["high_reward"],
                    target_type="program",
                    confidence=min(0.6, len(high_reward_traces) * 0.08),
                    n_triggers=len(high_reward_traces),
                )
                self.rules[rid] = rule
                new_rules.append(rule)

        return new_rules

    def should_create(self, context: Dict) -> Optional[ForgeRule]:
        """Verifica se alguma regra deve ser ativada no contexto atual."""
        for rule in self.rules.values():
            if rule.confidence < 0.3:
                continue
            # Checar se padrao corresponde ao contexto
            pattern = rule.trigger_pattern
            if not pattern:
                continue
            context_ops = context.get("recent_operators", [])
            if any(p in context_ops for p in pattern):
                return rule
            if "high_reward" in pattern and context.get("reward", 0) > 0.6:
                return rule
        return None

    def get_state(self) -> Dict:
        return {
            "rules": len(self.rules),
            "recent": [r.to_dict() for r in list(self.rules.values())[-3:]],
        }
