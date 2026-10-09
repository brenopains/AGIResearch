"""
CognitiveProgram — Programas Cognitivos como Entidades de Primeira Classe

Um programa cognitivo pode ser:
  um loop, uma estrategia, um habito, um vies, um protocolo,
  uma heuristica, um ritual

Cada programa e um no no CGE com:
  - HDV proprio (similaridade entre programas)
  - gatilhos (contextos onde deve ser ativado)
  - efeitos (modulacao de atencao, erro, memoria, simulacao)
  - historico de performance
  - confianca
  - versoes

Programas sao CRIADOS pela Forja (abstracao de experiencias),
AVALIADOS pelo LoopEvaluator, e APOSENTADOS quando nao performam.
"""
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class CognitiveProgram:
    """Um programa cognitivo — entidade de primeira classe no CGE.

    Atributos:
      id: identificador unico
      hdv_hash: hash do HDV para similaridade (se CGE nao disponivel)
      trigger_context: padrao de contexto que ativa este programa
      effect_profile: dict de como afeta atencao/erro/simulacao/etc
      performance: historico de recompensas
      confidence: 0..1, quao confiavel
      versions: lista de versoes anteriores
      parent_id: se foi derivado de outro programa
      created_at: timestamp
    """
    id: str
    hdv_hash: str = ""
    trigger_context: Dict[str, float] = field(default_factory=dict)
    effect_profile: Dict[str, float] = field(default_factory=dict)
    performance: List[float] = field(default_factory=list)
    confidence: float = 0.5
    versions: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    domain_tags: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    total_trials: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    retired: bool = False

    @property
    def avg_performance(self) -> float:
        if not self.performance:
            return 0.0
        return sum(self.performance[-20:]) / max(1, len(self.performance[-20:]))

    def record_outcome(self, reward: float):
        self.performance.append(reward)
        self.total_trials += 1
        self.last_used = time.time()
        # Atualizar confianca (media movel)
        n = len(self.performance[-20:])
        self.success_rate = sum(1 for r in self.performance[-20:] if r > 0.5) / max(1, n)
        self.confidence = self.confidence * 0.9 + self.success_rate * 0.1

    def matches_context(self, context: Dict[str, float],
                        threshold: float = 0.6) -> float:
        """Quanto este programa e adequado para o contexto atual."""
        if not self.trigger_context:
            return 0.3  # programa universal
        matches = 0
        total = 0
        for key, val in context.items():
            if key in self.trigger_context:
                matches += 1.0 - min(1.0, abs(val - self.trigger_context[key]))
                total += 1
        return matches / max(1, total) if total > 0 else 0.3


class ProgramForge:
    """Cria programas cognitivos por abstracao de experiencias.

    Quando uma sequencia de acoes leva consistentemente a bons resultados,
    a Forja pode abstrai-la em um novo CognitiveProgram.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.experience_buffer: List[Dict] = deque(maxlen=500)
        self.patterns: Dict[str, List[Dict]] = {}

    def record_episode(self, context: Dict, actions: List[str],
                       reward: float, outcome: Dict):
        """Registra um episodio para futura abstracao."""
        self.experience_buffer.append({
            "context": context,
            "actions": actions,
            "reward": reward,
            "outcome": outcome,
            "time": time.time(),
        })

    def detect_patterns(self) -> List[Dict]:
        """Detecta padroes recorrentes no buffer de experiencias."""
        patterns = []

        # 1. Detectar padroes por sequencia de acoes (independente de contexto)
        action_seqs = {}
        for exp in self.experience_buffer:
            seq_key = "_".join(exp["actions"])
            if seq_key not in action_seqs:
                action_seqs[seq_key] = []
            action_seqs[seq_key].append(exp["reward"])

        for seq_key, rewards in action_seqs.items():
            if len(rewards) >= 2 and sum(rewards) / len(rewards) > 0.5:
                ctx = {"novelty": 0.5, "error": 0.5}  # contexto universal
                patterns.append({
                    "context_prefix": self._context_hash(ctx),
                    "action_sequence": seq_key.split("_")[:5],
                    "avg_reward": sum(rewards) / len(rewards),
                    "count": len(rewards),
                })

        # 2. Detectar padroes por contexto similar
        contexts_seen = {}
        for exp in self.experience_buffer:
            ctx_key = self._context_hash(exp["context"])
            if ctx_key not in contexts_seen:
                contexts_seen[ctx_key] = []
            contexts_seen[ctx_key].append(exp)

        for ctx_key, episodes in contexts_seen.items():
            if len(episodes) < 2:
                continue
            seq_groups = {}
            for ep in episodes:
                seq_key = "_".join(ep["actions"][:3])
                if seq_key not in seq_groups:
                    seq_groups[seq_key] = []
                seq_groups[seq_key].append(ep["reward"])

            for seq_key, rewards in seq_groups.items():
                if len(rewards) >= 2 and sum(rewards) / len(rewards) > 0.5:
                    patterns.append({
                        "context_prefix": ctx_key,
                        "action_sequence": seq_key.split("_"),
                        "avg_reward": sum(rewards) / len(rewards),
                        "count": len(rewards),
                    })

        return patterns

    def forge_program(self, pattern: Dict, parent_id: Optional[str] = None) -> CognitiveProgram:
        """Cria um CognitiveProgram a partir de um padrao detectado."""
        prog_id = f"prog_{pattern['context_prefix'][:8]}_{int(time.time())}"
        effect_profile = self._actions_to_effects(pattern["action_sequence"])
        trigger_context = self._parse_context_prefix(pattern["context_prefix"])
        program = CognitiveProgram(
            id=prog_id,
            trigger_context=trigger_context,
            effect_profile=effect_profile,
            confidence=0.3,  # comeca baixo
            parent_id=parent_id,
        )
        return program

    def _context_hash(self, ctx: Dict) -> str:
        """Hash do contexto com quantizacao para agregar similares."""
        return "_".join(f"{k}_{round(v * 2) / 2:.1f}"
                       for k, v in sorted(ctx.items())[:4])

    def _parse_context_prefix(self, prefix: str) -> Dict:
        parts = prefix.split("_")
        ctx = {}
        for i in range(0, len(parts) - 1, 2):
            try:
                ctx[parts[i]] = float(parts[i + 1])
            except (ValueError, IndexError):
                pass
        return ctx

    def _actions_to_effects(self, actions: List[str]) -> Dict[str, float]:
        """Mapeia uma sequencia de acoes para perfil de efeitos."""
        effects = {
            "exploration_weight": 0.5,
            "error_sensitivity": 0.5,
            "simulation_depth": 0.5,
            "certainty_threshold": 0.5,
            "attention_width": 0.5,
        }
        for a in actions:
            if a in ("explore", "MOVE", "measure"):
                effects["exploration_weight"] = min(1.0, effects["exploration_weight"] + 0.2)
                effects["certainty_threshold"] = max(0.0, effects["certainty_threshold"] - 0.1)
            elif a in ("hypothesize", "predict", "TEST"):
                effects["simulation_depth"] = min(1.0, effects["simulation_depth"] + 0.2)
                effects["error_sensitivity"] = min(1.0, effects["error_sensitivity"] + 0.1)
            elif a in ("consolidate", "reflect"):
                effects["attention_width"] = max(0.0, effects["attention_width"] - 0.2)
                effects["certainty_threshold"] = min(1.0, effects["certainty_threshold"] + 0.1)
        return effects

    def get_state(self) -> Dict:
        return {
            "buffer_size": len(self.experience_buffer),
            "patterns_detected": len(self.patterns),
        }


class LoopEvaluator:
    """Avalia e evolui programas cognitivos.

    Mantem uma biblioteca de programas, mede desempenho,
    decide quais manter, especializar, combinar ou aposentar.
    """

    def __init__(self, max_programs: int = 32):
        self.programs: Dict[str, CognitiveProgram] = {}
        self.max_programs = max_programs
        self.selection_history: List[Dict] = []

    def register(self, program: CognitiveProgram) -> str:
        """Registra um programa na biblioteca."""
        if len(self.programs) >= self.max_programs:
            self._prune()
        self.programs[program.id] = program
        return program.id

    def select(self, context: Dict[str, float],
               mode: str = "best") -> Optional[CognitiveProgram]:
        """Seleciona o melhor programa para o contexto."""
        candidates = [p for p in self.programs.values()
                     if not p.retired and p.confidence > 0.2]

        if not candidates:
            return None

        # Pontuar cada programa
        scored = []
        for p in candidates:
            match = p.matches_context(context)
            perf = p.avg_performance
            conf = p.confidence
            score = match * 0.4 + perf * 0.4 + conf * 0.2
            scored.append((score, p))

        if mode == "explore":
            # As vezes explora programas menos usados
            if random.random() < 0.15:
                return random.choice(candidates)

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        self.selection_history.append({
            "context": context,
            "selected": best.id,
            "score": scored[0][0],
        })
        return best

    def evaluate_outcome(self, program_id: str, reward: float):
        """Atualiza programa com resultado da execucao."""
        prog = self.programs.get(program_id)
        if prog:
            prog.record_outcome(reward)

    def specialize(self, base_id: str, domain: str) -> Optional[str]:
        """Cria uma versao especializada de um programa para um dominio."""
        base = self.programs.get(base_id)
        if not base:
            return None
        new_id = f"{base_id}_{domain}_{int(time.time())}"
        new_prog = CognitiveProgram(
            id=new_id,
            trigger_context=dict(base.trigger_context),
            effect_profile=dict(base.effect_profile),
            parent_id=base_id,
            confidence=base.confidence * 0.8,
            domain_tags=[domain],
            versions=[base_id],
        )
        base.versions.append(new_id)
        self.register(new_prog)
        return new_id

    def combine(self, id_a: str, id_b: str) -> Optional[str]:
        """Combina dois programas em um novo (crossover)."""
        a = self.programs.get(id_a)
        b = self.programs.get(id_b)
        if not a or not b:
            return None
        new_id = f"combo_{id_a[:6]}_{id_b[:6]}_{int(time.time())}"
        # Misturar trigger_context e effect_profile
        mixed_triggers = {}
        for k in set(a.trigger_context) | set(b.trigger_context):
            va = a.trigger_context.get(k, 0.5)
            vb = b.trigger_context.get(k, 0.5)
            mixed_triggers[k] = (va + vb) / 2

        mixed_effects = {}
        for k in set(a.effect_profile) | set(b.effect_profile):
            va = a.effect_profile.get(k, 0.5)
            vb = b.effect_profile.get(k, 0.5)
            mixed_effects[k] = (va + vb) / 2

        new_prog = CognitiveProgram(
            id=new_id,
            trigger_context=mixed_triggers,
            effect_profile=mixed_effects,
            parent_id=id_a,
        )
        self.register(new_prog)
        return new_id

    def retire(self, program_id: str):
        """Aposenta um programa com baixa performance."""
        prog = self.programs.get(program_id)
        if prog:
            prog.retired = True
            prog.confidence *= 0.3

    def _prune(self):
        """Remove os programas com pior desempenho."""
        scored = [(p.avg_performance, p_id) for p_id, p in self.programs.items()
                 if not p.retired]
        scored.sort()
        # Remover os 25% piores
        n_remove = max(1, len(scored) // 4)
        for _, p_id in scored[:n_remove]:
            self.retire(p_id)

    def get_state(self) -> Dict:
        active = [p for p in self.programs.values() if not p.retired]
        return {
            "total_programs": len(self.programs),
            "active": len(active),
            "retired": sum(1 for p in self.programs.values() if p.retired),
            "best": max(active, key=lambda p: p.avg_performance).id
                     if active else None,
        }
