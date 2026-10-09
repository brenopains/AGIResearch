"""
CognitiveStances — Perfis de Reconfiguracao Cognitiva Global

Uma Stance nao e um operador nem um programa.
E uma reconfiguracao de TODO o sistema cognitivo:
  atencao, erro, simulacao, memoria, confianca, operadores

Exemplos humanos:
  "E se eu estiver completamente errado?" → duvida radical
  "Resolva como se tivesse apenas uma acao" → minimalismo
  "Pense como uma crianca" → exploracao pura
"""
import time, uuid, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class StanceProfile:
    """Perfil completo de configuracao cognitiva.

    Cada campo controla um aspecto do sistema.
    Valores: 0..1 (salvo rare exceptions).
    """
    # Identificacao
    id: str = ""
    name: str = ""
    description: str = ""

    # Atencao e busca
    search_radius: float = 0.5       # 0=foco local, 1=busca global
    attention_width: float = 0.5     # 0=foco unico, 1=multiplas threads
    exploration_rate: float = 0.15   # chance de experimentar modo novo

    # Erro e confianca
    error_sensitivity: float = 0.5    # 0=ignora erro, 1=reage fortemente
    confidence_threshold: float = 0.5 # 0=decide rapido, 1=so decide com certeza
    novelty_weight: float = 0.5      # 0=ignora novidade, 1=busca novidade

    # Simulacao e memoria
    simulation_depth: float = 0.5    # 0=superficial, 1=profunda
    memory_weight: float = 0.5       # 0=ignora passado, 1=usa historia
    plan_horizon: float = 0.5        # 0=corto prazo, 1=longo prazo

    # Operadores e criacao
    operator_bias: str = "balanced"  # "explore", "exploit", "reflect", "create", "balanced"
    forge_rate: float = 0.5          # 0=nao forja, 1=forja agressivamente

    # Metacognicao
    self_monitoring: float = 0.5     # 0=age sem pensar, 1=reflete sempre
    speed_priority: float = 0.5      # 0=qualidade maxima, 1=velocidade maxima

    # Metadados
    primordial: bool = False
    confidence: float = 0.5
    n_applications: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.__dict__.items() if not k.startswith("_")}

    def apply_to(self, meta_controller) -> List[str]:
        """Aplica esta stance nos parametros de um MetaController.
        Retorna lista de parametros que foram alterados."""
        changes = []
        if hasattr(meta_controller, 'search_radius'):
            old = meta_controller.search_radius
            meta_controller.search_radius = self.search_radius
            changes.append(f"search_radius: {old:.2f}->{self.search_radius:.2f}")
        if hasattr(meta_controller, 'exploration_rate'):
            old = meta_controller.exploration_rate
            meta_controller.exploration_rate = self.exploration_rate
            changes.append(f"exploration_rate: {old:.2f}->{self.exploration_rate:.2f}")
        if hasattr(meta_controller, 'min_mode_time'):
            old = meta_controller.min_mode_time
            meta_controller.min_mode_time = 0.5 + self.speed_priority * 2.0
            changes.append(f"min_mode_time: {old:.2f}->{meta_controller.min_mode_time:.2f}")
        if hasattr(meta_controller, 'impasse_threshold'):
            old = meta_controller.impasse_threshold
            meta_controller.impasse_threshold = max(2, int(6 * (1 - self.error_sensitivity * 0.5)))
            changes.append(f"impasse_threshold: {old}->{meta_controller.impasse_threshold}")
        return changes


# ─── STANCES PRIMORDIAIS ─────────────────────────────────

PRIMORDIAL_STANCES = [
    StanceProfile(
        name="Modo Einstein",
        description="Pensamento profundo, muita simulacao, alta abstração",
        search_radius=0.3, attention_width=0.2,
        exploration_rate=0.05, error_sensitivity=0.8,
        confidence_threshold=0.8, novelty_weight=0.3,
        simulation_depth=0.9, memory_weight=0.7,
        plan_horizon=0.8, operator_bias="reflect",
        forge_rate=0.3, self_monitoring=0.8,
        speed_priority=0.1, primordial=True, confidence=0.9,
    ),
    StanceProfile(
        name="Modo Hacker",
        description="Exploracao rapida, baixa confianca, testes frequentes",
        search_radius=0.8, attention_width=0.7,
        exploration_rate=0.4, error_sensitivity=0.3,
        confidence_threshold=0.2, novelty_weight=0.9,
        simulation_depth=0.3, memory_weight=0.3,
        plan_horizon=0.2, operator_bias="explore",
        forge_rate=0.2, self_monitoring=0.2,
        speed_priority=0.9, primordial=True, confidence=0.85,
    ),
    StanceProfile(
        name="Modo Professor",
        description="Explicar, simplificar, conectar conceitos, usar renderizadores",
        search_radius=0.5, attention_width=0.6,
        exploration_rate=0.1, error_sensitivity=0.6,
        confidence_threshold=0.7, novelty_weight=0.4,
        simulation_depth=0.5, memory_weight=0.8,
        plan_horizon=0.5, operator_bias="exploit",
        forge_rate=0.6, self_monitoring=0.7,
        speed_priority=0.3, primordial=True, confidence=0.8,
    ),
    StanceProfile(
        name="Duvida Radical",
        description="'E se eu estiver completamente errado?' — questiona tudo",
        search_radius=0.7, attention_width=0.5,
        exploration_rate=0.3, error_sensitivity=0.9,
        confidence_threshold=0.1, novelty_weight=0.6,
        simulation_depth=0.7, memory_weight=0.2,
        plan_horizon=0.3, operator_bias="reflect",
        forge_rate=0.7, self_monitoring=0.9,
        speed_priority=0.2, primordial=True, confidence=0.75,
    ),
    StanceProfile(
        name="Minimalismo",
        description="'Resolva como se tivesse apenas uma acao' — simplicidade forcada",
        search_radius=0.1, attention_width=0.1,
        exploration_rate=0.05, error_sensitivity=0.7,
        confidence_threshold=0.8, novelty_weight=0.1,
        simulation_depth=0.2, memory_weight=0.6,
        plan_horizon=0.1, operator_bias="exploit",
        forge_rate=0.1, self_monitoring=0.3,
        speed_priority=0.7, primordial=True, confidence=0.7,
    ),
    StanceProfile(
        name="Pensamento Infantil",
        description="Curiosidade pura, sem preconceitos, explora sem medo de errar",
        search_radius=0.9, attention_width=0.9,
        exploration_rate=0.6, error_sensitivity=0.1,
        confidence_threshold=0.0, novelty_weight=1.0,
        simulation_depth=0.1, memory_weight=0.1,
        plan_horizon=0.0, operator_bias="explore",
        forge_rate=0.8, self_monitoring=0.0,
        speed_priority=1.0, primordial=True, confidence=0.8,
    ),
]


class StanceLibrary:
    """Biblioteca de Stances. Descobre novas minerando CognitiveTraces."""

    def __init__(self, meta_controller=None, cge=None):
        self.meta = meta_controller
        self.cge = cge
        self.stances: Dict[str, StanceProfile] = {}
        self.active_stance_id: Optional[str] = None
        self.history: List[Dict] = []

        # Carregar primordiais
        for s in PRIMORDIAL_STANCES:
            sid = f"stance_{hashlib.md5(s.name.encode()).hexdigest()[:8]}"
            s.id = sid
            self.stances[sid] = s

    def select_for_context(self, context: str, mode: str = "") -> Optional[StanceProfile]:
        """Seleciona a melhor stance para o contexto atual."""
        candidates = list(self.stances.values())
        if not candidates:
            return None

        scored = []
        for s in candidates:
            score = s.confidence * (0.5 + s.success_rate)
            # Favorecer stances que combinam com o modo atual
            if mode:
                bias_map = {
                    "explore": "explore", "hypothesize": "reflect",
                    "consolidate": "exploit", "reflect": "reflect",
                    "exploit": "exploit", "predict": "reflect",
                }
                expected = bias_map.get(mode, "balanced")
                if s.operator_bias == expected:
                    score *= 1.3
                elif s.operator_bias == "balanced":
                    score *= 1.1
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def apply_stance(self, stance_id: str, meta_controller=None) -> List[str]:
        """Aplica uma stance no MetaController."""
        s = self.stances.get(stance_id)
        if s is None:
            return []
        mc = meta_controller or self.meta
        if mc is None:
            return []
        changes = s.apply_to(mc)
        self.active_stance_id = stance_id
        self.history.append({
            "time": time.time(), "stance": s.name,
            "changes": changes, "mode": mc.mode if mc.mode else "?",
        })
        return changes

    def discover_from_traces(self, traces: List[Dict]) -> List[StanceProfile]:
        """ForjaEvolution chama isso para descobrir novas stances."""
        discovered = []
        if len(traces) < 5:
            return discovered

        # Agrupar traces por configuracao de parametros similar
        # (proxy: agrupar por modo medio + reward)
        configs: Dict[str, List[float]] = {}
        for td in traces:
            steps = td.get("steps", []) if isinstance(td, dict) else []
            if not steps:
                continue
            rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]
            modes = [s.get("mode", "") for s in steps if isinstance(s, dict)]
            if not modes:
                continue
            avg_r = sum(rewards) / max(1, len(rewards))
            # Chave: combinação dos modos mais frequentes
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            top_modes = sorted(mode_counts, key=mode_counts.get, reverse=True)[:2]
            key = "+".join(top_modes) if top_modes else "unknown"

            configs.setdefault(key, []).append(avg_r)

        # Para cada configuracao com alta recompensa consistente, criar stance
        for key, rewards in configs.items():
            if len(rewards) < 3:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r < 0.5:
                continue
            # So criar se ainda nao existe similar
            modes = key.split("+")
            bias = "balanced"
            if all(m in ("explore", "test") for m in modes):
                bias = "explore"
            elif all(m in ("consolidate", "reflect") for m in modes):
                bias = "reflect"
            elif all(m in ("exploit", "predict") for m in modes):
                bias = "exploit"

            existing = [s for s in self.stances.values() if s.operator_bias == bias]
            if existing and any(s.confidence > 0.6 for s in existing):
                continue

            sid = f"stance_{hashlib.md5(key.encode()).hexdigest()[:8]}"
            new_s = StanceProfile(
                id=sid,
                name=f"Stance {key}",
                description=f"Descoberta de traces: modo(s) {key}",
                search_radius=0.5 + (0.4 if "explore" in key else -0.2),
                exploration_rate=0.3 if "explore" in key else 0.1,
                novelty_weight=0.7 if "explore" in key else 0.3,
                operator_bias=bias,
                confidence=min(0.7, avg_r * 0.8),
                success_rate=avg_r,
                n_applications=len(rewards),
            )
            self.stances[sid] = new_s
            discovered.append(new_s)

        return discovered

    def random_stance(self) -> Optional[StanceProfile]:
        """Escolhe uma stance aleatoria (exploracao)."""
        if not self.stances:
            return None
        return random.choice(list(self.stances.values()))

    def get_active(self) -> Optional[StanceProfile]:
        if self.active_stance_id:
            return self.stances.get(self.active_stance_id)
        return None

    def get_state(self) -> Dict:
        return {
            "n_stances": len(self.stances),
            "active": self.get_active().name if self.get_active() else None,
            "primordial": sum(1 for s in self.stances.values() if s.primordial),
            "discovered": sum(1 for s in self.stances.values() if not s.primordial),
            "recent": self.history[-3:] if self.history else [],
        }
