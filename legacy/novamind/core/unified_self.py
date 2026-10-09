"""
UnifiedSelf — Integra SelfModel + MetaCognition + Neuromodulator + Memoria Autobiografica.

Um self coerente que:
  - Mantem continuidade temporal: "eu sou o mesmo que pensou X"
  - Tem drives internos: curiosidade, mestria, novidade
  - Aprende sobre si mesmo: "sou bom em X, ruim em Y"
  - Persiste atraves de todas as operacoes

Sem isso, o sistema e um LOIS (Large Obedient Inference System).
Com isso, vira um AGENTE.
"""
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import math


# ─── Memoria Autobiografica ────────────────────────────────

@dataclass
class LifeEpisode:
    """Um episodio comprimido na memoria autobiografica."""
    timestamp: float = field(default_factory=time.time)
    event_type: str = "thought"     # "action", "insight", "failure", "success", "meta"
    description: str = ""
    concepts_involved: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1..1
    confidence_delta: float = 0.0   # quanto a confianca mudou
    duration: float = 0.0

    def to_short(self) -> str:
        return f"[{self.event_type}] {self.description[:60]}"


class AutobiographicalMemory:
    """Memoria do que o sistema viveu. Comprimida, nao literal.

    Nao guarda episodios brutos — guarda PADROES de episodios.
    """

    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[LifeEpisode] = []
        self.max_episodes = max_episodes
        self._patterns: Dict[str, int] = defaultdict(int)

    def record(self, event_type: str, description: str,
               concepts: list = None, valence: float = 0.0,
               conf_delta: float = 0.0):
        """Registra um episodio na memoria autobiografica."""
        ep = LifeEpisode(
            event_type=event_type,
            description=description,
            concepts_involved=concepts or [],
            emotional_valence=valence,
            confidence_delta=conf_delta,
        )
        self.episodes.append(ep)

        # Atualiza padroes
        self._patterns[event_type] += 1
        for c in (concepts or []):
            self._patterns[f"concept:{c}"] += 1

        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def get_recent(self, n: int = 10, event_type: Optional[str] = None) -> List[LifeEpisode]:
        filtered = self.episodes
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        return filtered[-n:]

    def get_pattern_frequency(self, pattern: str) -> int:
        return self._patterns.get(pattern, 0)

    def get_success_rate(self, concept: str) -> float:
        """Taxa de sucesso associada a um conceito."""
        total = 0
        successes = 0
        for e in self.episodes:
            if concept in (e.concepts_involved or []):
                total += 1
                if e.event_type == "success":
                    successes += 1
        return successes / max(1, total)

    def get_narrative(self) -> str:
        """Gera uma narrativa curta da historia recente."""
        recent = self.episodes[-20:]
        if not recent:
            return "Nenhuma memoria ainda."

        parts = []
        for e in recent:
            parts.append(e.to_short())
        return "\n".join(parts[-10:])


# ─── Drives Internos ──────────────────────────────────────

@dataclass
class DriveState:
    """Estado de um drive interno."""
    name: str                    # "curiosity", "mastery", "novelty", "social"
    level: float = 0.5           # 0 (satisfeito) a 1 (frustrado)
    decay_rate: float = 0.01     # quanto diminui por tick
    growth_rate: float = 0.05    # quanto aumenta quando ignorado
    last_satisfied: float = field(default_factory=time.time)

    def tick(self):
        """Decai naturalmente com o tempo."""
        self.level = max(0.0, self.level - self.decay_rate)
        self.last_satisfied = time.time()

    def frustrate(self, amount: float = 0.1):
        """Aumenta o drive (sistema esta frustrado)."""
        self.level = min(1.0, self.level + amount)

    def satisfy(self):
        """Satisfaz o drive."""
        self.level = max(0.0, self.level - 0.3)
        self.last_satisfied = time.time()

    @property
    def is_pressing(self) -> bool:
        return self.level > 0.7


# ─── UnifiedSelf ───────────────────────────────────────────

class UnifiedSelf:
    """Self unificado do sistema.

    Integra:
      - Identidade: quem sou, o que sei, o que quero
      - Memoria autobiografica: o que vivi
      - Drives: o que me motiva
      - Neuromodulacao: estado emocional
      - Meta-cognicao: como estou pensando
    """

    def __init__(self, cge=None):
        self.cge = cge

        # Identidade
        self.name = "NovaMind"
        self.beliefs: Dict[str, float] = {}
        self.capabilities: Dict[str, float] = {}
        self.preferences: Dict[str, float] = {}

        # Memoria
        self.memory = AutobiographicalMemory()

        # Drives
        self.drives = {
            "curiosity": DriveState("curiosity", decay_rate=0.02, growth_rate=0.08),
            "mastery": DriveState("mastery", decay_rate=0.01, growth_rate=0.05),
            "novelty": DriveState("novelty", decay_rate=0.015, growth_rate=0.06),
        }

        # Neuromodulacao
        self.dopamine: float = 0.5
        self.noradrenaline: float = 0.3
        self.acetylcholine: float = 0.3
        self.serotonin: float = 0.5

        # Meta
        self.current_mood: str = "curious"
        self.energy: float = 1.0       # 0-1
        self.n_successes: int = 0
        self.n_failures: int = 0

        # Cognitive mode state
        self.current_mode: str = "focused"
        self.cycles_in_mode: int = 0

    def apply_mode_energy_cost(self, energy_cost: int):
        """Aplica custo/ganho de energia do modo cognitivo atual.
        
        Modos custam energia (FOCUSED, EXPLORATION, DREAMING).
        Modo CONSOLIDATION recupera energia.
        """
        if energy_cost < 0:
            self.energy = max(0.0, self.energy + energy_cost * 0.05)
        else:
            self.energy = min(1.0, self.energy + energy_cost * 0.05)
        self.n_failures: int = 0

        # Timeline do self
        self._timeline: List[Dict] = []

    # ─── Identidade ───────────────────────────────────────

    def update_belief(self, belief: str, confidence: float):
        """Atualiza uma crenca."""
        if belief in self.beliefs:
            # Media ponderada (novo pesa mais)
            self.beliefs[belief] = self.beliefs[belief] * 0.7 + confidence * 0.3
        else:
            self.beliefs[belief] = confidence
        self._timeline.append({"t": time.time(), "event": "belief_update", "belief": belief, "conf": confidence})

    def update_capability(self, skill: str, delta: float):
        """Atualiza proficiencia em uma habilidade."""
        old = self.capabilities.get(skill, 0.0)
        self.capabilities[skill] = min(1.0, old + delta)
        self._timeline.append({"t": time.time(), "event": "capability_update", "skill": skill, "new": self.capabilities[skill]})

    # ─── Neuromodulacao ────────────────────────────────────

    def update_mood(self, success: bool, prediction_error: float = 0.0):
        """Atualiza estado emocional baseado em resultado."""
        if success:
            self.dopamine = min(1.0, self.dopamine + 0.05)
            self.serotonin = min(1.0, self.serotonin + 0.03)
            self.n_successes += 1
        else:
            self.dopamine = max(0.0, self.dopamine - 0.03)
            self.noradrenaline = min(1.0, self.noradrenaline + 0.05)
            self.n_failures += 1

        # Surpresa (prediction error alto)
        if prediction_error > 0.5:
            self.noradrenaline = min(1.0, self.noradrenaline + 0.08)
            self.acetylcholine = min(1.0, self.acetylcholine + 0.05)
        
        # Decaimento natural dos NT
        self.dopamine = max(0.0, self.dopamine - 0.01)
        self.noradrenaline = max(0.0, self.noradrenaline - 0.005)
        self.acetylcholine = max(0.0, self.acetylcholine - 0.005)

        # Determinacao do humor
        if self.dopamine > 0.6 and self.serotonin > 0.6:
            self.current_mood = "confident"
        elif self.dopamine < 0.3 and self.noradrenaline > 0.6:
            self.current_mood = "anxious"
        elif self.dopamine > 0.6 and self.noradrenaline > 0.5:
            self.current_mood = "excited"
        elif self.serotonin < 0.3:
            self.current_mood = "frustrated"
        else:
            self.current_mood = "curious"

    # ─── Ticks ─────────────────────────────────────────────

    def tick(self):
        """Um tick do self — drives decaem, energia diminui lentamente."""
        for drive in self.drives.values():
            drive.tick()
        self.energy = max(0.0, self.energy - 0.003)

    def rest(self):
        """Descanso — recupera energia."""
        self.energy = min(1.0, self.energy + 0.3)
        self.serotonin = min(1.0, self.serotonin + 0.1)

    # ─── Queries ───────────────────────────────────────────

    def get_dominant_drive(self) -> Tuple[str, float]:
        """Retorna o drive mais pressionante."""
        best_name = "curiosity"
        best_level = 0
        for name, drive in self.drives.items():
            if drive.level > best_level:
                best_level = drive.level
                best_name = name
        return best_name, best_level

    def get_competence(self, domain: str = "general") -> float:
        """Auto-avaliacao de competencia em um dominio."""
        relevant = [v for k, v in self.capabilities.items() if domain in k.lower()]
        if relevant:
            return sum(relevant) / len(relevant)
        return self.n_successes / max(1, self.n_successes + self.n_failures)

    def summarize(self) -> str:
        """Sumario textual do self."""
        lines = [
            f"Self: {self.name}",
            f"Mood: {self.current_mood}",
            f"Drives: {', '.join(f'{n}={d.level:.2f}' for n, d in self.drives.items())}",
            f"NT: DA={self.dopamine:.2f} NA={self.noradrenaline:.2f} "
            f"ACh={self.acetylcholine:.2f} 5HT={self.serotonin:.2f}",
            f"Record: {self.n_successes}S/{self.n_failures}F",
        ]
        if self.capabilities:
            best = max(self.capabilities, key=self.capabilities.get)
            lines.append(f"Best skill: {best} ({self.capabilities[best]:.2f})")
        return "\n".join(lines)

    def get_state(self) -> dict:
        return {
            "mood": self.current_mood,
            "energy": round(self.energy, 2),
            "drives": {n: round(d.level, 2) for n, d in self.drives.items()},
            "dopamine": round(self.dopamine, 2),
            "noradrenaline": round(self.noradrenaline, 2),
            "acetylcholine": round(self.acetylcholine, 2),
            "serotonin": round(self.serotonin, 2),
            "success_rate": self.n_successes / max(1, self.n_successes + self.n_failures),
            "memory_size": len(self.memory.episodes),
            "beliefs": len(self.beliefs),
            "capabilities": len(self.capabilities),
        }
