"""
CognitiveModeManager — 5 modos operacionais que alteram o comportamento
de TODOS os modulos da arquitetura.

Cada modo define:
  - temperature: 0.0 (rigido) a 1.0 (fluido)
  - beam_width: largura da busca
  - search_depth: profundidade maxima
  - constraint_strictness: 0.0 (relaxado) a 1.0 (rigido)
  - concept_compatibility_threshold: quao facil conectar conceitos distantes
  - hypothesis_count: quantas hipoteses gerar
  - energy_cost: consumo por ciclo

Modos:
  FOCUSED     → tarefa externa, busca estreita, restricoes altas
  EXPLORATION → falha/surpresa, busca ampla, restricoes medias
  INCUBATION  → sem tarefa, recombina cao, restricoes baixas
  DREAMING    → combinacoes quase aleatorias, sem restricoes
  CONSOLIDATION → pos-atividade, consolida memoria, baixo consumo
"""
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class CognitiveMode(Enum):
    FOCUSED = "focused"
    EXPLORATION = "exploration"
    INCUBATION = "incubation"
    DREAMING = "dreaming"
    CONSOLIDATION = "consolidation"


@dataclass
class ModeConfig:
    """Parametros de um modo operacional."""
    name: str
    temperature: float          # 0.0 (rigido) a 1.0 (fluido)
    beam_width: int             # largura da busca no synthesizer
    max_search_depth: int       # profundidade maxima
    constraint_strictness: float  # 0.0 (relaxado) a 1.0 (rigido)
    concept_compatibility_threshold: float  # quao facil conectar
    hypothesis_count: int       # quantas hipoteses gerar
    energy_cost: int            # consumo por ciclo (-) ou recuperacao (+)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.name,
            "temperature": self.temperature,
            "beam_width": self.beam_width,
            "max_search_depth": self.max_search_depth,
            "constraint_strictness": self.constraint_strictness,
            "concept_threshold": self.concept_compatibility_threshold,
            "hypothesis_count": self.hypothesis_count,
            "energy_cost": self.energy_cost,
        }


# Configuracoes de cada modo
MODE_CONFIGS = {
    CognitiveMode.FOCUSED: ModeConfig(
        name="FOCUSED",
        temperature=0.2,
        beam_width=5,
        max_search_depth=4,
        constraint_strictness=0.9,
        concept_compatibility_threshold=0.7,
        hypothesis_count=3,
        energy_cost=-1,
        description="Tarefa externa: busca estreita, restricoes altas",
    ),
    CognitiveMode.EXPLORATION: ModeConfig(
        name="EXPLORATION",
        temperature=0.5,
        beam_width=12,
        max_search_depth=6,
        constraint_strictness=0.5,
        concept_compatibility_threshold=0.4,
        hypothesis_count=6,
        energy_cost=-2,
        description="Falha/surpresa: busca ampla, restricoes medias",
    ),
    CognitiveMode.INCUBATION: ModeConfig(
        name="INCUBATION",
        temperature=0.6,
        beam_width=8,
        max_search_depth=3,
        constraint_strictness=0.3,
        concept_compatibility_threshold=0.3,
        hypothesis_count=8,
        energy_cost=-1,
        description="Sem tarefa: recombina cao, restricoes baixas",
    ),
    CognitiveMode.DREAMING: ModeConfig(
        name="DREAMING",
        temperature=0.9,
        beam_width=20,
        max_search_depth=2,
        constraint_strictness=0.1,
        concept_compatibility_threshold=0.1,
        hypothesis_count=12,
        energy_cost=-3,
        description="Combinacoes quase aleatorias, sem restricoes",
    ),
    CognitiveMode.CONSOLIDATION: ModeConfig(
        name="CONSOLIDATION",
        temperature=0.3,
        beam_width=3,
        max_search_depth=2,
        constraint_strictness=0.7,
        concept_compatibility_threshold=0.6,
        hypothesis_count=2,
        energy_cost=2,
        description="Pos-atividade: consolida, baixo consumo",
    ),
}


class CognitiveModeManager:
    """Gerencia o modo operacional atual e as transicoes.

    O modo muda baseado em:
      - presenca de tarefa externa
      - taxa de sucesso/erro
      - surpresa (prediction error do WorldModel)
      - energia disponivel
      - ciclos desde ultima consolidacao

    Uso:
        mgr = CognitiveModeManager()
        mgr.tick(has_task=True, success_rate=0.3, surprise=0.8)
        config = mgr.get_current_config()
    """

    def __init__(self, initial_mode: CognitiveMode = CognitiveMode.FOCUSED):
        self.current_mode = initial_mode
        self.previous_mode = initial_mode
        self.config = MODE_CONFIGS[initial_mode]

        # Contadores de estado
        self.cycles_in_mode: int = 0
        self.cycles_since_task: int = 0
        self.cycles_since_consolidation: int = 0
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0

        # Historico de modos
        self.mode_history: list = []
        self.transition_history: list = []

        # Callbacks de transicao
        self._on_transition: Optional[Callable] = None

        self.last_tick: float = time.time()

    def set_transition_callback(self, fn: Callable):
        """Callback chamado quando o modo muda."""
        self._on_transition = fn

    def get_current_config(self) -> ModeConfig:
        """Retorna a configuracao do modo atual."""
        return self.config

    def get_current_mode(self) -> CognitiveMode:
        return self.current_mode

    def tick(self, has_task: bool = False,
             success_rate: float = 0.5,
             surprise: float = 0.0,
             energy: float = 1.0,
             force_mode: Optional[CognitiveMode] = None) -> CognitiveMode:
        """Um tick do mode manager.

        Args:
            has_task: se ha uma tarefa externa pendente
            success_rate: taxa de sucesso recente (0-1)
            surprise: prediction error (0-1)
            energy: energia atual (0-1)
            force_mode: forca um modo especifico

        Returns: novo modo apos a transicao.
        """
        self.previous_mode = self.current_mode
        self.cycles_in_mode += 1
        self.last_tick = time.time()

        if has_task:
            self.cycles_since_task = 0
        else:
            self.cycles_since_task += 1

        self.cycles_since_consolidation += 1

        # Atualiza contadores de sucesso/falha
        if success_rate < 0.3:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        elif success_rate > 0.7:
            self.consecutive_successes += 1
            self.consecutive_failures = 0

        # Forca modo se especificado
        if force_mode is not None:
            self._transition_to(force_mode)
            return self.current_mode

        # Logica de transicao
        new_mode = self._decide_mode(has_task, success_rate, surprise, energy)

        if new_mode != self.current_mode:
            self._transition_to(new_mode)

        return self.current_mode

    def _decide_mode(self, has_task: bool, success_rate: float,
                     surprise: float, energy: float) -> CognitiveMode:
        """Decide para qual modo transicionar."""
        # Se ha tarefa e energia suficiente → FOCUSED
        if has_task and energy > 0.2:
            if self.consecutive_failures >= 3 or surprise > 0.6:
                return CognitiveMode.EXPLORATION
            return CognitiveMode.FOCUSED

        # Se falha repetida mesmo sem tarefa → EXPLORATION
        if self.consecutive_failures >= 5 and energy > 0.3:
            return CognitiveMode.EXPLORATION

        # Se surpresa alta → EXPLORATION
        if surprise > 0.7 and energy > 0.3:
            return CognitiveMode.EXPLORATION

        # Sem tarefa: verificar condicoes mais especificas primeiro
        if not has_task:
            if self.cycles_since_task > 30 and energy > 0.6:
                return CognitiveMode.DREAMING
            if self.cycles_since_task > 10 and energy > 0.5:
                return CognitiveMode.INCUBATION
            if self.cycles_since_consolidation > 20 or energy < 0.3:
                return CognitiveMode.CONSOLIDATION
            if self.cycles_since_task > 3:
                return CognitiveMode.INCUBATION

        # Mantem o modo atual se nenhuma condicao de transicao for atendida
        return self.current_mode

    def _transition_to(self, new_mode: CognitiveMode):
        """Executa transicao de modo."""
        self.mode_history.append({
            "from": self.current_mode.value,
            "to": new_mode.value,
            "cycles_in_mode": self.cycles_in_mode,
            "time": time.time(),
        })
        self.transition_history.append({
            "from_mode": self.current_mode.value,
            "to_mode": new_mode.value,
            "reason": "auto_transition",
        })

        self.previous_mode = self.current_mode
        self.current_mode = new_mode
        self.config = MODE_CONFIGS[new_mode]
        self.cycles_in_mode = 0

        # Callback
        if self._on_transition:
            self._on_transition(self.previous_mode, new_mode)

    def get_mode_cycle_count(self) -> int:
        """Retorna quantas vezes o modo atual foi executado."""
        return self.cycles_in_mode

    def get_stats(self) -> dict:
        return {
            "current_mode": self.current_mode.value,
            "previous_mode": self.previous_mode.value,
            "cycles_in_mode": self.cycles_in_mode,
            "cycles_since_task": self.cycles_since_task,
            "cycles_since_consolidation": self.cycles_since_consolidation,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "config": self.config.to_dict(),
            "total_transitions": len(self.mode_history),
            "recent_transitions": self.mode_history[-5:] if self.mode_history else [],
        }
