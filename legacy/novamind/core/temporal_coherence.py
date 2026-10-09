"""
temporal_coherence.py — O Campo de Coerência Temporal
======================================================
Inspirado pelo insight humano: "aqui tudo se comunica de forma
instantanea por código, isso não seria um problema?"

SIM, é um problema. No cérebro:
  - Sinal do olho demora ~50ms, ouvido ~10ms, tacto ~30ms
  - Esta assincronia não é defeito — é MECANISMO de computação
  - Sincronia gamma (~40Hz) liga percepções num objeto único (binding)
  - Janelas temporais definem o que "pertence junto"

Se tudo comunica instantaneamente, não há ressonância possível — só
chamadas de função. Este módulo introduz a dimensão temporal como
CRITÉRIO de binding, não só como performance.

Mecanismo:
  - Cada fonte perceptual tem uma FASE própria (fase oscilatória)
  - Binding só ocorre quando fases ALINHAM dentro de uma janela
  - "vermelho" + "redondo" ligam-se à maçã só porque chegaram em coerência
  - Percepções fora de fase são inibidas lateralmente

Isto transforma o meio cimático de "gerador de padrões" em
"campo de coerência" — exatamente o "algo que arbitra os contratos"
entre módulos cognitivos.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SignalPacket:
    """Pacote de informação percebida com fase temporal."""
    source: str           # "visual", "audio", "causal", "memory", etc.
    content: Any          # payload (vetor HDV, dict, etc.)
    phase: float          # fase oscilatória 0..2π
    amplitude: float = 1.0
    timestamp: float = field(default_factory=time.time)
    decay: float = 0.95   # decaimento por tick


@dataclass
class CoherenceEvent:
    """Quando múltiplos sinais entram em coerência, é um evento de binding."""
    members: List[str]    # fontes que coeriram
    time: float
    coherence_strength: float
    bound_concept: Optional[str] = None  # conceito formado por este binding


class TemporalCoherenceField:
    """
    Campo global de coerência temporal entre módulos cognitivos.

    Os módulos postam SignalPackets. O campo:
      1. Evolui as fases (cada fonte tem frequência própria)
      2. Detecta coerência: fontes cujas fases convergem numa janela
      3. Liga (binds) as coerentes num conceito único
      4. Inibe as não-coerentes (lateral inhibition temporal)

    Isto É a base do binding no sistema. Sem isto, tudo é instantâneo
    e arbitrário; com isto, o tempo vira critério.
    """

    def __init__(self, gamma_freq: float = 40.0,
                 coherence_window: float = 0.3):
        """
        Args:
            gamma_freq: frequência base de sincronização (Hz). 40Hz = gamma
                        cerebral. Define o "tic-tac" do binding.
            coherence_window: largura da janela temporal para considerar
                              sinais "ao mesmo tempo" (em radianos de fase).
        """
        self.gamma_freq = gamma_freq
        self.coherence_window = coherence_window
        self.gamma_period = 1.0 / gamma_freq  # 25ms

        # Estado: sinais ativos pendurados no campo
        self.active_signals: Dict[str, SignalPacket] = {}
        # Histórico de eventos de coerência (binding)
        self.events: List[CoherenceEvent] = []
        # Fase global do campo (evolui com tick)
        self.global_phase = 0.0
        self.tick_count = 0

        # Fase própria de cada fonte (frequências diferentes → assincronia)
        self.source_phases: Dict[str, float] = {}
        self.source_freqs: Dict[str, float] = {
            "visual": gamma_freq,        # 40Hz (rápido)
            "audio": gamma_freq * 0.8,   # 32Hz (diferente → chega em tempo dif)
            "tactile": gamma_freq * 0.6,
            "causal": gamma_freq * 0.3,  # raciocínio causal é lento
            "memory": gamma_freq * 0.4,  # memória é ainda mais lenta
            "motor": gamma_freq * 0.7,
        }

    def post(self, source: str, content: Any,
             amplitude: float = 1.0) -> SignalPacket:
        """
        Um módulo posta um sinal no campo. A fase é determinada pela
        frequência própria da fonte e pelo tempo global.

        Isto introduz assincronia: duas fontes postando "ao mesmo tempo"
        no código ainda têm fases DIFERENTES porque têm frequências próprias
        diferentes — exatamente como olho e ouvido.
        """
        freq = self.source_freqs.get(source, self.gamma_freq)
        phase = (self.global_phase * freq / self.gamma_freq) % (2 * math.pi)
        packet = SignalPacket(
            source=source, content=content, phase=phase,
            amplitude=amplitude,
        )
        self.active_signals[source] = packet
        return packet

    def tick(self, dt: float = None) -> List[CoherenceEvent]:
        """
        Evolui o campo um step temporal. Detecta coerência entre fontes.

        dt default = gamma_period (25ms). Pode ser maior para "pular" tempo.
        """
        if dt is None:
            dt = self.gamma_period

        self.global_phase += 2 * math.pi * dt * self.gamma_freq
        self.global_phase %= 2 * math.pi
        self.tick_count += 1

        # Atualiza fases das fontes ativas
        for source, packet in self.active_signals.items():
            freq = self.source_freqs.get(source, self.gamma_freq)
            packet.phase = (packet.phase + 2 * math.pi * dt * freq) % (2 * math.pi)
            packet.amplitude *= packet.decay

        # Remove sinais decaídos
        self.active_signals = {
            s: p for s, p in self.active_signals.items() if p.amplitude > 0.05
        }

        # Detecta coerência entre fontes ativas
        events = self._detect_coherence()
        self.events.extend(events)
        return events

    def _detect_coherence(self) -> List[CoherenceEvent]:
        """
        Agrupa fontes cujas fases estão dentro da janela de coerência.
        Cada grupo forma um evento de binding.

        Sinais fora de coerência são inibidos lateralmente (amplitude reduzida).
        """
        if len(self.active_signals) < 2:
            return []

        sources = list(self.active_signals.keys())
        events = []

        # Para cada par, mede diferença de fase circular
        # Fontes com diferença < coherence_window são "em coerência"
        visited = set()
        for i, s1 in enumerate(sources):
            if s1 in visited:
                continue
            group = [s1]
            visited.add(s1)
            p1 = self.active_signals[s1]

            for j, s2 in enumerate(sources):
                if s2 in visited or s2 == s1:
                    continue
                p2 = self.active_signals[s2]
                phase_diff = self._circular_distance(p1.phase, p2.phase)
                if phase_diff < self.coherence_window:
                    group.append(s2)
                    visited.add(s2)

            if len(group) >= 2:
                # Coerência = média de amplitude × quão próximas as fases
                amps = [self.active_signals[s].amplitude for s in group]
                phases = [self.active_signals[s].phase for s in group]
                phase_spread = max(phases) - min(phases)
                coherence = float(
                    (sum(amps) / len(amps)) * (1.0 - phase_spread / self.coherence_window)
                )
                coherence = max(0.0, min(1.0, coherence))

                event = CoherenceEvent(
                    members=group,
                    time=time.time(),
                    coherence_strength=coherence,
                )
                events.append(event)

        # Inibição lateral: fontes não agrupadas perdem amplitude
        grouped = set()
        for e in events:
            grouped.update(e.members)
        for s in self.active_signals:
            if s not in grouped:
                self.active_signals[s].amplitude *= 0.7  # inibida

        return events

    def _circular_distance(self, a: float, b: float) -> float:
        """Distância circular entre duas fases (0..π)."""
        diff = abs(a - b) % (2 * math.pi)
        return min(diff, 2 * math.pi - diff)

    # ─── Binding explícito em conceito ────────────────────────────────

    def bind_to_concept(self, event: CoherenceEvent,
                        concept_name: str = None) -> str:
        """
        Dado um evento de coerência, cria um conceito bound.
        Retorna o nome do conceito formado.

        Este é o "vermelho + redondo + doce → maçã" do sistema.
        """
        if concept_name is None:
            # Nome composto das fontes que coeriram
            concept_name = "BIND_" + "_".join(sorted(event.members))

        event.bound_concept = concept_name
        return concept_name

    def summary(self) -> str:
        lines = [f"TemporalCoherenceField (tick={self.tick_count})"]
        lines.append(f"  Sinais ativos: {len(self.active_signals)}")
        for s, p in self.active_signals.items():
            lines.append(f"    {s}: fase={p.phase:.2f} amp={p.amplitude:.2f}")
        lines.append(f"  Eventos de coerência: {len(self.events)}")
        recent = self.events[-3:]
        for e in recent:
            lines.append(f"    {e.members} força={e.coherence_strength:.2f}"
                        + (f" → {e.bound_concept}" if e.bound_concept else ""))
        return "\n".join(lines)
