"""
HippocampalBuffer — Buffer de Memória Episódica Contínua com Replay
====================================================================
Armazena os últimos N segundos de percepção como vetores HDC + estado
neuromodulador. Suporta pausar, rebobinar, refletir e detectar Eureka.

Diferença do TemporalBuffer:
  - TemporalBuffer: guarda chunks de TEXTO (7 slots) para geração textual
  - HippocampalBuffer: guarda VETORES HDC de percepção contínua (frames/áudio)
    com neuromoduladores associados, suporta replay e detecção de surpresa

Mecanismos:
  1. Circular: buffer de capacidade fixa, descarta o mais antigo
  2. Surprise: prediction error entre frames consecutivos
  3. Replay: "rebobina" o buffer em ordem reversa para reflexão
  4. Pause/Resume: controla se gravação está ativa
  5. Eureka detection: queda súbita de FE entre hypothesis atual e target

Uso:
    hb = HippocampalBuffer(capacity_seconds=10, fps=15)
    hb.push(hdc_vector, neuromodulators={"DA": 0.8, "NA": 0.2})
    if hb.should_reflect():
        for past_vec in hb.rewind():
            maestro.reflect(past_vec)
"""

import time
import torch
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Iterator, Tuple


class HippocampalEntry:
    """Uma entrada no buffer hipocampal: vetor HDC + estado naquele momento."""
    __slots__ = ("vector", "neuromodulators", "timestamp", "tick", "metadata")

    def __init__(self, vector: torch.Tensor, neuromodulators: dict = None,
                 timestamp: float = None, tick: int = 0, metadata: dict = None):
        self.vector = vector.detach().cpu().float() if torch.is_tensor(vector) else torch.tensor(vector)
        self.neuromodulators = neuromodulators or {}
        self.timestamp = timestamp or time.time()
        self.tick = tick
        self.metadata = metadata or {}


class HippocampalBuffer:
    """
    Buffer circular de percepção contínua com suporte a replay e surpresa.

    Args:
        capacity_seconds: quantos segundos de experiência manter (default 10)
        fps: frames por segundo esperado (default 15)
        dim: dimensão do vetor HDC (default 10000)
        device: device torch
        surprise_threshold: limiar de similaridade para considerar "surpresa"
    """

    def __init__(self, capacity_seconds: int = 10, fps: int = 15,
                 dim: int = 10000, device: Optional[str] = None,
                 surprise_threshold: float = 0.3):
        self.capacity = capacity_seconds * fps
        self.fps = fps
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.surprise_threshold = surprise_threshold

        self.buffer: deque = deque(maxlen=self.capacity)
        self.tick = 0

        # Estado de gravação
        self._paused = False
        self._pause_time = 0.0

        # Métricas de surpresa
        self.surprise_history: deque = deque(maxlen=self.capacity)
        self.total_surprise: float = 0.0
        self.avg_surprise: float = 0.0

        # Detecção de Eureka
        self._baseline_fe: float = 1.0
        self._eureka_threshold: float = 0.15
        self._consecutive_low_fe: int = 0
        self.eureka_moment: bool = False
        self.eureka_tick: int = 0

    # ─── Escrita ─────────────────────────────────────────────────

    def push(self, vector: torch.Tensor, neuromodulators: dict = None,
             metadata: dict = None) -> "HippocampalEntry":
        """Adiciona um frame de percepção ao buffer."""
        if self._paused:
            return self.buffer[-1] if self.buffer else None

        self.tick += 1
        entry = HippocampalEntry(
            vector=vector,
            neuromodulators=neuromodulators or {},
            timestamp=time.time(),
            tick=self.tick,
            metadata=metadata or {},
        )
        self.buffer.append(entry)

        # Computa surpresa se houver pelo menos 2 entradas
        if len(self.buffer) >= 2:
            prev = self.buffer[-2].vector
            curr = entry.vector
            sim = float(torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)))
            surprise = 1.0 - sim
            self.surprise_history.append(surprise)
            self.total_surprise = sum(self.surprise_history)
            self.avg_surprise = self.total_surprise / max(1, len(self.surprise_history))
        else:
            surprise = 0.0
            self.surprise_history.append(0.0)

        # Detecção de Eureka: queda brusca de FE
        if neuromodulators:
            da = neuromodulators.get("DA", 0.5)
            na = neuromodulators.get("NA", 0.5)
            fe = 1.0 - da  # FE inversa da dopamina

            if fe < self._eureka_threshold:
                self._consecutive_low_fe += 1
                if self._consecutive_low_fe >= 3 and not self.eureka_moment:
                    self.eureka_moment = True
                    self.eureka_tick = self.tick
                    entry.metadata["eureka"] = True
            else:
                self._consecutive_low_fe = 0

        return entry

    # ─── Controles ───────────────────────────────────────────────

    def pause(self):
        """Pausa a gravação. O buffer mantém o que tem."""
        self._paused = True
        self._pause_time = time.time()

    def resume(self):
        """Retoma a gravação."""
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def clear(self):
        """Limpa o buffer inteiro."""
        self.buffer.clear()
        self.surprise_history.clear()
        self.total_surprise = 0.0
        self.avg_surprise = 0.0
        self.eureka_moment = False
        self._consecutive_low_fe = 0

    # ─── Replay / Reflexão ──────────────────────────────────────

    def rewind(self) -> Iterator[HippocampalEntry]:
        """
        Rebobina: itera sobre o buffer em ordem reversa (mais recente → mais antigo).
        Usado para o Maestro "refletir" sobre o que acabou de acontecer.
        """
        for entry in reversed(self.buffer):
            yield entry

    def replay(self) -> Iterator[HippocampalEntry]:
        """
        Replay: itera sobre o buffer em ordem cronológica.
        """
        for entry in self.buffer:
            yield entry

    def reflect(self, n_frames: int = None) -> List[HippocampalEntry]:
        """
        Reflexão: retorna os últimos n_frames para análise do Maestro.
        Se n_frames for None, retorna o buffer inteiro.
        """
        if n_frames is None:
            return list(self.buffer)
        return list(self.buffer)[-n_frames:]

    def get_frame_at(self, tick: int) -> Optional[HippocampalEntry]:
        """Retorna a entrada em um tick específico."""
        for entry in self.buffer:
            if entry.tick == tick:
                return entry
        return None

    def get_recent_by_time(self, seconds: float) -> List[HippocampalEntry]:
        """Retorna entradas dos últimos N segundos."""
        cutoff = time.time() - seconds
        return [e for e in self.buffer if e.timestamp >= cutoff]

    # ─── Análise ─────────────────────────────────────────────────

    def should_reflect(self, threshold: float = 0.3) -> bool:
        """
        Retorna True se a surpresa média está alta o suficiente para
        justificar uma pausa para reflexão.
        """
        return self.avg_surprise > threshold or self.eureka_moment

    def get_salient_moment(self, top_k: int = 1) -> List[HippocampalEntry]:
        """
        Retorna os momentos mais marcantes (maior surpresa ou maior
        variação de neuromoduladores).
        """
        if len(self.buffer) < 2:
            return list(self.buffer)

        scored = []
        for i in range(1, len(self.buffer)):
            prev = self.buffer[i - 1]
            curr = self.buffer[i]
            sim = float(torch.nn.functional.cosine_similarity(
                prev.vector.unsqueeze(0), curr.vector.unsqueeze(0)))
            surprise = 1.0 - sim

            # Variação de neuromoduladores
            nm_change = 0.0
            if prev.neuromodulators and curr.neuromodulators:
                for k in set(prev.neuromodulators) | set(curr.neuromodulators):
                    nm_change += abs(
                        prev.neuromodulators.get(k, 0.5) -
                        curr.neuromodulators.get(k, 0.5)
                    )

            score = surprise + nm_change * 0.3
            scored.append((score, curr))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def estimate_duration(self, n_frames: int = None) -> float:
        """Estima a duração em segundos dos últimos n_frames."""
        frames = self.reflect(n_frames)
        if len(frames) < 2:
            return 0.0
        return frames[-1].timestamp - frames[0].timestamp

    # ─── Eureka Detection ───────────────────────────────────────

    def check_eureka(self, target_vec: torch.Tensor) -> bool:
        """
        Verifica se alguma entrada no buffer tem alta similaridade
        com um vetor alvo (a "solução" que a AGI estava buscando).
        Se sim, é um momento Eureka.
        """
        if len(self.buffer) == 0:
            return False

        target = target_vec.detach().cpu().float()
        for entry in self.buffer:
            sim = float(torch.nn.functional.cosine_similarity(
                entry.vector.unsqueeze(0), target.unsqueeze(0)))
            if sim > 0.85:
                entry.metadata["eureka"] = True
                self.eureka_moment = True
                self.eureka_tick = entry.tick
                return True
        return False

    def reset_eureka(self):
        """Reseta o flag de Eureka."""
        self.eureka_moment = False
        self._consecutive_low_fe = 0

    # ─── Estado ──────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Retorna estado completo do buffer para monitoramento."""
        return {
            "capacity": self.capacity,
            "filled": len(self.buffer),
            "usage_pct": round(len(self.buffer) / max(1, self.capacity) * 100, 1),
            "tick": self.tick,
            "paused": self._paused,
            "avg_surprise": round(self.avg_surprise, 4),
            "total_surprise": round(self.total_surprise, 4),
            "eureka": self.eureka_moment,
            "eureka_tick": self.eureka_tick,
            "duration_seconds": round(self.estimate_duration(), 2),
            "fps_estimate": round(
                len(self.buffer) / max(0.1, self.estimate_duration()), 1
            ) if len(self.buffer) > 1 else 0,
        }

    def __len__(self) -> int:
        return len(self.buffer)
