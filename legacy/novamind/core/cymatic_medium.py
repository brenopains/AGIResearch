"""
cymatic_medium.py — O Meio Cimático da NovaMind
=================================================
"Som gera forma. A forma gera estrutura. A estrutura, se for
cognitivamente útil, vira parte da arquitetura."

O substrato HDV ±1 da NovaMind JÁ é um meio de onda (duality de Fourier:
multiplicação bipolar = convolução espectral). Este modulo EXPLORA esse
meio dirigindo-o com frequências e deixando a interferência auto-organizar-se.

Diferenca crucial vs. ALife/evolução aleatória:
  - Mutação aleatória: amostra o espaco de configurações de forma cega
  - Ressonância: enviesada para a COERÊNCIA (estruturas estáveis sob
    interferência construtiva). Nao é aleatória — é dirigida pela física.

Mas ressonância encontra ESTABILIDADE, nao ADEQUAÇÃO. Por isso cada padrão
emergente é avaliado por VALOR COGNITIVO (ressona com conceitos do CGE?
Passa em trials? Reduz energia livre?) antes de ser incorporado.

O GARGALO é rastreado assim: para cada trial que falha, faz-se um sweep de
frequência. Se a freq X destrava o trial Y, então Y estava limitado pela
ausência de estrutura ressonante em X. A frequência APONTA o gargalo.
"""
from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ResonantPattern:
    """
    Padrão estável que emergiu da interferência de frequências no meio HDV.

    IMPORTANTE: nao foi programado. Foi DETECTADO após o meio evoluir.
    A estabilidade é medida por quao pouco o padrão muda entre iteracões
    (attractor) — exatamente como numa placa de Chladni.
    """
    pattern_id: str
    vector: np.ndarray              # estado HDV estabilizado (±1)
    driving_freqs: Tuple[float, ...]  # frequências que o geraram
    stability: float                # 0..1, quao estável ficou (1 = attractor)
    energy: float                   # energia do padrão
    emerged_at: float = field(default_factory=time.time)
    cognitive_value: float = 0.0    # preenchido apos avaliação (0 = nao avaliado)
    notes: str = ""


class CymaticMedium:
    """
    Meio vibracional sobre o espaco HDV bipolar.

    Estado: um vetor bipolar N-dimensional que evolui sob driving oscilatório.
    A cada step:
      1. Compoe ondas senoidais nas frequências ativas
      2. Bipolariza (sign) — mantém o contrato HDC ±1
      3. Deixa iterar até estabilizar (attractor) ou divergir
      4. Detecta atratores = padrões emergentes

    Funciona standalone (sem CGE) — permite testar a capacidade pura do
    meio sem depender de estado global.
    """

    def __init__(self, dim: int = 10000, seed: int = 2026):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # CAMPO CONTINUO interno. Crucial: bipolarizar a cada step (sign())
        # destrói a informação de frequência — o estado antigo domina e trava
        # num ponto fixo independente do driving. A física real funciona assim:
        # a placa de Chladni vibra continuamente; só a AREIA (representação
        # discreta) é ±1. Mantemos o campo contínuo e bipolarizamos só o
        # atrator final ao extrair um padrão.
        self._field = self.rng.normal(0, 1, dim).astype(np.float32)
        # state bipolar é derivado do field (a "areia")
        self.state = np.sign(self._field)
        self.history: List[np.ndarray] = []
        self.patterns: Dict[str, ResonantPattern] = {}
        self.step_count = 0

    # ─── Driving oscilatório ──────────────────────────────────────────

    def _compose_wave(self, freqs: List[float],
                      phases: Optional[List[float]] = None,
                      amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """
        Compoe uma onda complexa no espaco HDV. Cada frequência k produz
        k ciclos ao longo das N dimensoes. A composição é a soma coerente.
        """
        n = self.dim
        idx = np.arange(n, dtype=np.float32)
        if phases is None:
            phases = [0.0] * len(freqs)
        if amplitudes is None:
            amplitudes = [1.0 / len(freqs)] * len(freqs)

        wave = np.zeros(n, dtype=np.float32)
        for k, phi, amp in zip(freqs, phases, amplitudes):
            wave += amp * np.sin(2.0 * math.pi * k * idx / n + phi)
        return wave

    def drive(self, freqs: List[float], n_steps: int = 10,
              damping: float = 0.1,
              inject_noise: float = 0.0) -> List[float]:
        """
        Excita o meio com frequências e deixa evoluir por n_steps.

        Evolução do CAMPO CONTÍNUO (não bipolarizado a cada step):
          field_{t+1} = (1-d)·field_t + d·wave + ruído
        A estabilidade é medida na representação bipolar derivada (sign),
        que é o que importa para o contrato HDC — mas o driving age sobre
        o campo contínuo, preservando a informação de frequência.

        Retorna serie de estabilidades por step (1 = parou de mudar).
        """
        if isinstance(freqs, (int, float)):
            freqs = [float(freqs)]

        stabilities = []
        prev_bipolar = np.sign(self._field)

        for step in range(n_steps):
            wave = self._compose_wave(freqs)
            # Driving age sobre o campo contínuo
            self._field = (1.0 - damping) * self._field + damping * wave
            if inject_noise > 0:
                self._field += self.rng.normal(0, inject_noise, self.dim).astype(np.float32)

            # Estabilidade medida na areia (bipolar derivada)
            new_bipolar = np.sign(self._field)
            unchanged = np.mean(new_bipolar == prev_bipolar)
            stabilities.append(float(unchanged))
            prev_bipolar = new_bipolar
            self.step_count += 1

        self.state = np.sign(self._field)
        self.history.append(self.state.copy())
        return stabilities

    # ─── Detecção de padrões emergentes ───────────────────────────────

    def detect_resonances(self, min_stability: float = 0.92,
                          freqs: Optional[List[float]] = None,
                          stability_series: Optional[List[float]] = None) -> Optional[ResonantPattern]:
        """
        Se o meio estabilizou acima de min_stability nos últimos steps,
        registra um padrão emergente. Caso contrário, retorna None.

        Isto é a detecção do "attractor" — exatamente como a areia que
        para de se mover na placa de Chladni.

        stability_series (opcional): se passado, usa a estabilidade final
        da série retornada por drive() (mais confiável que comparar histórico).
        """
        last = self.history[-1] if self.history else self.state
        if stability_series is not None and stability_series:
            stability = float(stability_series[-1])
        elif len(self.history) >= 2:
            stability = float(np.mean(last == self.history[-2]))
        else:
            return None

        if stability < min_stability:
            return None

        energy = float(np.mean(np.abs(np.fft.fft(last.astype(np.float64)))[:self.dim // 2]))
        pid = f"pat_{uuid.uuid4().hex[:8]}"
        pattern = ResonantPattern(
            pattern_id=pid,
            vector=last.copy(),
            driving_freqs=tuple(freqs) if freqs else (),
            stability=stability,
            energy=energy,
            notes=f"emerged after {self.step_count} steps, stability={stability:.3f}",
        )
        self.patterns[pid] = pattern
        return pattern

    # ─── Varredura de gargalo (a análise honesta) ─────────────────────

    def frequency_sweep(self, freq_range: Tuple[float, float] = (1.0, 50.0),
                        n_freqs: int = 20, steps_per_freq: int = 8,
                        evaluator: Optional[callable] = None) -> List[Dict]:
        """
        Varre frequências e mede, para cada uma:
          - estabilidade alcançada
          - energia do padrão
          - valor cognitivo (se evaluator fornecido)

        Se evaluator for uma função pattern->float, mede se a frequência
        produz estruturas com valor cognitivo. A frequência com maior
        valor cognitivo APONTA o gargalo: aquela estrutura faltava.
        """
        results = []
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

        for f in freqs:
            # Reset do meio a cada frequência para isolar o efeito
            self._reset_to_noise()
            stab_series = self.drive([float(f)], n_steps=steps_per_freq)
            final_stab = stab_series[-1] if stab_series else 0.0

            pattern = self.detect_resonances(min_stability=0.5, freqs=[float(f)],
                                            stability_series=stab_series)

            cog_value = 0.0
            if pattern is not None and evaluator is not None:
                try:
                    cog_value = float(evaluator(pattern)) or 0.0
                except Exception:
                    cog_value = 0.0

            results.append({
                "freq": float(f),
                "stability": final_stab,
                "cognitive_value": cog_value,
                "pattern_id": pattern.pattern_id if pattern else None,
                "energy": pattern.energy if pattern else 0.0,
            })

        return results

    def _reset_to_noise(self):
        self._field = self.rng.normal(0, 1, self.dim).astype(np.float32)
        self.state = np.sign(self._field)

    def reset(self):
        self._reset_to_noise()
        self.history.clear()
        self.step_count = 0

    # ─── Ponte para a massinha cognitiva ──────────────────────────────

    def pattern_to_clay(self, pattern: ResonantPattern,
                        name: str = None) -> Dict:
        """
        Converte um padrão ressonante em candidato a objeto da massinha.

        A conversão é HONESTA: extrai propriedades do padrão que emergiu,
        não inventa. Forma vem da simetria do padrão; propriedades vêm
        do espectro de energia; coerência vem da estabilidade.
        """
        vec = pattern.vector
        n = len(vec)

        # Simetria: compara primeira metade com segunda metade invertida
        half = n // 2
        symmetry = float(np.mean(vec[:half] == vec[half:][::-1]))

        # Densidade espectral de baixa vs alta frequência
        spectrum = np.abs(np.fft.fft(vec.astype(np.float64)))[:n // 2]
        low_energy = float(np.sum(spectrum[:n // 8]))
        high_energy = float(np.sum(spectrum[n // 8:]))
        spectral_centroid = low_energy / (low_energy + high_energy + 1e-9)

        # Forma inferida da simetria
        if symmetry > 0.7:
            shape = "circle"      # alta simetria = fechado/redondo
        elif symmetry > 0.55:
            shape = "rectangle"   # simetria média = estruturado
        else:
            shape = "blob"        # baixa simetria = orgânico

        # Propriedades físicas inferidas do espectro
        # (centroid alto = energia concentrada em alta freq = mais "dinâmico")
        props = {
            "rigidity": min(1.0, symmetry),           # simétrico = rígido
            "mobility": spectral_centroid,             # alta freq = móvel
            "temperature": 1.0 - spectral_centroid,    # baixa freq = "quente"? inverso
            "density": min(1.0, pattern.energy / 100.0),
            "fragility": 1.0 - symmetry,
        }

        return {
            "name": name or f"cymatic_{pattern.pattern_id}",
            "shape": shape,
            "physical_props": props,
            "symmetry": symmetry,
            "spectral_centroid": spectral_centroid,
            "source": "cymatic_emergent",
            "pattern_id": pattern.pattern_id,
            "stability": pattern.stability,
        }

    def summary(self) -> str:
        lines = [f"CymaticMedium(dim={self.dim}, steps={self.step_count})"]
        lines.append(f"  Padrões emergentes detectados: {len(self.patterns)}")
        for p in list(self.patterns.values())[-5:]:
            lines.append(f"    {p.pattern_id}: stab={p.stability:.3f} "
                         f"cog={p.cognitive_value:.3f} freqs={p.driving_freqs}")
        return "\n".join(lines)
