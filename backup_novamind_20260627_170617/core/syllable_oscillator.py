"""
SyllableOscillator — Oscilações Teta-Gama para Ritmo Silábico
===============================================================
A produção de sílabas no cérebro humano é ritmada por oscilações
teta (~4-8 Hz) no córtex motor pré-central.

Cada ciclo teta = uma sílaba.
O conteúdo fonêmico da sílaba é carregado em sub-ciclos gama (~30-50 Hz).

Referência: "Temporal structure of the language hierarchy" (2026, arXiv)
Giraud & Poeppel (2012) — cortical oscillations for speech
"""

import numpy as np
import time
from typing import Dict, Optional, List


class SyllableOscillator:
    """
    Gera ritmo silábico via oscilador teta.

    Cada tick() avança a fase teta. Quando cruza o limiar de início
    de sílaba, retorna um burst de gestos gama para o planejamento
    fonêmico dentro da sílaba.

    Frequência teta típica: 4-8 Hz (bebês mais lentos ~3Hz, adultos ~5Hz)
    """
    def __init__(self, theta_freq: float = 4.5):
        self.theta_freq = theta_freq  # Hz
        self.theta_phase = 0.0
        self.gamma_freq = 35.0  # Hz
        self.last_tick_time = time.time()
        self.syllable_count = 0
        self.gesture_phase = 0.0
        self.is_syllable_onset = False

    def tick(self, dt: Optional[float] = None) -> Dict:
        """
        Avança o oscilador. Retorna estado atual.

        Returns:
            dict com:
              - syllable_onset: True se este tick inicia uma nova sílaba
              - gesture_window: duração da janela de gestos em ms
              - phase: fase teta atual (0-2pi)
              - syllable_index: índice da sílaba atual
        """
        if dt is None:
            now = time.time()
            dt = now - self.last_tick_time
            self.last_tick_time = now
            dt = min(dt, 0.5)

        old_phase = self.theta_phase
        self.theta_phase += dt * self.theta_freq * 2 * np.pi
        self.theta_phase %= 2 * np.pi

        self.is_syllable_onset = old_phase > self.theta_phase or (
            self.theta_phase < dt * self.theta_freq * 2 * np.pi
        )

        if self.is_syllable_onset:
            self.syllable_count += 1
            self.gesture_phase = 0.0

        gesture_window_ms = (1000.0 / self.theta_freq) * 0.8
        gamma_pulses = self._gamma_burst(dt)

        return {
            "syllable_onset": self.is_syllable_onset,
            "gesture_window_ms": gesture_window_ms,
            "theta_phase": float(self.theta_phase),
            "theta_phase_normalized": float(self.theta_phase / (2 * np.pi)),
            "syllable_index": self.syllable_count,
            "gamma_pulses": gamma_pulses,
        }

    def _gamma_burst(self, dt: float) -> List[float]:
        """
        Gera pulsos gama dentro da janela da sílaba atual.
        Cada pulso gama = oportunidade para um gesto fonêmico.
        Retorna tempos relativos (0-1) dentro da sílaba.
        """
        self.gesture_phase += dt * self.gamma_freq * 2 * np.pi
        pulses = []
        for i in range(int(self.gamma_freq / self.theta_freq)):
            pulse_phase = (self.gesture_phase + i * 2 * np.pi * self.theta_freq / self.gamma_freq) % (2 * np.pi)
            relative_position = pulse_phase / (2 * np.pi)
            if relative_position < (self.theta_freq / self.gamma_freq):
                pulses.append(float(relative_position))
        return pulses[:5]

    def reset(self):
        """Reinicia o oscilador."""
        self.theta_phase = 0.0
        self.syllable_count = 0
        self.last_tick_time = time.time()

    def get_state(self) -> Dict:
        return {
            "theta_freq_hz": self.theta_freq,
            "gamma_freq_hz": self.gamma_freq,
            "syllable_count": self.syllable_count,
            "current_phase_deg": float(self.theta_phase * 180 / np.pi),
            "is_syllable_onset": self.is_syllable_onset,
        }
