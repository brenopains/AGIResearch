"""
Neuromodulator — Vetor Multi-Dimensional [DA, NA, ACh, 5HT]
===========================================================
Implementa o sistema límbico artificial da NovaMind.

Referência: dados de fMRI 7T, ECoG, optogenética 2023-2026.
  - DA (Dopamina): progresso, exploração
  - NA (Noradrenalina): surpresa, foco atencional
  - ACh (Acetilcolina): plasticidade, janela de aprendizado
  - 5HT (Serotonina): confiança, inibição de impulsos

Uso:
    nm = Neuromodulator()
    nm.step(prediction_error=0.3, delta_pe=-0.1)
    if nm.should_forge(): forge.trigger()
    if nm.should_sleep(): sleep()
"""

import math
import time
from typing import Dict, Tuple


class Neuromodulator:
    """
    Vetor de 4 neuromoduladores com dinâmica biologicamente inspirada.
    Cada modulador ∈ [0, 1] com decaimento e regras de disparo.
    """

    # Nomes dos moduladores
    DA = 0  # Dopamine / Progress
    NA = 1  # Noradrenaline / Surprise
    ACh = 2  # Acetylcholine / Plasticity
    S5HT = 3  # Serotonin / Confidence

    def __init__(self):
        # Vetor principal [DA, NA, ACh, 5HT]
        self.v = [0.5, 0.0, 0.3, 0.5]  # estado inicial (repouso vigilante)

        # Rastreamento para detecção de padrões temporais
        self.pe_history = []       # prediction error recente
        self.pe_window = 10        # janela temporal
        self.consecutive_high_pe = 0
        self.consecutive_low_pe = 0

        # Estado emocional textual (para debug/UI)
        self._mood = "neutro"
        self._last_trigger = ""

    def step(self, prediction_error: float,
             delta_pe: float = 0.0,
             force_learn: bool = False) -> Dict[str, float]:
        """
        Atualiza todos os moduladores baseado no prediction error atual.

        Args:
            prediction_error: 0.0 (perfeito) a 1.0 (máxima surpresa)
            delta_pe: mudança no PE desde o último ciclo
            force_learn: se True, ACh sobe artificialmente (para debugging)

        Returns:
            Dict com valores atuais de [DA, NA, ACh, 5HT]
        """
        # Atualiza histórico
        self.pe_history.append(prediction_error)
        if len(self.pe_history) > self.pe_window:
            self.pe_history.pop(0)

        pe_medio = sum(self.pe_history) / len(self.pe_history)

        # Contadores de sequência
        if prediction_error > 0.6:
            self.consecutive_high_pe += 1
            self.consecutive_low_pe = 0
        elif prediction_error < 0.3:
            self.consecutive_low_pe += 1
            self.consecutive_high_pe = 0
        else:
            self.consecutive_high_pe = 0
            self.consecutive_low_pe = 0

        # === DA (Dopamina): Progresso ===
        # Sobe quando PE diminui (progresso), desce quando aumenta
        if delta_pe < -0.05:   # PE caiu → progresso
            self.v[self.DA] = min(1.0, self.v[self.DA] + 0.15)
        elif delta_pe > 0.05:  # PE subiu → regressão
            self.v[self.DA] = max(0.0, self.v[self.DA] - 0.08)
        # Decaimento lento
        self.v[self.DA] *= 0.997
        # Piso: nunca zera completamente (tônus dopaminérgico basal)
        self.v[self.DA] = max(0.05, self.v[self.DA])

        # === NA (Noradrenalina): Surpresa ===
        # Dispara em resposta a |PE| alto ou |delta_PE| abrupto
        na_spike = 0.0
        if prediction_error > 0.7:
            na_spike = 0.4
        elif prediction_error > 0.5:
            na_spike = 0.2
        if abs(delta_pe) > 0.15:
            na_spike = max(na_spike, 0.3)
        self.v[self.NA] = min(1.0, self.v[self.NA] + na_spike)
        # Decaimento rápido (noradrenalina é fásica)
        self.v[self.NA] *= 0.85
        self.v[self.NA] = max(0.0, self.v[self.NA])

        # === ACh (Acetilcolina): Plasticidade ===
        # Acumula com PE sustentado, dispara a janela de aprendizado
        ach_up = 0.0
        if prediction_error > 0.6:
            ach_up = 0.03  # acúmulo lento
        if self.consecutive_high_pe >= 3:
            ach_up += 0.05  # erro sustentado acelera plasticidade
        if force_learn:
            ach_up = 0.3
        self.v[self.ACh] = min(1.0, self.v[self.ACh] + ach_up)
        # Decaimento lento
        self.v[self.ACh] *= 0.99
        self.v[self.ACh] = max(0.1, self.v[self.ACh])  # tônus colinérgico basal

        # === 5HT (Serotonina): Confiança ===
        # Sobe com sucesso sustentado, desce com falha repetida
        if self.consecutive_low_pe >= 3:
            self.v[self.S5HT] = min(1.0, self.v[self.S5HT] + 0.08)
        if self.consecutive_high_pe >= 3:
            self.v[self.S5HT] = max(0.0, self.v[self.S5HT] - 0.1)
        # Decaimento lento
        self.v[self.S5HT] *= 0.998
        self.v[self.S5HT] = max(0.1, min(1.0, self.v[self.S5HT]))

        # Atualiza estado emocional
        self._update_mood()

        return self.as_dict()

    def _update_mood(self):
        """Estado emocional textual baseado no vetor atual."""
        da, na, ach, s5ht = self.v
        if na > 0.6 and ach > 0.5:
            self._mood = "frustrado"  # surpresa + plasticidade = confusão ativa
        elif da > 0.7 and s5ht > 0.6:
            self._mood = "confiante"   # progresso + confiança
        elif da < 0.2 and ach > 0.6:
            self._mood = "entediado"   # sem progresso, precisa de mudança
        elif na > 0.5:
            self._mood = "alerta"      # surpresa moderada
        elif s5ht < 0.3:
            self._mood = "inseguro"    # baixa confiança
        else:
            self._mood = "neutro"

    # === Gatilhos Comportamentais ===

    def should_forge(self) -> Tuple[bool, str]:
        """
        Dispara a Forja quando Surpresa + Plasticidade simultâneas.
        A Forja cria novos archetypes para reduzir o PE.
        """
        if self.v[self.NA] > 0.55 and self.v[self.ACh] > 0.5:
            self._last_trigger = "forge"
            return True, "forge"
        return False, ""

    def should_explore(self) -> bool:
        """Alta Dopamina → explorar (buscar novidade)."""
        return self.v[self.DA] > 0.65

    def should_exploit(self) -> bool:
        """Baixa Dopamina → aprofundar (refinar)."""
        return self.v[self.DA] < 0.3

    def should_sleep(self) -> Tuple[bool, str]:
        """
        Sono quando confiança baixa + plasticidade alta.
        O cérebro precisa consolidar antes de continuar.
        """
        if self.v[self.S5HT] < 0.3 and self.v[self.ACh] > 0.6:
            self._last_trigger = "sleep"
            return True, "sleep"
        return False, ""

    def should_switch_topic(self) -> bool:
        """Tédio: sem progresso (DA baixo) mas confiança normal."""
        return self.v[self.DA] < 0.2 and self.v[self.S5HT] > 0.4

    def attention_narrow(self) -> bool:
        """NA alto → foco estreito (ignorar distrações)."""
        return self.v[self.NA] > 0.5

    # === Acesso ===

    def as_dict(self) -> Dict[str, float]:
        return {
            "DA": round(self.v[self.DA], 3),
            "NA": round(self.v[self.NA], 3),
            "ACh": round(self.v[self.ACh], 3),
            "5HT": round(self.v[self.S5HT], 3),
        }

    @property
    def mood(self) -> str:
        return self._mood

    @property
    def last_trigger(self) -> str:
        return self._last_trigger

    def reset(self):
        """Reset para estado de repouso."""
        self.__init__()

    def __repr__(self) -> str:
        d = self.as_dict()
        return f"<Neuromodulator {d['DA']:.2f}/{d['NA']:.2f}/{d['ACh']:.2f}/{d['5HT']:.2f} [{self._mood}]>"
