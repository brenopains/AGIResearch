"""
NeuroField — Campo Neuromodulator Continuo

Substitui os triggers booleanos do Neuromodulator (should_forge, should_sleep)
por PARAMETROS CONTINUOS que modulam diretamente a dinamica do sistema.

Diferenca fundamental:
  Antes: "if NA > 0.55 and ACh > 0.5 -> should_forge() = True"
  Depois: "plasticity = sigmoid(NA * ACh - 0.3)" (continuo, 0..1)

Nao existem modos. Existem valores fisicos continuos que alteram
a temperatura do Gating Field, a taxa de aprendizado, a forca
dos atratores, a intensidade da atencao, a pressao de consolidacao.

O comportamento de "exploracao" ou "foco" emerge desses valores,
NAO e decidido por if/else.
"""
import time, math
from typing import Dict, Tuple, Optional


class NeuroField:
    """Campo continuo de neuromodulacao.

    Entrada: prediction_error, novelty, uncertainty, violation, fatigue
    Saida: parametros continuos que modulam o sistema.

    NENHUM trigger booleano. Tudo e continuo.
    """

    def __init__(self):
        # Estado interno dos neuromoduladores (0..1)
        self.DA: float = 0.5    # Dopamina: sinal de progresso
        self.NA: float = 0.0    # Noradrenalina: surpresa/alerta
        self.ACh: float = 0.3   # Acetilcolina: plasticidade/foco
        self.S5HT: float = 0.5  # Serotonina: confianca/estabilidade

        # Derivadas temporais (velocidade de mudanca)
        self.dDA: float = 0.0
        self.dNA: float = 0.0
        self.dACh: float = 0.0
        self.dS5HT: float = 0.0

        # Historico para calculo de tendencias
        self._history: Dict[str, list] = {
            "DA": [], "NA": [], "ACh": [], "5HT": []
        }
        self._max_history: int = 20
        self._last_update: float = time.time()

    # ─── Atualizacao ───────────────────────────────────────

    def update(self, prediction_error: float = 0.0,
               novelty: float = 0.0, uncertainty: float = 0.0,
               violation: float = 0.0, fatigue: float = 0.0,
               reward: float = 0.0):
        """Atualiza todos os neuromoduladores baseado nos erros.

        REGRA: Nao ha thresholds para "modos".
        Os valores mudam continuamente e o efeito e gradual.
        """
        # Salva estado anterior para derivadas
        old = (self.DA, self.NA, self.ACh, self.S5HT)

        # DA (Dopamina): progresso = reducao de erro ou recompensa
        # Sobe com progresso, desce lentamente
        if reward > 0:
            self.DA += reward * 0.2
        elif prediction_error > 0:
            self.DA -= prediction_error * 0.05
        self.DA = max(0.05, min(1.0, self.DA * 0.995))

        # NA (Noradrenalina): surpresa = erro alto ou novidade alta
        self.NA = max(0.0, min(1.0, prediction_error * 0.6 + novelty * 0.4))
        self.NA *= 0.85  # decaimento fasico rapido

        # ACh (Acetilcolina): plasticidade = erro + incerteza sustentados
        self.ACh += prediction_error * 0.03 + uncertainty * 0.02
        self.ACh = max(0.1, min(1.0, self.ACh * 0.995))

        # 5HT (Serotonina): confianca = baixo erro sustentado
        if prediction_error < 0.2 and novelty < 0.3:
            self.S5HT += 0.02
        elif prediction_error > 0.5:
            self.S5HT -= 0.03
        self.S5HT = max(0.1, min(1.0, self.S5HT * 0.998))

        # Derivadas temporais
        self.dDA = self.DA - old[0]
        self.dNA = self.NA - old[1]
        self.dACh = self.ACh - old[2]
        self.dS5HT = self.S5HT - old[3]

        # Historico
        for name, val in [("DA", self.DA), ("NA", self.NA),
                          ("ACh", self.ACh), ("5HT", self.S5HT)]:
            self._history[name].append(val)
            if len(self._history[name]) > self._max_history:
                self._history[name] = self._history[name][-self._max_history:]

        self._last_update = time.time()

    # ─── Parametros Continuos de Modulacao ─────────────────
    # NENHUM if/else. Tudo e calculo continuo.

    @property
    def gating_temperature(self) -> float:
        """Temperatura do Gating Field (softmax).

        Alta = multiplas portas abertas (exploracao difusa)
        Baixa = uma porta dominante (foco extremo)

        Determinantes: NA (abre), ACh (fecha)
        """
        t = 0.5 + self.NA * 0.5 - self.ACh * 0.3
        return max(0.1, min(1.0, t))

    @property
    def exploration_drive(self) -> float:
        """Drive exploratorio (0..1).

        Alto = sistema busca novidade
        Baixo = sistema aprofunda o que tem

        Formula: DA alto + NA alto + 5HT baixo = exploracao
        """
        return max(0.0, min(1.0, (self.DA * 0.4 + self.NA * 0.5 - self.S5HT * 0.3)))

    @property
    def focus_intensity(self) -> float:
        """Intensidade do foco atencional (0..1).

        Alta = spotlight estreito, ignora distracoes
        Baixa = atencao difusa, multiplos estimulos

        Determinantes: ACh alto (foco), NA baixo (sem surpresa)
        """
        return max(0.0, min(1.0, self.ACh * 0.6 + (1 - self.NA) * 0.4))

    @property
    def plasticity(self) -> float:
        """Taxa de plasticidade / aprendizado (0..1).

        Alta = CGE cria/modifica arestas facilmente
        Baixa = CGE preserva conhecimento existente

        Determinantes: ACh alto + erro sustentado
        """
        ach_erro = self.ACh * (1 - self.S5HT)
        return max(0.0, min(1.0, ach_erro * 0.8))

    @property
    def consolidation_pressure(self) -> float:
        """Pressao para consolidar/abstrair (0..1).

        Alta = sistema precisa organizar memoria
        Baixa = sistema pode continuar aprendendo

        Determinantes: fadiga alta + plasticidade baixa
        """
        fatigue = self._history["DA"][-1] if self._history["DA"] else 0.5
        return max(0.0, min(1.0, (1 - self.plasticity) * 0.5 + fatigue * 0.3))

    @property
    def creativity(self) -> float:
        """Indice de criatividade (0..1).

        Alta = sistema combina conceitos distantes
        Baixa = sistema segue caminhos conhecidos

        Determinantes: NA + exploracao - foco
        """
        return max(0.0, min(1.0, (self.exploration_drive * 0.6 +
                                  (1 - self.focus_intensity) * 0.4)))

    # ─── Flutuacao Estrutural (Anti-Cristalizacao) ─────────

    @property
    def fluctuation_noise(self) -> float:
        """Ruido estruturado para evitar cristalizacao cognitiva.

        Quando o sistema esta muito estavel (baixo erro, alta confianca,
        baixa temperatura), ele pode cair em mínimos locais.
        A flutuacao adiciona ruido PROPORCIONAL a:
          - (1 - exploration_drive): sistemas estaveis precisam de mais ruido
          - (1 - gating_temperature): foco extremo precisa de mais perturbacao
          - fator aleatorio gaussiano

        A formula garante que:
          - Altamente instavel (erro alto): ruido minimo (ja esta explorando)
          - Altamente estavel (erro baixo, foco alto): ruido maximo (evitar cristalizar)
        """
        stability = (1 - self.exploration_drive) * (1 - self.gating_temperature)
        noise = stability * 0.15 * (1 + 0.3 * (hash(str(time.time() % 1)) % 100) / 100)
        return min(0.15, max(0.0, noise))

    def perturb_state(self) -> Dict[str, float]:
        """Aplica flutuacao aos neuromoduladores (micro-mutacao).

        Retorna o quanto cada neuromodulador foi perturbado.
        A perturbacao e proporcional a estabilidade do sistema:
          - Sistemas estaveis recebem mais ruido (anti-cristalizacao)
          - Sistemas instaveis recebem menos ruido (ja estao explorando)
        """
        noise = self.fluctuation_noise
        import random

        delta = {
            "DA": random.gauss(0, noise),
            "NA": random.gauss(0, noise * 0.5),
            "ACh": random.gauss(0, noise * 0.3),
            "5HT": random.gauss(0, noise * 0.2),
        }
        self.DA = max(0.05, min(1.0, self.DA + delta["DA"]))
        self.NA = max(0.0, min(1.0, self.NA + delta["NA"]))
        self.ACh = max(0.1, min(1.0, self.ACh + delta["ACh"]))
        self.S5HT = max(0.1, min(1.0, self.S5HT + delta["5HT"]))
        return delta

    # ─── Estado ─────────────────────────────────────────────

    def get_state(self) -> Dict:
        """Estado completo serializavel."""
        return {
            "DA": round(self.DA, 3), "NA": round(self.NA, 3),
            "ACh": round(self.ACh, 3), "5HT": round(self.S5HT, 3),
            "dDA": round(self.dDA, 3), "dNA": round(self.dNA, 3),
            "dACh": round(self.dACh, 3), "dS5HT": round(self.dS5HT, 3),
            "gating_temperature": round(self.gating_temperature, 3),
            "exploration_drive": round(self.exploration_drive, 3),
            "focus_intensity": round(self.focus_intensity, 3),
            "plasticity": round(self.plasticity, 3),
            "consolidation_pressure": round(self.consolidation_pressure, 3),
            "creativity": round(self.creativity, 3),
        }
