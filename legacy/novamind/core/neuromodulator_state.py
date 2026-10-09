"""
NeuromodulatorState — Camada de Computação Neuromoduladora Contextual
=====================================================================
Expande o neuromodulator.py (DA, NA, ACh, 5HT) com:
  - CRH (cortisol/ameaça) e OXT (ocitocina/confiança social)
  - Assinatura NT por conceito/arquétipo (cada conceito carrega NT base)
  - Modificadores causais: mesmo conceito, NT diferente por contexto
  - Ganho variável por operação: learning_rate, attention_width, etc.
  - Atualização via sensores corporais

Hipótese central:
    Sentimentos = dosagens de NTs acopladas a arquétipos num contexto causal.
    "Fogo" na cozinha ≠ "Fogo" com vazamento de gás.

Referências: Damasio (Somatic Marker), Barrett (Constructed Emotion),
             Friston (Active Inference + Interoception).
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─── Índices dos neurotransmissores ────────────────────────────────────────────
DA   = 0  # Dopamina      — recompensa, motivação, exploração
NA   = 1  # Noradrenalina — surpresa, foco atencional, signal-to-noise
ACh  = 2  # Acetilcolina  — plasticidade, janela de aprendizado, modo sensorial
S5HT = 3  # Serotonina    — horizonte temporal, confiança, inibição de impulso
CRH  = 4  # Cortisol/CRH  — ameaça, stress, vigilância de perigo
OXT  = 5  # Ocitocina     — confiança social, cooperação, apego

NT_NAMES = ["DA", "NA", "ACh", "5HT", "CRH", "OXT"]
N_NTS    = 6

# Estado de repouso basal (normativa fisiológica)
BASAL_STATE = np.array([0.50, 0.05, 0.30, 0.50, 0.10, 0.30], dtype=np.float32)


# ─── Assinaturas NT base por conceito ─────────────────────────────────────────
# Formato: {concept_key: np.array([DA, NA, ACh, 5HT, CRH, OXT])}
# Valores em [0, 1]. 0.5 = neutro/basal.
CONCEPT_NT_BASE: Dict[str, np.ndarray] = {
    # Perigo e ameaça
    "fire":     np.array([0.3, 0.6, 0.5, 0.3, 0.7, 0.1], dtype=np.float32),
    "danger":   np.array([0.2, 0.8, 0.4, 0.2, 0.9, 0.1], dtype=np.float32),
    "pain":     np.array([0.1, 0.7, 0.6, 0.2, 0.9, 0.1], dtype=np.float32),
    "fall":     np.array([0.2, 0.8, 0.5, 0.2, 0.8, 0.1], dtype=np.float32),
    "cold":     np.array([0.3, 0.5, 0.4, 0.4, 0.5, 0.2], dtype=np.float32),
    "heat":     np.array([0.3, 0.5, 0.4, 0.3, 0.6, 0.2], dtype=np.float32),

    # Recompensa e prazer
    "reward":   np.array([0.9, 0.3, 0.4, 0.7, 0.1, 0.4], dtype=np.float32),
    "success":  np.array([0.8, 0.2, 0.4, 0.8, 0.1, 0.4], dtype=np.float32),
    "food":     np.array([0.8, 0.2, 0.3, 0.6, 0.1, 0.3], dtype=np.float32),
    "rest":     np.array([0.4, 0.1, 0.2, 0.8, 0.1, 0.5], dtype=np.float32),

    # Social
    "social":   np.array([0.6, 0.3, 0.4, 0.6, 0.1, 0.8], dtype=np.float32),
    "trust":    np.array([0.6, 0.2, 0.3, 0.7, 0.0, 0.9], dtype=np.float32),
    "threat":   np.array([0.2, 0.8, 0.5, 0.2, 0.9, 0.0], dtype=np.float32),

    # Cognitivo
    "unknown":  np.array([0.4, 0.6, 0.7, 0.3, 0.3, 0.2], dtype=np.float32),
    "learn":    np.array([0.6, 0.4, 0.8, 0.5, 0.1, 0.3], dtype=np.float32),
    "boring":   np.array([0.2, 0.1, 0.2, 0.5, 0.1, 0.3], dtype=np.float32),
    "curious":  np.array([0.7, 0.5, 0.7, 0.5, 0.1, 0.3], dtype=np.float32),

    # Ferramentas/ambiente neutro
    "water":    np.array([0.5, 0.2, 0.3, 0.6, 0.0, 0.3], dtype=np.float32),
    "tool":     np.array([0.5, 0.3, 0.5, 0.5, 0.1, 0.3], dtype=np.float32),
    "code":     np.array([0.6, 0.3, 0.7, 0.5, 0.1, 0.2], dtype=np.float32),
}

# ─── Modificadores causais de contexto ────────────────────────────────────────
# Formato: {concept_key: [{context: str, delta: np.array}]}
# delta é ADICIONADO à assinatura base quando o contexto está ativo.
CAUSAL_MODIFIERS: Dict[str, List[Dict]] = {
    "fire": [
        # Fogo na cozinha = controlado, útil → CRH baixa, DA sobe levemente
        {"context": "kitchen",   "delta": np.array([ 0.1, -0.2,  0.0, 0.0, -0.4,  0.0], dtype=np.float32)},
        # Fogo + gás = perigo iminente → CRH dispara, NA dispara
        {"context": "gas_leak",  "delta": np.array([-0.2,  0.4,  0.1, -0.2, 0.5, -0.1], dtype=np.float32)},
        # Fogo na lareira = conforto → CRH cai, OXT sobe
        {"context": "fireplace", "delta": np.array([ 0.2, -0.3, -0.1,  0.2, -0.4,  0.3], dtype=np.float32)},
        # Incêndio = emergência máxima
        {"context": "fire_alarm", "delta": np.array([-0.3, 0.5,  0.2, -0.3,  0.6, -0.2], dtype=np.float32)},
    ],
    "water": [
        {"context": "ocean",     "delta": np.array([ 0.0,  0.3,  0.2, -0.1,  0.3, -0.1], dtype=np.float32)},
        {"context": "drinking",  "delta": np.array([ 0.2, -0.1,  0.0,  0.2, -0.1,  0.0], dtype=np.float32)},
        {"context": "flooding",  "delta": np.array([-0.2,  0.5,  0.3, -0.2,  0.5, -0.1], dtype=np.float32)},
    ],
    "unknown": [
        {"context": "safe",      "delta": np.array([ 0.1, -0.1,  0.1,  0.1, -0.1,  0.1], dtype=np.float32)},
        {"context": "dark",      "delta": np.array([-0.1,  0.3,  0.1, -0.1,  0.3, -0.1], dtype=np.float32)},
    ],
    "reward": [
        {"context": "earned",    "delta": np.array([ 0.2,  0.0,  0.1,  0.1,  0.0,  0.1], dtype=np.float32)},
        {"context": "random",    "delta": np.array([ 0.1,  0.2,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)},
    ],
}


class NeuromodulatorState:
    """
    Estado neuromodulador completo da AGI com 6 dimensões.

    Controla:
    - Assinaturas NT por conceito ativo
    - Modificação causal por contexto
    - Ganho adaptativo de operações cognitivas
    - Atualização por sinais corporais (dor, recompensa, equilíbrio)
    - Decay temporal (cada NT tem meia-vida própria)
    """

    # Meias-vidas de cada NT (em segundos) — quanto tempo leva para cair 50%
    HALF_LIVES = np.array([
        60.0,   # DA   — decai lento (motivação sustentada)
         5.0,   # NA   — decai rápido (alerta é fásico)
        30.0,   # ACh  — médio (janela de plasticidade)
       120.0,   # 5HT  — muito lento (estado de humor)
        15.0,   # CRH  — médio-rápido (stress agudo)
        90.0,   # OXT  — lento (vínculo social)
    ], dtype=np.float32)

    def __init__(self):
        # Estado atual dos 6 NTs
        self.v: np.ndarray = BASAL_STATE.copy()

        # Contextos ativos (lista de strings)
        self.active_contexts: List[str] = []

        # Histórico para diagnóstico
        self.history: List[np.ndarray] = []
        self.max_history = 100

        # Momento da última atualização
        self._last_tick: float = time.time()

        # Mood textual
        self._mood: str = "neutro"

    # ─── Atualização temporal (decay) ─────────────────────────────────────────

    def tick(self, dt: Optional[float] = None):
        """
        Aplica decay temporal a todos os NTs.
        Se dt=None, usa tempo real desde o último tick.
        """
        now = time.time()
        if dt is None:
            dt = now - self._last_tick
        self._last_tick = now

        # Decay exponencial: v *= 0.5 ^ (dt / half_life)
        decay = np.power(0.5, dt / self.HALF_LIVES)
        # Decai em direção ao estado basal, não ao zero
        self.v = BASAL_STATE + (self.v - BASAL_STATE) * decay
        self.v = np.clip(self.v, 0.0, 1.0)
        self._update_mood()

    # ─── Resolução contextual de conceitos ────────────────────────────────────

    def resolve_concept_nt(self,
                           concept_key: str,
                           context_keys: Optional[List[str]] = None) -> np.ndarray:
        """
        Retorna o vetor NT resultante para um conceito dado o contexto ativo.

        Args:
            concept_key:  chave do conceito (ex: 'fire', 'reward')
            context_keys: lista de contextos ativos (ex: ['kitchen', 'gas_leak'])

        Returns:
            np.ndarray de 6 dims com NT modificado por contexto.
        """
        # Base: assinatura do conceito (ou basal se desconhecido)
        base = CONCEPT_NT_BASE.get(concept_key, BASAL_STATE).copy()

        # Aplica modificadores causais relevantes
        if context_keys:
            modifiers = CAUSAL_MODIFIERS.get(concept_key, [])
            for mod in modifiers:
                # Checa se algum contexto ativo é substring do modificador
                for ctx in context_keys:
                    if ctx in mod["context"] or mod["context"] in ctx:
                        base = self.blend(base, mod["delta"], alpha=0.8)
                        break

        return np.clip(base, 0.0, 1.0)

    def activate_concept(self,
                         concept_key: str,
                         context_keys: Optional[List[str]] = None,
                         strength: float = 0.3):
        """
        Ativa um conceito: injeta sua assinatura NT no estado atual.

        Args:
            concept_key:  conceito a ativar
            context_keys: contextos que modificam a assinatura
            strength:     quanto da assinatura é injetado (0.0–1.0)
        """
        sig = self.resolve_concept_nt(concept_key, context_keys)
        # Blending: mistura o estado atual com a assinatura do conceito
        self.v = (1.0 - strength) * self.v + strength * sig
        self.v = np.clip(self.v, 0.0, 1.0)
        self._update_mood()

        # Registra no histórico
        self.history.append(self.v.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def set_contexts(self, contexts: List[str]):
        """Atualiza a lista de contextos globais ativos."""
        self.active_contexts = contexts

    # ─── Atualização por sensores corporais ───────────────────────────────────

    def update_from_body(self, sensor_dict: Dict[str, float]):
        """
        Atualiza NTs baseado em leituras de sensores corporais.

        Mapeamentos:
          pain    → CRH sobe, DA cai, OXT cai
          reward  → DA sobe, 5HT sobe, CRH cai
          balance → NA: desequilíbrio gera NA (surpresa proprioceptiva)
          temperature (extrema) → CRH sobe, NA sobe
          energy_level (baixo)  → DA cai (fome reduz motivação)
        """
        pain    = float(sensor_dict.get("pain", 0.0))
        reward  = float(sensor_dict.get("reward", 0.0))
        balance = float(sensor_dict.get("balance", 1.0))
        temp    = abs(float(sensor_dict.get("temperature", 0.0)))  # magnitude
        energy  = float(sensor_dict.get("energy_level", 1.0))

        delta = np.zeros(N_NTS, dtype=np.float32)

        # Dor → stress agudo
        if pain > 0.3:
            delta[CRH] += pain * 0.5
            delta[DA]  -= pain * 0.3
            delta[NA]  += pain * 0.4
            delta[OXT] -= pain * 0.2

        # Recompensa → dopamina + serotonina
        if reward > 0.0:
            delta[DA]   += reward * 0.5
            delta[S5HT] += reward * 0.3
            delta[CRH]  -= reward * 0.2

        # Desequilíbrio → noradrenalina (surpresa proprioceptiva)
        imbalance = 1.0 - balance
        if imbalance > 0.3:
            delta[NA]  += imbalance * 0.4
            delta[CRH] += imbalance * 0.2

        # Temperatura extrema → stress
        if temp > 0.6:
            delta[CRH] += (temp - 0.6) * 0.5
            delta[NA]  += (temp - 0.6) * 0.3

        # Energia baixa → motivação cai
        if energy < 0.3:
            delta[DA]  -= (0.3 - energy) * 0.3

        # Aplica delta com saturação
        self.v = np.clip(self.v + delta, 0.0, 1.0)
        self._update_mood()

    # ─── Ganho adaptativo por operação ────────────────────────────────────────

    def get_gain(self, operation: str) -> float:
        """
        Retorna o fator de ganho para uma operação cognitiva.

        Operations:
          'learning_rate'   → DA + ACh modulam velocidade de aprendizado
          'attention_width' → NA modula foco (alto NA = foco estreito)
          'memory_mode'     → ACh decide: sensorial (ACh alto) vs memória (ACh baixo)
          'time_horizon'    → 5HT: alto = longo prazo; baixo = impulsivo
          'threat_bias'     → CRH: alto = tudo parece perigoso
          'trust_level'     → OXT: alto = confia em inputs externos
          'exploration'     → DA: alto = explora; baixo = aproveita o conhecido
        """
        da, na, ach, s5ht, crh, oxt = self.v

        gains = {
            "learning_rate":   float(0.3 + 0.5 * da + 0.3 * ach - 0.2 * crh),
            "attention_width": float(1.0 - 0.7 * na),   # alto NA = 0.3 (estreito), baixo = 1.0 (amplo)
            "memory_mode":     float(ach),                # >0.5: sensorial; <0.5: memória interna
            "time_horizon":    float(s5ht),               # >0.5: longo prazo
            "threat_bias":     float(crh),                # >0.5: interpreta como ameaça
            "trust_level":     float(oxt),                # >0.5: confia em inputs
            "exploration":     float(da),                 # >0.5: exploração
        }
        return float(np.clip(gains.get(operation, 0.5), 0.0, 1.0))

    # ─── Gatilhos comportamentais ──────────────────────────────────────────────

    def should_freeze(self) -> bool:
        """CRH muito alto → paralisia (freeze response)."""
        return float(self.v[CRH]) > 0.85

    def should_flee(self) -> bool:
        """CRH alto + NA alto → fuga."""
        return float(self.v[CRH]) > 0.7 and float(self.v[NA]) > 0.6

    def should_approach(self) -> bool:
        """DA alto + OXT alto → aproximação/curiosidade social."""
        return float(self.v[DA]) > 0.65 and float(self.v[OXT]) > 0.5

    def should_consolidate(self) -> bool:
        """ACh alto + DA baixo → hora de consolidar memória (sleep)."""
        return float(self.v[ACh]) > 0.6 and float(self.v[DA]) < 0.3

    def salience(self) -> float:
        """
        Saliência emocional: quanto este momento merece ser lembrado.
        Alta saliência → L1→L2 na plasticidade multi-escala.
        """
        return float(max(self.v[DA], self.v[CRH], self.v[NA]))

    # ─── Utilitários ──────────────────────────────────────────────────────────

    @staticmethod
    def blend(base: np.ndarray, delta: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Mistura base com delta de forma suave.
        alpha controla o quanto o delta influencia.
        """
        return base + alpha * delta

    def inject(self, nt_index: int, amount: float):
        """Injeta uma quantidade num NT específico (para debug/experimentação)."""
        self.v[nt_index] = float(np.clip(self.v[nt_index] + amount, 0.0, 1.0))
        self._update_mood()

    def _update_mood(self):
        """Estado emocional textual (para UI e logs)."""
        da, na, ach, s5ht, crh, oxt = self.v
        if crh > 0.8:
            self._mood = "apavorado"
        elif crh > 0.6 and na > 0.5:
            self._mood = "alerta-perigo"
        elif da > 0.7 and s5ht > 0.6:
            self._mood = "confiante"
        elif da > 0.6 and oxt > 0.6:
            self._mood = "social-curioso"
        elif na > 0.6 and ach > 0.5:
            self._mood = "frustrado"
        elif da < 0.2 and ach > 0.6:
            self._mood = "entediado"
        elif s5ht < 0.3:
            self._mood = "inseguro"
        elif crh > 0.4:
            self._mood = "ansioso"
        elif da > 0.5 and crh < 0.2:
            self._mood = "motivado"
        else:
            self._mood = "neutro"

    def as_dict(self) -> Dict[str, float]:
        return {name: round(float(self.v[i]), 3) for i, name in enumerate(NT_NAMES)}

    @property
    def mood(self) -> str:
        return self._mood

    def __repr__(self) -> str:
        d = self.as_dict()
        return (f"<NTState DA={d['DA']:.2f} NA={d['NA']:.2f} "
                f"ACh={d['ACh']:.2f} 5HT={d['5HT']:.2f} "
                f"CRH={d['CRH']:.2f} OXT={d['OXT']:.2f} [{self._mood}]>")


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nts = NeuromodulatorState()

    print("=== Teste de Neuromodulação Contextual ===\n")

    # Fogo em contexto neutro
    fire_neutral = nts.resolve_concept_nt("fire", [])
    print(f"Fogo (neutro):     CRH={fire_neutral[CRH]:.2f}  DA={fire_neutral[DA]:.2f}")

    # Fogo na cozinha
    fire_kitchen = nts.resolve_concept_nt("fire", ["kitchen"])
    print(f"Fogo (cozinha):    CRH={fire_kitchen[CRH]:.2f}  DA={fire_kitchen[DA]:.2f}")

    # Fogo + vazamento de gás
    fire_gas = nts.resolve_concept_nt("fire", ["gas_leak"])
    print(f"Fogo (gás):        CRH={fire_gas[CRH]:.2f}  DA={fire_gas[DA]:.2f}")

    print()
    # Atualiza via sensores
    nts.update_from_body({"pain": 0.8, "balance": 0.2, "reward": 0.0, "energy_level": 0.9})
    print(f"Após dor+queda:    {nts}")
    print(f"Should flee:       {nts.should_flee()}")
    print(f"Learning rate:     {nts.get_gain('learning_rate'):.2f}")
    print(f"Threat bias:       {nts.get_gain('threat_bias'):.2f}")
