"""
BodyEncoder — Codificação de Percepções Corporais em HDV Único da AGI
======================================================================
Codifica leituras de sensores corporais em vetores HDC de 10k dims
usando uma geometria ÚNICA e FIXA (seed da AGI).

Propriedade fundamental:
  Leituras similares → vetores similares no hiperespaço da AGI.
  Essa geometria não foi ensinada. Emerge da projeção com seed fixo.

Após 1000 quedas:
  encode_sensors({pain: 0.8, balance: 0.0, ...})
  → vetor próximo de "queda", "perigo", "recuperar"
  Sem ninguém ter dito isso explicitamente.

Isso É grounding: significado emergindo de experiência física repetida.
"""

import math
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# Importa tipos do simulated_body (sem criar dependência circular)
try:
    from .simulated_body import SensorReading, Actuator
except ImportError:
    from simulated_body import SensorReading, Actuator

DIM = 10_000

# Conceitos corporais primitivos (para decode_similarity)
BODY_CONCEPTS = {
    "falling":      {"pain": 0.0, "balance": 0.0, "velocity": [0, -5, 0], "reward": 0.0},
    "stable":       {"pain": 0.0, "balance": 1.0, "velocity": [0, 0, 0],  "reward": 0.0},
    "pain_high":    {"pain": 0.9, "balance": 0.5, "velocity": [0, 0, 0],  "reward": 0.0},
    "reward_zone":  {"pain": 0.0, "balance": 1.0, "velocity": [1, 0, 1],  "reward": 0.9},
    "danger_zone":  {"pain": 0.7, "balance": 0.8, "velocity": [0, 0, 0],  "reward": 0.0},
    "moving_fast":  {"pain": 0.0, "balance": 0.6, "velocity": [5, 0, 5],  "reward": 0.0},
    "resting":      {"pain": 0.0, "balance": 1.0, "velocity": [0, 0, 0],  "reward": 0.0},
    "exhausted":    {"pain": 0.1, "balance": 0.8, "velocity": [0, 0, 0],  "reward": 0.0},
    "impact":       {"pain": 0.8, "balance": 0.3, "velocity": [0, -8, 0], "reward": 0.0},
    "warmth":       {"pain": 0.0, "temperature": 0.6, "balance": 1.0,     "reward": 0.1},
    "cold":         {"pain": 0.1, "temperature": -0.7, "balance": 1.0,    "reward": 0.0},
}


class BodyEncoder:
    """
    Codificador de percepções corporais em HDV de 10k dims.

    A matriz de projeção é gerada com seed fixo (identidade da AGI)
    e nunca muda. Garante geometria consistente e única.
    """

    SENSOR_DIM = 20  # dimensão do vetor de sensores raw (ver SensorReading.to_array())
    ACTION_DIM = 4   # [dx, dy, dz, grasp]

    def __init__(self, seed: int = 88):
        self.seed = seed
        self.dim  = DIM
        rng       = np.random.default_rng(seed)

        # Matriz de projeção sensor→HDV (FIXA, identidade da AGI)
        # Inicializa com dimensão máxima segura (suficiente para os vetores raw)
        self._proj_sensor = rng.standard_normal((DIM, 25)).astype(np.float32)
        self._proj_action = rng.standard_normal((DIM, self.ACTION_DIM)).astype(np.float32)

        # Vetores de conceitos corporais pré-computados (para decode)
        self._concept_vecs: Dict[str, np.ndarray] = {}
        self._build_concept_vecs()

    def _build_concept_vecs(self):
        """Pré-computa vetores HDC dos conceitos corporais primitivos."""
        for name, attrs in BODY_CONCEPTS.items():
            # Constrói reading artificial com os atributos do conceito
            reading = SensorReading(
                pain        = attrs.get("pain", 0.0),
                temperature = attrs.get("temperature", 0.0),
                balance     = attrs.get("balance", 1.0),
                velocity    = attrs.get("velocity", [0.0, 0.0, 0.0]),
                reward      = attrs.get("reward", 0.0),
                energy_level = attrs.get("energy_level", 1.0),
            )
            self._concept_vecs[name] = self.encode_sensors(reading)

    def _project(self, raw_vec: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
        """Projeta vetor raw no hiperespaço HDC e binariza."""
        n_features = len(raw_vec)
        proj = proj_matrix[:, :n_features]
        result = proj @ raw_vec
        return np.sign(result).astype(np.float32)

    def encode_sensors(self, reading: SensorReading) -> np.ndarray:
        """
        Codifica leitura de sensores em vetor HDC bipolar de DIM dims.

        A geometria é única da AGI (seed=88). Leituras similares →
        vetores similares no hiperespaço. Isso é fundamentação corporal.

        Args:
            reading: SensorReading do SimulatedBody

        Returns:
            np.ndarray bipolar {-1, +1} de shape (DIM,)
        """
        raw = reading.to_array()
        return self._project(raw, self._proj_sensor)

    def encode_action(self, actuator: Actuator) -> np.ndarray:
        """
        Codifica ação de atuação em vetor HDC.

        Args:
            actuator: Actuator com move, grasp, etc.

        Returns:
            np.ndarray bipolar de shape (DIM,)
        """
        raw = np.array([
            actuator.move[0], actuator.move[1], actuator.move[2],
            actuator.grasp,
        ], dtype=np.float32)
        return self._project(raw, self._proj_action)

    def encode_state_action(self, reading: SensorReading, actuator: Actuator) -> np.ndarray:
        """
        Codifica par (estado, ação) para o Forward Model do Cerebelo.
        Usa XOR (bind) dos dois vetores.
        """
        s_vec = self.encode_sensors(reading)
        a_vec = self.encode_action(actuator)
        return (s_vec * a_vec).astype(np.float32)  # bind HDC

    def decode_similarity(self, vec: np.ndarray,
                          candidates: Optional[List[str]] = None) -> str:
        """
        Retorna o conceito corporal mais próximo do vetor dado.

        Args:
            vec:        vetor HDC a decodificar
            candidates: se fornecido, busca só nesses conceitos

        Returns:
            Nome do conceito mais similar (ex: 'falling', 'stable')
        """
        pool = candidates if candidates else list(self._concept_vecs.keys())
        best_sim  = -1.0
        best_name = "unknown"

        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-8:
            return "unknown"

        for name in pool:
            c_vec = self._concept_vecs.get(name)
            if c_vec is None:
                continue
            sim = float(np.dot(vec, c_vec) / (norm_vec * np.linalg.norm(c_vec) + 1e-8))
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        return best_name

    def get_body_concept(self, reading: SensorReading) -> str:
        """
        Retorna label semântico emergente para uma leitura de sensores.
        Prioriza casos extremos (dor alta, recompensa alta, etc.).
        """
        # Regras de prioridade (casos extremos primeiro)
        if reading.pain > 0.7:
            return "pain_high" if reading.balance > 0.5 else "impact"
        if reading.reward > 0.6:
            return "reward_zone"
        if reading.balance < 0.2:
            return "falling"
        if reading.energy_level < 0.2:
            return "exhausted"
        if abs(reading.temperature) > 0.6:
            return "warmth" if reading.temperature > 0 else "cold"
        vel_speed = math.sqrt(sum(v**2 for v in reading.velocity))
        if vel_speed > 3.0:
            return "moving_fast"
        if vel_speed < 0.1 and reading.balance > 0.9:
            return "resting" if reading.energy_level > 0.8 else "stable"

        # Fallback: similaridade com conceitos
        vec = self.encode_sensors(reading)
        return self.decode_similarity(vec)

    def get_state(self) -> Dict:
        return {
            "dim":            self.dim,
            "seed":           self.seed,
            "sensor_dim":     self.SENSOR_DIM,
            "concept_count":  len(self._concept_vecs),
            "concepts":       list(self._concept_vecs.keys()),
        }


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from simulated_body import SimulatedBody, Actuator
    except ImportError:
        from novamind.core.simulated_body import SimulatedBody, Actuator

    encoder = BodyEncoder(seed=88)
    body    = SimulatedBody(start_pos=[0.0, 3.0, 0.0])

    print("=== BodyEncoder: Geometria Corporal da AGI ===\n")

    # Simula queda
    for _ in range(20):
        reading = body.apply_actuator(Actuator(move=[0.0, 0.0, 0.0]))

    concept = encoder.get_body_concept(reading)
    vec     = encoder.encode_sensors(reading)
    print(f"Após queda: conceito='{concept}' pain={reading.pain:.2f} balance={reading.balance:.2f}")

    # Move para zona de recompensa
    body.reset(pos=[4.0, 0.0, 4.0])
    reading2 = body.read_sensors()
    concept2 = encoder.get_body_concept(reading2)
    print(f"Perto de reward: conceito='{concept2}' reward={reading2.reward:.2f}")

    # Similaridade entre duas leituras opostas
    v1 = encoder.encode_sensors(reading)
    v2 = encoder.encode_sensors(reading2)
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"\nSimilaridade queda vs recompensa: {sim:.3f} (deve ser baixa)")

    print(f"\nEstado do encoder: {encoder.get_state()}")
