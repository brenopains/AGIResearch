"""
SimulatedBody — Corpo Físico Simulado com Sensores Fundamentados
================================================================
Implementa um corpo digital com sensores análogos aos humanos.
O objetivo NÃO é realismo gráfico — é causalidade fiel:
  "Pesado" significa que resistência a movimento é proporcional à massa.
  "Dor" significa que colisão de alta energia gera sinal nociceptivo.
  "Equilíbrio" significa que a força gravitacional desequilibra o corpo.

Princípio: Inteligência é efeito colateral de sensorimotor loops.
Sem corpo → sem grounding → sem significado real.

Referências:
  Brooks (1991) - Intelligence without representation
  Damasio (1994) - Descartes' Error (Somatic Marker Hypothesis)
  Pfeifer & Bongard (2006) - How the body shapes the way we think
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ─── Dataclasses de Sensores e Atuadores ──────────────────────────────────────

@dataclass
class SensorReading:
    """Snapshot completo de todos os sensores num instante."""

    # Somatossensorial
    pain:            float = 0.0   # 0.0 (nenhuma) → 1.0 (máxima dor)
    temperature:     float = 0.0   # -1.0 (frio extremo) → +1.0 (calor extremo)
    pressure:        float = 0.0   # força sobre superfície corporal (N normalizado)

    # Propriocepção / Espacial
    proprioception:  List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x, y, z]
    balance:         float = 1.0   # 0.0 (caindo) → 1.0 (estável e firme)
    gravity_vec:     List[float] = field(default_factory=lambda: [0.0, -9.8, 0.0])
    velocity:        List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration:    List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Temporal / Circadiano
    circadian:       float = 0.0   # fase 0 → 2π (representa hora do dia interna)
    duration:        float = 0.0   # tempo gasto nesta ação (segundos)
    urgency:         float = 0.0   # pressão temporal percebida (0=calmo, 1=urgente)

    # Hedônico / Homeostático
    reward:          float = 0.0   # sinal hedônico -1.0 → +1.0
    energy_level:    float = 1.0   # 0.0 (esgotado) → 1.0 (pleno)

    def to_array(self) -> np.ndarray:
        """Converte para vetor numpy para codificação HDC."""
        return np.array([
            self.pain, self.temperature, self.pressure,
            *self.proprioception,
            self.balance,
            *self.gravity_vec,
            *self.velocity,
            *self.acceleration,
            self.circadian / (2 * math.pi),   # normaliza para [0,1]
            self.duration / 60.0,              # normaliza por minuto
            self.urgency,
            self.reward,
            self.energy_level,
        ], dtype=np.float32)

    def to_dict(self) -> Dict:
        return {
            "pain": self.pain, "temperature": self.temperature,
            "pressure": self.pressure, "proprioception": self.proprioception,
            "balance": self.balance, "gravity_vec": self.gravity_vec,
            "velocity": self.velocity, "acceleration": self.acceleration,
            "circadian": self.circadian, "duration": self.duration,
            "urgency": self.urgency, "reward": self.reward,
            "energy_level": self.energy_level,
        }


@dataclass
class Actuator:
    """Comandos de atuação enviados pela AGI para o corpo."""
    move:     List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # vetor de movimento
    look_at:  List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # direção da atenção
    grasp:    float = 0.0       # força de preensão 0→1
    vocalize: str   = ""        # output de linguagem


# ─── SimulatedBody ────────────────────────────────────────────────────────────

class SimulatedBody:
    """
    Corpo físico simulado com sensores causalmente fundamentados.

    Estado interno:
      - Posição 3D (x, y, z)
      - Velocidade 3D
      - Aceleração 3D
      - Energia (decresce com atividade, recupera com descanso)
      - Fase circadiana (avança com o tempo)
      - Temperatura ambiente (pode variar por zona)
    """

    # Constantes físicas
    GRAVITY        = np.array([0.0, -9.8, 0.0], dtype=np.float32)
    MASS           = 1.0           # kg (normalizado)
    FRICTION       = 0.8           # coeficiente de fricção com chão
    MAX_SPEED      = 5.0           # m/s
    ENERGY_DECAY   = 0.001         # por tick de ação
    ENERGY_REGEN   = 0.0005        # por tick de descanso
    HEIGHT_FLOOR   = 0.0           # altura do chão
    PAIN_THRESHOLD = 5.0           # aceleração (m/s²) que começa a causar dor

    # Zonas de recompensa (posições que geram reward)
    REWARD_ZONES = [
        (np.array([5.0, 0.0, 5.0], dtype=np.float32), 1.0, 0.8),    # pos, raio, reward
        (np.array([-3.0, 0.0, -3.0], dtype=np.float32), 0.5, 0.5),
    ]
    # Zonas de perigo (posições que geram dor)
    DANGER_ZONES = [
        (np.array([0.0, 0.0, 8.0], dtype=np.float32), 0.8, 0.9),    # fogo
        (np.array([-5.0, 0.0, -5.0], dtype=np.float32), 1.2, 0.6),  # obstáculo
    ]

    def __init__(self, start_pos: Optional[List[float]] = None, world_temp: float = 0.2):
        # Cinemática
        self.pos  = np.array(start_pos or [0.0, 0.0, 0.0], dtype=np.float32)
        self.vel  = np.zeros(3, dtype=np.float32)
        self.acc  = np.zeros(3, dtype=np.float32)
        self.prev_vel = np.zeros(3, dtype=np.float32)

        # Estado interno
        self.energy       = 1.0
        self.temperature  = world_temp    # temperatura percebida pelo corpo
        self.circadian    = 0.0           # fase (avanças com time.time())
        self.birth_time   = time.time()
        self.last_tick    = time.time()

        # Contato com chão
        self.on_ground    = True

        # Histórico para cálculo de urgência
        self._pain_history:   List[float] = []
        self._reward_history: List[float] = []

        # Última leitura
        self._last_reading: Optional[SensorReading] = None
        self._action_start  = time.time()

    # ─── Física ───────────────────────────────────────────────────────────────

    def _physics_step(self, dt: float, force: np.ndarray):
        """Integração de Euler simples com gravidade e fricção."""
        self.prev_vel = self.vel.copy()

        # Força total = força aplicada + gravidade (se no ar)
        if self.on_ground:
            # No chão: cancela componente Y da gravidade, aplica fricção
            total_force = force.copy()
            total_force[1] = max(0.0, total_force[1])  # não afunda no chão
            friction_force = -self.vel * self.FRICTION
            total_force += friction_force
        else:
            total_force = force + self.GRAVITY * self.MASS

        # Integração
        self.acc = total_force / self.MASS
        self.vel = self.vel + self.acc * dt
        # Limita velocidade máxima
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = self.vel * (self.MAX_SPEED / speed)

        self.pos = self.pos + self.vel * dt

        # Colisão com chão
        if self.pos[1] <= self.HEIGHT_FLOOR:
            self.pos[1] = self.HEIGHT_FLOOR
            self.on_ground = True
            # Energia de impacto → dor se queda brusca
            impact = abs(self.vel[1])
            if impact > 2.0:
                self._record_impact(impact)
            self.vel[1] = 0.0
        else:
            self.on_ground = self.pos[1] <= self.HEIGHT_FLOOR + 0.01

    def _record_impact(self, impact_speed: float):
        """Registra impacto físico (para cálculo de dor)."""
        self._pain_history.append(min(1.0, impact_speed / 10.0))

    # ─── Sensores ─────────────────────────────────────────────────────────────

    def _compute_pain(self) -> float:
        """Dor: impactos recentes + zonas de perigo."""
        pain = 0.0
        # Impactos recentes
        if self._pain_history:
            pain = max(self._pain_history[-3:]) if len(self._pain_history) >= 3 else self._pain_history[-1]
            # Decay de dor
            self._pain_history = [p * 0.7 for p in self._pain_history[-10:] if p > 0.01]

        # Proximidade de zonas de perigo
        for zone_pos, zone_radius, zone_pain in self.DANGER_ZONES:
            dist = np.linalg.norm(self.pos - zone_pos)
            if dist < zone_radius:
                pain = max(pain, zone_pain * (1.0 - dist / zone_radius))

        return float(np.clip(pain, 0.0, 1.0))

    def _compute_reward(self) -> float:
        """Recompensa: proximidade de zonas de recompensa."""
        reward = 0.0
        for zone_pos, zone_radius, zone_reward in self.REWARD_ZONES:
            dist = np.linalg.norm(self.pos - zone_pos)
            if dist < zone_radius:
                reward = max(reward, zone_reward * (1.0 - dist / zone_radius))
        return float(np.clip(reward, 0.0, 1.0))

    def _compute_balance(self) -> float:
        """Equilíbrio: 1.0 se no chão e estável; decresce com velocidade e queda."""
        if not self.on_ground:
            # No ar: equilíbrio proporcional ao controle
            speed = np.linalg.norm(self.vel)
            return float(max(0.0, 1.0 - speed / self.MAX_SPEED))
        # No chão: equilíbrio baseado na velocidade lateral
        lateral_speed = math.sqrt(self.vel[0]**2 + self.vel[2]**2)
        return float(max(0.0, 1.0 - lateral_speed / (self.MAX_SPEED * 0.5)))

    def _compute_urgency(self) -> float:
        """Urgência: baseada na taxa de mudança de dor recente."""
        if len(self._pain_history) < 2:
            return 0.0
        delta = self._pain_history[-1] - self._pain_history[-2]
        return float(np.clip(delta * 5.0, 0.0, 1.0))

    def _compute_circadian(self) -> float:
        """Fase circadiana: avanças continuamente com o tempo real."""
        elapsed = time.time() - self.birth_time
        # 1 dia real = 1 ciclo circadiano (para acelerar: usar fator)
        phase = (elapsed / 86400.0) * 2 * math.pi
        return float(phase % (2 * math.pi))

    # ─── Interface pública ────────────────────────────────────────────────────

    def read_sensors(self) -> SensorReading:
        """Lê o estado atual de todos os sensores."""
        acc_magnitude = float(np.linalg.norm(self.acc))

        reading = SensorReading(
            pain           = self._compute_pain(),
            temperature    = float(np.clip(self.temperature, -1.0, 1.0)),
            pressure       = float(np.clip(acc_magnitude / 20.0, 0.0, 1.0)),
            proprioception = [float(self.pos[0]), float(self.pos[1]), float(self.pos[2])],
            balance        = self._compute_balance(),
            gravity_vec    = [0.0, -9.8, 0.0],
            velocity       = [float(self.vel[0]), float(self.vel[1]), float(self.vel[2])],
            acceleration   = [float(self.acc[0]), float(self.acc[1]), float(self.acc[2])],
            circadian      = self._compute_circadian(),
            duration       = float(time.time() - self._action_start),
            urgency        = self._compute_urgency(),
            reward         = self._compute_reward(),
            energy_level   = float(self.energy),
        )
        self._last_reading = reading
        return reading

    def apply_actuator(self, actuator: Actuator, dt: float = 0.016) -> SensorReading:
        """
        Aplica comandos de atuação e retorna a leitura dos sensores pós-ação.

        Args:
            actuator: comandos de movimento, atenção, preensão e vocalização
            dt:       timestep de simulação (default: 16ms ≈ 60fps)

        Returns:
            SensorReading após a ação.
        """
        self._action_start = time.time()

        # Converte comando de movimento em força
        move = np.array(actuator.move, dtype=np.float32)
        force_magnitude = np.linalg.norm(move)
        if force_magnitude > 1.0:
            move = move / force_magnitude  # normaliza
        force = move * 10.0  # escala de força (N)

        # Custo energético da ação
        action_cost = float(force_magnitude) * self.ENERGY_DECAY
        if force_magnitude < 0.01:
            # Descanso: regenera energia
            self.energy = min(1.0, self.energy + self.ENERGY_REGEN)
        else:
            self.energy = max(0.0, self.energy - action_cost)

        # Passo de física
        self._physics_step(dt, force)

        # Lê sensores pós-ação
        return self.read_sensors()

    def set_temperature(self, temp: float):
        """Define temperatura percebida pelo corpo (-1.0 a +1.0)."""
        self.temperature = float(np.clip(temp, -1.0, 1.0))

    def apply_external_force(self, force: List[float]):
        """Aplica força externa (vento, colisão, etc.)."""
        self._physics_step(0.016, np.array(force, dtype=np.float32))

    def reset(self, pos: Optional[List[float]] = None):
        """Reseta o corpo para posição inicial."""
        self.pos   = np.array(pos or [0.0, 0.0, 0.0], dtype=np.float32)
        self.vel   = np.zeros(3, dtype=np.float32)
        self.acc   = np.zeros(3, dtype=np.float32)
        self.energy = 1.0
        self._pain_history = []
        self._reward_history = []
        self.on_ground = True

    def get_state(self) -> Dict:
        return {
            "pos":       self.pos.tolist(),
            "vel":       self.vel.tolist(),
            "energy":    float(self.energy),
            "on_ground": self.on_ground,
            "circadian": self._compute_circadian(),
            "pain_level": self._compute_pain(),
            "reward":    self._compute_reward(),
        }


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    body = SimulatedBody(start_pos=[0.0, 5.0, 0.0])  # começa no ar

    print("=== Simulação de Corpo Físico ===\n")
    print(f"Pos inicial: {body.pos.tolist()}")

    # Queda livre (sem ação)
    for i in range(30):
        s = body.apply_actuator(Actuator(move=[0.0, 0.0, 0.0]))
        if i % 5 == 0:
            print(f"t={i}: pos_y={s.proprioception[1]:.2f} "
                  f"balance={s.balance:.2f} pain={s.pain:.2f}")

    print(f"\nAtingiu o chão. Pos: {body.pos.tolist()}")
    print(f"Pain pós-impacto: {body.read_sensors().pain:.2f}")

    # Move em direção à zona de recompensa
    body.reset(pos=[0.0, 0.0, 0.0])
    for i in range(20):
        s = body.apply_actuator(Actuator(move=[0.25, 0.0, 0.25]))
        if i % 5 == 0:
            print(f"Movendo... pos={[round(p,1) for p in s.proprioception]} reward={s.reward:.2f}")
