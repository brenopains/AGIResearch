"""
PhysicsSim — Simulador Físico 3D Leve em Numpy para AGI
======================================================
Simulador físico simplificado de corpos rígidos 3D:
- Integração de Euler para movimento
- Gravidade global (eixo Z negativo)
- Atrito com o ar e com o solo
- Colisões com plano do solo (Z=0)
- Colisões entre esferas
"""

import numpy as np
from typing import Dict, List, Union

class PhysicsSim:
    """
    Simulador de física 3D leve para suportar o loop de incorporação.
    Gerencia gravidade, colisões e forças aplicadas.
    """
    def __init__(self, gravity: float = 9.8, friction: float = 0.1, ground_friction: float = 0.5):
        self.gravity = gravity
        self.friction = friction
        self.ground_friction = ground_friction
        self.objects: List[Dict] = []
        self.restitution = 0.2  # coeficiente de restituição (quique)

    def add_object(self, pos: Union[list, np.ndarray], mass: float, radius: float) -> int:
        """
        Adiciona um objeto rígido esférico ao simulador.
        Retorna o ID do objeto.
        """
        obj_id = len(self.objects)
        self.objects.append({
            "pos": np.array(pos, dtype=np.float32),
            "vel": np.zeros(3, dtype=np.float32),
            "accel": np.zeros(3, dtype=np.float32),
            "forces": np.zeros(3, dtype=np.float32),
            "mass": max(1e-5, float(mass)),
            "radius": float(radius)
        })
        return obj_id

    def apply_force(self, obj_id: int, force: Union[list, np.ndarray]):
        """Aplica uma força instantânea ao acumulador de forças do objeto."""
        if 0 <= obj_id < len(self.objects):
            self.objects[obj_id]["forces"] += np.array(force, dtype=np.float32)

    def step(self, dt: float = 0.016):
        """
        Avança o tempo na simulação (1 tick).
        Aplica gravidade, forças acumuladas, atrito e resolve colisões.
        """
        for obj_id, obj in enumerate(self.objects):
            # Gravidade global (Z negativo)
            # F = m * g -> força peso
            weight_force = np.array([0.0, 0.0, -self.gravity * obj["mass"]], dtype=np.float32)
            obj["forces"] += weight_force

            # Fricção do ar (amortecimento linear)
            drag = -self.friction * obj["vel"]
            obj["forces"] += drag

            # Cálculo de aceleração (a = F / m)
            obj["accel"] = obj["forces"] / obj["mass"]

            # Integração de Euler
            obj["vel"] += obj["accel"] * dt
            obj["pos"] += obj["vel"] * dt

            # Resolução de colisão com solo plano (Z = 0)
            if obj["pos"][2] - obj["radius"] < 0:
                obj["pos"][2] = obj["radius"]  # Impede penetrar o solo
                
                # Se houver velocidade de impacto negativa significativa, quica
                if obj["vel"][2] < -0.1:
                    obj["vel"][2] = -obj["vel"][2] * self.restitution
                else:
                    obj["vel"][2] = 0.0
                    
                # Atrito do solo nas direções X e Y
                obj["vel"][0] *= (1.0 - self.ground_friction * dt * 10)
                obj["vel"][1] *= (1.0 - self.ground_friction * dt * 10)

            # Zera forças acumuladas para o próximo frame
            obj["forces"] = np.zeros(3, dtype=np.float32)

        # Resolução de colisões entre objetos (esferas)
        self._resolve_inter_collisions()

    def _resolve_inter_collisions(self):
        """Resolve colisões simples elásticas entre esferas no espaço 3D."""
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                o1 = self.objects[i]
                o2 = self.objects[j]
                
                dist_vec = o2["pos"] - o1["pos"]
                dist = np.linalg.norm(dist_vec)
                min_dist = o1["radius"] + o2["radius"]
                
                if dist < min_dist and dist > 1e-6:
                    normal = dist_vec / dist
                    penetration = min_dist - dist
                    
                    # Empurra os objetos para fora um do outro (proporcional às massas)
                    total_mass = o1["mass"] + o2["mass"]
                    o1["pos"] -= normal * penetration * (o2["mass"] / total_mass)
                    o2["pos"] += normal * penetration * (o1["mass"] / total_mass)
                    
                    # Velocidade relativa
                    rel_vel = o2["vel"] - o1["vel"]
                    vel_along_normal = np.dot(rel_vel, normal)
                    
                    # Só resolve se estiverem se aproximando
                    if vel_along_normal < 0:
                        # Impulso elástico
                        restitution = min(self.restitution, 0.5)
                        impulse_scalar = -(1 + restitution) * vel_along_normal
                        impulse_scalar /= (1 / o1["mass"] + 1 / o2["mass"])
                        
                        impulse = impulse_scalar * normal
                        o1["vel"] -= impulse / o1["mass"]
                        o2["vel"] += impulse / o2["mass"]

    def get_object_state(self, obj_id: int) -> dict:
        """Retorna o estado do objeto (posição, velocidade e aceleração)."""
        if 0 <= obj_id < len(self.objects):
            obj = self.objects[obj_id]
            return {
                "pos": obj["pos"].tolist(),
                "vel": obj["vel"].tolist(),
                "accel": obj["accel"].tolist()
            }
        return {}

    def check_collision(self, obj_id: int) -> list:
        """
        Retorna uma lista de objetos com os quais o objeto atual colidiu.
        Pode conter 'ground' se colidir com o solo.
        """
        collisions = []
        if 0 <= obj_id < len(self.objects):
            obj = self.objects[obj_id]
            
            # Solo
            if obj["pos"][2] - obj["radius"] <= 1e-2:
                collisions.append("ground")
                
            # Outros objetos
            for i, other in enumerate(self.objects):
                if i != obj_id:
                    dist = np.linalg.norm(obj["pos"] - other["pos"])
                    if dist <= (obj["radius"] + other["radius"]):
                        collisions.append(i)
                        
        return collisions

    def get_ground_contact(self, obj_id: int) -> float:
        """Retorna 1.0 se o objeto estiver tocando o chão firme, 0.0 caso contrário."""
        if 0 <= obj_id < len(self.objects):
            obj = self.objects[obj_id]
            # Se a distância Z ao solo for igual ou menor ao raio + tolerância, está em contato
            if obj["pos"][2] - obj["radius"] <= 1e-2:
                return 1.0
        return 0.0
