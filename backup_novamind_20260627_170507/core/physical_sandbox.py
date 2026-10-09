"""
PhysicalSandbox — Simulador Genérico por Propriedades (não hardcoded)
======================================================================
Qualquer conceito pode ter propriedades arbitrárias (chave → vetor HDC).
Archetypes são operadores que transformam propriedades.
A AGI APRENDE archetypes por observação de padrões, não por física
hardcoded. Novos archetypes podem ser criados quando descobre novas
relações causais.

Fluxo:
  1. instantiate(concept, properties) → Manifold com propriedades
  2. apply(archetype, manifold) → novo manifold transformado
  3. simulate(steps) → aplica archetypes sequencialmente
  4. learn_archetype(observations) → infere novo archetype de padrões

Não há física hardcoded. Gravidade é só um operador que liga
"Massa" → "Aceleração". A AGI pode criar "GravidadeQuântica" se
descobrir o padrão.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class Property:
    """Uma propriedade de um conceito (ex: Peso:1, Fragilidade:Alta)."""
    name: str
    vector: torch.Tensor       # vetor HDC do valor
    strength: float = 1.0      # intensidade 0-1

    def to_dict(self):
        return {
            "name": self.name,
            "strength": self.strength,
        }


@dataclass
class Archetype:
    """
    Operador que transforma propriedades.
    Ex: Gravidade(entrada="Massa", saída="Aceleração", coeficiente=9.8)
    A AGI aprende archetypes por regressão: observa pares (antes, depois)
    e infere a transformação.
    """
    name: str
    input_keys: List[str]         # propriedades de entrada
    output_keys: List[str]        # propriedades de saída
    transform: Optional[Callable] = None  # função de transformação (aprendida)
    transform_params: torch.Tensor = field(default_factory=lambda: torch.eye(10000))
    confidence: float = 0.3       # 0-1, aumenta com validações
    n_validations: int = 0
    is_learned: bool = False      # True se foi aprendido, não hardcoded
    description: str = ""

    def apply(self, props: Dict[str, Property]) -> Dict[str, Property]:
        """Aplica o archetype a um conjunto de propriedades."""
        result = {}
        for out_key in self.output_keys:
            if self.transform:
                # Usa função aprendida
                out_vec = self.transform(props, self.transform_params)
            else:
                # Fallback: bind dos inputs
                bound = None
                for in_key in self.input_keys:
                    if in_key in props:
                        v = props[in_key].vector
                        bound = v if bound is None else torch.sign(bound * v)
                out_vec = bound if bound is not None else torch.randn(10000)
            result[out_key] = Property(
                name=out_key, vector=out_vec, strength=0.5)
        return result


@dataclass
class SimulatedObject:
    """Objeto instanciado no sandbox com propriedades atuais."""
    id: str
    concept_name: str
    concept_vector: torch.Tensor
    properties: Dict[str, Property] = field(default_factory=dict)
    archetypes_applied: List[str] = field(default_factory=list)
    created_at: float = 0.0

    def get_prop(self, name: str) -> Optional[Property]:
        return self.properties.get(name)

    def set_prop(self, name: str, vector: torch.Tensor, strength: float = 1.0):
        self.properties[name] = Property(name=name, vector=vector, strength=strength)

    def copy(self) -> "SimulatedObject":
        return SimulatedObject(
            id=self.id + "_copy",
            concept_name=self.concept_name,
            concept_vector=self.concept_vector.clone(),
            properties={k: Property(v.name, v.vector.clone(), v.strength)
                       for k, v in self.properties.items()},
            archetypes_applied=self.archetypes_applied.copy(),
            created_at=self.created_at,
        )


class PhysicalSandbox:
    """
    Simulador genérico baseado em propriedades.

    NÃO contém física hardcoded. Archetypes são aprendidos por
    observação. A AGI pode criar novos archetypes quando descobre
    relações causais.

    Uso:
        sandbox = PhysicalSandbox()
        obj = sandbox.instantiate("maca", {"Peso": 0.3, "Fragilidade": 0.9})
        obj = sandbox.apply_archetype("Gravidade", obj)
        result = sandbox.simulate([obj], steps=10)
    """

    def __init__(self, engine=None, device: Optional[str] = None):
        self.engine = engine
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Archetypes: dicionário de operadores
        self.archetypes: Dict[str, Archetype] = {}

        # Objetos simulados
        self.objects: Dict[str, SimulatedObject] = {}

        # Histórico de simulações para aprendizado
        self.simulation_history: List[dict] = []

        # Archetypes padrão (mínimos, apenas para bootstrap)
        self._init_bootstrap_archetypes()

    def _init_bootstrap_archetypes(self):
        """
        Archetypes de bootstrap — apenas os mais fundamentais.
        Servem como ponto de partida para a AGI criar os próprios.
        """
        # Identity: apenas copia propriedades
        self.add_archetype(Archetype(
            name="identity",
            input_keys=["*"],
            output_keys=["*"],
            confidence=1.0,
            is_learned=False,
            description="Mantém propriedades inalteradas",
        ))

        # Bind genérico: combina duas propriedades
        self.add_archetype(Archetype(
            name="bind",
            input_keys=["a", "b"],
            output_keys=["bound"],
            confidence=1.0,
            is_learned=False,
            description="Faz bind HDC de duas propriedades",
        ))

        # O resto a AGI aprende por observação

    def add_archetype(self, archetype: Archetype):
        self.archetypes[archetype.name] = archetype

    def get_archetype(self, name: str) -> Optional[Archetype]:
        return self.archetypes.get(name)

    def instantiate(
        self,
        concept_name: str,
        concept_vector: Optional[torch.Tensor] = None,
        properties: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Instancia um conceito com propriedades no sandbox.

        Args:
            concept_name: nome do conceito (ex: "maca")
            concept_vector: vetor HDC opcional
            properties: dict nome → intensidade (ex: {"Peso": 0.3})

        Returns:
            object_id
        """
        obj_id = f"obj_{concept_name}_{int(time.time())}_{len(self.objects)}"

        if concept_vector is None and self.engine:
            try:
                mf = self.engine.encode_text(concept_name)
                concept_vector = mf.tensor.float()
            except Exception:
                concept_vector = torch.randn(10000, device=self.device)
        elif concept_vector is None:
            concept_vector = torch.randn(10000, device=self.device)

        obj = SimulatedObject(
            id=obj_id,
            concept_name=concept_name,
            concept_vector=concept_vector.float(),
            created_at=time.time(),
        )

        # Codifica propriedades como vetores HDC
        if properties:
            for prop_name, strength in properties.items():
                prop_vec = self._encode_property(concept_vector, prop_name, strength)
                obj.set_prop(prop_name, prop_vec, strength)

        self.objects[obj_id] = obj
        print(f"[SANDBOX] Instanciado: {concept_name} ({obj_id}) "
              f"com {len(properties or {})} propriedades")
        return obj_id

    def _encode_property(
        self, concept_vec: torch.Tensor, prop_name: str, strength: float
    ) -> torch.Tensor:
        """Codifica uma propriedade como vetor HDC: concept * property_name."""
        if self.engine:
            try:
                mf = self.engine.encode_text(prop_name)
                prop_vec = mf.tensor.float()
                # Bind: concept * property
                return torch.sign(concept_vec * prop_vec)
            except Exception:
                pass
        # Fallback: permutation indexada
        idx = hash(prop_name) % 10000
        perm = torch.cat([concept_vec[idx:], concept_vec[:idx]])
        return torch.sign(perm * strength)

    def apply_archetype(self, archetype_name: str, obj_id: str) -> Optional[str]:
        """
        Aplica um archetype a um objeto.

        O archetype transforma as propriedades de entrada em
        propriedades de saída. O resultado é um NOVO objeto
        (não modifica o original — princípio da imutabilidade
        para permitir backtracking).
        """
        obj = self.objects.get(obj_id)
        if obj is None:
            return None

        archetype = self.archetypes.get(archetype_name)
        if archetype is None:
            print(f"[SANDBOX] Archetype desconhecido: {archetype_name}")
            return None

        # Cria cópia do objeto
        new_obj = obj.copy()
        new_obj.id = f"{obj.id}_{archetype_name}"

        # Aplica transformação
        new_props = archetype.apply(obj.properties)

        # Merge das novas propriedades
        for k, v in new_props.items():
            new_obj.set_prop(k, v.vector, v.strength)

        new_obj.archetypes_applied = obj.archetypes_applied + [archetype_name]

        self.objects[new_obj.id] = new_obj

        # Registra no histórico
        self.simulation_history.append({
            "timestamp": time.time(),
            "archetype": archetype_name,
            "input_obj": obj_id,
            "output_obj": new_obj.id,
            "input_props": {k: p.strength for k, p in obj.properties.items()},
            "output_props": {k: p.strength for k, p in new_obj.properties.items()},
        })

        return new_obj.id

    def simulate(
        self,
        obj_ids: List[str],
        steps: int = 5,
        archetype_sequence: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Executa simulação: aplica archetypes sequencialmente.

        Se archetype_sequence for None, aplica archetypes em ordem
        decrescente de confiança.

        Args:
            obj_ids: objetos a simular
            steps: número de passos
            archetype_sequence: sequência de archetypes a aplicar

        Returns:
            dict obj_id → lista de descendentes (linhagem)
        """
        lineages = {oid: [oid] for oid in obj_ids}

        if archetype_sequence is None:
            # Usa archetypes mais confiáveis
            sorted_archs = sorted(
                self.archetypes.values(), key=lambda a: -a.confidence)
            archetype_sequence = [a.name for a in sorted_archs[:5]]

        for step in range(steps):
            for obj_id in obj_ids:
                current = lineages[obj_id][-1]
                for arch_name in archetype_sequence:
                    new_id = self.apply_archetype(arch_name, current)
                    if new_id:
                        lineages[obj_id].append(new_id)
                        current = new_id
                        break  # aplica só o primeiro archetype por passo

        return lineages

    def get_state(self, obj_id: str) -> Optional[dict]:
        """Retorna estado atual de um objeto."""
        obj = self.objects.get(obj_id)
        if obj is None:
            return None
        return {
            "id": obj.id,
            "concept": obj.concept_name,
            "properties": {k: p.strength for k, p in obj.properties.items()},
            "archetypes_applied": obj.archetypes_applied,
        }

    def measure_similarity(self, obj_id_1: str, obj_id_2: str) -> float:
        """Similaridade entre dois objetos (média das similaridades de props)."""
        o1 = self.objects.get(obj_id_1)
        o2 = self.objects.get(obj_id_2)
        if o1 is None or o2 is None:
            return 0.0

        all_keys = set(o1.properties.keys()) | set(o2.properties.keys())
        if not all_keys:
            return float(F.cosine_similarity(
                o1.concept_vector.flatten().unsqueeze(0),
                o2.concept_vector.flatten().unsqueeze(0),
            ))

        sims = []
        for key in all_keys:
            v1 = o1.properties.get(key)
            v2 = o2.properties.get(key)
            if v1 is not None and v2 is not None:
                s = float(F.cosine_similarity(
                    v1.vector.flatten().unsqueeze(0),
                    v2.vector.flatten().unsqueeze(0),
                ))
                sims.append(s)
            else:
                sims.append(0.0)

        return float(np.mean(sims)) if sims else 0.0

    # ─── Aprendizado de Archetypes ──────────────────────────────

    def learn_archetype_from_observations(
        self,
        observations: List[Tuple[Dict[str, float], Dict[str, float]]],
        name: Optional[str] = None,
    ) -> str:
        """
        Aprende um NOVO archetype a partir de observações.
        Cada observação é (propriedades_antes, propriedades_depois).

        A AGI infere quais propriedades de entrada se transformam
        em quais propriedades de saída por correlação.

        Args:
            observations: lista de (before_dict, after_dict)
            name: nome opcional para o archetype

        Returns:
            nome do archetype criado
        """
        if not observations:
            return ""

        # Coleta todas as chaves de entrada e saída
        input_keys = sorted(set(k for b, a in observations for k in b.keys()))
        output_keys = sorted(set(k for b, a in observations for k in a.keys()))

        # Infere correlações: entrada → saída
        arch_name = name or f"learned_arch_{len(self.archetypes)}_{int(time.time())}"

        # Cria vetor de parâmetros da transformação (média dos binds)
        param_vec = torch.zeros(10000, device=self.device)
        n = 0
        for before, after in observations:
            for ik in input_keys:
                if ik in before:
                    for ok in output_keys:
                        if ok in after:
                            # Bind das propriedades
                            b_val = torch.tensor([before[ik]], device=self.device)
                            a_val = torch.tensor([after[ok]], device=self.device)
                            param_vec = param_vec + torch.sign(b_val * a_val)
                            n += 1

        if n > 0:
            param_vec = torch.sign(param_vec)

        archetype = Archetype(
            name=arch_name,
            input_keys=input_keys,
            output_keys=output_keys,
            transform_params=param_vec,
            confidence=0.3,
            n_validations=len(observations),
            is_learned=True,
            description=f"Archetype aprendido de {len(observations)} observacoes",
        )

        self.add_archetype(archetype)
        print(f"[SANDBOX] Novo archetype aprendido: {arch_name} "
              f"({len(input_keys)} entrada → {len(output_keys)} saida, "
              f"{len(observations)} obs)")
        return arch_name

    def validate_archetype(self, name: str, new_observations: List[Tuple]) -> bool:
        """Valida um archetype com novas observações. Aumenta confiança se consistente."""
        arch = self.archetypes.get(name)
        if not arch:
            return False

        consistent = 0
        for before, after in new_observations:
            for ik in arch.input_keys:
                if ik in before:
                    for ok in arch.output_keys:
                        if ok in after:
                            consistent += 1

        ratio = consistent / max(1, len(new_observations) * len(arch.input_keys))
        arch.confidence = min(1.0, arch.confidence + ratio * 0.1)
        arch.n_validations += len(new_observations)
        print(f"[SANDBOX] Archetype {name} validado: "
              f"confidence={arch.confidence:.2f} ({arch.n_validations} obs)")
        return True

    def get_stats(self) -> dict:
        return {
            "n_objects": len(self.objects),
            "n_archetypes": len(self.archetypes),
            "n_learned": sum(1 for a in self.archetypes.values() if a.is_learned),
            "n_simulations": len(self.simulation_history),
        }

    def save(self, path: str):
        """Salva estado do sandbox."""
        data = {
            "archetypes": [
                {
                    "name": a.name,
                    "input_keys": a.input_keys,
                    "output_keys": a.output_keys,
                    "confidence": a.confidence,
                    "n_validations": a.n_validations,
                    "is_learned": a.is_learned,
                    "description": a.description,
                }
                for a in self.archetypes.values()
            ],
            "simulation_history": self.simulation_history[-1000:],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
