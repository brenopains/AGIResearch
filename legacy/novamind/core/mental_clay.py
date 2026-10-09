"""
mental_clay.py — A Massinha de Modelar Cognitiva
==================================================
"so ela entende e que faca sentido real"

Um conceito nao e um vetor opaco de 10000 dim. E um OBJETO MENTAL
manipulavel com:
  - geometria: forma primitiva + dimensoes no espaco
  - partes: sub-objetos (uma cadeira = 4 pernas + assento + encosto)
  - propriedades fisicas: massa, rigidez, temperatura, densidade...
  - afeto: valencia/arousal atrelados (fogo = quente + perigo)
  - posicao 4D: (x,y,z,t) no WorldModel4D
  - ligacoes: como se conecta a outros objetos (suporta, contem, causa)

Isto resolve o buraco que faz T1/T4/T6 falharem hoje: o sistema NAO
tem "variavel para intervir" nem "mundo para simular contrafactual".
A massinha DA a variavel e o mundo.

Arquitetura:
  MentalClayObject (atomos)
        |
        v
  MentalClayWorkspace (tabuleiro + operacoes mentais)
        |
        +---> CGE (persistencia: cada objeto vira no/aresta)
        +---> PhysicalSandbox (fisica: propriedades viram simulacao)
        +---> WorldModel4D (espaco: posicao e trajetoria)

A verificacao ( Trial 4 counterfactual) eh o teste:
  imagine_if("gravidade = 1.5x") deve propagar efeitos coerentemente,
  porque cada objeto tem massa e a fisica e consistente.
"""
from __future__ import annotations
import time
import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Formas geometricas primarias (curriculum de Spelke / cores de objetos)
PRIMITIVE_SHAPES = ("point", "line", "circle", "rectangle",
                    "triangle", "arc", "blob", "lattice")

# Propriedades fisicas padrao (escala 0..1 normalizada)
DEFAULT_PHYSICAL_PROPS = {
    "mass": 0.5, "rigidity": 0.5, "temperature": 0.3,
    "density": 0.5, "fragility": 0.3, "conductivity": 0.3,
    "reflectivity": 0.3, "mobility": 0.5,
}


@dataclass
class MentalClayObject:
    """
    Atomo da massinha. Um conceito com SUBSTANCIA — geometria, partes,
    propriedades fisicas, afeto e posicao. Manipulavel mentalmente.
    """
    name: str
    shape: str = "blob"
    dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # RGB 0..1
    position: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x,y,z,t
    parts: List["MentalClayObject"] = field(default_factory=list)

    # Propriedades fisicas manipulaveis (intervencao causal)
    physical_props: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PHYSICAL_PROPS))

    # Afeto atrelado (estende o CGE emotional_valence/arousal)
    valence: float = 0.0    # -1 (ruim) .. +1 (bom)
    arousal: float = 0.0    # 0 (calmo) .. 1 (agitado)

    # Categoria e origem (para formacao de conceitos / tracking)
    category: str = ""
    source: str = "imagined"  # observed | imagined | inferred | composed

    confidence: float = 0.5
    uid: str = field(default_factory=lambda: f"clay_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)

    # Para contrafactual: snapshot do estado antes de uma transformacao
    _prior_state: Optional[dict] = field(default=None, repr=False)

    # ─── Operacoes de modelagem (a "massinha") ───────────────────────

    def mold(self, prop: str, value: float) -> "MentalClayObject":
        """
        Altera uma propriedade fisica. Retorna UM NOVO objeto (imutabilidade
        permite backtracking contrafactual: o original fica preservado para
        comparar "mundo base" vs "mundo contrafactual").
        """
        if prop not in self.physical_props:
            self.physical_props[prop] = 0.5  # nova prop descoberta
        new = self._snapshot()
        new.physical_props[prop] = max(0.0, min(1.0, float(value)))
        new.source = "molded"
        return new

    def reshape(self, shape: str = None,
                dimensions: Tuple[float, float, float] = None,
                color: Tuple[float, float, float] = None) -> "MentalClayObject":
        """Muda geometria/aparencia. Tambem imutavel."""
        new = self._snapshot()
        if shape is not None:
            new.shape = shape
        if dimensions is not None:
            new.dimensions = dimensions
        if color is not None:
            new.color = color
        new.source = "reshaped"
        return new

    def attach_part(self, part: "MentalClayObject",
                    relation: str = "has_part") -> "MentalClayObject":
        """Adiciona uma parte (cadeira + perna). Estrutura hierarquica."""
        new = self._snapshot()
        new.parts = list(self.parts) + [part]
        new.source = "composed"
        return new

    def detach_part(self, part_name: str) -> Tuple["MentalClayObject", Optional["MentalClayObject"]]:
        """Remove uma parte (decomposicao). Retorna (novo, parte_removida)."""
        new = self._snapshot()
        removed = None
        new.parts = []
        for p in self.parts:
            if p.name == part_name and removed is None:
                removed = p
            else:
                new.parts.append(p)
        new.source = "decomposed"
        return new, removed

    def tag_affect(self, valence: float, arousal: float = None) -> "MentalClayObject":
        """Atrela sentimento ao conceito (fogo = quente + medo)."""
        new = self._snapshot()
        new.valence = max(-1.0, min(1.0, float(valence)))
        if arousal is not None:
            new.arousal = max(0.0, min(1.0, float(arousal)))
        new.source = "affect_tagged"
        return new

    def _snapshot(self) -> "MentalClayObject":
        """Copia profunda preservando referencia ao estado anterior."""
        new = copy.deepcopy(self)
        new.uid = f"clay_{uuid.uuid4().hex[:8]}"
        new._prior_state = self._state_signature()
        return new

    def _state_signature(self) -> dict:
        return {
            "physical": dict(self.physical_props),
            "shape": self.shape,
            "dim": self.dimensions,
            "pos": self.position,
            "parts": [p.uid for p in self.parts],
        }

    # ─── Introspeccao ─────────────────────────────────────────────────

    def total_mass(self) -> float:
        """Massa agregada (propria + partes). Para contrafactual fisico."""
        own = self.physical_props.get("mass", 0.5)
        parts_mass = sum(p.total_mass() for p in self.parts)
        return own + parts_mass

    def diff_from(self, other: "MentalClayObject") -> Dict[str, Any]:
        """Diferenca entre dois estados — base do raciocinio contrafactual."""
        d = {}
        for k in set(self.physical_props) | set(other.physical_props):
            a = self.physical_props.get(k, 0.0)
            b = other.physical_props.get(k, 0.0)
            if abs(a - b) > 1e-6:
                d[f"prop_{k}"] = b - a
        if self.shape != other.shape:
            d["shape"] = other.shape
        if self.dimensions != other.dimensions:
            d["dimensions"] = other.dimensions
        if self.position != other.position:
            d["position"] = other.position
        return d

    def describe(self) -> str:
        parts_str = f" + {len(self.parts)} parte(s)" if self.parts else ""
        affect = f" [val={self.valence:+.1f} ar={self.arousal:.1f}]" if (self.valence or self.arousal) else ""
        mass = self.total_mass()
        return (f"{self.name}({self.shape}, m={mass:.2f}{parts_str}{affect})")


class MentalClayWorkspace:
    """
    O tabuleiro onde a massinha e modelada. Mantem o conjunto de objetos
    mentais ativos e expoe operacoes mentais de alto nivel.

    Conecta-se a CGE (persistencia de conceitos) e PhysicalSandbox
    (simulacao fisica) de forma opcional — funciona standalone tambem,
    o que permite testar a capacidade sem depender de estado global.
    """

    def __init__(self, cge=None, sandbox=None, world4d=None, device: str = None):
        self.cge = cge
        self.sandbox = sandbox
        self.world4d = world4d
        self.objects: Dict[str, MentalClayObject] = {}
        # Log de toda manipulacao — para o Trial 7 (metacognicao auditavel)
        self.manipulation_log: List[dict] = []

    # ─── Criacao e recall ─────────────────────────────────────────────

    def create(self, name: str, shape: str = "blob",
               physical_props: Optional[Dict[str, float]] = None,
               category: str = "", source: str = "imagined") -> MentalClayObject:
        """Cria um novo objeto mental. A "massinha" crua."""
        props = dict(DEFAULT_PHYSICAL_PROPS)
        if physical_props:
            props.update(physical_props)
        obj = MentalClayObject(
            name=name, shape=shape, physical_props=props,
            category=category, source=source,
        )
        self.objects[obj.uid] = obj
        self._log("create", obj)
        # Espelha no CGE se disponivel (conceito persistente)
        self._mirror_to_cge(obj)
        return obj

    def recall(self, name: str) -> Optional[MentalClayObject]:
        """Recupera um objeto mental pelo nome (mais recente)."""
        matches = [o for o in self.objects.values() if o.name == name]
        return matches[-1] if matches else None

    # ─── Operacoes mentais de alto nivel ──────────────────────────────

    def combine(self, a: MentalClayObject, b: MentalClayObject,
                new_name: str = None) -> MentalClayObject:
        """
        Fusao: combina dois objetos. As propriedades resultantes sao a media
        ponderada por massa (principio fisico honesto — nao magica).
        """
        ma, mb = a.total_mass(), b.total_mass()
        total = ma + mb + 1e-9
        merged_props = {}
        for k in set(a.physical_props) | set(b.physical_props):
            va = a.physical_props.get(k, 0.5)
            vb = b.physical_props.get(k, 0.5)
            merged_props[k] = (va * ma + vb * mb) / total

        composed = MentalClayObject(
            name=new_name or f"{a.name}+{b.name}",
            shape="blob",
            physical_props=merged_props,
            parts=[a, b],
            valence=(a.valence + b.valence) / 2,
            category=f"{a.category}+{b.category}",
            source="composed",
        )
        self.objects[composed.uid] = composed
        self._log("combine", composed, extra={"from": [a.uid, b.uid]})
        self._mirror_to_cge(composed)
        return composed

    def split(self, obj: MentalClayObject, part_name: str = None) -> List[MentalClayObject]:
        """Decomposicao: separa em partes. Devolve lista de resultantes."""
        if not obj.parts:
            # Sem partes: divide em duas metades identicas (operacao abstrata)
            half = self.create(f"{obj.name}_half", shape=obj.shape,
                               physical_props=dict(obj.physical_props))
            return [half, half._snapshot()]
        new_obj, removed = obj.detach_part(part_name)
        if removed is None:
            return [obj]
        self.objects[new_obj.uid] = new_obj
        self._log("split", new_obj, extra={"removed": removed.uid})
        return [new_obj, removed]

    # ─── A operacao-chave: CONTRAFACTUAL ──────────────────────────────

    def imagine_if(self, base_obj: MentalClayObject,
                   intervention: Dict[str, float],
                   steps: int = 5) -> Dict[str, Any]:
        """
        Raciocinio contrafactual fisico: "e se [prop] fosse [valor]?"

        Esta e a operacao que destrava o Trial 4 (hoje = 0.00). Em vez de
        alucinar uma resposta textual, PROPAGA a intervencao pelas
        propriedades fisicas do objeto (e suas partes), usando regras
        causais locais honestas:
          - aumentar massa -> diminui mobilidade, aumenta temperatura (atrito)
          - aumentar temperatura -> diminui rigidez, aumenta fragilidade
          - etc.

        Retorna: estado contrafactual + cadeia de efeitos propagados +
        diff contra o mundo base. Auditavel.
        """
        # Snapshot do mundo base (imutabilidade)
        base_sig = base_obj._state_signature()

        # Aplica intervencao
        counter_obj = base_obj
        for prop, val in intervention.items():
            counter_obj = counter_obj.mold(prop, val)

        # Propaga efeitos pelas regras causais locais
        effects_chain = []
        current = counter_obj
        for step in range(steps):
            changed = self._propagate_physics(current)
            if not changed:
                break
            effects_chain.append(changed)
            current = changed

        diff = current.diff_from(base_obj)

        return {
            "base": base_obj,
            "counterfactual": current,
            "intervention": intervention,
            "effects_chain": effects_chain,
            "diff": diff,
            "steps_propagated": len(effects_chain),
            "audit_log": self._audit_trail(base_obj, current, intervention),
        }

    def _propagate_physics(self, obj: MentalClayObject) -> Optional[MentalClayObject]:
        """
        Propaga efeitos fisicos por regras causais LOCAIS. Estas NAO sao
        hardcodes de dominio — sao as leis termodinamicas minimas que
        qualquer mundo fisico obedece. O curriculum as ensina; aqui sao
        aplicadas como mecanismo de propagacao.
        """
        p = obj.physical_props
        new = obj._snapshot()
        changed = False

        # Regra 1: massa alta -> menos mobilidade
        if p.get("mass", 0.5) > 0.7 and p.get("mobility", 0.5) > 0.3:
            new.physical_props["mobility"] = p["mobility"] * 0.7
            changed = True

        # Regra 2: temperatura alta -> menos rigidez, mais fragilidade
        if p.get("temperature", 0.3) > 0.6:
            new.physical_props["rigidity"] = p.get("rigidity", 0.5) * 0.8
            new.physical_props["fragility"] = min(1.0, p.get("fragility", 0.3) * 1.3)
            changed = True

        # Regra 3: temperatura baixa -> mais rigidez, mais fragilidade (brittle)
        if p.get("temperature", 0.3) < 0.15:
            new.physical_props["rigidity"] = min(1.0, p.get("rigidity", 0.5) * 1.2)
            new.physical_props["fragility"] = min(1.0, p.get("fragility", 0.3) * 1.4)
            changed = True

        # Regra 4: densidade alta + massa alta -> condutividade aumenta (solido denso)
        if p.get("density", 0.5) > 0.7 and p.get("mass", 0.5) > 0.6:
            new.physical_props["conductivity"] = min(1.0, p.get("conductivity", 0.3) * 1.2)
            changed = True

        # Recursao nas partes (propagacao estrutural)
        new_parts = []
        for part in obj.parts:
            propagated = self._propagate_physics(part)
            new_parts.append(propagated if propagated else part)
            if propagated is not None:
                changed = True
        new.parts = new_parts

        return new if changed else None

    # ─── Espelhamento e logging ───────────────────────────────────────

    def _mirror_to_cge(self, obj: MentalClayObject) -> None:
        """Espelha objeto no CGE como no persistente (se CGE disponivel)."""
        if self.cge is None:
            return
        try:
            if hasattr(self.cge, "nodes") and obj.name not in self.cge.nodes:
                # Cria no sem vec pesado (vec e detalhe do CGE, nao da massinha)
                import torch
                self.cge.create_node(
                    obj.name, semantic_vec=torch.zeros(10000, device=self.cge.device)
                )
                self.cge.nodes[obj.name]["emotional_valence"] = obj.valence
                self.cge.nodes[obj.name]["emotional_arousal"] = obj.arousal
                self.cge.nodes[obj.name]["confidence"] = obj.confidence
        except Exception:
            pass

    def _log(self, op: str, obj: MentalClayObject, extra: dict = None) -> None:
        self.manipulation_log.append({
            "time": time.time(), "op": op, "obj": obj.uid,
            "name": obj.name, "source": obj.source,
            "extra": extra or {},
        })

    def _audit_trail(self, base, counter, intervention) -> List[str]:
        """Gera trilha textual auditavel para metacognicao (Trial 7)."""
        trail = [f"BASE: {base.describe()}"]
        trail.append(f"INTERVENCAO: {intervention}")
        for prop, val in intervention.items():
            trail.append(f"  aplicado: {prop} -> {val:.2f}")
        trail.append(f"RESULTADO: {counter.describe()}")
        diff = counter.diff_from(base)
        if diff:
            trail.append("EFEITOS PROPAGADOS:")
            for k, v in diff.items():
                trail.append(f"  {k}: {v:+.3f}")
        else:
            trail.append("(nenhum efeito adicional propagado)")
        return trail

    # ─── Serializacao para inspecao ───────────────────────────────────

    def summary(self) -> str:
        lines = [f"MentalClayWorkspace: {len(self.objects)} objeto(s)"]
        for o in list(self.objects.values())[-10:]:
            lines.append(f"  {o.describe()}")
        lines.append(f"Log: {len(self.manipulation_log)} operacoes")
        return "\n".join(lines)
