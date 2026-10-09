"""
DynamicCognitiveSpace — Criação Dinâmica de Espaços Mentais
=============================================================
Permite que o sistema crie, modifique e destrua espaços cognitivos
em tempo de execução — a verdadeira neuroplasticidade arquitetural.

Cada CognitiveSpace é um "canvas mental" temporário com:
  - Próprio manifold HDV 10000D
  - Próprio grafo de conceitos tipados
  - Conjunto de leis físicas/causais (pode sobrescrever as do WorldModel)
  - Capacidade de modificar outros espaços e o CGE principal

Isto permite:
  1. Imaginar objetos novos (criar espaço → adicionar conceitos → modificar)
  2. Simular cenários sem contaminar o CGE principal
  3. Misturar conceitos de domínios diferentes (blending)
  4. Modificar a própria arquitetura (meta-espaco)
"""
import time
import uuid
import copy
import random
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from collections import defaultdict
from dataclasses import dataclass, field

import torch

try:
    from .coglang import Manifold, bind_roles, unbind_role, ROLE_SUBJECT, ROLE_VERB, ROLE_OBJECT
    COGLANG_OK = True
except ImportError:
    COGLANG_OK = False

try:
    from .typed_graph import TypedCognitiveGraph, RELATION_TYPES
    TYPED_GRAPH_OK = True
except ImportError:
    TYPED_GRAPH_OK = False

try:
    from .mental_state import get_mental_state, MentalState
    MENTAL_STATE_OK = True
except ImportError:
    MENTAL_STATE_OK = False


@dataclass
class ImaginaryObject:
    """Um objeto criado dentro de um CognitiveSpace para manipulacao mental."""
    id: str = ""
    name: str = ""
    concept_hdv: 'Manifold' = None
    properties: Dict[str, float] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    position: Tuple[float, ...] = (0.0, 0.0)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    shape: str = "circle"
    size: float = 1.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "properties": dict(self.properties),
            "relations": list(self.relations),
            "position": list(self.position),
            "color": list(self.color),
            "shape": self.shape, "size": self.size,
            "velocity": list(self.velocity),
            "valence": self.emotional_valence,
            "arousal": self.emotional_arousal,
            "confidence": self.confidence,
        }


class CognitiveSpace:
    """
    Um espaço cognitivo dinâmico — um "bloco de imaginação" temporário.
    
    Diferente do CGE principal, este espaço pode ser criado, modificado
    livremente e descartado sem efeitos colaterais. Isto permite que o
    sistema IMAGINE sem consequências.
    
    Cada espaço tem:
    - Manifold proprio (HDV 10000D isolado)
    - Seu proprio grafo de conceitos (TypedCognitiveGraph leve)
    - Leis causais que podem ser diferentes do WorldModel global
    - Lista de objetos imaginarios com posicao, forma, cor, movimento
    """

    def __init__(self, space_id: str = "", name: str = "unnamed",
                 parent_space_id: Optional[str] = None):
        self.id = space_id or str(uuid.uuid4())[:8]
        self.name = name
        self.parent_id = parent_space_id
        self.created_at = time.time()
        self.last_access = time.time()
        self.ttl_seconds: float = 300.0  # espacos expiram se nao usados
        self.active: bool = True

        self.objects: Dict[str, ImaginaryObject] = {}
        self.concepts: Dict[str, 'Manifold'] = {}
        self.concept_metadata: Dict[str, dict] = {}

        self.local_laws: Dict[str, Callable] = {}
        self.local_law_descriptions: Dict[str, str] = {}

        self.typed_graph = None
        if TYPED_GRAPH_OK:
            try:
                self.typed_graph = TypedCognitiveGraph(dim=50, curvature=1.0)
            except Exception:
                self.typed_graph = None

        self._mutation_history: List[dict] = []

    def create_object(self, name: str, properties: Optional[dict] = None) -> ImaginaryObject:
        """Cria um novo objeto imaginario neste espaco."""
        obj = ImaginaryObject(
            id=str(uuid.uuid4())[:8],
            name=name,
            properties=properties or {},
            concept_hdv=self._make_concept_hdv(name),
        )
        self.objects[obj.id] = obj
        self.last_access = time.time()
        return obj

    def _make_concept_hdv(self, name: str) -> Optional['Manifold']:
        """Gera um HDV puro para um conceito, sem encoder externo."""
        if not COGLANG_OK:
            return None
        hdv = Manifold()
        hdv.tensor = hdv.tensor * (hash(name) % 3 - 1)
        return hdv

    def add_concept(self, name: str, properties: Optional[dict] = None) -> bool:
        """Adiciona conceito ao espaco."""
        hdv = self._make_concept_hdv(name)
        if hdv is None:
            return False
        self.concepts[name] = hdv
        self.concept_metadata[name] = {
            "properties": properties or {},
            "created_at": time.time(),
            "access_count": 0,
        }
        if self.typed_graph is not None:
            try:
                self.typed_graph.add_concept(name, vector=hdv.tensor.cpu().numpy()[:50])
            except Exception:
                pass
        self.last_access = time.time()
        return True

    def add_relation(self, from_name: str, to_name: str,
                     rel_type: str = "CORRELATED", confidence: float = 0.5) -> bool:
        """Adiciona relacao tipada entre conceitos no espaco."""
        if from_name not in self.concepts or to_name not in self.concepts:
            return False
        if self.typed_graph is not None:
            try:
                self.typed_graph.add_relation(from_name, to_name, rel_type, confidence)
            except Exception:
                pass
        self.last_access = time.time()
        return True

    def modify_object(self, obj_id: str, **modifications) -> bool:
        """Modifica propriedades de um objeto imaginario."""
        if obj_id not in self.objects:
            return False
        obj = self.objects[obj_id]
        for key, val in modifications.items():
            if hasattr(obj, key):
                setattr(obj, key, val)
            else:
                obj.properties[key] = val
        self._mutation_history.append({
            "time": time.time(), "action": "modify",
            "object": obj_id, "changes": modifications
        })
        self.last_access = time.time()
        return True

    def combine_objects(self, obj_a_id: str, obj_b_id: str,
                        new_name: str, blend_ratio: float = 0.5) -> Optional[ImaginaryObject]:
        """Combina dois objetos em um novo (blending conceitual)."""
        if obj_a_id not in self.objects or obj_b_id not in self.objects:
            return None
        a = self.objects[obj_a_id]
        b = self.objects[obj_b_id]

        new_props = {}
        all_keys = set(a.properties.keys()) | set(b.properties.keys())
        for k in all_keys:
            va = a.properties.get(k, 0.0)
            vb = b.properties.get(k, 0.0)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                new_props[k] = va * (1 - blend_ratio) + vb * blend_ratio
            elif isinstance(va, str) and blend_ratio < 0.5:
                new_props[k] = va
            else:
                new_props[k] = vb if isinstance(vb, str) else va

        new_obj = ImaginaryObject(
            id=str(uuid.uuid4())[:8],
            name=new_name,
            properties=new_props,
            color=tuple(
                a.color[i] * (1 - blend_ratio) + b.color[i] * blend_ratio
                for i in range(3)
            ),
            shape=a.shape if blend_ratio < 0.5 else b.shape,
            size=a.size * (1 - blend_ratio) + b.size * blend_ratio,
        )
        self.objects[new_obj.id] = new_obj
        self._mutation_history.append({
            "time": time.time(), "action": "combine",
            "parents": [obj_a_id, obj_b_id],
            "child": new_obj.id, "ratio": blend_ratio
        })
        self.last_access = time.time()
        return new_obj

    def set_law(self, law_name: str, law_fn: Callable, description: str = "") -> None:
        """Define uma lei causal local (substitui WorldModel)."""
        self.local_laws[law_name] = law_fn
        self.local_law_descriptions[law_name] = description

    def simulate(self, n_steps: int = 5, dt: float = 0.1) -> List[dict]:
        """Simula a dinamica dos objetos neste espaco por N passos."""
        history = []
        for step in range(n_steps):
            for obj_id, obj in list(self.objects.items()):
                x, y = obj.position
                vx, vy = obj.velocity
                x += vx * dt
                y += vy * dt
                obj.position = (x, y)
                for law_name, law_fn in self.local_laws.items():
                    try:
                        law_fn(obj, dt)
                    except Exception:
                        pass
            snapshot = {
                "step": step,
                "objects": {oid: o.to_dict() for oid, o in self.objects.items()}
            }
            history.append(snapshot)
        self.last_access = time.time()
        return history

    def inject_into_cge(self, cge: Any, prefix: str = "imaginado_") -> int:
        """Injeta conceitos deste espaco no CGE principal como novos nos."""
        count = 0
        for name, hdv in self.concepts.items():
            try:
                cge_name = f"{prefix}{name}"
                if cge_name not in getattr(cge, 'nodes', {}):
                    cge.create_node(
                        cge_name,
                        semantic_vec=hdv.tensor,
                        perceptual_vec=hdv.tensor,
                    )
                    count += 1
            except Exception:
                pass
        self.last_access = time.time()
        return count

    def get_state(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "active": self.active,
            "age": time.time() - self.created_at,
            "n_objects": len(self.objects),
            "n_concepts": len(self.concepts),
            "n_laws": len(self.local_laws),
            "n_mutations": len(self._mutation_history),
        }


class SpaceRegistry:
    """
    Gerenciador global de todos os CognitiveSpaces.
    
    O sistema cria, acessa e descarta espacos atraves deste registro.
    Espacos inativos expiram apos TTL.
    """

    def __init__(self, cge: Any = None, world_model: Any = None,
                 mental_state: Any = None):
        self.spaces: Dict[str, CognitiveSpace] = {}
        self.cge = cge
        self.world_model = world_model
        self.mental_state = mental_state
        self._space_counter = 0
        self._max_spaces = 20
        self._cleanup_interval = 60.0
        self._last_cleanup = time.time()

        if MENTAL_STATE_OK and mental_state is None:
            try:
                from .mental_state import get_mental_state
                self.mental_state = get_mental_state()
            except Exception:
                pass

    def create_space(self, name: str = "unnamed",
                     parent_id: Optional[str] = None) -> Optional[CognitiveSpace]:
        """Cria um novo CognitiveSpace."""
        if len(self.spaces) >= self._max_spaces:
            self._evict_oldest()
        space_id = f"space_{self._space_counter:04d}"
        self._space_counter += 1
        space = CognitiveSpace(space_id=space_id, name=name,
                               parent_space_id=parent_id)
        self.spaces[space_id] = space
        self._sync_to_mental_state()
        return space

    def get_space(self, space_id: str) -> Optional[CognitiveSpace]:
        """Acessa um espaco pelo ID."""
        space = self.spaces.get(space_id)
        if space is not None:
            space.last_access = time.time()
        return space

    def delete_space(self, space_id: str) -> bool:
        """Remove um espaco."""
        if space_id in self.spaces:
            del self.spaces[space_id]
            self._sync_to_mental_state()
            return True
        return False

    def list_spaces(self) -> List[dict]:
        """Lista todos os espacos ativos."""
        return [s.get_state() for s in self.spaces.values() if s.active]

    def get_or_create_imagination_space(self) -> CognitiveSpace:
        """Retorna ou cria o espaco padrao de imaginacao."""
        for s in self.spaces.values():
            if s.name == "imagination" and s.active:
                return s
        return self.create_space(name="imagination")

    def inject_all_to_cge(self, prefix: str = "imaginado_") -> int:
        """Injeta todos os conceitos de todos os espacos no CGE."""
        total = 0
        for space in self.spaces.values():
            if space.active and self.cge is not None:
                total += space.inject_into_cge(self.cge, prefix)
        return total

    def _evict_oldest(self):
        """Remove o espaco mais antigo."""
        if not self.spaces:
            return
        oldest_id = min(self.spaces.keys(),
                       key=lambda sid: self.spaces[sid].last_access)
        del self.spaces[oldest_id]

    def cleanup_expired(self) -> int:
        """Remove espacos expirados."""
        now = time.time()
        expired = [
            sid for sid, s in self.spaces.items()
            if s.active and (now - s.last_access) > s.ttl_seconds
        ]
        for sid in expired:
            del self.spaces[sid]
        self._last_cleanup = now
        self._sync_to_mental_state()
        return len(expired)

    def _sync_to_mental_state(self):
        """Sincroniza estado com MentalState."""
        if self.mental_state is not None:
            try:
                self.mental_state.write("cognitive_spaces", {
                    "count": len(self.spaces),
                    "spaces": [s.get_state() for s in self.spaces.values()],
                })
            except Exception:
                pass

    def tick(self) -> dict:
        """Tick de manutencao. Chamado pelo Maestro."""
        now = time.time()
        expired = 0
        if (now - self._last_cleanup) > self._cleanup_interval:
            expired = self.cleanup_expired()
        return {
            "active_spaces": len(self.spaces),
            "expired": expired,
            "total_created": self._space_counter,
        }


class ImaginationCanvas:
    """
    Interface de alto nivel para o sistema 'imaginar'.
    
    Oferece operacoes mentais que humanos usam ao pensar:
    - criar_objeto: cria algo novo
    - modificar: muda forma, cor, tamanho, posicao
    - combinar: junta dois objetos em um
    - animar: da movimento
    - sentir: atribui valencia emocional
    - simular: executa fisica local e ve o que acontece
    - refletir: injeta resultado de volta no CGE
    """

    def __init__(self, registry: SpaceRegistry):
        self.registry = registry
        self._active_canvas_id: Optional[str] = None

    @property
    def _space(self) -> CognitiveSpace:
        if self._active_canvas_id and self._active_canvas_id in self.registry.spaces:
            return self.registry.spaces[self._active_canvas_id]
        space = self.registry.get_or_create_imagination_space()
        self._active_canvas_id = space.id
        return space

    def criar_objeto(self, nome: str, forma: str = "esfera",
                     cor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                     tamanho: float = 1.0,
                     propriedades: Optional[dict] = None) -> Optional[ImaginaryObject]:
        """Cria um objeto imaginario com atributos."""
        obj = self._space.create_object(nome, propriedades)
        self._space.modify_object(
            obj.id,
            shape=forma,
            color=cor,
            size=tamanho,
        )
        return obj

    def modificar(self, obj_id: str, **atributos) -> bool:
        """Modifica atributos de um objeto: forma, cor, tamanho, posicao, etc."""
        return self._space.modify_object(obj_id, **atributos)

    def combinar(self, obj_a_id: str, obj_b_id: str,
                 novo_nome: str, razao: float = 0.5) -> Optional[ImaginaryObject]:
        """Combina dois objetos em um novo."""
        return self._space.combine_objects(obj_a_id, obj_b_id, novo_nome, razao)

    def animar(self, obj_id: str, vx: float, vy: float) -> bool:
        """Atribui velocidade a um objeto."""
        return self._space.modify_object(obj_id, velocity=(vx, vy))

    def sentir(self, obj_id: str, valencia: float, arousal: float) -> bool:
        """Atribui sentimento a um objeto."""
        return self._space.modify_object(
            obj_id, emotional_valence=valencia, emotional_arousal=arousal
        )

    def simular(self, n_passos: int = 10, dt: float = 0.1) -> List[dict]:
        """Simula a dinamica dos objetos no canvas."""
        return self._space.simulate(n_passos, dt)

    def refletir(self, prefixo: str = "imaginado_") -> int:
        """Injeta conceitos imaginados de volta no CGE."""
        if self.registry.cge is None:
            return 0
        return self._space.inject_into_cge(self.registry.cge, prefixo)

    def limpar(self) -> bool:
        """Limpa o canvas (remove todos os objetos)."""
        if self._active_canvas_id in self.registry.spaces:
            self.registry.delete_space(self._active_canvas_id)
            self._active_canvas_id = None
            return True
        return False

    def listar_objetos(self) -> List[dict]:
        """Lista todos os objetos no canvas atual."""
        return [o.to_dict() for o in self._space.objects.values()]

    def estado(self) -> dict:
        """Retorna estado completo do canvas."""
        return self._space.get_state()
