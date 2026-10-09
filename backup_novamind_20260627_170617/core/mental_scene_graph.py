"""
MentalSceneGraph — Representacao Estrutural Intermediaria

Entre o estado cognitivo (CGE + SimEngine) e os renderizadores.
Todos os renderers consomem MentalSceneGraph:
  VisualRenderer: scene graph → imagem
  VideoRenderer:  scene graph temporal → video
  AudioRenderer:  scene graph → audio params
  TextRenderer:   scene graph → descricao
"""
import time, math, random, hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field


@dataclass
class SceneObject:
    id: str = ""
    type: str = "entity"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 0.1
    color: Tuple[float, float, float] = (0.5, 0.5, 0.8)
    rotation: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"id": self.id, "type": self.type, "position": list(self.position),
                "size": self.size, "color": list(self.color), "rotation": self.rotation,
                "properties": dict(self.properties), "children": list(self.children)}


@dataclass
class SceneRelation:
    subject: str = ""
    predicate: str = ""
    object: str = ""
    weight: float = 1.0

    def to_dict(self) -> Dict:
        return {"subject": self.subject, "predicate": self.predicate,
                "object": self.object, "weight": self.weight}


@dataclass
class SceneEvent:
    description: str = ""
    time: float = 0.0
    objects_involved: List[str] = field(default_factory=list)
    type: str = "transform"


class MentalSceneGraph:
    objects: Dict[str, SceneObject] = field(default_factory=dict)
    relations: List[SceneRelation] = field(default_factory=list)
    events: List[SceneEvent] = field(default_factory=list)
    focus: str = ""
    timestamp: float = 0.0

    def __init__(self):
        self.objects = {}
        self.relations = []
        self.events = []
        self.focus = ""
        self.timestamp = time.time()

    def add_object(self, obj: SceneObject):
        self.objects[obj.id] = obj

    def get(self, oid: str) -> Optional[SceneObject]:
        return self.objects.get(oid)

    def add_relation(self, rel: SceneRelation):
        self.relations.append(rel)

    def add_event(self, evt: SceneEvent):
        self.events.append(evt)

    @classmethod
    def from_cge(cls, cge, concept: str = "") -> "MentalSceneGraph":
        """Constroi grafo de cena a partir do CGE."""
        scene = cls()
        scene.focus = concept
        scene.timestamp = time.time()

        # Nó central: o conceito principal
        obj = SceneObject(
            id=concept or "focus",
            type="concept",
            position=(0.5, 0.5, 0.0),
            size=0.15,
            color=_color_from_seed(concept, 0),
        )
        scene.add_object(obj)

        if not cge or not cge.nodes:
            return scene

        # Filhos: nós relacionados no CGE
        related = set()
        for e in cge.edges[-80:]:
            if e["from"] == concept and e["to"] in cge.nodes:
                related.add(e["to"])
                scene.add_relation(SceneRelation(
                    subject=concept, predicate=e.get("edge_type", "related"),
                    object=e["to"], weight=e.get("weight", 0.5)))
            elif e["to"] == concept and e["from"] in cge.nodes:
                related.add(e["from"])
                scene.add_relation(SceneRelation(
                    subject=e["from"], predicate=e.get("edge_type", "related"),
                    object=concept, weight=e.get("weight", 0.5)))

        angle_step = 2 * math.pi / max(len(related), 1)
        for i, rel_name in enumerate(list(related)[:12]):
            angle = angle_step * i
            dist = 0.25 + (i % 4) * 0.04
            child = SceneObject(
                id=rel_name,
                type="concept",
                position=(0.5 + dist * math.cos(angle), 0.5 + dist * math.sin(angle), 0.0),
                size=0.06 + cge.nodes.get(rel_name, {}).get("confidence", 0.5) * 0.04,
                color=_color_from_seed(rel_name, 1),
            )
            scene.add_object(child)

        return scene

    @classmethod
    def from_simulation(cls, branches: list, focus: str = "") -> "MentalSceneGraph":
        """Constroi grafo de cena a partir de branches de simulacao."""
        scene = cls()
        scene.focus = focus or "simulation"
        scene.timestamp = time.time()

        for i, br in enumerate(branches[:6]):
            prob = br.get("probability", 0.5) if isinstance(br, dict) else getattr(br, "probability", 0.5)
            desc = br.get("description", "") if isinstance(br, dict) else getattr(br, "description", "")
            obj = SceneObject(
                id=br.get("id", f"branch_{i}") if isinstance(br, dict) else getattr(br, "id", f"branch_{i}"),
                type="branch",
                position=(0.15 + i * 0.14, 0.5, 0.0),
                size=0.05 + prob * 0.08,
                color=(0.0, prob * 0.8, 1.0 - prob * 0.5),
                properties={"probability": prob, "description": desc},
            )
            scene.add_object(obj)

        return scene

    @classmethod
    def from_error(cls, error_state: dict, focus: str = "") -> "MentalSceneGraph":
        """Constroi grafo de cena a partir de sinais de erro."""
        scene = cls()
        scene.focus = focus or "error"
        for key in ("prediction_error", "novelty", "uncertainty", "violation"):
            val = error_state.get(key, 0.0) if isinstance(error_state, dict) else 0.0
            obj = SceneObject(
                id=key,
                type="error",
                position=(0.1 + ["prediction_error", "novelty", "uncertainty", "violation"].index(key) * 0.26, 0.5, 0.0),
                size=0.04 + val * 0.1,
                color=(val * 0.9, 0.1, 0.3),
                properties={"value": val},
            )
            scene.add_object(obj)
        return scene

    def to_temporal(self, n_frames: int = 16) -> List["MentalSceneGraph"]:
        """Gera sequencia temporal para video (animacao de pensamento)."""
        frames = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            frame = MentalSceneGraph()
            frame.focus = self.focus
            frame.timestamp = self.timestamp + t

            for oid, obj in self.objects.items():
                wobble = math.sin(t * 2 * math.pi * 2 + hash(oid) % 100 * 0.1) * 0.02
                new_pos = (obj.position[0] + wobble, obj.position[1] + math.cos(t * 2 * math.pi * 1.5 + hash(oid) % 50 * 0.1) * 0.02, obj.position[2])
                new_size = obj.size * (0.9 + 0.2 * math.sin(t * 2 * math.pi * 3 + hash(oid) % 30 * 0.1))
                frame.add_object(SceneObject(
                    id=oid, type=obj.type, position=new_pos,
                    size=new_size, color=obj.color,
                    rotation=obj.rotation + t * 0.5,
                    properties=dict(obj.properties),
                    children=list(obj.children),
                ))

            for rel in self.relations:
                frame.add_relation(SceneRelation(rel.subject, rel.predicate, rel.object, rel.weight))

            frame.add_event(SceneEvent(
                description=f"frame_{i}", time=t,
                type="animation", objects_involved=list(self.objects.keys())[:3]))

            frames.append(frame)
        return frames

    def to_dict(self) -> Dict:
        return {
            "focus": self.focus,
            "objects": [o.to_dict() for o in self.objects.values()],
            "relations": [r.to_dict() for r in self.relations],
            "events": [{"description": e.description, "time": e.time, "type": e.type} for e in self.events],
            "timestamp": self.timestamp,
        }

    @staticmethod
    def merge(graphs: List["MentalSceneGraph"]) -> "MentalSceneGraph":
        """Mescla multiplos grafos de cena em um."""
        merged = MentalSceneGraph()
        merged.focus = " + ".join(g.focus for g in graphs if g.focus)
        for g in graphs:
            for oid, obj in g.objects.items():
                if oid not in merged.objects:
                    merged.add_object(obj)
            merged.relations.extend(g.relations)
            merged.events.extend(g.events)
        return merged


def _color_from_seed(name: str, offset: int = 0) -> Tuple[float, float, float]:
    s = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) + offset * 137
    rng = random.Random(s)
    return (rng.random(), rng.random(), rng.random())
