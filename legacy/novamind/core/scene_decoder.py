"""
SceneDecoder — Representacao Intermediaria de Cena + Tres Decodificadores

Pipeline correto:
  Imagem → SceneRepresentation → HDV  (percepcao)
  HDV → SceneRepresentation → Imagem  (reconstrucao)
  Simulacao → sequencia HDV → SceneRepresentation → Video  (imaginacao)
  WorldModel + Goal → melhor cena → diagrama  (comunicacao)

Cada decoder tem objetivo de treino DISTINTO.
Nenhum deles aprende pixel.
A SceneRepresentation e o intermediario que todos compartilham.
"""
import time, math, random, hashlib, io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── CENA: REPRESENTACAO INTERMEDIARIA ─────────────────────

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
        return {"id": self.id, "type": self.type,
                "position": list(self.position), "size": self.size,
                "color": list(self.color), "rotation": self.rotation,
                "properties": dict(self.properties)}


@dataclass
class SceneRelation:
    subject: str = ""
    predicate: str = ""
    object: str = ""
    weight: float = 1.0


@dataclass
class SceneEvent:
    description: str = ""
    time: float = 0.0
    objects_involved: List[str] = field(default_factory=list)
    type: str = "transform"


class MentalState:
    """Estado cognitivo completo para o decoder.

    Diferenca de SceneRepresentation:
      SceneRepresentation = objeto sendo representado (cena)
      MentalState = estado completo da mente (cena + atencao + contexto + objetivo)

    O decoder recebe MentalState, nao apenas HDV.
    """
    def __init__(self, focus: str = "", scene: "SceneRepresentation" = None,
                 hdvs: Optional[list] = None,
                 attention: Optional[Dict] = None,
                 context: Optional[Dict] = None,
                 goal: str = "",
                 simulation_state: Optional[Dict] = None,
                 workspace: Optional[List] = None,
                 error_signals: Optional[Dict] = None,
                 confidence: float = 0.5,
                 creativity: float = 0.5):
        self.focus = focus
        self.scene = scene or SceneRepresentation()
        self.hdvs = hdvs or []
        self.attention = attention or {}
        self.context = context or {}
        self.goal = goal
        self.simulation_state = simulation_state or {}
        self.workspace = workspace or []
        self.error_signals = error_signals or {}
        self.confidence = confidence
        self.creativity = creativity
        self.timestamp = time.time()

    @classmethod
    def from_state_dict(cls, state: Dict) -> "MentalState":
        """Converte dict de estado (do run_agi_real) para MentalState."""
        focus = state.get("focus", "?")
        from core.mental_scene_graph import MentalSceneGraph as SG
        cge = state.get("_cge", None)
        scene = SG.from_cge(cge, focus) if cge else SceneRepresentation()
        hdvs = state.get("hdvs", [])
        hdv_tensors = []
        for h in hdvs:
            if isinstance(h, list):
                import torch; hdv_tensors.append(torch.tensor(h, dtype=torch.float32))
            else:
                hdv_tensors.append(h)
        return cls(
            focus=focus, scene=scene, hdvs=hdv_tensors,
            attention=state.get("attention", {}),
            context=state.get("context", {}),
            goal=state.get("_goal", ""),
            simulation_state=state.get("_simulation", {}),
        )


class SceneRepresentation:
    """Representacao intermediaria entre percepcao/cognicao e decoders.

    Nao e HDV. Nao e pixel.
    E uma cena: objetos com posicao, relacoes, eventos.

    Tanto o encoder (imagem → HDV) quanto o decoder (HDV → imagem)
    passam por esta representacao.
    """

    def __init__(self):
        self.objects: Dict[str, SceneObject] = {}
        self.relations: List[SceneRelation] = []
        self.events: List[SceneEvent] = []
        self.focus: str = ""
        self.timestamp: float = time.time()

    def add_object(self, obj: SceneObject):
        self.objects[obj.id] = obj

    def add_relation(self, rel: SceneRelation):
        self.relations.append(rel)

    def to_dict(self) -> Dict:
        return {
            "focus": self.focus,
            "objects": [o.to_dict() for o in self.objects.values()],
            "relations": [{"s": r.subject, "p": r.predicate, "o": r.object, "w": r.weight}
                         for r in self.relations],
        }

    @classmethod
    def from_state(cls, focus: str = "", hdvs: Optional[list] = None,
                   cge=None, workspace=None) -> "SceneRepresentation":
        """Constroi representacao de cena a partir do estado cognitivo."""
        scene = cls()
        scene.focus = focus
        scene.timestamp = time.time()

        # Objeto central: o foco
        obj = SceneObject(id=focus or "focus", type="concept",
                         position=(0.5, 0.5, 0.0), size=0.15,
                         color=_color_from_seed(str(focus), 0))
        scene.add_object(obj)

        if cge and focus in cge.nodes:
            related = set()
            for e in cge.edges[-80:]:
                if e["from"] == focus and e["to"] in cge.nodes:
                    related.add(e["to"])
                    scene.add_relation(SceneRelation(focus, "related", e["to"]))
                elif e["to"] == focus and e["from"] in cge.nodes:
                    related.add(e["from"])
                    scene.add_relation(SceneRelation(e["from"], "related", focus))

            step = 2 * math.pi / max(len(related), 1)
            for i, name in enumerate(list(related)[:12]):
                angle = step * i
                dist = 0.25 + (i % 4) * 0.04
                child = SceneObject(id=name, type="concept",
                    position=(0.5 + dist * math.cos(angle), 0.5 + dist * math.sin(angle), 0.0),
                    size=0.06, color=_color_from_seed(str(name), 1))
                scene.add_object(child)
        return scene

    def to_temporal(self, n_frames: int = 16) -> List["SceneRepresentation"]:
        """Gera sequencia temporal (Mental Movie)."""
        frames = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            frame = SceneRepresentation()
            frame.focus = self.focus
            for oid, obj in self.objects.items():
                wobble = math.sin(t * 2 * math.pi * 2 + hash(oid) % 100 * 0.1) * 0.02
                np = (obj.position[0] + wobble, obj.position[1] +
                      math.cos(t * 2 * math.pi * 1.5 + hash(oid) % 50 * 0.1) * 0.02, 0.0)
                ns = obj.size * (0.9 + 0.2 * math.sin(t * 2 * math.pi * 3 + hash(oid)%30*0.1))
                frame.add_object(SceneObject(id=oid, type=obj.type, position=np,
                    size=ns, color=obj.color, rotation=obj.rotation + t * 0.5,
                    properties=dict(obj.properties)))
            for r in self.relations:
                frame.add_relation(SceneRelation(r.subject, r.predicate, r.object, r.weight))
            frames.append(frame)
        return frames


# ─── DECODER 1: RECONSTRUCAO ───────────────────────────────

class ReconstructionDecoder:
    """Decoder de RECONSTRUCAO.

    Input: HDV perceptivo + workspace
    Output: imagem reconstruida (proxima do original)
    Ground truth: SIM (imagem original)
    Proposito: Observabilidade — "foi isso que voce viu?"

    NAO aprende pixel. Aprende HDV → SceneRepresentation → pixels.
    Usa VisualPipeline.decode_bipolar() quando disponivel.
    """

    def __init__(self, visual_pipeline=None, device: str = "cpu"):
        self.device = torch.device(device)
        self.visual_pipeline = visual_pipeline
        self._trained = visual_pipeline is not None

    def decode(self, scene: SceneRepresentation, hdvs: Optional[list] = None) -> np.ndarray:
        """Decodifica representacao de cena em imagem."""
        if self.visual_pipeline is not None and hdvs is not None and len(hdvs) > 0:
            try:
                vec = hdvs[0].flatten().float().to(self.device)
                img_tensor = self.visual_pipeline.decode_bipolar(vec)
                img = (img_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                return img
            except Exception:
                pass
        return self._placeholder_render(scene)

    def _placeholder_render(self, scene: SceneRepresentation) -> np.ndarray:
        """Placeholder honesto: 'decoder nao treinado'."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = [5, 8, 16]
        if _HAS_CV2:
            cv2.putText(img, "RECONSTRUCTION", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)
            cv2.putText(img, "not yet trained", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            img[80:120, 40:180] = [0, 210, 255]
            img[120:130, 50:170] = [100, 100, 100]
        return img

    @property
    def ready(self) -> bool:
        return self._trained


# ─── DECODER 2: IMAGINACAO ─────────────────────────────────

class ImaginationDecoder:
    """Decoder de IMAGINACAO.

    Input: sequencia de HDVs da SimulationEngine (Mental Movie)
    Output: video / imagem novA (nunca existiu)
    Ground truth: NAO (criacao original)
    Proposito: Expressividade — traduzir pensamentos em midia
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._trained = False

    def decode(self, scene: SceneRepresentation,
               temporal: bool = False) -> np.ndarray:
        """Decodifica cena de imaginacao em imagem ou video."""
        if temporal:
            frames = scene.to_temporal(16)
            return np.stack([self._placeholder_render(f) for f in frames])
        return self._placeholder_render(scene)

    def _placeholder_render(self, scene: SceneRepresentation) -> np.ndarray:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = [16, 8, 5]
        if _HAS_CV2:
            cv2.putText(img, scene.focus[:20] if scene.focus else "IMAGINATION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 0), 2)
            cv2.putText(img, "decoder not trained", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            img[60:100, 20:200] = [255, 210, 0]
            img[120:130, 40:180] = [100, 100, 100]
        return img

    @property
    def ready(self) -> bool:
        return self._trained


# ─── DECODER 3: COMUNICACAO ───────────────────────────────

class CommunicationDecoder:
    """Decoder de COMUNICACAO.

    Input: WorldModel + Goal + Context + Audience
    Output: diagrama / animacao / documentario explicativo
    Ground truth: NAO (avaliado por efetividade)
    Proposito: Explicabilidade — melhor explicacao para um humano
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._trained = False

    def decode(self, scene: SceneRepresentation,
               goal: Optional[str] = None) -> np.ndarray:
        return self._placeholder_render(scene, goal or "")

    def _placeholder_render(self, scene: SceneRepresentation, goal: str = "") -> np.ndarray:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = [8, 16, 5]
        label = goal[:30] if goal else (scene.focus[:30] if scene.focus else "EXPLANATION")
        if _HAS_CV2:
            cv2.putText(img, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 210), 2)
            cv2.putText(img, "COMUNICATION", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 150), 1)
            cv2.putText(img, "decoder not trained", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            img[60:100, 20:200] = [0, 255, 210]
            img[120:130, 40:180] = [100, 100, 100]
        return img

    @property
    def ready(self) -> bool:
        return self._trained


# ─── MENTAL DECODER (UNIFICADO) ───────────────────────────

class MentalDecoder:
    """Interface unificada dos tres decodificadores.

    Uso:
        decoder = MentalDecoder()
        img = decoder.reconstruct(state)    # Reconstrói percepcao
        vid = decoder.imagine(state)        # Gera imaginacao
        dia = decoder.communicate(state)    # Explica para humano
    """

    def __init__(self, reconstruction=None, imagination=None, communication=None,
                 reconstruction_kwargs=None):
        if reconstruction is None and reconstruction_kwargs:
            reconstruction = ReconstructionDecoder(**reconstruction_kwargs)
        self.reconstruction = reconstruction or ReconstructionDecoder()
        self.imagination = imagination or ImaginationDecoder()
        self.communication = communication or CommunicationDecoder()

    def reconstruct(self, state: Dict) -> bytes:
        """Decoder 1: Reconstrucao fiel do que foi percebido."""
        mental = MentalState.from_state_dict(state)
        arr = self.reconstruction.decode(mental.scene, hdvs=mental.hdvs)
        return _arr_to_png(arr)

    def imagine(self, state: Dict) -> bytes:
        """Decoder 2: Imaginacao — cria algo que nunca existiu."""
        mental = MentalState.from_state_dict(state)
        arr = self.imagination.decode(mental.scene)
        return _arr_to_png(arr)

    def communicate(self, state: Dict) -> bytes:
        """Decoder 3: Comunicacao — melhor explicacao para humano."""
        mental = MentalState.from_state_dict(state)
        arr = self.communication.decode(mental.scene, goal=mental.goal)
        return _arr_to_png(arr)

    def get_state(self) -> Dict:
        return {
            "reconstruction": self.reconstruction.ready,
            "imagination": self.imagination.ready,
            "communication": self.communication.ready,
        }


# ─── UTILS ────────────────────────────────────────────────

def _color_from_seed(name: str, offset: int = 0) -> Tuple[float, float, float]:
    s = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) + offset * 137
    rng = random.Random(s)
    return (rng.random(), rng.random(), rng.random())

def _arr_to_png(arr: np.ndarray) -> bytes:
    """Converte numpy array (H,W,3) uint8 para PNG bytes."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    H, W = arr.shape[:2]
    import struct, zlib
    px = bytearray(W * H * 3)
    for y in range(H):
        for x in range(W):
            idx = (y * W + x) * 3
            px[idx] = arr[y, x, 0]
            px[idx+1] = arr[y, x, 1]
            px[idx+2] = arr[y, x, 2]
    def chk(ct, d):
        c = ct + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(px[y * W * 3:(y+1) * W * 3]) for y in range(H))
    return (b"\x89PNG\r\n\x1a\n" +
            chk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chk(b"IDAT", zlib.compress(raw)) +
            chk(b"IEND", b""))

# Tentar importar cv2 para placeholders mais bonitos
_HAS_CV2 = False
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    pass
