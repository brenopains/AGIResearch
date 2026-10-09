"""
ARC Object-Centric Perception Layer
=====================================
Extrai objetos de grids ARC, calcula propriedades e relacoes,
e gera hipoteses de transformacao qualitativamente diferentes.

Isto permite que a Competition receba hipoteses como:
  - rotate_90, mirror_horizontal, crop_top, delete_color_3, copy_object, etc.
em vez de apenas "color_5_grey" vs "color_4_yellow".
"""
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .cognition.contracts import (
    Hypothesis, EvidenceItem, EvidenceSource, EvidenceRelation,
)


# ─── Object Representation ──────────────────────────────────

@dataclass
class GridObject:
    """Um objeto contiguo extraido de um grid ARC."""
    id: str
    color: int          # 0-9
    pixels: List[Tuple[int, int]]  # (row, col)
    bbox: Tuple[int, int, int, int]  # r_min, c_min, r_max, c_max
    area: int = 0
    shape_type: str = "unknown"  # rectangle, L_shape, dot, line_h, line_v, blob

    @property
    def center(self) -> Tuple[float, float]:
        r = sum(p[0] for p in self.pixels) / max(1, len(self.pixels))
        c = sum(p[1] for p in self.pixels) / max(1, len(self.pixels))
        return (r, c)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "color": self.color,
            "area": len(self.pixels),
            "bbox": self.bbox,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "width": self.width,
            "height": self.height,
            "shape": self.shape_type,
        }


# ─── ARC Grid Parser ────────────────────────────────────────

class ARCGridParser:
    """Extrai objetos conectados de grids ARC.

    Usa flood fill (BFS) para identificar componentes conectados
    (4-vizinhanca) para cada cor.
    """

    @staticmethod
    def extract_objects(grid: np.ndarray) -> List[GridObject]:
        """Extrai todos os objetos de um grid."""
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects: List[GridObject] = []
        obj_counter = 0

        for r in range(h):
            for c in range(w):
                if visited[r, c]:
                    continue
                color = int(grid[r, c])
                if color == 0:  # background preto nao e objeto
                    visited[r, c] = True
                    continue

                # BFS flood fill
                pixels: List[Tuple[int, int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                r_min, r_max = r, r
                c_min, c_max = c, c

                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))
                    r_min = min(r_min, cr)
                    r_max = max(r_max, cr)
                    c_min = min(c_min, cc)
                    c_max = max(c_max, cc)

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                shape = ARCGridParser._classify_shape(pixels, r_min, r_max, c_min, c_max)
                obj = GridObject(
                    id=f"obj_{obj_counter}",
                    color=color,
                    pixels=pixels,
                    bbox=(r_min, c_min, r_max, c_max),
                    shape_type=shape,
                )
                objects.append(obj)
                obj_counter += 1

        return objects

    @staticmethod
    def _classify_shape(pixels: List[Tuple[int, int]],
                        r_min: int, r_max: int,
                        c_min: int, c_max: int) -> str:
        """Classifica a forma aproximada do objeto."""
        n = len(pixels)
        if n == 1:
            return "dot"
        h = r_max - r_min + 1
        w = c_max - c_min + 1
        if h == 1:
            return "line_h"
        if w == 1:
            return "line_v"
        # Preenchimento do bounding box
        bbox_area = h * w
        fill_ratio = n / bbox_area if bbox_area > 0 else 0
        if fill_ratio > 0.8:
            return "rectangle"
        if fill_ratio > 0.5:
            return "L_shape"
        return "blob"


# ─── Relation Extractor ─────────────────────────────────────

class RelationExtractor:
    """Extrai relacoes espaciais entre objetos."""

    @staticmethod
    def extract(objects: List[GridObject]) -> List[dict]:
        """Lista de relacoes entre todos os pares de objetos."""
        relations: List[dict] = []
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                rels = RelationExtractor._pair_relations(a, b)
                relations.extend(rels)
        return relations

    @staticmethod
    def _pair_relations(a: GridObject, b: GridObject) -> List[dict]:
        rels = []
        ca = a.center
        cb = b.center

        # Relacoes direcionais
        if ca[0] < cb[0] - 0.5:
            rels.append({"from": a.id, "to": b.id, "type": "above"})
        elif ca[0] > cb[0] + 0.5:
            rels.append({"from": a.id, "to": b.id, "type": "below"})

        if ca[1] < cb[1] - 0.5:
            rels.append({"from": a.id, "to": b.id, "type": "left_of"})
        elif ca[1] > cb[1] + 0.5:
            rels.append({"from": a.id, "to": b.id, "type": "right_of"})

        # Sobreposicao
        a_bbox = a.bbox
        b_bbox = b.bbox
        if (a_bbox[0] <= b_bbox[2] and a_bbox[2] >= b_bbox[0] and
                a_bbox[1] <= b_bbox[3] and a_bbox[3] >= b_bbox[1]):
            # Verifica contencao
            a_area = a.width * a.height
            b_area = b.width * b.height
            overlap_r = max(0, min(a_bbox[2], b_bbox[2]) - max(a_bbox[0], b_bbox[0]) + 1)
            overlap_c = max(0, min(a_bbox[3], b_bbox[3]) - max(a_bbox[1], b_bbox[1]) + 1)
            overlap_area = overlap_r * overlap_c
            if overlap_area > 0:
                if a_area > b_area and overlap_area / b_area > 0.5:
                    rels.append({"from": a.id, "to": b.id, "type": "contains"})
                elif b_area > a_area and overlap_area / a_area > 0.5:
                    rels.append({"from": b.id, "to": a.id, "type": "contains"})
                else:
                    rels.append({"from": a.id, "to": b.id, "type": "overlaps"})

        # Similaridade de cor
        if a.color == b.color:
            rels.append({"from": a.id, "to": b.id, "type": "same_color"})

        # Similaridade de tamanho
        a_pixels = len(a.pixels)
        b_pixels = len(b.pixels)
        if abs(a_pixels - b_pixels) / max(1, max(a_pixels, b_pixels)) < 0.2:
            rels.append({"from": a.id, "to": b.id, "type": "same_size"})

        return rels


# ─── Transformation Hypotheses Generator ────────────────────

SUPPORTED_TRANSFORMATIONS = [
    "rotate_90", "rotate_180", "rotate_270",
    "mirror_horizontal", "mirror_vertical",
    "crop_top", "crop_bottom", "crop_left", "crop_right",
    "delete_color",
    "recolor",
    "copy_object",
    "move_object",
    "scale_up", "scale_down",
    "identity",
]


class TransformationHypothesisGenerator:
    """Gera hipoteses de transformacao entre grids de entrada e saida.

    Compara objetos no input com objetos no output e infere
    transformacoes candidatas.
    """

    def __init__(self):
        self.parser = ARCGridParser()
        self.relation_extractor = RelationExtractor()

    def generate(self, input_grids: List[np.ndarray],
                 output_grids: List[np.ndarray]) -> List[Hypothesis]:
        """Gera hipoteses de transformacao baseadas nos K exemplos."""
        if not input_grids or not output_grids:
            return []

        hypotheses: List[Hypothesis] = []

        # Analisa cada par input/output
        for idx, (inp, out) in enumerate(zip(input_grids, output_grids)):
            input_objects = self.parser.extract_objects(inp)
            output_objects = self.parser.extract_objects(out)
            input_relations = self.relation_extractor.extract(input_objects)
            output_relations = self.relation_extractor.extract(output_objects)

            # Detecta transformacoes
            transforms = self._detect_transformations(
                input_objects, output_objects,
                input_relations, output_relations,
            )

            for t_name, confidence, detail in transforms:
                hyp = Hypothesis(
                    name=t_name,
                    category="transformation",
                    description=f"{t_name}: {detail}",
                    evidence_for=[EvidenceItem(
                        source=EvidenceSource.PERCEPTUAL,
                        relation=EvidenceRelation.RELATED,
                        confidence=confidence,
                        raw_score=confidence,
                        trace=f"Example {idx}: {detail[:80]}",
                        metadata={
                            "example_idx": idx,
                            "transformation": t_name,
                            "input_objects": len(input_objects),
                            "output_objects": len(output_objects),
                        },
                    )],
                    confidence=confidence,
                    evidence_total=confidence,
                )
                hypotheses.append(hyp)

        # Consolida hipoteses: agrupa pelo mesmo nome, soma confianca
        consolidated = self._consolidate(hypotheses)

        return consolidated

    def generate_single(self, input_grid: np.ndarray,
                        output_grid: np.ndarray) -> List[Hypothesis]:
        """Gera hipoteses para um unico par input/output."""
        return self.generate([input_grid], [output_grid])

    def _detect_transformations(
        self,
        input_objects: List[GridObject],
        output_objects: List[GridObject],
        input_relations: List[dict],
        output_relations: List[dict],
    ) -> List[Tuple[str, float, str]]:
        """Detecta transformacoes possiveis entre input e output."""
        transforms: List[Tuple[str, float, str]] = []

        n_in = len(input_objects)
        n_out = len(output_objects)

        # --- Contagem de objetos mudou? ---
        if n_out < n_in:
            conf = min(0.8, 0.3 + 0.1 * (n_in - n_out))
            transforms.append(("delete_object", conf,
                               f"objects decreased: {n_in} -> {n_out}"))
        elif n_out > n_in:
            conf = min(0.8, 0.3 + 0.1 * (n_out - n_in))
            transforms.append(("copy_object", conf,
                               f"objects increased: {n_in} -> {n_out}"))

        # --- Cores mudaram? ---
        in_colors = set(o.color for o in input_objects)
        out_colors = set(o.color for o in output_objects)
        if in_colors != out_colors:
            removed = in_colors - out_colors
            added = out_colors - in_colors
            if removed:
                c = list(removed)[0]
                transforms.append((f"delete_color_{c}", 0.5,
                                   f"color {c} present in input but not output"))
            if added:
                c = list(added)[0]
                transforms.append((f"recolor_to_{c}", 0.4,
                                   f"color {c} present in output but not input"))
            transforms.append(("recolor", 0.3,
                               f"colors changed: {in_colors} -> {out_colors}"))

        # --- Posicoes mudaram? (possivel rotacao/translacao) ---
        if n_in > 0 and n_out > 0 and n_in == n_out:
            # Tenta parear objetos por cor e tamanho
            pairs = self._pair_objects(input_objects, output_objects)
            if pairs:
                # Verifica se todos moveram na mesma direcao (translacao)
                drs = [b.center[0] - a.center[0] for a, b in pairs]
                dcs = [b.center[1] - a.center[1] for a, b in pairs]
                if abs(np.std(drs)) < 0.5 and abs(np.std(dcs)) < 0.5:
                    avg_dr = np.mean(drs)
                    avg_dc = np.mean(dcs)
                    if abs(avg_dr) > 1 or abs(avg_dc) > 1:
                        transforms.append(("move_object", 0.5,
                                           f"translated by ({avg_dr:.0f}, {avg_dc:.0f})"))

        # --- Tamanho do grid mudou? (crop/scale) ---
        if input_objects and output_objects:
            total_in_pixels = sum(len(o.pixels) for o in input_objects)
            total_out_pixels = sum(len(o.pixels) for o in output_objects)
            if total_out_pixels < total_in_pixels * 0.5:
                transforms.append(("crop", 0.4,
                                   f"output has {total_out_pixels} vs {total_in_pixels} input pixels"))
            elif total_out_pixels > total_in_pixels * 1.5:
                transforms.append(("scale_up", 0.3,
                                   f"output larger: {total_out_pixels} vs {total_in_pixels}"))

        # --- Simetria? (mirror) ---
        if n_in > 0 and n_out == n_in:
            # Heuristica: se objetos estao invertidos horizontalmente
            if pairs:
                mirror_count = 0
                for a, b in pairs:
                    if abs(a.center[0] - b.center[0]) < 1:
                        mirror_count += 1
                if mirror_count > n_in * 0.5:
                    transforms.append(("mirror", 0.3, "objects horizontally aligned"))

        # --- Fallback: identity ---
        transforms.append(("identity", 0.1, "no transformation detected"))

        return transforms

    def _pair_objects(self, input_objs: List[GridObject],
                      output_objs: List[GridObject]) -> List[Tuple[GridObject, GridObject]]:
        """Pareia objetos do input com output por cor e tamanho."""
        pairs = []
        used_out = set()
        for inp in sorted(input_objs, key=lambda o: -len(o.pixels)):
            best = None
            best_score = 0
            for j, out in enumerate(output_objs):
                if j in used_out:
                    continue
                # Score baseado em cor + tamanho
                color_match = 1.0 if inp.color == out.color else 0.0
                size_sim = 1.0 - abs(len(inp.pixels) - len(out.pixels)) / max(1, len(inp.pixels), len(out.pixels))
                score = color_match * 0.6 + size_sim * 0.4
                if score > best_score:
                    best_score = score
                    best = j
            if best is not None and best_score > 0.3:
                pairs.append((inp, output_objs[best]))
                used_out.add(best)
        return pairs

    def _consolidate(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Agrupa hipoteses com mesmo nome, soma confianca."""
        groups: Dict[str, Hypothesis] = {}
        for h in hypotheses:
            if h.name in groups:
                existing = groups[h.name]
                existing.evidence_for.extend(h.evidence_for)
                existing.confidence = min(1.0, existing.confidence + h.confidence * 0.3)
                existing.evidence_total = existing.confidence
            else:
                groups[h.name] = h

        result = list(groups.values())
        result.sort(key=lambda h: h.confidence, reverse=True)
        return result


# ─── Grid Visualizer (para debug) ──────────────────────────

class GridVisualizer:
    """Visualiza objetos extraidos sobre o grid original."""

    @staticmethod
    def overlay_objects(grid: np.ndarray, objects: List[GridObject]) -> np.ndarray:
        """Retorna grid RGB com bounding boxes dos objetos."""
        h, w = grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Mapeia cores ARC para RGB
        arc_colors = [
            (0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 255, 0), (128, 128, 128), (255, 0, 255),
            (255, 165, 0), (0, 255, 255), (128, 0, 128),
        ]
        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if color < len(arc_colors):
                    rgb[r, c] = arc_colors[color]

        # Desenha bounding boxes
        for obj in objects:
            r_min, c_min, r_max, c_max = obj.bbox
            color = (255, 255, 255)
            # Bordas horizontais
            if r_min < h:
                for cc in range(max(0, c_min), min(w, c_max + 1)):
                    rgb[r_min, cc] = color
                    if r_max < h:
                        rgb[r_max, cc] = color
            # Bordas verticais
            if c_min < w:
                for rr in range(max(0, r_min), min(h, r_max + 1)):
                    rgb[rr, c_min] = color
                    if c_max < w:
                        rgb[rr, c_max] = color

        return rgb
