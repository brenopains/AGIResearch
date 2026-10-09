"""
DeltaExtractor — Extrai a DIFERENCA entre conjuntos de objetos input/output.

Nao classifica transformacoes. Apenas computa o delta objetivo:

  objetos_removidos:  estavam no input, nao estao no output
  objetos_adicionados: estao no output, nao estavam no input
  objetos_transformados: estao em ambos mas com atributos diferentes
  relacoes_mudadas: relacoes entre objetos que mudaram

O delta e a materia-prima para o ProgramSynthesizer e o PrimitiveDiscovery.
"""
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from ..arc_object_centric import GridObject, ARCGridParser, RelationExtractor


@dataclass
class ObjectDelta:
    """Delta de um unico objeto entre input e output."""
    input_obj: Optional[GridObject] = None
    output_obj: Optional[GridObject] = None
    exists_in_input: bool = False
    exists_in_output: bool = False

    # Atributos que mudaram
    color_changed: bool = False
    color_from: int = -1
    color_to: int = -1

    position_changed: bool = False
    dr: float = 0.0  # delta row
    dc: float = 0.0  # delta col

    size_changed: bool = False
    area_from: int = 0
    area_to: int = 0

    shape_changed: bool = False
    shape_from: str = ""
    shape_to: str = ""

    def describe(self) -> str:
        parts = []
        if not self.exists_in_output:
            return "removed"
        if not self.exists_in_input:
            return "added"
        if self.color_changed:
            parts.append(f"recolor: {self.color_from}->{self.color_to}")
        if self.position_changed:
            parts.append(f"move: ({self.dr:.0f},{self.dc:.0f})")
        if self.size_changed:
            parts.append(f"resize: {self.area_from}->{self.area_to}")
        if self.shape_changed:
            parts.append(f"reshape: {self.shape_from}->{self.shape_to}")
        return ", ".join(parts) if parts else "unchanged"


@dataclass
class SceneDelta:
    """Delta completo entre uma cena de input e output."""
    objects_removed: List[ObjectDelta] = field(default_factory=list)
    objects_added: List[ObjectDelta] = field(default_factory=list)
    objects_transformed: List[ObjectDelta] = field(default_factory=list)
    objects_unchanged: List[ObjectDelta] = field(default_factory=list)
    n_input: int = 0
    n_output: int = 0
    grid_h_changed: bool = False
    grid_w_changed: bool = False
    grid_h_from: int = 0
    grid_w_from: int = 0
    grid_h_to: int = 0
    grid_w_to: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.objects_removed or self.objects_added or self.objects_transformed)

    @property
    def transformation_type(self) -> str:
        """Classificacao ampla do tipo de transformacao (para debug)."""
        if self.objects_removed and not self.objects_added and not self.objects_transformed:
            return "deletion"
        if self.objects_added and not self.objects_removed and not self.objects_transformed:
            return "addition"
        if not self.objects_removed and not self.objects_added and self.objects_transformed:
            return "modification"
        if self.objects_removed and self.objects_added:
            return "replacement"
        return "mixed"

    def summarize(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_removed": len(self.objects_removed),
            "n_added": len(self.objects_added),
            "n_transformed": len(self.objects_transformed),
            "n_unchanged": len(self.objects_unchanged),
            "type": self.transformation_type,
        }

    def to_description(self) -> str:
        parts = []
        if self.objects_removed:
            colors = set(str(o.input_obj.color) for o in self.objects_removed if o.input_obj)
            parts.append(f"removed {len(self.objects_removed)} object(s) of color(s) {','.join(sorted(colors))}")
        if self.objects_added:
            colors = set(str(o.output_obj.color) for o in self.objects_added if o.output_obj)
            parts.append(f"added {len(self.objects_added)} object(s) of color(s) {','.join(sorted(colors))}")
        if self.objects_transformed:
            for o in self.objects_transformed[:3]:
                parts.append(o.describe())
        if self.grid_h_changed or self.grid_w_changed:
            parts.append(f"grid resized: ({self.grid_h_from},{self.grid_w_from})->({self.grid_h_to},{self.grid_w_to})")
        return " | ".join(parts) if parts else "no change"


class DeltaExtractor:
    """Extrai SceneDelta entre grids de input e output.

    Usa o ARCGridParser para extrair objetos de ambos os grids,
    entao computa o pareamento e as diferencas.
    """

    def __init__(self):
        self.parser = ARCGridParser()

    def extract(self, input_grid: np.ndarray, output_grid: np.ndarray) -> SceneDelta:
        """Extrai o delta entre dois grids."""
        input_objects = self.parser.extract_objects(input_grid)
        output_objects = self.parser.extract_objects(output_grid)

        delta = SceneDelta(
            n_input=len(input_objects),
            n_output=len(output_objects),
            grid_h_from=input_grid.shape[0],
            grid_w_from=input_grid.shape[1],
            grid_h_to=output_grid.shape[0],
            grid_w_to=output_grid.shape[1],
        )
        delta.grid_h_changed = input_grid.shape[0] != output_grid.shape[0]
        delta.grid_w_changed = input_grid.shape[1] != output_grid.shape[1]

        # Pareia objetos (mesma cor e posicao similar)
        paired_input = set()
        paired_output = set()

        for i, inp in enumerate(input_objects):
            best_match = None
            best_dist = float("inf")

            for j, out in enumerate(output_objects):
                if j in paired_output:
                    continue
                # Distancia: posicao + cor
                dr = inp.center[0] - out.center[0]
                dc = inp.center[1] - out.center[1]
                pos_dist = (dr ** 2 + dc ** 2) ** 0.5
                color_penalty = 0 if inp.color == out.color else 1.0
                total_dist = pos_dist + color_penalty * 3

                if total_dist < best_dist:
                    best_dist = total_dist
                    best_match = j

            # Threshold maior permite parear objetos que so mudaram de cor
            if best_match is not None and best_dist < 8.0:
                paired_input.add(i)
                paired_output.add(best_match)

                inp_obj = input_objects[i]
                out_obj = output_objects[best_match]
                obj_delta = ObjectDelta(
                    input_obj=inp_obj,
                    output_obj=out_obj,
                    exists_in_input=True,
                    exists_in_output=True,
                    color_changed=inp_obj.color != out_obj.color,
                    color_from=inp_obj.color,
                    color_to=out_obj.color,
                    position_changed=abs(inp_obj.center[0] - out_obj.center[0]) > 0.5 or abs(inp_obj.center[1] - out_obj.center[1]) > 0.5,
                    dr=out_obj.center[0] - inp_obj.center[0],
                    dc=out_obj.center[1] - inp_obj.center[1],
                    size_changed=len(inp_obj.pixels) != len(out_obj.pixels),
                    area_from=len(inp_obj.pixels),
                    area_to=len(out_obj.pixels),
                    shape_changed=inp_obj.shape_type != out_obj.shape_type,
                    shape_from=inp_obj.shape_type,
                    shape_to=out_obj.shape_type,
                )
                if any([obj_delta.color_changed, obj_delta.position_changed,
                        obj_delta.size_changed, obj_delta.shape_changed]):
                    delta.objects_transformed.append(obj_delta)
                else:
                    delta.objects_unchanged.append(obj_delta)

        # Objetos nao pareados
        for i, inp in enumerate(input_objects):
            if i not in paired_input:
                delta.objects_removed.append(ObjectDelta(
                    input_obj=inp, exists_in_input=True, exists_in_output=False,
                ))
        for j, out in enumerate(output_objects):
            if j not in paired_output:
                delta.objects_added.append(ObjectDelta(
                    output_obj=out, exists_in_input=False, exists_in_output=True,
                ))

        return delta
