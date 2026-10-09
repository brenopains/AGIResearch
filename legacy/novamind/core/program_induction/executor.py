"""
ProgramExecutor — Executa programas em grids e valida resultados.

Nao contem NENHUMA transformacao hardcoded.
So aplica as primitives registradas no PrimitiveRegistry.
"""
from typing import List, Optional, Dict
import numpy as np

from .primitive import Primitive, PrimitiveRegistry
from .synthesis import Program
from ..arc_object_centric import ARCGridParser


class ProgramExecutor:
    """Executor de programas.

    Aplica programas a grids e valida resultados.
    """

    def __init__(self, registry: Optional[PrimitiveRegistry] = None):
        self.registry = registry or PrimitiveRegistry()
        self.parser = ARCGridParser()

    def execute(self, program: Program, grid: np.ndarray) -> np.ndarray:
        """Executa um programa em um grid.

        Returns grid resultante (ou grid original se falhar).
        """
        try:
            return program.execute(grid)
        except Exception:
            return grid.copy()

    def validate(self, program: Program, input_grid: np.ndarray,
                 expected_output: np.ndarray) -> dict:
        """Valida um programa comparando resultado com esperado.

        Returns dict com metricas de validacao.
        """
        result = self.execute(program, input_grid)

        # Exact match
        exact = np.array_equal(result, expected_output) and result.shape == expected_output.shape

        # Pixel accuracy (se shapes baterem)
        pixel_acc = 0.0
        if result.shape == expected_output.shape:
            total = int(expected_output.size)
            wrong = int((result != expected_output).sum())
            pixel_acc = 1.0 - wrong / max(1, total)
        else:
            pixel_acc = 0.0

        # Object-level accuracy
        result_objs = self.parser.extract_objects(result)
        expected_objs = self.parser.extract_objects(expected_output)
        obj_acc = min(len(result_objs), len(expected_objs)) / max(1, len(expected_objs)) if expected_objs else 1.0

        return {
            "exact_match": bool(exact),
            "pixel_accuracy": round(pixel_acc, 4),
            "object_accuracy": round(obj_acc, 4),
            "result_shape": list(result.shape),
            "expected_shape": list(expected_output.shape),
        }

    def validate_program_on_batch(self, program: Program,
                                  examples: List[tuple]) -> float:
        """Valida um programa em multiplos exemplos.

        Args:
            program: o programa
            examples: lista de (input_grid, output_grid)

        Returns: taxa de acerto media (0-1)
        """
        scores = []
        for inp, out in examples:
            v = self.validate(program, inp, out)
            scores.append(1.0 if v["exact_match"] else v["pixel_accuracy"])
        return sum(scores) / max(1, len(scores))
