"""
Primitive — Bloco fundamental de transformacao descoberto.

NENHUMA transformacao e hardcoded. Primitives sao descobertas
pela experiencia e armazenam:

  - structure_type: o TIPO de operacao (ex: "recolor", "remove")
  - parameter_schema: DEFINE o espaco de parametros validos
  - constraints: relacoes entre parametros (ex: from_color != to_color)
  - solver: funcao que INFERE parametros de input/output

Isto permite que o sistema:
  1. Descubra o tipo de transformacao
  2. Defina o espaco de parametros
  3. RESOLVA os parametros (nao busque) por constraint satisfaction
"""
from typing import List, Optional, Callable, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class ParameterSchema:
    """Define um parametro que uma primitiva aceita.

    name: nome do parametro (ex: "from_color")
    dtype: tipo ("categorical", "int", "float", "bool")
    domain: valores validos (ex: list(range(10)) para categorical)
    description: explicacao textual
    """
    name: str
    dtype: str = "categorical"
    domain: List[Any] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        if not self.domain and self.dtype == "categorical":
            self.domain = list(range(10))

    def is_valid(self, value: Any) -> bool:
        if self.dtype == "categorical":
            return value in self.domain
        elif self.dtype == "int":
            return self.domain[0] <= value <= self.domain[1]
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "domain": self.domain[:10] if len(self.domain) > 10 else self.domain,
        }


@dataclass
class Constraint:
    """Uma restricao entre parametros.

    Exemplos:
      - "from_color != to_color"
      - "from_color in input_colors"
      - "dr != 0 or dc != 0"
    """
    description: str
    check_fn: Optional[Callable] = None

    def check(self, params: Dict, input_grid: np.ndarray = None,
              output_grid: np.ndarray = None) -> bool:
        if self.check_fn:
            return self.check_fn(params, input_grid, output_grid)
        return True


def _categorical_in_domain(params, inp, out):
    return True  # ja validado pelo schema


class Primitive:
    """Uma primitiva de transformacao descoberta.

    Em vez de armazenar parametros fixos (ex: "recolor 2->8"),
    armazena a ESTRUTURA e INFERE parametros quando necessario.
    """

    def __init__(self, name: str, structure_type: str,
                 description: str = "",
                 parameter_schemas: List[ParameterSchema] = None,
                 constraints: List[Constraint] = None,
                 solver_fn: Callable = None):
        self.name = name
        self.structure_type = structure_type
        self.description = description
        self.parameter_schemas = parameter_schemas or []
        self.constraints = constraints or []
        self._solver_fn = solver_fn

        # Metadados de aprendizado
        self.confidence: float = 0.3
        self.n_examples: int = 0
        self.success_count: int = 0
        self.attempt_count: int = 0
        self.first_seen: float = time.time()
        self.last_used: float = time.time()

    def infer_params(self, input_grid: np.ndarray,
                     output_grid: np.ndarray) -> Optional[Dict]:
        """INFERE parametros a partir de input/output usando constraint solving.

        Nao busca. Nao enumera. Resolve.
        """
        if self._solver_fn:
            params = self._solver_fn(input_grid, output_grid)
            if params is not None and self._validate_params(params, input_grid, output_grid):
                return params
        return None

    def apply(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Aplica a primitiva com parametros INFERIDOS."""
        return self._execute(grid, params or {})

    def generate_hypothesis(self, input_grid: np.ndarray,
                           output_grid: np.ndarray) -> Optional['PrimitiveInstance']:
        """Gera uma instancia concreta da primitiva para um par input/output.

        Retorna None se a primitiva nao se aplicar a este par.
        """
        params = self.infer_params(input_grid, output_grid)
        if params is None:
            return None

        # Valida constraints
        if not self._validate_params(params, input_grid, output_grid):
            return None

        return PrimitiveInstance(primitive=self, params=params)

    def _validate_params(self, params: Dict, input_grid: np.ndarray = None,
                         output_grid: np.ndarray = None) -> bool:
        """Valida parametros contra schemas e constraints."""
        for schema in self.parameter_schemas:
            if schema.name not in params:
                return False
            if not schema.is_valid(params[schema.name]):
                return False

        for constraint in self.constraints:
            if not constraint.check(params, input_grid, output_grid):
                return False

        return True

    def _execute(self, grid: np.ndarray, params: Dict) -> np.ndarray:
        """Executa a transformacao. Deve ser sobrescrito por primitives especificas."""
        return grid.copy()

    def record_attempt(self, success: bool):
        self.attempt_count += 1
        self.last_used = time.time()
        if success:
            self.success_count += 1
        self.confidence = self.success_count / max(1, self.attempt_count)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.attempt_count)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "structure_type": self.structure_type,
            "confidence": round(self.confidence, 3),
            "n_examples": self.n_examples,
            "schemas": [s.to_dict() for s in self.parameter_schemas],
            "n_constraints": len(self.constraints),
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class PrimitiveInstance:
    """Uma instancia CONCRETA de uma primitiva com parametros resolvidos.

    Isso e o que o synthesis realmente usa: uma primitiva + parametros.
    """
    primitive: Primitive
    params: Dict

    @property
    def name(self) -> str:
        return f"{self.primitive.structure_type}({self._params_str})"

    @property
    def _params_str(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return self.primitive.apply(grid, self.params)

    def to_dict(self) -> dict:
        return {"primitive": self.primitive.name, "params": self.params}


class PrimitiveRegistry:
    """Registro central de primitives descobertas.

    Cresce organicamente com a experiencia do sistema.
    """

    def __init__(self):
        self._primitives: Dict[str, Primitive] = {}

    @property
    def count(self) -> int:
        return len(self._primitives)

    def register(self, primitive: Primitive):
        if primitive.name in self._primitives:
            existing = self._primitives[primitive.name]
            existing.n_examples += 1
            existing.confidence = min(1.0, existing.confidence + 0.05)
        else:
            self._primitives[primitive.name] = primitive

    def get(self, name: str) -> Optional[Primitive]:
        return self._primitives.get(name)

    def get_all(self, min_confidence: float = 0.0) -> List[Primitive]:
        return [p for p in self._primitives.values() if p.confidence >= min_confidence]

    def get_by_type(self, structure_type: str) -> List[Primitive]:
        return [p for p in self._primitives.values() if p.structure_type == structure_type]

    def get_best(self, n: int = 10) -> List[Primitive]:
        sorted_p = sorted(self._primitives.values(), key=lambda p: p.confidence, reverse=True)
        return sorted_p[:n]

    def get_stats(self) -> dict:
        types = {}
        for p in self._primitives.values():
            types[p.structure_type] = types.get(p.structure_type, 0) + 1
        avg_conf = sum(p.confidence for p in self._primitives.values()) / max(1, self.count)
        return {
            "total": self.count,
            "types": types,
            "avg_confidence": round(avg_conf, 3),
        }
