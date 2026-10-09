"""
OperatorGroundingBridge — Ponte entre CGE (conceitos) e operadores executaveis.

Fluxo:
  1. CGE sugere conceitos que podem ser transformacoes (via CGE query)
  2. A bridge gera OPERATOR PROPOSALS (hipoteses de operador)
  3. Cada proposal entra como candidato na beam search
  4. A execucao VALIDA ou REJEITA o proposal
  5. Se validado → vira primitiva real no CompositionGraph

NENHUM operador e valido ate ser validado por execucao.
O CGE da a dica estrutural; a execucao da a confirmacao.
"""
from typing import List, Optional, Dict, Callable
import numpy as np

from .primitive import Primitive, PrimitiveInstance, PrimitiveRegistry, ParameterSchema
from .composition_graph import CompositionGraph
from .operator_proposals import get_all_proposals, detect_applicable_operators


class OperatorGroundingBridge:
    """Ponte CGE -> Operadores.

    Gera propostas de operadores a partir de:
      1. Conceitos do CGE (se disponivel)
      2. Operadores espaciais gerais (rotacao, reflexao, etc.)
      3. Invariantes observadas no par input/output

    Cada proposta so vira primitiva se VALIDADA por execucao.
    """

    def __init__(self, registry: PrimitiveRegistry,
                 composition_graph: CompositionGraph,
                 cge=None):
        self.registry = registry
        self.graph = composition_graph
        self.cge = cge  # Pode ser None (funciona sem CGE tambem)

    def generate_proposals(self, input_grid: np.ndarray,
                           output_grid: np.ndarray) -> List['OperatorProposal']:
        """Gera propostas de operadores para este par input/output.

        Returns lista de OperatorProposal, ordenados por confianca.
        """
        proposals: List[OperatorProposal] = []

        # 1. Detecta operadores espaciais que aplicam
        spatial_matches = self._check_spatial_operators(input_grid, output_grid)
        proposals.extend(spatial_matches)

        # 2. Consulta CGE por conceitos de transformacao
        if self.cge is not None:
            cge_proposals = self._query_cge_concepts(input_grid, output_grid)
            proposals.extend(cge_proposals)

        # 3. Detecta invariantes (contagem, estrutura)
        invariant_proposals = self._detect_invariants(input_grid, output_grid)
        proposals.extend(invariant_proposals)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def promote_if_validated(self, proposal: 'OperatorProposal',
                              input_grid: np.ndarray,
                              target: np.ndarray) -> bool:
        """Valida uma proposta por execucao. Se funcionar, vira primitiva."""
        result = proposal.execute(input_grid)
        if result.shape == target.shape and np.array_equal(result, target):
            # Validado! Vira primitiva real
            prim = self._proposal_to_primitive(proposal)
            self.registry.register(prim)
            return True
        return False

    def _check_spatial_operators(self, inp: np.ndarray,
                                  out: np.ndarray) -> List['OperatorProposal']:
        """Verifica operadores espaciais por execucao direta."""
        proposals = []
        for op in get_all_proposals():
            try:
                result = op["function"](inp)
                if result.shape == out.shape:
                    score = 1.0 if np.array_equal(result, out) else 0.0
                    proposals.append(OperatorProposal(
                        name=op["name"],
                        description=op["description"],
                        execute_fn=op["function"],
                        confidence=score,
                        source="spatial",
                    ))
            except Exception:
                pass
        return proposals

    def _query_cge_concepts(self, inp: np.ndarray,
                             out: np.ndarray) -> List['OperatorProposal']:
        """Consulta CGE por conceitos de transformacao."""
        if self.cge is None:
            return []

        proposals = []
        # Palavras-chave de transformacao espacial no CGE
        transform_keywords = [
            "rotate", "rotation", "turn",
            "mirror", "reflect", "reflection", "flip",
            "symmetry", "symmetric",
            "translate", "shift", "move",
            "scale", "resize", "grow", "shrink",
            "transpose", "swap",
            "identity", "same",
        ]

        for keyword in transform_keywords:
            # Tenta encontrar conceito no CGE
            try:
                results = self.cge.query_attention(
                    keyword, top_k=3
                ) if hasattr(self.cge, "query_attention") else []
                if results:
                    for node_id, node_data, sim in results:
                        if sim > 0.5:
                            proposals.append(OperatorProposal(
                                name=f"cge_{node_id}",
                                description=f"CGE concept: {node_id}",
                                execute_fn=None,
                                confidence=float(sim) * 0.5,
                                source="cge",
                                cge_concept=node_id,
                            ))
            except Exception:
                pass

        return proposals

    def _detect_invariants(self, inp: np.ndarray,
                            out: np.ndarray) -> List['OperatorProposal']:
        """Detecta invariantes estruturais (contagem, preservacao)."""
        proposals = []

        # Invariante de contagem
        in_objs = (inp > 0).sum()
        out_objs = (out > 0).sum()
        if in_objs == out_objs:
            proposals.append(OperatorProposal(
                name="preserve_count",
                description="same number of non-zero pixels",
                execute_fn=lambda g: g.copy(),
                confidence=0.3,
                source="invariant",
            ))

        # Invariante de cores
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        out_colors = set(int(c) for c in np.unique(out) if c != 0)
        if in_colors == out_colors:
            proposals.append(OperatorProposal(
                name="preserve_colors",
                description="same set of colors",
                execute_fn=lambda g: g.copy(),
                confidence=0.3,
                source="invariant",
            ))

        # Invariante de bounding box
        in_bbox = self._bbox(inp)
        out_bbox = self._bbox(out)
        if in_bbox == out_bbox:
            proposals.append(OperatorProposal(
                name="preserve_bbox",
                description="same bounding box",
                execute_fn=lambda g: g.copy(),
                confidence=0.3,
                source="invariant",
            ))

        return proposals

    def _proposal_to_primitive(self, proposal: 'OperatorProposal') -> Primitive:
        """Converte proposta validada em primitiva real."""
        # Gera nome unico
        name = f"validated_{proposal.name}"

        prim = Primitive(
            name=name,
            structure_type=proposal.name,
            description=proposal.description,
            parameter_schemas=[],
            constraints=[],
            solver_fn=None,
        )

        # Se tem funcao de execucao, usa
        if proposal.execute_fn:
            def _make_exec(fn):
                def _exec(g, p):
                    try:
                        return fn(g)
                    except Exception:
                        return g.copy()
                return _exec
            prim._execute = _make_exec(proposal.execute_fn)

        prim.confidence = proposal.confidence
        return prim

    def _bbox(self, grid: np.ndarray) -> tuple:
        rows, cols = np.where(grid > 0)
        if len(rows) == 0:
            return (0, 0, 0, 0)
        return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


class OperatorProposal:
    """Uma PROPOSTA de operador (ainda nao validada).

    So vira primitiva se a execucao validar.
    """

    def __init__(self, name: str, description: str = "",
                 execute_fn: Callable = None,
                 confidence: float = 0.0,
                 source: str = "unknown",
                 cge_concept: str = ""):
        self.name = name
        self.description = description
        self.execute_fn = execute_fn
        self.confidence = confidence
        self.source = source
        self.cge_concept = cge_concept

    def execute(self, grid: np.ndarray) -> np.ndarray:
        if self.execute_fn:
            try:
                return self.execute_fn(grid)
            except Exception:
                return grid.copy()
        return grid.copy()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description[:60],
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "cge_concept": self.cge_concept,
        }
