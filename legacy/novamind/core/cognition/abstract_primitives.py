"""
AbstractPrimitives — Operadores cognitivos abstratos descobertos da experiencia.

Assim como as primitives visuais (recolor, remove, move) sao descobertas
de deltas entre grids, as primitives abstratas sao descobertas de deltas
entre estados COGNITIVOS (CGE).

Cada primitiva abstrata e um operador que TRANSFORMA O CONHECIMENTO:
  - infer: se A→B e B→C, propoe A→C (transitividade)
  - analogize: se A:B :: C:D, propoe D
  - contradict: se A→X e A→¬X, flag contradicao
  - generalize: se A→X e B→X, cria conceito C com X
  - specialize: se C→X e A e subtipo de C, propoe A→X

NENHUM operador e hardcoded. Cada um e DESCOBERTO quando o sistema
observa o mesmo padrao de delta cognitivo em contextos diferentes.
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time

from .cognitive_delta import CognitiveDelta, CGEState


# ─── Schemas de parametros para primitives abstratas ──────

ABSTRACT_SCHEMAS = {
    "infer": {
        "params": ["from_concept", "via_concept", "to_concept"],
        "description": "transitive inference: if A->B and B->C, infer A->C",
        "constraints": ["via_concept connected to both"],
    },
    "analogize": {
        "params": ["source_a", "source_b", "target_a"],
        "description": "analogy: A:B :: target_a:?",
        "constraints": ["relation(source_a, source_b) exists",
                        "relation(source_a, target_a) exists"],
    },
    "generalize": {
        "params": ["instance_a", "instance_b", "property"],
        "description": "abstraction: A and B share property -> create concept",
        "constraints": ["property connected to both"],
    },
    "contradict": {
        "params": ["concept", "statement_a", "statement_b"],
        "description": "contradiction: concept has mutually exclusive properties",
        "constraints": ["statement_a and statement_b cannot both be true"],
    },
    "specialize": {
        "params": ["category", "instance", "property"],
        "description": "inheritance: if category has property, instance inherits",
        "constraints": ["instance is_a category"],
    },
}


# ─── Solvers (inferem parametros de deltas cognitivos) ─────

def _solve_infer(delta: CognitiveDelta, cge) -> Optional[Dict]:
    """Infere operacao de inferencia transitiva.

    Se o delta contem arestas novas que conectam conceitos
    que ja tinham um caminho indireto, e uma inferencia.

    Importante: busca no GRAFO SEM as arestas novas para
    encontrar o caminho indireto ORIGINAL.
    """
    # Cria conjunto de chaves das arestas novas para excluir da busca
    new_edge_keys = set()
    for e in delta.edges_added:
        new_edge_keys.add((e.get("from", ""), e.get("to", "")))

    for edge in delta.edges_added:
        frm = edge.get("from", "")
        to = edge.get("to", "")
        # Procura caminho indireto IGNORANDO arestas novas
        path = _find_path_excluding(cge, frm, to, exclude_keys=new_edge_keys, max_depth=3)
        if len(path) >= 3:
            return {
                "from_concept": frm,
                "via_concept": path[1] if len(path) > 2 else "",
                "to_concept": to,
                "new_edge_type": edge.get("edge_type", "related"),
            }
    return None


def _solve_analogize(delta: CognitiveDelta, cge) -> Optional[Dict]:
    """Infere operacao de analogia.

    Se o delta adicionou uma relacao que completa um padrao analogico
    A:B :: C:D, extrai os pares.
    """
    for edge in delta.edges_added:
        frm = edge.get("from", "")
        to = edge.get("to", "")
        etype = edge.get("edge_type", "related")

        # Procura outra aresta do mesmo tipo de frm para outro no
        for e2 in cge.edges:
            if (e2.get("from") == frm and e2.get("edge_type") == etype
                    and e2.get("to") != to):
                # Temos A(source) -> B e A(source) -> target_a
                return {
                    "source_a": frm,
                    "source_b": e2.get("to", ""),
                    "target_a": to,
                    "relation_type": etype,
                }
    return None


def _solve_generalize(delta: CognitiveDelta, cge) -> Optional[Dict]:
    """Infere operacao de generalizacao.

    Se dois nos agora compartilham uma propriedade que antes
    so um tinha, pode ser generalizacao.
    """
    # Procura propriedades que apareceram em multiplos nos
    prop_to_nodes: Dict[str, List[str]] = {}
    for edge in cge.edges:
        if edge.get("edge_type") in ("has_property", "is_a"):
            prop = edge.get("to", "")
            node = edge.get("from", "")
            if prop not in prop_to_nodes:
                prop_to_nodes[prop] = []
            prop_to_nodes[prop].append(node)

    for prop, nodes in prop_to_nodes.items():
        if len(nodes) >= 2:
            return {
                "instance_a": nodes[0],
                "instance_b": nodes[1],
                "property": prop,
            }
    return None


def _solve_contradict(delta: CognitiveDelta, cge) -> Optional[Dict]:
    """Infere contradicao.

    Se um no tem arestas para conceitos mutualmente exclusivos.
    """
    for edge in delta.edges_added:
        frm = edge.get("from", "")
        to = edge.get("to", "")
        etype = edge.get("edge_type", "")

        # Procura aresta conflitante existente
        for e2 in cge.edges:
            if (e2.get("from") == frm and e2.get("edge_type") != etype
                    and _is_exclusive(etype, e2.get("edge_type", ""))):
                return {
                    "concept": frm,
                    "statement_a": f"{frm} --[{etype}]--> {to}",
                    "statement_b": f"{frm} --[{e2['edge_type']}]--> {e2['to']}",
                }
    return None


def _solve_specialize(delta: CognitiveDelta, cge) -> Optional[Dict]:
    """Infere especializacao (heranca).

    Se uma categoria ganha uma propriedade, instancias
    dessa categoria tambem deveriam te-la.
    """
    for edge in delta.edges_added:
        frm = edge.get("from", "")
        to = edge.get("to", "")
        etype = edge.get("edge_type", "")

        if etype == "has_property":
            # Procura instancias de frm
            for e2 in cge.edges:
                if (e2.get("edge_type") in ("is_a", "part_of")
                        and e2.get("to") == frm):
                    return {
                        "category": frm,
                        "instance": e2.get("from", ""),
                        "property": to,
                    }
    return None


# ─── Helpers ───────────────────────────────────────────────

def _find_path(cge, start: str, end: str, max_depth: int = 3) -> List[str]:
    """BFS simples para encontrar caminho indireto."""
    return _find_path_excluding(cge, start, end, set(), max_depth)


def _find_path_excluding(cge, start: str, end: str,
                          exclude_keys: set, max_depth: int = 3) -> List[str]:
    """BFS para encontrar caminho ignorando arestas especificadas."""
    visited = {start}
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        if node == end and len(path) > 1:
            return path
        if len(path) >= max_depth:
            continue
        for e in cge.edges:
            efrom = e.get("from", "")
            eto = e.get("to", "")
            if (efrom, eto) in exclude_keys:
                continue  # Pula aresta que esta sendo testada
            if efrom == node and eto not in visited:
                visited.add(eto)
                queue.append((eto, path + [eto]))
    return []


def _is_exclusive(type_a: str, type_b: str) -> bool:
    """Duas relacoes sao mutualmente exclusivas?"""
    exclusive = [
        ("is_a", "part_of"),
        ("causes", "caused_by"),
        ("above", "below"),
        ("inside", "outside"),
        ("before", "after"),
        ("increases", "decreases"),
    ]
    return (type_a, type_b) in exclusive or (type_b, type_a) in exclusive


# ─── Fabrica de primitives abstratas ───────────────────────

# Mapeia transformation_type do CognitiveDelta para solver
ABSTRACT_SOLVERS: Dict[str, Callable] = {
    "inference": _solve_infer,
    "expansion": _solve_generalize,  # novos conceitos = generalizacao
    "revision": _solve_contradict,   # remocao = contradicao
}


def get_abstract_solver(delta_type: str) -> Optional[Callable]:
    """Retorna o solver apropriado para um tipo de delta."""
    return ABSTRACT_SOLVERS.get(delta_type)


def get_abstract_primitive_names() -> List[str]:
    return list(ABSTRACT_SCHEMAS.keys())
