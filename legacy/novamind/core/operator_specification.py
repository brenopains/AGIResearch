"""
OperatorSpecificationEngine — Gera ESPECIFICACOES de operadores a partir de falhas.

A separacao conceitual:
  Antes: falha → gera STUB (implementacao direta, falha)
  Depois: falha → EXTRAI INVARIANTES → gera SPEC → busca implementacao depois

Uma OperatorSpec NAO e codigo. E uma descricao formal de:
  - O que o operador RECEBE (input structure)
  - O que o operador PRODUZ (output structure)
  - O que ele PRESERVA (invariants)
  - O que ele TRANSFORMA (transformations)
  - Restricoes (constraints)

Isso permite que a IMPLEMENTACAO seja descoberta DEPOIS,
por separado, por busca composicional ou sintese.
"""
from typing import List, Optional, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import time


# ─── Invariant Types ───────────────────────────────────────

@dataclass
class Invariant:
    """Algo que o operador PRESERVA."""
    type: str          # "color_set", "object_count", "connectivity", "shape", "bbox"
    description: str
    value: Any = None

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description,
                "value": str(self.value)[:60] if self.value else ""}


@dataclass
class Transformation:
    """Algo que o operador TRANSFORMA."""
    type: str          # "recolor", "move", "add", "remove", "resize", "replicate"
    description: str
    delta: Any = None   # O que muda (ex: cores adicionadas, objetos a mais)

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description,
                "delta": str(self.delta)[:60] if self.delta else ""}


# ─── OperatorSpec — a especificacao formal ────────────────

@dataclass
class OperatorSpec:
    """Especificacao formal de um operador.

    Nao contem codigo. Contem apenas DESCRICAO do que o operador faz.
    """
    name: str
    failure_type: str
    description: str

    # Estrutura
    input_structure: Dict = field(default_factory=dict)
    output_structure: Dict = field(default_factory=dict)

    # O que preserva vs transforma
    invariants: List[Invariant] = field(default_factory=list)
    transformations: List[Transformation] = field(default_factory=list)

    # Exemplos concretos (input, output) que deram origem a spec
    examples: List[Tuple] = field(default_factory=list)

    # Metadados
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    n_examples: int = 0

    @property
    def invariant_summary(self) -> str:
        return "; ".join(i.description for i in self.invariants[:3])

    @property
    def transformation_summary(self) -> str:
        return "; ".join(t.description for t in self.transformations[:3])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "failure_type": self.failure_type,
            "description": self.description,
            "invariants": [i.to_dict() for i in self.invariants],
            "transformations": [t.to_dict() for t in self.transformations],
            "input_structure": {k: str(v)[:40] for k, v in self.input_structure.items()},
            "output_structure": {k: str(v)[:40] for k, v in self.output_structure.items()},
            "confidence": round(self.confidence, 3),
            "n_examples": self.n_examples,
        }

    def matches(self, other: 'OperatorSpec') -> float:
        """Quao similar esta spec e a outra? (para deduplicacao)."""
        score = 0.0
        if self.failure_type == other.failure_type:
            score += 0.3
        # Invariantes similares
        my_inv = set(i.type for i in self.invariants)
        other_inv = set(i.type for i in other.invariants)
        if my_inv and other_inv:
            score += 0.3 * len(my_inv & other_inv) / max(1, len(my_inv | other_inv))
        # Transformacoes similares
        my_trans = set(t.type for t in self.transformations)
        other_trans = set(t.type for t in other.transformations)
        if my_trans and other_trans:
            score += 0.4 * len(my_trans & other_trans) / max(1, len(my_trans | other_trans))
        return score


# ─── InvariantExtractor — descobre invariantes de exemplos ─

class InvariantExtractor:
    """Extrai INVARIANTES de pares input/output.

    Dado um conjunto de exemplos (input, output), descobre:
      - O que PRESERVA (cores, contagem, posicoes relativas)
      - O que TRANSFORMA (mudancas estruturais)

    Nao gera codigo. So extrai propriedades.
    """

    def extract_all(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                    failure_type: str) -> Tuple[List[Invariant], List[Transformation]]:
        """Extrai invariantes e transformacoes de exemplos."""
        if not examples:
            return [], []

        invariants = self._extract_invariants(examples)
        transformations = self._extract_transformations(examples, failure_type)

        return invariants, transformations

    def _extract_invariants(self, examples):
        """Extrai o que e PRESERVADO entre input e output."""
        invariants = []
        if not examples:
            return invariants

        inp, out = examples[0]

        # Invariante de cor: mesma cor nos dois?
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        out_colors = set(int(c) for c in np.unique(out) if c != 0)
        common = in_colors & out_colors
        if common:
            invariants.append(Invariant(
                type="color_set",
                description=f"colors preserved: {common}",
                value=common,
            ))

        # Invariante de dimensao: mesmo shape?
        if all(i.shape == o.shape for i, o in examples):
            invariants.append(Invariant(
                type="shape",
                description=f"grid shape preserved: {inp.shape}",
                value=inp.shape,
            ))

        # Invariante de objetos: mesma contagem?
        in_objs = (inp > 0).sum()
        out_objs = (out > 0).sum()
        if in_objs == out_objs:
            invariants.append(Invariant(
                type="object_count",
                description=f"object count preserved: {in_objs}",
                value=in_objs,
            ))

        # Invariante de bounding box?
        def _bbox(g):
            rows, cols = np.where(g > 0)
            if len(rows) == 0:
                return None
            return (rows.min(), cols.min(), rows.max(), cols.max())

        in_bb = _bbox(inp)
        out_bb = _bbox(out)
        if in_bb and out_bb and in_bb == out_bb:
            invariants.append(Invariant(
                type="bbox",
                description=f"bounding box preserved: {in_bb}",
                value=in_bb,
            ))

        # Invariante de conectividade?
        if self._check_connectivity_preserved(inp, out):
            invariants.append(Invariant(
                type="connectivity",
                description="connected components preserved",
            ))

        return invariants

    def _extract_transformations(self, examples, failure_type):
        """Extrai o que TRANSFORMA entre input e output."""
        transformations = []
        if not examples:
            return transformations

        inp, out = examples[0]

        # Diferenca de cores
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        out_colors = set(int(c) for c in np.unique(out) if c != 0)
        added = out_colors - in_colors
        removed = in_colors - out_colors
        if added:
            transformations.append(Transformation(
                type="recolor",
                description=f"colors added: {added}",
                delta={"added": list(added)},
            ))
        if removed:
            transformations.append(Transformation(
                type="recolor",
                description=f"colors removed: {removed}",
                delta={"removed": list(removed)},
            ))

        # Diferenca no numero de objetos
        in_objs = (inp > 0).sum()
        out_objs = (out > 0).sum()
        if in_objs != out_objs:
            transformations.append(Transformation(
                type="object_count",
                description=f"object count: {in_objs} -> {out_objs}",
                delta={"from": int(in_objs), "to": int(out_objs)},
            ))

        # Diferenca de dimensao
        if inp.shape != out.shape:
            transformations.append(Transformation(
                type="resize",
                description=f"grid resize: {inp.shape} -> {out.shape}",
                delta={"from": list(inp.shape), "to": list(out.shape)},
            ))

        # Diferenca de posicao (centroide)
        def _center(g):
            rows, cols = np.where(g > 0)
            if len(rows) == 0:
                return (0, 0)
            return (rows.mean(), cols.mean())

        in_c = _center(inp)
        out_c = _center(out)
        dr = round(out_c[0] - in_c[0])
        dc = round(out_c[1] - in_c[1])
        if abs(dr) > 1 or abs(dc) > 1:
            transformations.append(Transformation(
                type="move",
                description=f"centroid shifted: ({dr}, {dc})",
                delta={"dr": int(dr), "dc": int(dc)},
            ))

        return transformations

    def _check_connectivity_preserved(self, inp, out) -> bool:
        """Verifica se componentes conectados sao preservados."""
        from scipy.ndimage import label
        _, in_features = label(inp > 0)
        _, out_features = label(out > 0)
        return in_features == out_features


# ─── SpecGenerator — gera OperatorSpec de cluster de falhas ─

class SpecGenerator:
    """Gera OperatorSpec a partir de um cluster de falhas.

    Nao gera implementacao. Gera ESPECIFICACAO.
    """

    def __init__(self):
        self.invariant_extractor = InvariantExtractor()
        self._specs: List[OperatorSpec] = []

    def generate(self, failure_type: str, name: str,
                 examples: List[Tuple[np.ndarray, np.ndarray]],
                 description: str = "") -> OperatorSpec:
        """Gera uma OperatorSpec a partir de exemplos de falha."""
        if not examples:
            return OperatorSpec(
                name=name, failure_type=failure_type,
                description=description or f"operator for {failure_type}",
            )

        # Extrai invariantes e transformacoes
        invariants, transformations = self.invariant_extractor.extract_all(
            examples, failure_type
        )

        # Estrutura de input/output do primeiro exemplo
        inp, out = examples[0]
        input_structure = {
            "shape": inp.shape,
            "n_objects": int((inp > 0).sum()),
            "colors": sorted(set(int(c) for c in np.unique(inp) if c != 0)),
        }
        output_structure = {
            "shape": out.shape,
            "n_objects": int((out > 0).sum()),
            "colors": sorted(set(int(c) for c in np.unique(out) if c != 0)),
        }

        spec = OperatorSpec(
            name=name,
            failure_type=failure_type,
            description=description or f"operator for {failure_type}",
            input_structure=input_structure,
            output_structure=output_structure,
            invariants=invariants,
            transformations=transformations,
            examples=examples[:3],  # So os primeiros exemplos
            n_examples=len(examples),
            confidence=min(1.0, len(examples) * 0.1),  # Mais exemplos = mais confianca
        )

        # Deduplica: se spec similar ja existe, mantem a mais antiga
        for existing in self._specs:
            if existing.matches(spec) > 0.7:
                existing.n_examples += spec.n_examples
                existing.confidence = min(1.0, existing.confidence + 0.05)
                return existing

        self._specs.append(spec)
        return spec

    def get_specs(self, min_confidence: float = 0.0) -> List[OperatorSpec]:
        return [s for s in self._specs if s.confidence >= min_confidence]

    def get_spec_by_failure(self, failure_type: str) -> Optional[OperatorSpec]:
        for s in self._specs:
            if s.failure_type == failure_type:
                return s
        return None

    def get_stats(self) -> dict:
        return {
            "total_specs": len(self._specs),
            "specs": [s.to_dict() for s in self._specs],
        }


# ─── Integracao com FailureDrivenLoop ─────────────────────

def integrate_with_failure_loop(loop, spec_generator: SpecGenerator = None):
    """Substitui a geracao de stubs por geracao de especificacao.

    Uso:
        loop = FailureDrivenLoop(registry, engine)
        spec_gen = SpecGenerator()
        integrate_with_failure_loop(loop, spec_gen)

    Agora, em vez de gerar stubs, o loop gera OperatorSpecs.
    """
    if spec_generator is None:
        spec_generator = SpecGenerator()

    # Hook: quando um cluster atinge tamanho minimo, gera spec
    original_on_failure = loop.on_failure

    def patched_on_failure(inp, out, current, gap, task_id="", primitives_tried=None):
        # Chama o original (para manter clusters e analise)
        candidate = original_on_failure(inp, out, current, gap, task_id, primitives_tried)

        # Gera spec para o cluster
        sig = loop.analyzer.analyze(inp, out, current, gap, task_id, primitives_tried)
        ftype = sig.failure_type
        if ftype in loop.clusters and loop.clusters[ftype].count >= 3:
            cluster = loop.clusters[ftype]
            # Coleta exemplos do cluster
            examples = []
            task_ids = set()
            for s in cluster.signatures:
                tid = s.task_id
                if tid and tid not in task_ids:
                    task_ids.add(tid)
                    if tid in loop._input_cache and tid in loop._output_cache:
                        examples.append((loop._input_cache[tid], loop._output_cache[tid]))

            if len(examples) >= 2:
                spec = spec_generator.generate(
                    failure_type=ftype,
                    name=f"spec_{ftype.lower()}",
                    examples=examples,
                    description=f"operator for {ftype}: {cluster.infer_pattern()}",
                )
                # Registra no loop para acesso posterior
                if not hasattr(loop, '_specs'):
                    loop._specs = []
                loop._specs.append(spec)

        return candidate

    loop.on_failure = patched_on_failure
    return spec_generator
