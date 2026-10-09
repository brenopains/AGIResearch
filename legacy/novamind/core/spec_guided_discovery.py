"""
SpecGuidedDiscovery — Busca implementacao para OperatorSpec no GeometryDataset.

Uma OperatorSpec diz O QUE falta. O GeometryDataset tem EXPERIENCIAS
de transformacoes. Este modulo conecta os dois:

  1. Recebe uma OperatorSpec (ex: "objetos 4→36, shape (2,2)→(6,6)")
  2. Busca no GeometryDataset exemplos que CASAM com a spec
  3. Alimenta o PrimitiveDiscovery com esses exemplos
  4. O Discovery INDUZ uma nova primitiva
  5. A primitiva e validada em tarefas ARC
  6. Se valida, adicionada ao registry

A implementacao nao e copiada do dataset. E INDUZIDA
pelo mesmo mecanismo de descoberta que ja existe.
"""
from typing import List, Optional, Tuple
import numpy as np

from .operator_specification import OperatorSpec, InvariantExtractor
from .program_induction import PrimitiveDiscovery, PrimitiveRegistry
from .geometry_dataset import GeometryTransformDataset, RelationalDataset
from .goal_driven_engine import GoalState, GoalComparator


class SpecGuidedDiscovery:
    """Conecta OperatorSpec -> exemplos do dataset -> nova primitiva."""

    def __init__(self, registry: PrimitiveRegistry,
                 discovery: PrimitiveDiscovery,
                 dataset_size: int = 2000):
        self.registry = registry
        self.discovery = discovery
        self.comparator = GoalComparator()
        self.invariant_extractor = InvariantExtractor()

        # Carrega datasets como FONTES DE EXPERIENCIA (nao catalogo)
        print(f"[SpecGuided] Carregando GeometryDataset ({dataset_size}, 20x20)...")
        self.geo = GeometryTransformDataset(size=dataset_size, h=20, w=20)
        self.rel = RelationalDataset(size=dataset_size // 2, h=20, w=20)

    def find_matching_examples(self, spec: OperatorSpec,
                                n_max: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Busca exemplos no dataset que CASAM com a especificacao.

        O casamento e por INVARIANTES, nao por implementacao.
        """
        matches = []
        sources = []

        # Busca no GeometryDataset
        for idx in range(min(len(self.geo), n_max * 10)):
            inp, out, label, _ = self.geo[idx]
            if self._example_matches_spec(inp, out, spec):
                matches.append((inp, out))
                sources.append(f"geo_{label}")
                if len(matches) >= n_max:
                    break

        if len(matches) < n_max:
            # Busca no RelationalDataset
            for idx in range(min(len(self.rel), n_max * 5)):
                inp, out, label, _ = self.rel[idx]
                if self._example_matches_spec(inp, out, spec):
                    matches.append((inp, out))
                    sources.append(f"rel_{label}")
                    if len(matches) >= n_max:
                        break

        return matches[:n_max]

    def _example_matches_spec(self, inp: np.ndarray, out: np.ndarray,
                               spec: OperatorSpec) -> bool:
        """Verifica se um exemplo do dataset satisfaz a especificacao.

        O casamento e por TIPO de transformacao (in/out objects mudam,
        cores mudam, shape muda), nao por valores exatos.
        """
        inv, trans = self.invariant_extractor.extract_all([(inp, out)], spec.failure_type)
        spec_trans_types = set(t.type for t in spec.transformations)
        ex_trans_types = set(t.type for t in trans)

        # Precisa ter pelo menos um tipo de transformacao em comum
        if not (spec_trans_types & ex_trans_types):
            return False

        # Verifica direcao: objetos aumentam/diminuem igual?
        spec_in = spec.input_structure.get("n_objects", 0)
        spec_out = spec.output_structure.get("n_objects", 0)
        ex_in = int((inp > 0).sum())
        ex_out = int((out > 0).sum())

        if spec_in > 0 and spec_out > 0 and ex_in > 0 and ex_out > 0:
            if (spec_out > spec_in) != (ex_out > ex_in):
                return False

        # Verifica cores: se spec tem recolor, exemplo deve ter mudanca de cor
        if "recolor" in spec_trans_types:
            has_color_change = any(
                "color" in t.type or "recolor" in t.type for t in trans
            )
            if not has_color_change:
                return False

        return True

    def discover_all_from_dataset(self) -> int:
        """Descobre TODAS as primitives possiveis do dataset.

        Alimenta o PrimitiveDiscovery com todas as experiencias
        disponiveis, deixando-o descobrir quaisquer padroes.

        Returns: numero de novas primitives descobertas.
        """
        n_before = self.registry.count
        examples_used = 0

        # 1. Geometry dataset
        for idx in range(len(self.geo)):
            inp, out, label, _ = self.geo[idx]
            result = self.discovery.observe(inp, out)
            if result is not None:
                examples_used += 1

        # 2. Relational dataset
        for idx in range(len(self.rel)):
            inp, out, label, _ = self.rel[idx]
            result = self.discovery.observe(inp, out)
            if result is not None:
                examples_used += 1

        n_after = self.registry.count
        n_new = n_after - n_before

        if n_new > 0:
            print(f"[SpecGuided] Dataset: {examples_used} exemplos processados, "
                  f"{n_new} novas primitives (+{n_before}->{n_after})")
            for p in self.registry.get_all():
                if p.confidence < 0.5:
                    print(f"  NOVA: {p.name}: {p.description[:60]}")

        return n_new

    def discover_from_spec(self, spec: OperatorSpec) -> bool:
        """Descobre nova primitiva a partir de uma OperatorSpec.

        Usa exemplos que casam com a spec para guiar a descoberta.

        Returns True se uma nova primitiva foi descoberta e adicionada.
        """
        examples = self.find_matching_examples(spec, n_max=50)
        if len(examples) < 3:
            return False

        n_before = self.registry.count
        for inp, out in examples:
            self.discovery.observe(inp, out)
        n_after = self.registry.count

        return n_after > n_before

    def get_new_primitives(self, before_count: int) -> list:
        """Retorna primitives descobertas depois de antes_count."""
        all_prims = self.registry.get_all()
        return all_prims[before_count:]

    def get_stats(self) -> dict:
        return {
            "geo_size": len(self.geo),
            "rel_size": len(self.rel),
            "total_experiences": len(self.geo) + len(self.rel),
        }
