"""
Multimodal Invariant Dataset — Gera exemplos de invariantes em MULTIPLOS dominios.

Cada invariante (branching, reflection, expansion, etc.) aparece em
diferentes modalidades/dominios para testar transferencia cross-domínio.

Estrutura:
  {
    "invariant_type": "branching",
    "domain": "biology",
    "input": ...,     # grid, texto, ou dict (depende da modalidade)
    "output": ...,
    "metadata": {"description": "...", "modality": "grid|text|program"}
  }

Uso:
    dataset = MultimodalInvariantDataset()
    examples = dataset.get_examples("branching")
    cross = dataset.get_cross_domain_examples("reflection", min_domains=3)
"""
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import random
import json


@dataclass
class InvariantExample:
    invariant_type: str
    domain: str
    modality: str
    input: any
    output: any
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        inp = self.input.tolist() if isinstance(self.input, np.ndarray) else self.input
        out = self.output.tolist() if isinstance(self.output, np.ndarray) else self.output
        return {
            "invariant_type": self.invariant_type,
            "domain": self.domain,
            "modality": self.modality,
            "input": inp,
            "output": out,
            "metadata": self.metadata,
        }


def _random_grid(h: int, w: int, n_colors: int = 3) -> np.ndarray:
    return np.random.randint(0, n_colors + 1, (h, w), dtype=np.int32)


def _copy_grid(g: np.ndarray) -> np.ndarray:
    return g.copy()


# ============================================================
# GERADORES POR DOMINIO
# ============================================================

# ---------- BRANCHING ----------

def _branching_biology() -> Tuple[np.ndarray, np.ndarray]:
    """Arvore: tronco → galhos."""
    inp = np.zeros((9, 7), dtype=np.int32)
    inp[2:9, 3] = 2  # tronco
    out = np.zeros((9, 7), dtype=np.int32)
    out[2:9, 3] = 2  # tronco
    out[1, 1] = 2; out[1, 2] = 2; out[1, 4] = 2; out[1, 5] = 2  # galhos
    out[0, 0] = 2; out[0, 6] = 2
    return inp, out


def _branching_geography() -> Tuple[np.ndarray, np.ndarray]:
    """Rio: curso principal → afluentes."""
    inp = np.zeros((7, 7), dtype=np.int32)
    inp[:, 3] = 1
    out = np.zeros((7, 7), dtype=np.int32)
    out[:, 3] = 1
    out[1, :2] = 1; out[3, 4:6] = 1; out[5, :2] = 1
    return inp, out


def _branching_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Fractal simples: quadrado → 3 quadrados."""
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1:4, 1:4] = 3
    out = np.zeros((9, 9), dtype=np.int32)
    out[0:3, 0:3] = 3; out[0:3, 6:9] = 3; out[3:6, 3:6] = 3
    return inp, out


def _branching_program() -> Tuple[dict, dict]:
    """Programa: if-else como estrutura de decisao."""
    inp = {"type": "control_flow", "structure": "linear", "branches": 0}
    out = {"type": "control_flow", "structure": "branching", "branches": 2}
    return inp, out


# ---------- REFLECTION ----------

def _reflection_geometry_h() -> Tuple[np.ndarray, np.ndarray]:
    """Flip horizontal."""
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 4
    out = np.fliplr(inp)
    return inp, out


def _reflection_geometry_v() -> Tuple[np.ndarray, np.ndarray]:
    """Flip vertical."""
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1:4, 1] = 4
    out = np.flipud(inp)
    return inp, out


def _reflection_physics() -> Tuple[dict, dict]:
    """Bola quicando: trajetoria inverte."""
    inp = {"object": "ball", "trajectory": [0, 1, 2, 3], "wall_at": 3}
    out = {"object": "ball", "trajectory": [0, 1, 2, 3, 2, 1, 0], "reflected": True}
    return inp, out


def _reflection_audio() -> Tuple[dict, dict]:
    """Eco: som reflete."""
    inp = {"signal": "pulse", "amplitude": [1, 0.8, 0.6, 0.4]}
    out = {"signal": "echo", "amplitude": [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1], "reflected": True}
    return inp, out


# ---------- EXPANSION ----------

def _expansion_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Tile: um objeto → multiplos."""
    inp = np.zeros((4, 4), dtype=np.int32)
    inp[1:3, 1:3] = 1
    out = np.zeros((8, 8), dtype=np.int32)
    out[1:3, 1:3] = 1; out[1:3, 5:7] = 1; out[5:7, 1:3] = 1; out[5:7, 5:7] = 1
    return inp, out


def _expansion_thermo() -> Tuple[dict, dict]:
    """Gas: volume aumenta com temperatura."""
    inp = {"state": "gas", "temperature": 300, "volume": 1.0}
    out = {"state": "gas", "temperature": 600, "volume": 2.0}
    return inp, out


def _expansion_biology() -> Tuple[np.ndarray, np.ndarray]:
    """Celula se divide."""
    inp = np.zeros((3, 3), dtype=np.int32)
    inp[1, 1] = 5
    out = np.zeros((6, 6), dtype=np.int32)
    out[1, 1] = 5; out[1, 4] = 5; out[4, 1] = 5; out[4, 4] = 5
    return inp, out


# ---------- CONTRACTION ----------

def _contraction_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Crop: objeto grande → bounding box."""
    inp = np.zeros((8, 8), dtype=np.int32)
    inp[2:6, 2:6] = 2
    out = np.zeros((4, 4), dtype=np.int32)
    out[:, :] = 2
    return inp, out


def _contraction_physics() -> Tuple[dict, dict]:
    """Gas: volume diminui com pressao."""
    inp = {"state": "gas", "pressure": 1.0, "volume": 3.0}
    out = {"state": "gas", "pressure": 3.0, "volume": 1.0}
    return inp, out


# ---------- SYMMETRY ----------

def _symmetry_bilateral() -> Tuple[np.ndarray, np.ndarray]:
    """Face com simetria bilateral."""
    inp = np.zeros((7, 7), dtype=np.int32)
    inp[1, 2:5] = 3; inp[3, 1:6] = 3
    out = np.zeros((7, 7), dtype=np.int32)
    out[1, :] = 3; out[3, :] = 3  # simetria perfeita
    # preserva o que ja era simetrico, completa o resto
    out[1, :] = 3; out[3, :] = 3
    out[2, 1] = 3; out[2, 5] = 3  # olhos
    return inp, out


def _symmetry_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Rotacao 180: simetria rotacional."""
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[0, 0] = 4; inp[0, 4] = 4; inp[4, 0] = 4; inp[4, 4] = 4
    out = inp.copy()  # ja simetrico
    return inp, out


# ---------- CAUSAL_CHAIN ----------

def _causal_physics() -> Tuple[dict, dict]:
    """Bola empurra bola."""
    inp = {"state": "A_em_repouso_B_em_repouso"}
    out = {"state": "A_move_B_move_apos_colisao"}
    return inp, out


def _causal_program() -> Tuple[dict, dict]:
    """Pipeline de programa."""
    inp = {"pipeline": [], "result": None}
    out = {"pipeline": ["load", "transform", "save"], "result": "output"}
    return inp, out


def _causal_daily() -> Tuple[dict, dict]:
    """Causa-efeito cotidiano."""
    inp = {"event": "noite", "state": "escuro"}
    out = {"event": "dia", "state": "claro"}
    return inp, out


# ---------- SPATIAL_TRANSFORMATION ----------

def _translate_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Move objeto."""
    inp = np.zeros((7, 7), dtype=np.int32)
    inp[1:3, 1:3] = 1
    out = np.zeros((7, 7), dtype=np.int32)
    out[4:6, 4:6] = 1
    return inp, out


def _rotate_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Rotaciona 90."""
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 2
    out = np.rot90(inp, k=-1)  # cw
    return inp, out


# ============================================================
# REGISTRO CENTRAL
# ============================================================

INVARIANT_GENERATORS: Dict[str, Dict[str, List[dict]]] = {
    "branching": {
        "grid": [
            {"fn": _branching_biology, "domain": "biology",
             "desc": "Tronco → galhos (arvore)"},
            {"fn": _branching_geography, "domain": "geography",
             "desc": "Rio → afluentes"},
            {"fn": _branching_geometry, "domain": "geometry",
             "desc": "Quadrado → fractal simples"},
        ],
        "program": [
            {"fn": _branching_program, "domain": "programming",
             "desc": "Linear → if-else branching"},
        ],
    },
    "reflection": {
        "grid": [
            {"fn": _reflection_geometry_h, "domain": "geometry",
             "desc": "Flip horizontal"},
            {"fn": _reflection_geometry_v, "domain": "geometry",
             "desc": "Flip vertical"},
        ],
        "symbolic": [
            {"fn": _reflection_physics, "domain": "physics",
             "desc": "Bola quicando (trajetoria inverte)"},
            {"fn": _reflection_audio, "domain": "audio",
             "desc": "Eco (som reflete)"},
        ],
    },
    "expansion": {
        "grid": [
            {"fn": _expansion_geometry, "domain": "geometry",
             "desc": "Tile: 1→4 objetos"},
            {"fn": _expansion_biology, "domain": "biology",
             "desc": "Celula se divide"},
        ],
        "symbolic": [
            {"fn": _expansion_thermo, "domain": "physics",
             "desc": "Gas expande com temperatura"},
        ],
    },
    "contraction": {
        "grid": [
            {"fn": _contraction_geometry, "domain": "geometry",
             "desc": "Crop: objeto → bounding box"},
        ],
        "symbolic": [
            {"fn": _contraction_physics, "domain": "physics",
             "desc": "Gas contrai com pressao"},
        ],
    },
    "symmetry": {
        "grid": [
            {"fn": _symmetry_bilateral, "domain": "biology",
             "desc": "Simetria bilateral (face)"},
            {"fn": _symmetry_geometry, "domain": "geometry",
             "desc": "Simetria rotacional 180"},
        ],
    },
    "causal_chain": {
        "symbolic": [
            {"fn": _causal_physics, "domain": "physics",
             "desc": "Colisao: A→B"},
            {"fn": _causal_program, "domain": "programming",
             "desc": "Pipeline de dados"},
            {"fn": _causal_daily, "domain": "daily_life",
             "desc": "Noite→dia"},
        ],
    },
    "spatial_transformation": {
        "grid": [
            {"fn": _translate_geometry, "domain": "geometry",
             "desc": "Translacao de objeto"},
            {"fn": _rotate_geometry, "domain": "geometry",
             "desc": "Rotacao 90"},
        ],
    },
}


class MultimodalInvariantDataset:
    """Dataset multimodal de invariantes.

    Gera exemplos on-the-fly ou carrega de arquivo.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self._examples: List[InvariantExample] = []
        self._by_type: Dict[str, List[InvariantExample]] = defaultdict(list)
        self._by_domain: Dict[str, List[InvariantExample]] = defaultdict(list)

        if data_path:
            self.load(data_path)
        else:
            self._build()

    def _build(self):
        """Gera exemplos de todos os invariantes registrados."""
        self._examples = []
        for inv_type, modalities in INVARIANT_GENERATORS.items():
            for modality, generators in modalities.items():
                for gen in generators:
                    try:
                        inp, out = gen["fn"]()
                        ex = InvariantExample(
                            invariant_type=inv_type,
                            domain=gen["domain"],
                            modality=modality,
                            input=inp,
                            output=out,
                            metadata={"description": gen["desc"]},
                        )
                        self._examples.append(ex)
                    except Exception as e:
                        print(f"[invariant_dataset] Erro gerando {inv_type}/{gen['domain']}: {e}")

        # Indexa
        self._by_type.clear()
        self._by_domain.clear()
        for ex in self._examples:
            self._by_type[ex.invariant_type].append(ex)
            self._by_domain[ex.domain].append(ex)

    def get_examples(self, invariant_type: str) -> List[InvariantExample]:
        return list(self._by_type.get(invariant_type, []))

    def get_cross_domain_examples(self, invariant_type: str,
                                   min_domains: int = 2) -> List[InvariantExample]:
        """Exemplos de um invariante que aparece em N dominios diferentes."""
        examples = self._by_type.get(invariant_type, [])
        domains = set(ex.domain for ex in examples)
        if len(domains) >= min_domains:
            return examples
        return []

    def get_domains_for_invariant(self, invariant_type: str) -> List[str]:
        return sorted(set(ex.domain for ex in self._by_type.get(invariant_type, [])))

    def get_all_types(self) -> List[str]:
        return sorted(self._by_type.keys())

    def get_all_domains(self) -> List[str]:
        return sorted(self._by_domain.keys())

    def get_by_domain(self, domain: str) -> List[InvariantExample]:
        return list(self._by_domain.get(domain, []))

    def get_by_modality(self, modality: str) -> List[InvariantExample]:
        return [ex for ex in self._examples if ex.modality == modality]

    @property
    def total_examples(self) -> int:
        return len(self._examples)

    def summary(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "invariant_types": len(self._by_type),
            "domains": len(self._by_domain),
            "by_type": {t: len(exs) for t, exs in self._by_type.items()},
            "by_domain": {d: len(exs) for d, exs in self._by_domain.items()},
            "cross_domain_invariants": [
                t for t in self._by_type
                if len(set(e.domain for e in self._by_type[t])) >= 2
            ],
        }

    def save(self, path: str):
        """Salva dataset em JSON."""
        data = [ex.to_dict() for ex in self._examples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[invariant_dataset] Salvo {len(data)} exemplos em {path}")

    def load(self, path: str):
        """Carrega dataset de JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._examples = []
        for item in data:
            inp = np.array(item["input"]) if isinstance(item["input"], list) else item["input"]
            out = np.array(item["output"]) if isinstance(item["output"], list) else item["output"]
            ex = InvariantExample(
                invariant_type=item["invariant_type"],
                domain=item["domain"],
                modality=item["modality"],
                input=inp,
                output=out,
                metadata=item.get("metadata", {}),
            )
            self._examples.append(ex)
        self._by_type.clear()
        self._by_domain.clear()
        for ex in self._examples:
            self._by_type[ex.invariant_type].append(ex)
            self._by_domain[ex.domain].append(ex)
        print(f"[invariant_dataset] Carregado {len(self._examples)} exemplos de {path}")
