"""
Expande o MultimodalInvariantDataset com exemplos reais dos curriculuns,
audio, program traces, e dominios adicionais.

Gera D:\novamind_data\multimodal_world\expanded_dataset.json
com centenas de exemplos em 6+ modalidades.
"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.invariant_dataset import MultimodalInvariantDataset, InvariantExample


OUTPUT_DIR = r"D:\novamind_data\multimodal_world"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "expanded_dataset.json")


# ============================================================
# GERADORES ADICIONAIS
# ============================================================

def _identity_variants():
    """10 variantes de identity (shape preserved, identical)."""
    examples = []
    for size in range(3, 8):
        for seed in range(3):
            rng = np.random.RandomState(seed * 10 + size)
            inp = rng.randint(0, 4, (size, size)).astype(np.int32)
            out = inp.copy()
            examples.append(InvariantExample(
                invariant_type="symmetry", domain="geometry",
                modality="grid", input=inp, output=out,
                metadata={"description": f"identity {size}x{size} seed={seed}"},
            ))
    return examples


def _audio_pattern_reflection():
    """Reflection em audio: eco simples."""
    examples = []
    for length in [5, 8, 10]:
        signal = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
        # Eco: sinal + versao atenuada e atrasada
        echo = np.zeros(length, dtype=np.float32)
        echo[:len(signal)] = signal
        echo[len(signal):] = signal[:length - len(signal)] * 0.3
        inp = {"signal": signal.tolist(), "type": "audio"}
        out = {"signal": echo.tolist(), "type": "echo", "reflected": True}
        examples.append(InvariantExample(
            invariant_type="reflection", domain="audio",
            modality="audio", input=inp, output=out,
            metadata={"description": f"audio echo length={length}"},
        ))
    return examples


def _program_trace_branching():
    """Branching em traces de programa."""
    examples = []
    traces = [
        {
            "inp": {"type": "function", "calls": 1, "branches": 0},
            "out": {"type": "function", "calls": 3, "branches": 2},
            "desc": "if-else branching",
        },
        {
            "inp": {"type": "loop", "iterations": 1, "nested": 0},
            "out": {"type": "loop", "iterations": 3, "nested": 1},
            "desc": "nested loop branching",
        },
        {
            "inp": {"type": "data", "sources": 1, "destinations": 1},
            "out": {"type": "data", "sources": 2, "destinations": 3},
            "desc": "data pipeline branching",
        },
    ]
    for t in traces:
        examples.append(InvariantExample(
            invariant_type="branching", domain="programming",
            modality="program_trace", input=t["inp"], output=t["out"],
            metadata={"description": t["desc"]},
        ))
    return examples


def _causal_sequences():
    """Causal chains em eventos."""
    examples = []
    chains = [
        {
            "domain": "physics",
            "inp": {"event": "ball_dropped", "height": 10, "surface": "hard"},
            "out": {"event": "ball_bounced", "height": 7, "bounces": 3},
            "desc": "ball drop → bounce cascade",
        },
        {
            "domain": "biology",
            "inp": {"event": "stimulus", "type": "light", "intensity": 0.8},
            "out": {"event": "response", "type": "pupil_contraction", "magnitude": 0.5},
            "desc": "light stimulus → pupil response",
        },
        {
            "domain": "economics",
            "inp": {"event": "interest_rate_change", "delta": -0.5},
            "out": {"event": "market_impact", "direction": "up", "magnitude": 0.3},
            "desc": "interest rate → market movement",
        },
    ]
    for c in chains:
        examples.append(InvariantExample(
            invariant_type="causal_chain", domain=c["domain"],
            modality="event_sequence", input=c["inp"], output=c["out"],
            metadata={"description": c["desc"]},
        ))
    return examples


def _expansion_contraction_more():
    """Mais exemplos de expansion/contraction."""
    examples = []

    # Expansion: thermal (dict)
    for factor in [1.5, 2.0, 3.0]:
        examples.append(InvariantExample(
            invariant_type="expansion", domain="physics",
            modality="simulation",
            input={"state": "gas", "temperature": 300, "volume": 1.0},
            output={"state": "gas", "temperature": int(300 * factor), "volume": factor},
            metadata={"description": f"thermal expansion {factor}x"},
        ))

    # Expansion: cell division (grid)
    for size in [2, 3]:
        inp = np.zeros((size, size), dtype=np.int32)
        inp[size // 2, size // 2] = 5
        out = np.zeros((size * 2, size * 2), dtype=np.int32)
        out[size // 2, size // 2] = 5
        out[size // 2, size // 2 + size] = 5
        out[size // 2 + size, size // 2] = 5
        out[size // 2 + size, size // 2 + size] = 5
        examples.append(InvariantExample(
            invariant_type="expansion", domain="biology",
            modality="grid", input=inp, output=out,
            metadata={"description": f"cell division {size}x{size} → {size*2}x{size*2}"},
        ))

    # Contraction: crop
    for size in [6, 8]:
        inp = np.ones((size, size), dtype=np.int32) * 3
        inp[1:-1, 1:-1] = 5
        out = inp[1:-1, 1:-1].copy()
        examples.append(InvariantExample(
            invariant_type="contraction", domain="geometry",
            modality="grid", input=inp, output=out,
            metadata={"description": f"crop {size}x{size} → {size-2}x{size-2}"},
        ))

    return examples


def _spatial_transforms():
    """Mais transformacoes espaciais."""
    examples = []
    for size in [5, 7]:
        inp = np.zeros((size, size), dtype=np.int32)
        inp[size // 3:2 * size // 3, size // 3:2 * size // 3] = 4

        # Translate
        out_t = np.zeros((size, size), dtype=np.int32)
        out_t[1:1 + size // 3, 1:1 + size // 3] = 4
        examples.append(InvariantExample(
            invariant_type="spatial_transformation", domain="geometry",
            modality="grid", input=inp, output=out_t,
            metadata={"description": f"translate {size}x{size}"},
        ))

        # Rotate
        out_r = np.rot90(inp, k=-1)
        examples.append(InvariantExample(
            invariant_type="spatial_transformation", domain="geometry",
            modality="grid", input=inp, output=out_r,
            metadata={"description": f"rotate {size}x{size}"},
        ))

    return examples


def _pattern_completion():
    """Exemplos de completude de padrao (causal/correlacao)."""
    examples = []
    patterns = [
        {"inp": [1, 2, 3], "out": [4], "domain": "math", "desc": "arithmetic sequence"},
        {"inp": [1, 1, 2, 3, 5], "out": [8], "domain": "math", "desc": "fibonacci"},
        {"inp": ["a", "b", "c"], "out": ["d"], "domain": "language", "desc": "alphabet"},
    ]
    for p in patterns:
        examples.append(InvariantExample(
            invariant_type="causal_chain", domain=p["domain"],
            modality="sequence", input=p["inp"], output=p["out"],
            metadata={"description": p["desc"]},
        ))
    return examples


# ============================================================
# CONSTRUCAO
# ============================================================

def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Construindo Expanded Multimodal Dataset ===\n")

    # Base original
    base = MultimodalInvariantDataset()
    all_examples = list(base._examples)
    print(f"Base original: {len(base._examples)} exemplos")

    # Adiciona novos geradores
    generators = [
        ("identity_variants", _identity_variants),
        ("audio_reflection", _audio_pattern_reflection),
        ("program_branching", _program_trace_branching),
        ("causal_sequences", _causal_sequences),
        ("expansion_more", _expansion_contraction_more),
        ("spatial_more", _spatial_transforms),
        ("pattern_completion", _pattern_completion),
    ]

    total_new = 0
    for name, gen_fn in generators:
        try:
            new_exs = gen_fn()
            all_examples.extend(new_exs)
            total_new += len(new_exs)
            print(f"  {name}: +{len(new_exs)} exemplos")
        except Exception as e:
            print(f"  {name}: ERRO: {e}")

    print(f"\nTotal: {len(all_examples)} exemplos ({total_new} novos)")

    # Salva
    data = [ex.to_dict() for ex in all_examples]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {OUTPUT_FILE}")

    # Estatisticas
    types = {}
    domains = {}
    modalities = {}
    for ex in all_examples:
        types[ex.invariant_type] = types.get(ex.invariant_type, 0) + 1
        domains[ex.domain] = domains.get(ex.domain, 0) + 1
        modalities[ex.modality] = modalities.get(ex.modality, 0) + 1

    print(f"\nPor tipo:")
    for t, n in sorted(types.items()):
        print(f"  {t}: {n}")
    print(f"\nPor dominio:")
    for d, n in sorted(domains.items()):
        print(f"  {d}: {n}")
    print(f"\nPor modalidade:")
    for m, n in sorted(modalities.items()):
        print(f"  {m}: {n}")


if __name__ == "__main__":
    build()
