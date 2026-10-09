"""
Teste: MultimodalInvariantDataset — geracao, cross-dominio, save/load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from core.invariant_dataset import MultimodalInvariantDataset


PASS = "[PASS]"
FAIL = "[FAIL]"


def test_build():
    ds = MultimodalInvariantDataset()
    summary = ds.summary()
    print(f"  Total: {summary['total_examples']} exemplos")
    print(f"  Tipos: {summary['invariant_types']}")
    print(f"  Dominios: {summary['domains']}")
    assert summary["total_examples"] > 0


def test_all_types_present():
    ds = MultimodalInvariantDataset()
    expected = {"branching", "reflection", "expansion", "contraction",
                "symmetry", "causal_chain", "spatial_transformation"}
    found = set(ds.get_all_types())
    missing = expected - found
    assert not missing, f"Faltando: {missing}"
    print(f"  Tipos: {found}")


def test_cross_domain():
    ds = MultimodalInvariantDataset()
    cross = ds.summary()["cross_domain_invariants"]
    print(f"  Cross-dominio: {cross}")
    domains = ds.get_domains_for_invariant("branching")
    print(f"  branching dominios: {domains}")
    assert len(domains) >= 2, "branching deveria ser cross-dominio"
    domains = ds.get_domains_for_invariant("reflection")
    print(f"  reflection dominios: {domains}")
    assert len(domains) >= 2, "reflection deveria ser cross-dominio"
    assert "spatial_transformation" not in cross, \
        "spatial_transformation tem 1 dominio, nao deveria ser cross"


def test_get_examples():
    ds = MultimodalInvariantDataset()
    exs = ds.get_examples("branching")
    assert len(exs) > 0
    for ex in exs:
        assert ex.invariant_type == "branching"
    print(f"  Branching: {len(exs)} exemplos")
    for ex in exs:
        print(f"    {ex.domain}/{ex.modality}: {ex.metadata.get('description', '')[:60]}")


def test_modalities():
    ds = MultimodalInvariantDataset()
    mods = set(ex.modality for ex in ds._examples)
    print(f"  Modalidades: {mods}")
    assert "grid" in mods
    assert len(mods) >= 2


def test_grid_examples():
    ds = MultimodalInvariantDataset()
    for ex in ds.get_by_modality("grid"):
        inp, out = ex.input, ex.output
        assert isinstance(inp, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert inp.ndim == 2 and out.ndim == 2
        assert inp.dtype == np.int32 and out.dtype == np.int32


def test_save_load():
    import tempfile, json
    ds = MultimodalInvariantDataset()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
        json.dump([], f)
    try:
        ds.save(tmp_path)
        loaded = MultimodalInvariantDataset(tmp_path)
        assert loaded.total_examples == ds.total_examples
        assert loaded.get_all_types() == ds.get_all_types()
        print(f"  Save/Load OK: {loaded.total_examples} exemplos")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("=== MultimodalInvariantDataset Tests ===\n")
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"{name}...")
        try:
            fn()
            print(f"  {PASS}\n")
        except Exception as e:
            print(f"  {FAIL}: {e}\n")
            import traceback
            traceback.print_exc()
