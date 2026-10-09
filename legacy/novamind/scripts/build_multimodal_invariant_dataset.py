"""
Script: Constroi e salva o MultimodalInvariantDataset em D:\novamind_data\invariant_dataset\
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.invariant_dataset import MultimodalInvariantDataset


OUTPUT_DIR = r"D:\novamind_data\invariant_dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "invariant_dataset.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Construindo MultimodalInvariantDataset ===")
    dataset = MultimodalInvariantDataset()
    summary = dataset.summary()

    print(f"\nTotal de exemplos: {summary['total_examples']}")
    print(f"Tipos de invariantes: {summary['invariant_types']}")
    print(f"Dominios: {summary['domains']}")
    print(f"\nPor tipo:")
    for t, n in summary["by_type"].items():
        domains = dataset.get_domains_for_invariant(t)
        print(f"  {t}: {n} exemplos, dominios: {domains}")
    print(f"\nInvariantes cross-dominio: {summary['cross_domain_invariants']}")

    # Salva
    dataset.save(OUTPUT_FILE)
    print(f"\nDataset salvo em: {OUTPUT_FILE}")

    # Load test
    print("\n=== Teste de load ===")
    loaded = MultimodalInvariantDataset(OUTPUT_FILE)
    print(f"Carregado: {loaded.total_examples} exemplos")
    print(f"Tipos: {loaded.get_all_types()}")
    print(f"Dominios: {loaded.get_all_domains()}")

    # Estatisticas por modalidade
    for mod in ["grid", "symbolic", "program"]:
        n = len(loaded.get_by_modality(mod))
        if n > 0:
            print(f"  Modalidade '{mod}': {n} exemplos")

    print("\nDone!")


if __name__ == "__main__":
    main()
