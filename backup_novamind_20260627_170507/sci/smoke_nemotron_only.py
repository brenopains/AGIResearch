"""
Smoke test SEM Anthropic — usa SO o Nemotron local. Prova que a infra (PubMed + LLM local + agentes)
funciona. Honestidade: 4B nao gera hipoteses de qualidade Co-Scientist. Mas demonstra a tubulacao.
"""
import sys, pathlib, re, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from shared.pubmed import search_and_fetch, format_for_llm
from shared.nemotron_client import ask, summarize_abstract

def main(disease="alopecia areata"):
    print(f"=== SMOKE TEST NEMOTRON-ONLY ===\ndoenca: {disease}\n")
    print("[1] PubMed: buscando 5 papers recentes...")
    papers = search_and_fetch(f"{disease} treatment mechanism", retmax=5)
    print(f"  -> {len(papers)} papers")
    for p in papers: print(f"    PMID:{p['pmid']} ({p['year']}) {p['title'][:80]}")

    print("\n[2] Nemotron: comprime cada abstract (papel utilitario)...")
    summaries = []
    for p in papers:
        if not p["abstract"]: continue
        s = summarize_abstract(p["abstract"], target_tokens=80)
        summaries.append(f"[PMID:{p['pmid']}] {s}")
        print(f"  PMID:{p['pmid']} -> {s[:120]}...")

    print("\n[3] Nemotron: propoe 3 hipoteses (papel principal aqui — limitado, mas demonstra fluxo)...")
    user = f"DOENCA: {disease}\n\nLITERATURA RESUMIDA:\n" + "\n".join(summaries) + \
           "\n\nProponha 3 hipoteses TESTAVEIS de tratamento, cada uma com mecanismo e intervencao. " \
           "Saida JSON: array de objetos com keys 'titulo','mecanismo','intervencao'."
    raw = ask([{"role":"user","content":user}], temperature=0.6, max_tokens=900)
    print("  RESPOSTA:")
    print(raw[:2000])

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("disease",nargs="?",default="alopecia areata"); a=ap.parse_args()
    main(a.disease)
