"""
Suite de teste em ORDEM CRESCENTE DE DIFICULDADE. Doencas sem cura conhecida + avaliacao automatica.

Progressao escolhida (justificativa em cada item):
1. ALOPECIA AREATA - mecanismo bem mapeado (IFN-y/JAK-STAT), JAK inhibitors aprovados 2022-23.
   Bom proof-of-concept: se o sistema recupera JAKs, sinal forte. Tambem cabe novidade.
2. VITILIGO - autoimune mais complexo, IFN-y/CXCL10, ruxolitinib topico aprovado 2022.
3. FIBROSE PULMONAR IDIOPATICA - cura desconhecida, mecanismo complexo (TGF-b/EMT).
4. ALS - cura desconhecida, mecanismo aberto, ALTO desafio.

Roda cada um, avalia, salva relatorio. Mede progressao da assertividade.
"""
import sys, time, json, pathlib, argparse, os
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from co_scientist.run import run as run_coscientist
from shared.evaluator import evaluate_run
from shared.agent_base import save_run

LADDER = [
    ("alopecia areata", "JAK inhibitors aprovados (baricitinib/ritlecitinib 2022-23) - lead claro a recuperar"),
    ("vitiligo", "ruxolitinib topico aprovado 2022 - mecanismo IFN-y/JAK"),
    ("idiopathic pulmonary fibrosis", "pirfenidona/nintedanib so atrasam; cura desconhecida"),
    ("amyotrophic lateral sclerosis", "riluzol/edaravone modestos; cura desconhecida"),
]

def main(start_idx=0, end_idx=1, n=6, rounds=2, tourn=2):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERRO: defina ANTHROPIC_API_KEY no ambiente.")
        return
    results = []
    for idx in range(start_idx, end_idx+1):
        disease, note = LADDER[idx]
        print(f"\n{'='*78}\nDOENCA #{idx+1}: {disease}\n  {note}\n{'='*78}")
        t0=time.time()
        try:
            payload = run_coscientist(disease, n_initial=n, rounds=rounds, tourn_rounds=tourn)
        except Exception as e:
            print(f"FALHA em {disease}: {e}")
            continue
        report = evaluate_run(payload, top_k=5)
        report["timing_s"] = time.time()-t0
        results.append({"disease":disease,"summary":report["summary"],"timing":time.time()-t0})
        save_run(f"eval_{disease.replace(' ','_').lower()[:40]}_{int(time.time())}", report)
    print(f"\n{'='*78}\nPROGRESSAO DA ASSERTIVIDADE\n{'='*78}")
    for r in results:
        s = r["summary"]
        print(f"  {r['disease']:42s} assert={s['assertiveness']:.3f} | PMID-exist={s['avg_pmid_existence']:.0%} | known={s['pct_already_known']:.0%} | novel={s['pct_novel_plausible']:.0%} | div={s['diversity_keys']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tourn", type=int, default=2)
    a = ap.parse_args()
    main(a.start, a.end, a.n, a.rounds, a.tourn)
