"""
VALIDACAO DO COMPOUNDING (sinal de AGI): conforme a biblioteca CRESCE (K atomos), o nº de tasks
resolvidas SEM LLM cresce — e a parcela por COMPOSICAO cresce SUPER-LINEARMENTE? Se sim, cada
primitiva nova destrava nao so suas tasks mas COMBINACOES novas = aquisicao aberta de habilidade.
Tudo sem LLM (so execucao). Ablacao: K = 4,8,13,19 atomos.
"""
import sys, json, time, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from v3_bridge import Library
from solver import ARC_DIR, compile_transform, verify_on_train, apply_to_test, test_correct
from agi_signal import engine_solve_depth

def run_with_k(fns_all, K, tasks):
    fns = fns_all[:K]
    solo=comp=correct=0
    for tid in tasks:
        t=json.load(open(ARC_DIR/tid)); t["_id"]=tid
        preds,how,names=engine_solve_depth(t,fns,max_depth=2)
        if how=="solo": solo+=1
        elif how=="compose2": comp+=1
        if preds is not None and test_correct(preds,t): correct+=1
    return solo, comp, correct

def main(n=400, Ks=(4,8,13,19)):
    lib=Library(); lib.seed()
    fns_all=[(p["category"], compile_transform(p["code"])[0]) for p in lib.prims]
    fns_all=[(nm,f) for nm,f in fns_all if f is not None]
    tasks=sorted(p.name for p in ARC_DIR.glob("*.json"))[:n]
    print(f"=== COMPOUNDING (sem LLM) | {n} tasks | K atomos = {Ks} ===", flush=True)
    print(f"{'K':>4} {'solo':>6} {'compose':>8} {'total':>6} {'%total':>7} {'compose/total':>14}", flush=True)
    rows=[]; t0=time.time()
    for K in Ks:
        if K>len(fns_all): K=len(fns_all)
        solo,comp,correct = run_with_k(fns_all, K, tasks)
        tot=solo+comp
        frac = comp/tot if tot else 0
        print(f"{K:>4} {solo:>6} {comp:>8} {tot:>6} {tot/n:>6.1%} {frac:>13.0%}", flush=True)
        rows.append({"K":K,"solo":solo,"compose":comp,"total":tot,"correct":correct,"compose_frac":frac})
    print(f"\n>>> ANALISE DO SINAL DE AGI (compounding):", flush=True)
    if len(rows)>=2:
        a,b=rows[0],rows[-1]
        dK=b["K"]/max(a["K"],1)
        d_solo=b["solo"]/max(a["solo"],1); d_comp=b["compose"]/max(a["compose"],1)
        print(f"  de K={a['K']} p/ K={b['K']} (x{dK:.1f} atomos):", flush=True)
        print(f"    solos x{d_solo:.1f}  |  composicoes x{d_comp:.1f}", flush=True)
        verdict = ("COMPOUNDING: composicao cresce MAIS RAPIDO que atomos -> cada primitiva nova destrava COMBINACOES (super-linear) = aquisicao aberta"
                   if d_comp>dK else
                   "composicao cresce ~linear ou sublinear nesta faixa -> generatividade presente mas nao super-linear ainda")
        print(f"  VEREDITO: {verdict}", flush=True)
    out={"n":n,"Ks":list(Ks),"rows":rows,"duration_s":time.time()-t0}
    json.dump(out, open(pathlib.Path(__file__).parent/"runs"/f"agi_compound_{int(t0)}.json","w"), indent=2)
    print(f"salvo. {time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=400)
    a=ap.parse_args(); main(a.n)
