"""
VALIDACAO DO SINAL DE AGI (sem LLM): quanto a arquitetura PURA (memoria associativa + composicao
nao-comutativa) resolve sozinha, e quanto disso e CAPACIDADE EMERGENTE (composicao de primitivas
que nenhuma resolve sozinha) vs memoria (primitiva unica). Roda sobre as 400 tasks de treino.

Sinal-chave de generatividade: tasks resolvidas por COMPOSICAO (peca nova de pecas velhas), SEM LLM.
Depois: COMPOUNDING — adicionar primitivas faz o nº de solves-sem-LLM CRESCER? (curva de aprendizado)
"""
import sys, json, glob, time, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from v3_bridge import Library
from solver import (ARC_DIR, compile_transform, run_with_timeout, normalize_grid,
                    verify_on_train, apply_to_test, test_correct)

def engine_solve_depth(task, fns, max_depth=2, max_pairs=400):
    """resolve compondo ate max_depth primitivas (ordem importa). retorna (preds, how, names)."""
    # depth 1: solo
    for name, fn in fns:
        if fn is None: continue
        solved, ok, tot = verify_on_train(fn, task)
        if solved: return apply_to_test(fn, task), "solo", [name]
    if max_depth < 2: return None, None, []
    # depth 2: composicao nao-comutativa (ordem importa)
    valid = [(n,f) for n,f in fns if f is not None]
    tried = 0
    for (na,fa),(nb,fb) in itertools.product(valid, valid):
        if na==nb: continue
        def comp(g, fa=fa, fb=fb):
            r=fa([list(r) for r in g])
            if hasattr(r,"tolist"): r=r.tolist()
            return fb([list(r) for r in r])
        try:
            solved, ok, tot = verify_on_train(comp, task)
        except Exception:
            solved=False
        tried+=1
        if solved: return apply_to_test(comp, task), "compose2", [na,nb]
        if tried>=max_pairs: break
    return None, None, []

def main(n=400, max_depth=2):
    lib=Library(); lib.seed()
    fns=[(p["category"], compile_transform(p["code"])[0]) for p in lib.prims]
    fns=[(n_,f) for n_,f in fns if f is not None]
    print(f"=== SINAL DE AGI — motor PURO (sem LLM) | {len(fns)} primitivas semente | depth<= {max_depth} ===", flush=True)
    tasks=sorted(p.name for p in ARC_DIR.glob("*.json"))[:n]
    solo=0; comp=0; none=0; correct=0; t0=time.time()
    comp_examples=[]
    for i,tid in enumerate(tasks,1):
        t=json.load(open(ARC_DIR/tid)); t["_id"]=tid
        preds,how,names=engine_solve_depth(t,fns,max_depth)
        if how=="solo": solo+=1
        elif how=="compose2":
            comp+=1
            if len(comp_examples)<15: comp_examples.append((tid,names))
        else: none+=1
        if preds is not None and test_correct(preds,t): correct+=1
        if i%50==0: print(f"  {i}/{len(tasks)} | solo={solo} compose={comp} | {time.time()-t0:.0f}s", flush=True)
    tot_solved=solo+comp
    print(f"\n{'='*60}")
    print(f"SINAL DE AGI — motor PURO sem LLM sobre {len(tasks)} tasks ARC:")
    print(f"  resolvidas (treino-consistente): {tot_solved}/{len(tasks)} = {tot_solved/len(tasks):.1%}")
    print(f"  TEST correto: {correct}/{len(tasks)} = {correct/len(tasks):.1%}  (baseline GrammarForge 25.2%)")
    print(f"  por MEMORIA (primitiva unica): {solo}")
    print(f"  por COMPOSICAO nao-comutativa (capacidade EMERGENTE de pecas velhas): {comp}")
    print(f"\n  >>> {comp} tasks resolvidas por COMPOSICAO = generatividade pura (peca nova de pecas velhas, SEM LLM)")
    print("  exemplos de composicao emergente:")
    for tid,names in comp_examples: print(f"    {tid}: {names[0]} >> {names[1]}")
    out={"n":len(tasks),"solo":solo,"compose":comp,"none":none,"test_correct":correct,
         "rate_solved":tot_solved/len(tasks),"rate_test":correct/len(tasks),
         "compose_examples":comp_examples,"n_primitives":len(fns),"duration_s":time.time()-t0}
    json.dump(out, open(pathlib.Path(__file__).parent/"runs"/f"agi_signal_{int(t0)}.json","w"), indent=2)
    print(f"\nsalvo. tempo {time.time()-t0:.0f}s")
    return out

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=400); ap.add_argument("--depth",type=int,default=2)
    a=ap.parse_args(); main(a.n,a.depth)
