"""
H85 — Diagnostico de LESAO/DISSOCIACAO do ensemble (a estrutura revelada por como QUEBRA)
==========================================================================================
Metodo emprestado da neuropsicologia (dupla dissociacao: Broca, prosopagnosia, neglect):
o cerebro foi mapeado vendo QUE faculdade some quando uma area e lesionada. Aplicamos o
mesmo ao nosso "cerebro" de 21,2%: ablamos cada base e medimos QUE tarefas do ARC morrem.

Produz uma MATRIZ de dissociacao por-tarefa {tarefa: {base: bool}} e dela extrai:
  - cobertura por base;
  - contribuicao UNICA marginal (coverage[b] - uniao(outras)) = "cortex eloquente"
    (lesiona-la custa solves que ninguem mais cobre);
  - histograma de redundancia (quantas bases resolvem cada tarefa);
  - o RESIDUO (~315 nao resolvidas) caracterizado (mesmo-shape? nº objetos? paleta? area)
    -> diz QUAL experimento (H86/H87/H88) e sequer aplicavel a cada fatia do residuo.

Nao e um solver; e o INSTRUMENTO. Pior caso: relata honestamente que o residuo e inalcancavel.
Bases deterministas (pulse, abduce, breadth, mb) sao RE-RODADAS ao vivo; nca/rc (estocasticas
/caras, ja validadas) sao lidas dos caches h70/h72. Use --fresh p/ forcar tudo do zero.
Refs: Carhart-Harris 2014 (entropic brain: modularidade comprometida, integracao global) p/ H86.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

HERE=os.path.dirname(__file__)
DATA=os.path.join(HERE,"arc_data","data","training")
H63=_load("exp_h63_efficiency_breadth.py")   # traz PU, AB, NEW, induce
PU=H63.PU; AB=H63.AB
MB=_load("exp_h68_mb_arc.py")                  # solve_mb, patches, mb_expand, Wproj

BASES=["pulse","abduce","breadth","mb","nca","rc"]

def ok_fn(fn, tests):
    try:
        for xi,yo in tests:
            r=fn(xi)
            if r is None or not isinstance(r,np.ndarray) or r.shape!=yo.shape or not np.array_equal(r,yo): return False
        return True
    except Exception: return False

def ok_outs(outs, tests):
    if outs is None: return False
    try:
        return all(p is not None and p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests))
    except Exception: return False

def run_pulse(train, tests):
    pool=dict(PU.FIXED)
    if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
    prog,_,_=PU.pulse_search(pool,train)
    return prog is not None and ok_fn(prog,tests)

def run_abduce(train, tests):
    render,_,_=AB.abduce(train)
    return render is not None and ok_fn(render,tests)

def run_breadth(train, tests):
    pool=dict(PU.FIXED)
    if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
    pool.update(H63.NEW)
    prog=H63.induce(train,pool)
    return prog is not None and ok_fn(prog,tests)

def run_mb(train, tests):
    if not all(i.shape==o.shape for i,o in train): return False
    return ok_outs(MB.solve_mb(train,tests), tests)

def task_features(train, tests):
    same=all(i.shape==o.shape for i,o in train)
    smaller=all(o.size<i.size for i,o in train)
    larger=all(o.size>i.size for i,o in train)
    cls="same_shape" if same else ("crop" if smaller else ("grow" if larger else "diff_shape"))
    nob=[]; pal=set(); area=[]
    for i,_ in train:
        try: nob.append(len(AB.parse(i,8,False)))
        except Exception: nob.append(-1)
        pal|=set(np.unique(i).tolist()); area.append(int(i.size))
    return {"shape_class":cls,"n_objs":round(float(np.mean(nob)),1),
            "palette":len(pal),"area":int(np.mean(area))}

def build_matrix(files, fresh=False):
    nca=set(json.load(open(os.path.join(HERE,"h70_nca_solved.json")))) if os.path.exists(os.path.join(HERE,"h70_nca_solved.json")) else set()
    rc=set(json.load(open(os.path.join(HERE,"h72_rcens_solved.json")))) if os.path.exists(os.path.join(HERE,"h72_rcens_solved.json")) else set()
    nca_live=rc_live=None
    if fresh:
        NCA=_load("exp_h70_nca.py"); RC=_load("exp_h72_rc_ensemble.py")
        nca_live=NCA; rc_live=RC; res=[RC.make_res(s,rho) for s,rho in RC.CONFIGS]
    matrix={}; feats={}
    for k,fp in enumerate(files):
        name=os.path.basename(fp)
        t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        row={}
        row["pulse"]=run_pulse(train,tests)
        row["abduce"]=run_abduce(train,tests)
        row["breadth"]=run_breadth(train,tests)
        row["mb"]=run_mb(train,tests)
        if fresh:
            row["nca"]=ok_outs(nca_live.solve_nca(train,tests),tests)
            cand=[o for o in (rc_live.solve_one(train,tests,Wr,Wi) for Wr,Wi in res) if o is not None]
            row["rc"]= bool(cand) and _rc_vote_ok(cand,tests)
        else:
            row["nca"]= name in nca
            row["rc"]= name in rc
        matrix[name]=row
        feats[name]=task_features(train,tests)
        if (k+1)%50==0: print(f"  ... {k+1}/400 tarefas avaliadas"); sys.stdout.flush()
    return matrix, feats

def _rc_vote_ok(cand, tests):
    from scipy import stats
    voted=[]
    for ti in range(len(tests)):
        stack=np.stack([c[ti] for c in cand]); voted.append(stats.mode(stack,axis=0,keepdims=False).mode)
    return ok_outs(voted,tests)

def report(matrix, feats):
    cov={b:set(t for t,r in matrix.items() if r[b]) for b in BASES}
    union=set().union(*cov.values())
    allt=set(matrix); residue=allt-union
    print("="*72)
    print(f"H85 Diagnostico de lesao/dissociacao do ensemble (400 tarefas treino)")
    print(f"  UNIAO (todas as bases): {len(union)}/400 = {100*len(union)/400:.1f}%")
    print("\n  Cobertura por base e CONTRIBUICAO UNICA marginal (cortex eloquente):")
    print(f"    {'base':>8} | {'cobre':>6} | {'unica':>6}  (lesionar custa N solves que ninguem mais cobre)")
    for b in BASES:
        others=set().union(*[cov[o] for o in BASES if o!=b]) if len(BASES)>1 else set()
        uniq=cov[b]-others
        print(f"    {b:>8} | {len(cov[b]):>6} | {len(uniq):>6}  {sorted(uniq)[:4]}")
    # redundancia
    from collections import Counter
    deg=Counter(sum(1 for b in BASES if r[b]) for r in matrix.values())
    print("\n  Redundancia (quantas bases resolvem cada tarefa):")
    for k in range(0,len(BASES)+1):
        print(f"    {k} base(s): {deg.get(k,0):>3} tarefas")
    single={t:[b for b in BASES if matrix[t][b]][0] for t in allt if sum(matrix[t][b] for b in BASES)==1}
    print(f"\n  Tarefas SINGLE-BASE (dissociacao pura): {len(single)}")
    # residuo caracterizado
    print(f"\n  RESIDUO (nao resolvido por nenhuma base): {len(residue)}/400")
    from collections import Counter as C
    rc_cls=C(feats[t]["shape_class"] for t in residue)
    print(f"    por shape_class: {dict(rc_cls)}")
    same_res=[t for t in residue if feats[t]["shape_class"]=="same_shape"]
    print(f"    mesmo-shape no residuo (alvo de H86/H87): {len(same_res)}")
    multi_obj=[t for t in residue if feats[t]["n_objs"]>=3]
    print(f"    residuo com >=3 objetos (alvo de H88 binding): {len(multi_obj)}")
    json.dump({"matrix":matrix,"feats":feats}, open(os.path.join(HERE,"h85_matrix.json"),"w"))
    json.dump(sorted(union), open(os.path.join(HERE,"h85_union.json"),"w"))
    print(f"\n  -> h85_matrix.json e h85_union.json salvos. union={len(union)} (referencia p/ H86-H88).")
    return union, residue

def main():
    fresh="--fresh" in sys.argv
    t0=time.time()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    print(f"H85: avaliando {len(files)} tarefas | fresh={fresh} (nca/rc {'ao vivo' if fresh else 'do cache'})")
    matrix,feats=build_matrix(files,fresh)
    report(matrix,feats)
    print(f"  tempo {time.time()-t0:.0f}s")

if __name__=="__main__": main()
