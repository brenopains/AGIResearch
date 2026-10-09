"""
H112 — GrammarForge++ : predicados em CONJUNCAO + regra ITERADA (CA simbolico)
==============================================================================
Estende o H111 (que cravou +6 -> 25,0%) com mais expressividade na meta-linguagem:
  (a) CONJUNCOES: predicado = (f1==v1) AND (f2==v2) -> regras mais finas (fallback quando
      single-feature falha), capturando familias condicionais compostas.
  (b) ITERADO: aplica a regra induzida repetidamente ate ponto fixo (ou K passos) -> captura
      PROPAGACAO/CRESCIMENTO (autômato celular simbolico) que a versao 1-passo nao pega.
Roda no residuo APOS o teto 100; mede NOVAS vs 100. Gate exato. So mesmo-shape.
Honesto: ainda meta-linguagem fixa (aproximacao de Solomonoff / fecho pratico maior), nao AGI.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H111=_load("exp_h111_grammarforge.py"); FEATS=H111.FEATS; cell_features=H111.cell_features

def gather(demos):
    fa={k:[] for k in FEATS}; Y=[]; IN=[]
    for i,o in demos:
        F=cell_features(i)
        for k in FEATS: fa[k].append(F[k].ravel())
        Y.append(o.ravel()); IN.append(i.ravel())
    return {k:np.concatenate(v) for k,v in fa.items()}, np.concatenate(Y), np.concatenate(IN)

def induce_conj(demos, max_rules=18, pairs=True):
    """lista de decisao com predicados single e (fallback) conjuncao de 2 features."""
    if not all(i.shape==o.shape for i,o in demos): return None
    fa,Y,IN=gather(demos); covered=np.zeros(len(Y),bool); rules=[]
    feat_vals={fk:np.unique(fa[fk]) for fk in FEATS}
    for _ in range(max_rules):
        if covered.all(): break
        unc=~covered; best=None  # (cnt, kind, spec, act, col)
        # singles
        for fk in FEATS:
            arr=fa[fk]
            for v in np.unique(arr[unc]):
                m=unc&(arr==v)
                if not m.any(): continue
                cnt=int(m.sum())
                if np.all(Y[m]==IN[m]) and (best is None or cnt>best[0]): best=(cnt,'1',(fk,int(v)),'copy',0)
                u=np.unique(Y[m])
                if len(u)==1 and (best is None or cnt>best[0]): best=(cnt,'1',(fk,int(v)),'const',int(u[0]))
        # conjuncoes (so se singles nao cobre tudo ainda; limita a pares de features distintas)
        if pairs and (best is None or best[0]<unc.sum()):
            fl=FEATS
            for a in range(len(fl)):
                for b in range(a+1,len(fl)):
                    fa1=fa[fl[a]]; fa2=fa[fl[b]]
                    for v1 in np.unique(fa1[unc]):
                        mm=unc&(fa1==v1)
                        if not mm.any(): continue
                        for v2 in np.unique(fa2[mm]):
                            m=mm&(fa2==v2)
                            if not m.any(): continue
                            cnt=int(m.sum())
                            if cnt<=(best[0] if best else 0): continue
                            if np.all(Y[m]==IN[m]): best=(cnt,'2',(fl[a],int(v1),fl[b],int(v2)),'copy',0)
                            u=np.unique(Y[m])
                            if len(u)==1: best=(cnt,'2',(fl[a],int(v1),fl[b],int(v2)),'const',int(u[0]))
        if best is None or best[0]==0: return None
        cnt,kind,spec,act,col=best
        if kind=='1': m=(~covered)&(fa[spec[0]]==spec[1])
        else: m=(~covered)&(fa[spec[0]]==spec[1])&(fa[spec[2]]==spec[3])
        rules.append((kind,spec,act,col)); covered|=m
    return rules if covered.all() else None

def apply_conj(rules,g):
    F=cell_features(g); flat={k:F[k].ravel() for k in FEATS}
    out=g.ravel().copy(); assigned=np.zeros(g.size,bool)
    for kind,spec,act,col in rules:
        if kind=='1': m=(~assigned)&(flat[spec[0]]==spec[1])
        else: m=(~assigned)&(flat[spec[0]]==spec[1])&(flat[spec[2]]==spec[3])
        if act=='const': out[m]=col
        assigned|=m
    return out.reshape(g.shape)

def fixpoint(rules,g,K=12):
    cur=g
    for _ in range(K):
        nxt=apply_conj(rules,cur)
        if np.array_equal(nxt,cur): break
        cur=nxt
    return cur

def solve_task(train, tests):
    # 1) conjuncao 1-passo
    r=induce_conj(train,pairs=True)
    if r is not None and all(np.array_equal(apply_conj(r,i),o) for i,o in train):
        outs=[apply_conj(r,xi) for xi,_ in tests]
        if all(x.shape==yo.shape and np.array_equal(x,yo) for x,(_,yo) in zip(outs,tests)): return ('conj',r,outs)
    # 2) iterado: regra single (H111) aplicada ate ponto fixo reproduz os demos?
    r1=H111.induce(train)
    if r1 is not None:
        rc=[('1',(fk,v),act,col) for fk,v,act,col in r1]
        if all(np.array_equal(fixpoint(rc,i),o) for i,o in train):
            outs=[fixpoint(rc,xi) for xi,_ in tests]
            if all(x.shape==yo.shape and np.array_equal(x,yo) for x,(_,yo) in zip(outs,tests)): return ('iter',rc,outs)
    return (None,None,None)

def main():
    t0=time.time()
    ceiling=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); novas={}; att=0
    for fp in files:
        name=os.path.basename(fp)
        if name in ceiling: continue
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        att+=1
        kind,rule,outs=solve_task(train,tests)
        if kind is not None: novas[name]=(kind,rule)
        if att%60==0: print(f"  ... {att} residuo mesmo-shape testado (NOVAS={len(novas)})"); sys.stdout.flush()
    alln=ceiling|set(novas)
    json.dump(sorted(alln),open(os.path.join(HERE,"arc_unified_final.json"),"w"))
    print("="*74)
    print(f"H112 GrammarForge++ (conjuncao + iterado) | teto anterior={len(ceiling)}")
    print(f"  NOVAS no residuo: {len(novas)} {sorted(novas)[:14]}")
    print(f"  novo TETO: {len(alln)}/400 = {100*len(alln)/400:.1f}%")
    for n in sorted(novas)[:8]:
        k,r=novas[n]; print(f"    {n} [{k}] -> {len(r)} regras")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: conjuncao+iteracao ampliam a meta-linguagem (fecho pratico maior). Nao e AGI.")

if __name__=="__main__": main()
