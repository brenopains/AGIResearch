"""
H111 — GrammarForge: induz REGRA SIMBOLICA POR-CELULA por MDL (motor mais expressivo)
======================================================================================
Em vez de recombinar familias fixas (PGA), INDUZ uma lista de decisao por-celula
[ (predicado -> acao), ... ] a partir dos demos, por compressao gulosa (MDL), gate exato.
Predicados ricos (vizinhanca/paridade/adjacencia-a-cor/borda/paridade-coord) capturam familias
que objeto-nivel (PGA/H100/H102) e transducers neurais (NCA/MB) nao pegam de forma exata:
ex. "se #vizinhos-mesma-cor par -> azul; senao -> vazio". Interpretavel, generaliza de 2-3 demos.

HONESTO (declarado uma vez): a meta-linguagem (predicados-base) e fixa = substrato; isto e uma
APROXIMACAO de Solomonoff (fecho pratico muito maior), NAO AGI nem escape do fecho. So mesmo-shape.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H102=_load("exp_h102_grammar.py"); PU=H102.PU
DIRS8=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
DIRS4=[(-1,0),(1,0),(0,-1),(0,1)]

def _sh(g,dy,dx):
    o=np.zeros_like(g); H,W=g.shape
    ys0=max(0,dy);ys1=min(H,H+dy);xs0=max(0,dx);xs1=min(W,W+dx);yd0=max(0,-dy);xd0=max(0,-dx)
    o[ys0:ys1,xs0:xs1]=g[yd0:yd0+(ys1-ys0),xd0:xd0+(xs1-xs0)]; return o

def cell_features(g):
    H,W=g.shape; nb8=[_sh(g,dy,dx) for dy,dx in DIRS8]; nb4=[_sh(g,dy,dx) for dy,dx in DIRS4]
    same8=sum((n==g).astype(np.int64) for n in nb8)
    same4=sum((n==g).astype(np.int64) for n in nb4)
    nz8=sum((n!=0).astype(np.int64) for n in nb8)
    F={'own':g.copy(),
       'same8':same8,'par_same8':same8%2,'same4':same4,'par_same4':same4%2,
       'nz8':nz8,'par_nz8':nz8%2,
       'border':np.zeros((H,W),np.int64),'rpar':(np.arange(H)[:,None]%2)*np.ones((H,W),np.int64),
       'cpar':(np.arange(W)[None,:]%2)*np.ones((H,W),np.int64)}
    F['border'][0,:]=1;F['border'][-1,:]=1;F['border'][:,0]=1;F['border'][:,-1]=1
    for c in range(1,10):
        F['adj%d'%c]=(sum((n==c).astype(np.int64) for n in nb8)>0).astype(np.int64)
    return F
FEATS=['own','par_same8','par_same4','same8','same4','par_nz8','nz8','border','rpar','cpar']+['adj%d'%c for c in range(1,10)]

def induce(demos, max_rules=16):
    if not all(i.shape==o.shape for i,o in demos): return None
    fa={k:[] for k in FEATS}; Y=[]; IN=[]
    for i,o in demos:
        F=cell_features(i)
        for k in FEATS: fa[k].append(F[k].ravel())
        Y.append(o.ravel()); IN.append(i.ravel())
    fa={k:np.concatenate(v) for k,v in fa.items()}; Y=np.concatenate(Y); IN=np.concatenate(IN)
    covered=np.zeros(len(Y),bool); rules=[]
    for _ in range(max_rules):
        if covered.all(): break
        unc=~covered; best=None
        for fk in FEATS:
            arr=fa[fk]
            for v in np.unique(arr[unc]):
                m=unc&(arr==v)
                if not m.any(): continue
                ys=Y[m]; cnt=int(m.sum())
                if np.all(Y[m]==IN[m]):     # acao copiar (mantem entrada)
                    if best is None or cnt>best[0]: best=(cnt,fk,int(v),'copy',0)
                u=np.unique(ys)
                if len(u)==1:               # acao cor constante (consistente, erro 0)
                    if best is None or cnt>best[0]: best=(cnt,fk,int(v),'const',int(u[0]))
        if best is None or best[0]==0: return None
        cnt,fk,v,act,col=best
        m=(~covered)&(fa[fk]==v); rules.append((fk,v,act,col)); covered|=m
    return rules if covered.all() else None

def apply_rule(rules,g):
    F=cell_features(g); flat={k:F[k].ravel() for k in FEATS}
    out=g.ravel().copy(); assigned=np.zeros(g.size,bool)
    for fk,v,act,col in rules:
        m=(~assigned)&(flat[fk]==v)
        if act=='const': out[m]=col
        assigned|=m
    return out.reshape(g.shape)

def readable(rules):
    return " ; ".join([f"{fk}=={v}->{'cor%d'%col if act=='const' else 'copia'}" for fk,v,act,col in rules])

def main():
    t0=time.time()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json"))))|set(json.load(open(os.path.join(HERE,"h89_evo_solved.json"))))
    union|=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json"))))|set(json.load(open(os.path.join(HERE,"h100_solved.json"))))
    try: union|=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    except: pass
    ceiling=union
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); solved={}; att=0
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        att+=1
        rules=induce(train)
        if rules is None:
            cm=PU.infer_cmap(train)   # tenta cor-mapeada antes da regra
            continue
        # gate: reproduz demos exato (por construcao) + teste exato
        if all(np.array_equal(apply_rule(rules,i),o) for i,o in train):
            outs=[apply_rule(rules,xi) for xi,_ in tests]
            if all(r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                solved[name]=rules
    novas={n:r for n,r in solved.items() if n not in ceiling}
    json.dump(sorted(set(solved)|ceiling),open(os.path.join(HERE,"h111_solved.json"),"w"))
    print("="*74)
    print(f"H111 GrammarForge (regra por-celula induzida por MDL) | mesmo-shape={att}/400 | teto={len(ceiling)}")
    print(f"  resolveu (total): {len(solved)} | NOVAS vs teto: {len(novas)} {sorted(novas)[:14]}")
    print(f"  novo TETO: {len(ceiling|set(solved))}/400 = {100*len(ceiling|set(solved))/400:.1f}%")
    for n in sorted(novas)[:10]:
        print(f"    NOVA {n}: regra = [{readable(novas[n])}]")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: regra simbolica por-celula induzida (aproximacao de Solomonoff/fecho pratico maior). NAO e AGI.")

if __name__=="__main__": main()
