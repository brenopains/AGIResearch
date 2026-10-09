"""
H88 — Frontend de LIGACAO POR SINCRONIA (oscilacoes/binding-by-synchrony) p/ ABDUCE
====================================================================================
Fries/Singer (binding-by-synchrony): caracteristicas que pertencem ao mesmo objeto sao
ligadas por neuronios que SINCRONIZAM (gamma). Implementamos como Kuramoto esparso: cada
celula nao-zero recebe uma fase; arestas acoplam por adjacencia + MESMA COR + colinear/
simetria. Apos assentar, clusters de fase = objetos. Esses objetos alimentam os schemas do
ABDUCE (Tier A/B/C) NO LUGAR de componentes conexos.

Diferenca crucial vs o connected-components que o ABDUCE ja faz (4/8-conn): os termos de
acoplamento de LONGO ALCANCE (mesma cor a distancia, parceiros simetricos) agrupam celulas
DESCONEXAS — exatamente o que CC nao consegue. Se colapsar para adjacencia, vira CC lento (+0).
Gate exato; metrica = tarefas que a coluna 'abduce' do H85 NAO resolve.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
AB=_load("exp_h55_abduce.py")

def _settle(g, n_iter=60, a_adj=1.0, a_col=0.35, dt=0.25):
    """Kuramoto esparso sobre celulas nao-zero. Retorna fases por celula (dict)."""
    H,W=g.shape; cells=[(r,c) for r in range(H) for c in range(W) if g[r,c]!=0]
    if not cells: return {},[]
    idx={rc:i for i,rc in enumerate(cells)}; n=len(cells)
    rng=np.random.default_rng(0); th=rng.uniform(0,2*np.pi,n)
    # arestas: adjacencia 8-conn (peso a_adj) + mesma-cor longo alcance (peso a_col)
    edges=[]
    for (r,c) in cells:
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nb=(r+dr,c+dc)
                if nb in idx and idx[nb]>idx[(r,c)]: edges.append((idx[(r,c)],idx[nb],a_adj))
    # mesma cor (longo alcance) — amostra para nao explodir
    from collections import defaultdict
    bycol=defaultdict(list)
    for (r,c) in cells: bycol[g[r,c]].append(idx[(r,c)])
    for col,ids in bycol.items():
        if len(ids)>1:
            for ii in range(len(ids)):
                for jj in range(ii+1,min(ii+6,len(ids))):  # vizinhos no indice: limita grau
                    edges.append((ids[ii],ids[jj],a_col))
    if not edges: return {rc:th[i] for rc,i in idx.items()}, cells
    E=np.array(edges); ai=E[:,0].astype(int); aj=E[:,1].astype(int); aw=E[:,2]
    for _ in range(n_iter):
        diff=np.sin(th[aj]-th[ai])*aw
        upd=np.zeros(n); np.add.at(upd,ai,diff); np.add.at(upd,aj,-diff)
        th=th+dt*upd
    return {rc:th[i] for i,rc in enumerate(cells)}, cells

def phase_objects(g, conn=4, same_color=True, nbins=16):
    ph,cells=_settle(g)
    if not cells: return []
    from collections import defaultdict
    groups=defaultdict(list)
    for rc in cells:
        b=int(((ph[rc])%(2*np.pi))/(2*np.pi)*nbins)%nbins
        groups[b].append(rc)
    objs=[]
    for b,gc in groups.items():
        if gc: objs.append(AB.Obj(g,gc))
    return objs

def main():
    t0=time.time()
    mat=json.load(open(os.path.join(HERE,"h85_matrix.json"))) if os.path.exists(os.path.join(HERE,"h85_matrix.json")) else {"matrix":{}}
    matrix=mat["matrix"]
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    ab_col=set(t for t,r in matrix.items() if r.get("abduce")) if matrix else set()
    # monkeypatch: ABDUCE passa a segmentar por fase
    orig_parse=AB.parse; AB.parse=phase_objects
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); pb=set()
    try:
        for fp in files:
            name=os.path.basename(fp); t=json.load(open(fp))
            train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
            tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
            try: render,_,_=AB.abduce(train)
            except Exception: render=None
            if render is None: continue
            try:
                if all((lambda r,yo:r is not None and isinstance(r,np.ndarray) and r.shape==yo.shape and np.array_equal(r,yo))(render(xi),yo) for xi,yo in tests):
                    pb.add(name)
            except Exception: pass
    finally:
        AB.parse=orig_parse
    json.dump(sorted(pb),open(os.path.join(HERE,"h88_pb_solved.json"),"w"))
    print("="*72)
    print(f"H88 Frontend de ligacao por sincronia (Kuramoto) -> ABDUCE")
    print(f"  phase-bind resolveu: {len(pb)} | vs coluna abduce(H85)={len(ab_col)}")
    print(f"  NOVAS vs abduce: {len(pb-ab_col)} {sorted(pb-ab_col)[:10]}")
    print(f"  NOVAS vs union(H85): {len(pb-union)} {sorted(pb-union)[:10]}")
    print(f"  nova UNIAO: {len(union|pb)}/400 = {100*len(union|pb)/400:.1f}%  (era {100*len(union)/400:.1f}%)")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: se NOVAS vs abduce>0, o binding de longo alcance (mesma-cor/simetria")
    print("  desconexa) agrupou o que CC nao agrupa. Se 0, colapsou para CC (como H64/H82).")

if __name__=="__main__": main()
