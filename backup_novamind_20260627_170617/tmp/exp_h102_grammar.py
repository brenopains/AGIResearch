"""
H102 — Gramatica de primitivas ENRIQUECIDA (relacionais/condicionais) — empurra o ARC
=======================================================================================
Estende o H100 com familias RELACIONAIS/CONDICIONAIS (o tipo "recolorir o objeto que cerca
outro", "objetos da borda vs interior", "mover ate o maior") — o que o PDF do Zealot citou como
primitivas relacionais complexas. Reusa todo o motor do H100 + H93 + ABDUCE. Gate exato; MDL por
ordem. Tambem serve de SOLVER ARC UNIFICADO (uniao de tudo que funcionou) p/ o numero final.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H100=_load("exp_h100_primitive_creation.py"); PU=H100.PU; AB=H100.AB; Z=H100.Z
objs=H100.objs; PS=H100.PS; exact=H100.exact

# ---------- familias relacionais/condicionais novas ----------
def fam_recolor_border(ps):
    def build(train):
        mp={}
        for gi,go in train:
            if gi.shape!=go.shape: return None
            H,W=gi.shape; os_=objs(gi,ps)
            if not os_: return None
            for o in os_:
                r0,c0,r1,c1=o.bbox; touch=(r0==0 or c0==0 or r1==H-1 or c1==W-1)
                cols={go[r,c] for r,c in o.cells}
                if len(cols)!=1: return None
                k=int(touch); c=next(iter(cols))
                if k in mp and mp[k]!=c: return None
                mp[k]=c
        def fn(g):
            H,W=g.shape; out=g.copy()
            for o in objs(g,ps):
                r0,c0,r1,c1=o.bbox; k=int(r0==0 or c0==0 or r1==H-1 or c1==W-1)
                if k not in mp: return None
                for r,c in o.cells: out[r,c]=mp[k]
            return out
        return fn
    return build

def fam_recolor_quadrant(ps):
    def build(train):
        mp={}
        for gi,go in train:
            if gi.shape!=go.shape: return None
            H,W=gi.shape; os_=objs(gi,ps)
            if not os_: return None
            for o in os_:
                cy,cx=o.centroid; q=(int(cy>=H/2),int(cx>=W/2))
                cols={go[r,c] for r,c in o.cells}
                if len(cols)!=1: return None
                c=next(iter(cols))
                if q in mp and mp[q]!=c: return None
                mp[q]=c
        def fn(g):
            H,W=g.shape; out=g.copy()
            for o in objs(g,ps):
                cy,cx=o.centroid; q=(int(cy>=H/2),int(cx>=W/2))
                if q not in mp: return None
                for r,c in o.cells: out[r,c]=mp[q]
            return out
        return fn
    return build

def _encloses(a,b):
    r0,c0,r1,c1=a.bbox; R0,C0,R1,C1=b.bbox
    return r0<=R0 and c0<=C0 and r1>=R1 and c1>=C1 and a.size>b.size
def fam_select_encloser(tname,ps):
    def build(train):
        col=None
        for gi,go in train:
            if gi.shape!=go.shape: return None
            os_=objs(gi,ps); enc=[a for a in os_ if any(_encloses(a,b) for b in os_ if b is not a)]
            if len(enc)!=1: return None
            sel=enc[0]
            if tname=='recolor':
                cs={go[r,c] for r,c in sel.cells}
                if len(cs)!=1: return None
                if col is None: col=next(iter(cs))
                elif col!=next(iter(cs)): return None
        def fn(g):
            os_=objs(g,ps); enc=[a for a in os_ if any(_encloses(a,b) for b in os_ if b is not a)]
            if len(enc)!=1: return None
            sel=enc[0]; out=g.copy()
            if tname=='keep':
                out=np.zeros_like(g)
                for r,c in sel.cells: out[r,c]=g[r,c]
            elif tname=='recolor':
                for r,c in sel.cells: out[r,c]=col
            elif tname=='delete':
                for r,c in sel.cells: out[r,c]=0
            return out
        return fn
    return build

def fam_move_to_largest(ps):
    def build(train):
        for gi,go in train:
            if gi.shape!=go.shape: return None
        def fn(g):
            os_=objs(g,ps)
            if len(os_)<2: return None
            big=max(os_,key=lambda x:x.size); out=np.zeros_like(g)
            for r,c in big.cells: out[r,c]=g[r,c]
            tgt=np.array(big.centroid)
            for o in os_:
                if o is big: continue
                cur=np.array(o.centroid); best=None
                for dy in range(-g.shape[0],g.shape[0]):
                    for dx in range(-g.shape[1],g.shape[1]):
                        ok=True
                        for r,c in o.cells:
                            nr,nc=r+dy,c+dx
                            if not(0<=nr<g.shape[0] and 0<=nc<g.shape[1]) or out[nr,nc]!=0: ok=False;break
                        if ok:
                            d=abs((o.centroid[0]+dy)-tgt[0])+abs((o.centroid[1]+dx)-tgt[1])
                            if best is None or d<best[0]: best=(d,dy,dx)
                if best is None: return None
                _,dy,dx=best
                for r,c in o.cells: out[r+dy,c+dx]=g[r,c]
            return out
        return fn
    return build

def new_families():
    fams=[]
    for ps in PS:
        fams.append((f"recolor_border_{ps[0]}{int(ps[1])}", fam_recolor_border(ps)))
        fams.append((f"recolor_quad_{ps[0]}{int(ps[1])}", fam_recolor_quadrant(ps)))
        for tn in ('keep','recolor','delete'):
            fams.append((f"encloser_{tn}_{ps[0]}{int(ps[1])}", fam_select_encloser(tn,ps)))
    return fams

def solve(train, tests):
    cm=PU.infer_cmap(train); cands=[]
    for nm,build in (H100.families()+new_families()):
        try: fn=build(train)
        except Exception: fn=None
        if fn is not None: cands.append((nm,fn))
    for nm,fn in Z.zealot_candidates(train): cands.append((nm,fn))
    for nm,fn in cands:
        if exact(fn,train): return nm,[fn(xi) for xi,_ in tests]
        if cm is not None:
            f2=lambda g,fn=fn:(lambda r:None if r is None else PU.cmap(r,cm))(fn(g))
            if exact(f2,train): return nm+"+cmap",[f2(xi) for xi,_ in tests]
    return None,None

def main():
    t0=time.time()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json"))))
    h89=set(json.load(open(os.path.join(HERE,"h89_evo_solved.json"))))
    h93=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json"))))
    h100=set(json.load(open(os.path.join(HERE,"h100_solved.json")))) if os.path.exists(os.path.join(HERE,"h100_solved.json")) else set()
    ceiling=union|h89|h93|h100
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); solved={}
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        fam,outs=solve(train,tests)
        if outs is not None and all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
            solved[name]=fam
    allnew=set(solved); novas=allnew-ceiling
    json.dump(sorted(allnew),open(os.path.join(HERE,"h102_solved.json"),"w"))
    from collections import Counter
    print("="*74)
    print(f"H102 Gramatica enriquecida (familias relacionais/condicionais)")
    print(f"  teto={len(ceiling)} | resolveu total {len(allnew)} | NOVAS {len(novas)} {sorted(novas)[:14]}")
    print(f"  novo TETO: {len(ceiling|allnew)}/400 = {100*len(ceiling|allnew)/400:.1f}%")
    print(f"  exemplos NOVAS->familia: {[(n,solved[n]) for n in sorted(novas)[:8]]}")
    print(f"  tempo {time.time()-t0:.0f}s")

if __name__=="__main__": main()
