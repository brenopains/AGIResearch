"""
H93 — ZEALOT-1: a "colisão" que FORÇA primitiva de fora do fecho (hipótese do usuário)
=======================================================================================
Hipótese "Subcortical Zealot": a primitiva nova nao e GERADA por busca/evolucao (todas deram
+0: H89-H92) — ela e a CICATRIZ que sobra quando o sistema e FORCADO a explicar um sinal que
NAO esta no seu espaco de primitivas. Operacionalizacao honesta e mais forte que a CNN aleatoria
do PDF (que tem risco alto de +0 por gerar ruido): o "Zealot" propoe FAMILIAS DE PRIMITIVA
PARAMETRICA que estao FORA do DSL atual (as que o residuo de fato precisa); o "Oracle" so adota
se a primitiva (param abduzido dos demos) reproduz TODOS os demos exatamente (colapso por MDL/
coerencia). Isto e, ao mesmo tempo, o unico lever que historicamente moveu o numero: LARGURA nova.

Familias novas propostas (fora do fecho FIXED/NEW/ABDUCE):
  Z1 periodic_complete : detecta o menor periodo consistente e preenche zeros (reparo de padrao)
  Z2 symmetrize        : completa simetria (h/v/ambos/diag/rot4) sobrepondo o espelho onde ha 0
  Z3 enclosed_fill     : preenche regioes de fundo CERCADAS (nao tocam a borda) com cor abduzida
  Z4 tile              : ladrilha a entrada k x m (e variante espelhada/caleidoscopio) p/ a saida
Cada uma testada sozinha, e composta com cmap e com 1 op FIXED (pre/pos). Gate exato; CPU.
Metrica: NOVAS vs teto atual (86). Honesto: se NOVAS>0, foi LARGURA (a cicatriz) — confirma que
adicionar a familia certa e o que move; se 0, nem essas familias estavam no residuo.
"""
import os, sys, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H63=_load("exp_h63_efficiency_breadth.py"); PU=H63.PU

def exact(fn, demos):
    try:
        for i,o in demos:
            r=fn(i)
            if r is None or not isinstance(r,np.ndarray) or r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False

# ---------------- Z1 periodic_complete ----------------
def periodic_complete(g):
    # detecta o menor (py,px) tal que as celulas nao-zero sao consistentes por classe de residuo,
    # e preenche os zeros pela classe. Repara padrao periodico com buracos.
    H,W=g.shape
    for py in range(1,H+1):
        for px in range(1,W+1):
            if py==H and px==W: continue
            classes={}; good=True
            for r in range(H):
                for c in range(W):
                    if g[r,c]!=0:
                        k=(r%py,c%px)
                        if k in classes and classes[k]!=g[r,c]: good=False; break
                        classes[k]=g[r,c]
                if not good: break
            if not good: continue
            # preenche
            cand=g.copy(); filled=True
            for r in range(H):
                for c in range(W):
                    if cand[r,c]==0:
                        k=(r%py,c%px)
                        if k in classes: cand[r,c]=classes[k]
                        else: filled=False
            if filled and (cand!=g).any():
                return cand
    return None

# ---------------- Z2 symmetrize ----------------
def _mirror(g,kind):
    if kind=='h': return g[:,::-1]
    if kind=='v': return g[::-1,:]
    if kind=='both': return g[::-1,::-1]
    if kind=='t' and g.shape[0]==g.shape[1]: return g.T
    if kind=='at' and g.shape[0]==g.shape[1]: return g[::-1,::-1].T
    return None
def symmetrize(kind):
    def fn(g):
        m=_mirror(g,kind)
        if m is None: return None
        out=g.copy(); out[out==0]=m[out==0]; return out
    return fn
def symmetrize_rot4(g):
    if g.shape[0]!=g.shape[1]: return None
    out=g.copy()
    for k in (1,2,3):
        m=np.rot90(g,k); out[out==0]=m[out==0]
    return out

# ---------------- Z3 enclosed_fill ----------------
def enclosed_fill(color):
    def fn(g):
        H,W=g.shape; bg=(g==0);
        # flood fill a partir da borda em celulas 0
        from collections import deque
        outside=np.zeros((H,W),bool); q=deque()
        for r in range(H):
            for c in (0,W-1):
                if bg[r,c] and not outside[r,c]: outside[r,c]=True; q.append((r,c))
        for c in range(W):
            for r in (0,H-1):
                if bg[r,c] and not outside[r,c]: outside[r,c]=True; q.append((r,c))
        while q:
            r,c=q.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and bg[nr,nc] and not outside[nr,nc]: outside[nr,nc]=True; q.append((nr,nc))
        enclosed=bg&(~outside)
        if not enclosed.any(): return None
        out=g.copy(); out[enclosed]=color; return out
    return fn

# ---------------- Z4 tile ----------------
def tile_fn(ky,kx,flip):
    def fn(g):
        if flip:
            row=[];
            for j in range(kx): row.append(g if j%2==0 else g[:,::-1])
            band=np.concatenate(row,axis=1); blocks=[]
            for i in range(ky): blocks.append(band if i%2==0 else band[::-1,:])
            return np.concatenate(blocks,axis=0)
        return np.tile(g,(ky,kx))
    return fn

def abduce_tile(train):
    i0,o0=train[0]
    if o0.shape[0]%i0.shape[0] or o0.shape[1]%i0.shape[1]: return None
    ky=o0.shape[0]//i0.shape[0]; kx=o0.shape[1]//i0.shape[1]
    if ky*kx<=1: return None
    for flip in (False,True):
        fn=tile_fn(ky,kx,flip)
        if exact(fn,train): return fn
    return None

# ---------------- Oracle: abduz familia + verifica exato ----------------
def zealot_candidates(train):
    cands=[]
    cands.append(("Z1_periodic", periodic_complete))
    for kind in ('h','v','both','t','at'): cands.append((f"Z2_sym_{kind}", symmetrize(kind)))
    cands.append(("Z2_rot4", symmetrize_rot4))
    # enclosed_fill: abduz a cor (testa cores presentes na saida)
    cols=set()
    for _,o in train: cols|=set(np.unique(o).tolist())
    for c in sorted(cols):
        if c!=0: cands.append((f"Z3_fill_{c}", enclosed_fill(c)))
    ft=abduce_tile(train)
    if ft is not None: cands.append(("Z4_tile", ft))
    return cands

def solve_zealot(train, tests):
    cm=PU.infer_cmap(train)
    pre_ops=[('id',lambda g:g)]+[(n,PU.FIXED[n]) for n in ('keep_largest','denoise','crop') if n in PU.FIXED]
    for nm,fam in zealot_candidates(train):
        # familia sozinha
        if exact(fam,train):
            return nm, [fam(xi) for xi,_ in tests]
        # familia ∘ cmap
        if cm is not None:
            f2=lambda g,fam=fam: (lambda r: None if r is None else PU.cmap(r,cm))(fam(g))
            if exact(f2,train): return nm+"+cmap",[f2(xi) for xi,_ in tests]
        # pre-op ∘ familia
        for pn,pf in pre_ops:
            f3=lambda g,fam=fam,pf=pf: (lambda r: None if r is None else fam(r))(pf(g))
            if exact(f3,train): return f"{pn}+{nm}",[f3(xi) for xi,_ in tests]
    return None, None

def main():
    t0=time.time()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    h89=set(json.load(open(os.path.join(HERE,"h89_evo_solved.json")))) if os.path.exists(os.path.join(HERE,"h89_evo_solved.json")) else set()
    base_union=union|h89
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    solved={}; fam_count={}
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        fam,outs=solve_zealot(train,tests)
        if outs is not None and all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
            solved[name]=fam; fam_count[fam.split('+')[0].split('_')[0] if '_' in fam else fam]=fam_count.get(fam.split('_')[0],0)+1
    allnew=set(solved); novas=allnew-base_union
    json.dump(sorted(allnew),open(os.path.join(HERE,"h93_zealot_solved.json"),"w"))
    from collections import Counter
    famc=Counter(v.split('+')[0] for v in solved.values())
    print("="*72)
    print(f"H93 ZEALOT-1: familias de primitiva FORA do fecho (a cicatriz da colisao)")
    print(f"  teto atual={len(base_union)} | resolveu (todas): {len(allnew)}")
    print(f"  NOVAS vs teto: {len(novas)} {sorted(novas)[:16]}")
    print(f"  novo TETO: {len(base_union|allnew)}/400 = {100*len(base_union|allnew)/400:.1f}%  (era {100*len(base_union)/400:.1f}%)")
    print(f"  por familia: {dict(famc)}")
    print(f"  exemplos NOVAS->familia: {[(n,solved[n]) for n in sorted(novas)[:10]]}")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: NOVAS>0 = LARGURA nova (a cicatriz) moveu o numero, exatamente a tese do")
    print("  Zealot e o unico lever que sempre funcionou (3,5->21%). 0 = nem essas familias faltavam.")

if __name__=="__main__": main()
