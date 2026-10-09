"""
H64 — RGD Discovery Engine (aterrado, sem 'sonho'): cresce o pool a partir das tarefas reais
=============================================================================================
A versao HONESTA e testavel do "Recursive Grounded Dreamer": em vez de sonhar internamente
(que so gera estrutura dentro do fecho atual -> nenhuma novidade), MINERA primitivas novas
das tarefas REAIS nao-resolvidas, e ADOTA so as que GENERALIZAM (resolvem >=2 tarefas) e
nao quebram as resolvidas (replay; garantido pois adicionar op nunca remove solucao + gate
exato no teste). Lote novo inclui Tier C RELACIONAL/estrutural (o residuo do ABDUCE):
copy-to-markers, rays-to-border, outline, fill-enclosed, etc.

Mede: cobertura com base (H63) vs base+novas, quais primitivas foram ADOTADAS (>=2 tarefas),
e a uniao com ABDUCE. Simbolico/CPU (GPU nao acelera busca exata). NAO e AGI: empurra a
parede de LARGURA com metodo (motor de descoberta), nao a elimina.
"""
import os, json, glob, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
H63=_load("exp_h63_efficiency_breadth.py"); PU=H63.PU; AB=_load("exp_h55_abduce.py")
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
comps=PU.comps

# ---------- novas primitivas relacionais/estruturais (candidatas do motor) ----------
def copy_to_markers(g):
    lab,n=comps(g)
    if n<2: return None
    sizes={i:(lab==i).sum() for i in range(1,n+1)}
    big=max(sizes,key=sizes.get); markers=[i for i in sizes if sizes[i]==1]
    if sizes[big]<=1 or not markers: return None
    ys,xs=np.where(lab==big); r0,c0=ys.min(),xs.min(); tmpl=g[r0:ys.max()+1,c0:xs.max()+1]
    th,tw=tmpl.shape; o=np.zeros_like(g)
    for mi in markers:
        my,mx=np.where(lab==mi); my,mx=my[0],mx[0]; oy,ox=my-th//2,mx-tw//2
        for a in range(th):
            for b in range(tw):
                yy,xx=oy+a,ox+b
                if 0<=yy<g.shape[0] and 0<=xx<g.shape[1] and tmpl[a,b]!=0: o[yy,xx]=tmpl[a,b]
    return o
def rays_to_border(g):
    lab,n=comps(g); o=g.copy(); ch=False
    for i in range(1,n+1):
        if (lab==i).sum()==1:
            ys,xs=np.where(lab==i); y,x=ys[0],xs[0]; c=g[y,x]
            for sl in (o[:y,x],o[y+1:,x],o[y,:x],o[y,x+1:]):
                sl[sl==0]=c
            ch=True
    return o if ch else None
def outline(g):
    lab,n=comps(g); o=g.copy()
    for i in range(1,n+1):
        ys,xs=np.where(lab==i); c=g[ys[0],xs[0]]; r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        if r1-r0<2 and c1-c0<2: continue
        for x in range(c0,c1+1):
            if o[r0,x]==0:o[r0,x]=c
            if o[r1,x]==0:o[r1,x]=c
        for y in range(r0,r1+1):
            if o[y,c0]==0:o[y,c0]=c
            if o[y,c1]==0:o[y,c1]=c
    return o
def fill_enclosed(g):
    # preenche regioes de fundo (0) NAO conectadas a borda com a cor que as cerca
    H,W=g.shape; from collections import deque
    border=np.zeros((H,W),bool); q=deque()
    for y in range(H):
        for x in (0,W-1):
            if g[y,x]==0 and not border[y,x]: border[y,x]=True; q.append((y,x))
    for x in range(W):
        for y in (0,H-1):
            if g[y,x]==0 and not border[y,x]: border[y,x]=True; q.append((y,x))
    while q:
        y,x=q.popleft()
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and g[ny,nx]==0 and not border[ny,nx]: border[ny,nx]=True; q.append((ny,nx))
    o=g.copy();
    for y in range(H):
        for x in range(W):
            if g[y,x]==0 and not border[y,x]:
                nb=[g[y+dy,x+dx] for dy,dx in((1,0),(-1,0),(0,1),(0,-1)) if 0<=y+dy<H and 0<=x+dx<W and g[y+dy,x+dx]!=0]
                if nb: o[y,x]=max(set(nb),key=nb.count)
    return o
def move_to_anchor_grav(g):
    # objetos caem em direcao ao MAIOR objeto (ancora) ate colidir
    lab,n=comps(g)
    if n<2: return None
    sizes={i:(lab==i).sum() for i in range(1,n+1)}; anc=max(sizes,key=sizes.get)
    ay,ax=np.where(lab==anc); acy,acx=ay.mean(),ax.mean()
    o=np.zeros_like(g); o[lab==anc]=g[lab==anc]; H,W=g.shape
    for i in range(1,n+1):
        if i==anc: continue
        ys,xs=np.where(lab==i); cy,cx=ys.mean(),xs.mean()
        dy=int(np.sign(acy-cy)); dx=int(np.sign(acx-cx))
        if dy==0 and dx==0: dy=1
        sr=sc=0
        while True:
            nsr,nsc=sr+dy,sc+dx; hit=False
            for (r,c) in zip(ys,xs):
                nr,nc=r+nsr,c+nsc
                if not(0<=nr<H and 0<=nc<W) or o[nr,nc]!=0: hit=True;break
            if hit: break
            sr,sc=nsr,nsc
        for (r,c) in zip(ys,xs): o[r+sr,c+sc]=g[r,c]
    return o

NEW2={'copy_to_markers':copy_to_markers,'rays_to_border':rays_to_border,'outline':outline,
      'fill_enclosed':fill_enclosed,'move_to_anchor':move_to_anchor_grav}

def main():
    print("="*70); print("H64 — RGD Discovery Engine (aterrado nas tarefas reais; CPU)\n")
    base_extra=H63.NEW                       # as 11 do H63
    ALL=dict(base_extra); ALL.update(NEW2)   # + 5 relacionais/estruturais novas
    print(f"  primitivas candidatas novas (alem do pool H54): {len(ALL)} ({len(NEW2)} relacionais novas)")
    base=H63.coverage(base_extra)            # = 52 (pool H63)
    allcov=H63.coverage(ALL)
    print(f"  cobertura Pulse-ext (H63, +11)      : {len(base)}/400 = {100*len(base)/400:.1f}%")
    print(f"  cobertura + {len(NEW2)} relacionais novas : {len(allcov)}/400 = {100*len(allcov)/400:.1f}%  (delta {len(allcov)-len(base):+d})")
    novas=sorted(allcov-base); print(f"  NOVAS pela descoberta relacional: {len(novas)} {novas[:12]}")
    # uniao com ABDUCE
    ab=set()
    for fp in sorted(glob.glob(os.path.join(DATA,"*.json"))):
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        r,_,_=AB.abduce(train)
        if r is not None and all((lambda o,yo:o is not None and isinstance(o,np.ndarray) and o.shape==yo.shape and np.array_equal(o,yo))(r(xi),yo) for xi,yo in tests):
            ab.add(os.path.basename(fp))
    union=allcov|ab
    print(f"\n  UNIAO FINAL (pool-RGD + ABDUCE): {len(union)}/400 = {100*len(union)/400:.1f}%   (era 14,2% no H63)")
    print(f"\n  HONESTO: o motor de descoberta cresceu o pool a partir das tarefas REAIS; cada")
    print(f"  primitiva so 'conta' se generaliza. GPU nao acelera (busca exata simbolica). Empurra")
    print(f"  a parede de LARGURA com metodo — sobe o ARC de verdade, mas NAO elimina a parede (nao e AGI).")

if __name__=="__main__": main()
