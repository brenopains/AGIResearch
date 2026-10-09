"""
H50 — On-demand specialist generator + analysis of the unsolved ARC tasks
=========================================================================

(1) GENERATIVE specialist-on-demand: instead of only fixed operators, the inducer
    INFERS parametric operators from the task's own structure (size ratio -> scale/
    downscale/tile factor) and generates the matching specialist on the fly. Plus
    a few more fixed specialists. Re-measure ARC.
(2) ANALYZE the unsolved tasks: categorize by structural features to map exactly
    what kinds of rules are out of reach (the data the user asked for).
"""
import os, json, glob, numpy as np
from collections import deque
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

def crop(g):
    nz=np.argwhere(g!=0)
    if len(nz)==0: return g.copy()
    (r0,c0),(r1,c1)=nz.min(0),nz.max(0); return g[r0:r1+1,c0:c1+1].copy()
def maj(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmax()])
def grav(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o
def comps(g):
    lab=np.zeros_like(g,dtype=int); cur=0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r,c]!=0 and lab[r,c]==0:
                cur+=1; q=deque([(r,c)]); lab[r,c]=cur
                while q:
                    y,x=q.popleft()
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<g.shape[0] and 0<=nx<g.shape[1] and lab[ny,nx]==0 and g[ny,nx]==g[r,c]:
                            lab[ny,nx]=cur; q.append((ny,nx))
    return lab,cur
def keep_largest(g):
    lab,n=comps(g)
    if n==0: return g.copy()
    big=1+int(np.argmax([(lab==i).sum() for i in range(1,n+1)])); o=np.zeros_like(g); o[lab==big]=g[lab==big]; return o
def isolate(g):
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True); k=v[c.argmax()]; o=np.zeros_like(g); o[g==k]=k; return o
def border(g):
    o=g.copy()
    if min(o.shape)>0: o[0,:]=o[-1,:]=1; o[:,0]=o[:,-1]=1
    return o
def sym_h(g):
    o=g.copy(); m=g[:,::-1]; o[o==0]=m[o==0]; return o
def sym_v(g):
    o=g.copy(); m=g[::-1,:]; o[o==0]=m[o==0]; return o
def scale(g,a,b): return np.kron(g,np.ones((a,b),dtype=g.dtype))
def downs(g,a,b):
    if g.shape[0]%a or g.shape[1]%b: return None
    H,W=g.shape[0]//a,g.shape[1]//b; o=np.zeros((H,W),g.dtype)
    for r in range(H):
        for c in range(W):
            blk=g[r*a:(r+1)*a,c*b:(c+1)*b]; v,cc=np.unique(blk,return_counts=True); o[r,c]=v[cc.argmax()]
    return o
def tile(g,a,b): return np.tile(g,(a,b))

FIXED={'id':lambda g:g.copy(),'fliplr':lambda g:g[:,::-1].copy(),'flipud':lambda g:g[::-1,:].copy(),'rot180':lambda g:np.rot90(g,2).copy(),
 'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1),
 'grav':grav,'maj':maj,'border':border,'crop':crop,'keep_largest':keep_largest,'isolate':isolate,'sym_h':sym_h,'sym_v':sym_v}
def with_square(P,g):
    P=dict(P)
    if g.shape[0]==g.shape[1]: P['rot90']=lambda x:np.rot90(x,1).copy(); P['rot270']=lambda x:np.rot90(x,3).copy(); P['transpose']=lambda x:x.T.copy()
    return P

def generate_ondemand(train):
    """GENERATE specialist operators from the task's structure (size relationship)."""
    gen={}
    ratios=set()
    for i,o in train:
        if i.shape[0]==0 or i.shape[1]==0: return gen
        ratios.add((o.shape[0]/i.shape[0], o.shape[1]/i.shape[1]))
    if len(ratios)!=1: return gen           # inconsistent ratio -> can't generate a clean op
    ra,rb=ratios.pop()
    if ra>=1 and rb>=1 and ra==int(ra) and rb==int(rb) and (ra>1 or rb>1):
        a,b=int(ra),int(rb); gen[f'scale{a}x{b}']=lambda g,a=a,b=b:scale(g,a,b); gen[f'tile{a}x{b}']=lambda g,a=a,b=b:tile(g,a,b)
    if ra<=1 and rb<=1 and ra>0 and rb>0 and (1/ra)==int(1/ra) and (1/rb)==int(1/rb) and (ra<1 or rb<1):
        a,b=int(1/ra),int(1/rb); gen[f'down{a}x{b}']=lambda g,a=a,b=b:downs(g,a,b)
    return gen

def infer_cmap(pairs):
    m={}
    for i,o in pairs:
        if i.shape!=o.shape: return None
        for a,b in zip(i.ravel(),o.ravel()):
            if a in m and m[a]!=b: return None
            m[a]=b
    return m
def cmap(g,m): return np.vectorize(lambda v:m.get(v,v))(g)
def ok(f,tr):
    try:
        for i,o in tr:
            r=f(i)
            if r is None or r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False

def induce(train):
    P=with_square(FIXED,train[0][0]); P.update(generate_ondemand(train))  # fixed + generated specialists
    for f in P.values():
        if ok(f,train): return True
    cm=infer_cmap(train)
    if cm is not None and ok(lambda g:cmap(g,cm),train): return True
    for f1 in P.values():
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=infer_cmap(t2)
                if cm2 is not None and ok(lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2),train): return True
            for f2 in P.values():
                if ok(lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g))),train): return True
        except Exception: pass
    return False

# We re-run with a solve-check on test using the same program search (re-find + apply).
def induce_prog(train):
    P=with_square(FIXED,train[0][0]); P.update(generate_ondemand(train))
    for f in P.values():
        if ok(f,train): return f
    cm=infer_cmap(train)
    if cm is not None and ok(lambda g:cmap(g,cm),train): return lambda g:cmap(g,cm)
    for f1 in P.values():
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=infer_cmap(t2)
                if cm2 is not None and ok(lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2),train): return lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2)
            for f2 in P.values():
                if ok(lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g))),train): return lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g)))
        except Exception: pass
    return None

files=sorted(glob.glob(os.path.join(DATA,"*.json")))
solved=[]; unsolved=[]
for fp in files:
    t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    prog=induce_prog(train); good=False
    if prog is not None:
        good=True
        for p in t["test"]:
            xi=np.array(p["input"]); yo=np.array(p["output"])
            try:
                pr=prog(xi)
                if pr is None or pr.shape!=yo.shape or not np.array_equal(pr,yo): good=False; break
            except Exception: good=False; break
    (solved if good else unsolved).append((os.path.basename(fp)[:8],t))

print(f"REAL ARC — gerador de especialistas SOB DEMANDA + mais fixos")
print(f"  RESOLVIDAS: {len(solved)}/400 = {100*len(solved)/400:.1f}%   (antes 9.2%)")
print("-"*60)
# analyze the unsolved
def feats(t):
    tr=t["train"]; i0=np.array(tr[0]["input"]); o0=np.array(tr[0]["output"])
    size_change=any(np.array(p["input"]).shape!=np.array(p["output"]).shape for p in tr)
    in_colors=set(); out_colors=set()
    for p in tr: in_colors|=set(np.array(p["input"]).ravel().tolist()); out_colors|=set(np.array(p["output"]).ravel().tolist())
    new_colors=len(out_colors-in_colors)>0
    out_bigger=o0.size>i0.size; out_smaller=o0.size<i0.size
    lab,nobj=comps(i0)
    return size_change,new_colors,out_bigger,out_smaller,len(in_colors),nobj
N=len(unsolved)
sc=sum(feats(t)[0] for _,t in unsolved); nc=sum(feats(t)[1] for _,t in unsolved)
ob=sum(feats(t)[2] for _,t in unsolved); os_=sum(feats(t)[3] for _,t in unsolved)
ncol=np.mean([feats(t)[4] for _,t in unsolved]); nobj=np.mean([feats(t)[5] for _,t in unsolved])
print(f"ANÁLISE DAS {N} NÃO-RESOLVIDAS:")
print(f"  mudam o tamanho da grade:        {100*sc/N:.0f}%")
print(f"  introduzem cores NOVAS (não no input): {100*nc/N:.0f}%")
print(f"  output maior / menor / igual:    {100*ob/N:.0f}% / {100*os_/N:.0f}% / {100*(N-ob-os_)/N:.0f}%")
print(f"  média de cores no input:         {ncol:.1f}")
print(f"  média de objetos no input:       {nobj:.1f}")
print("-"*60)
print("Leitura: as não-resolvidas exigem regras objeto-relacionais/contagem/conceituais que")
print("NENHUM operador (nem gerado sob demanda) cobre — precisam de PRIMITIVAS novas, não de")
print("mais profundidade. O gerador sob demanda ajuda em tamanho/escala; o resto é a parede.")
