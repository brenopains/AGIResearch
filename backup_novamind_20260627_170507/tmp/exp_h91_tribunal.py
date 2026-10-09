"""
H91 — "Tribunal evolutivo": ilhas em conflito + novidade(tedio) + hibrido macro/baixo-nivel
=============================================================================================
Junta num so motor todas as direcoes que sobreviveram ao teste honesto:
  - SUBAGENTES EM CONFLITO (ideia #4 do usuario; Minsky Society of Mind / Global Workspace):
    modelo de ILHAS — 4 personalidades evoluindo em paralelo e competindo, com MIGRACAO
    (o "tribunal" que troca descobertas):
        I0 conservador  -> explora o melhor (exploit, mutacao pequena, prefere macros)
        I1 explorador   -> seleciona por NOVIDADE de comportamento (nao por acerto)  [TEDIO, #5]
        I2 destruidor   -> mutacoes grandes/estruturais, programas longos
        I3 cientista    -> parcimonia (MDL): prefere o programa CURTO de alta aptidao
  - NOVIDADE/TEDIO (ideia #5; novelty search de Lehman&Stanley 2011): premia ser DIFERENTE,
    nao so estar certo -> escapa de otimos locais que a aptidao graduada trava.
  - HIBRIDO (o vies que faltou no H90): genoma mistura MACROS de alto nivel (primitivas que ja
    temos: FIXED+NEW+cmap) com INSTRUCOES de baixo nivel (registradores) -> deixa o sistema
    DESCOBRIR o macro que falta em vez de enumerar tudo as cegas.
  - Compute concentrado SO no RESIDUO (tarefas fora da uniao 86) — onde mora o nao-resolvido.
Gate exato; so-CPU (3050 ok; VRAM irrelevante). Metrica honesta = NOVAS vs uniao atual.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H63=_load("exp_h63_efficiency_breadth.py"); PU=H63.PU
random.seed(0); np.random.seed(0)
NR=4; LMAX=12; NLL=12

# ---------- baixo nivel (preserva shape) ----------
def _shift(g,dy,dx):
    o=np.zeros_like(g); H,W=g.shape
    ys0=max(0,dy); ys1=min(H,H+dy); xs0=max(0,dx); xs1=min(W,W+dx); yd0=max(0,-dy); xd0=max(0,-dx)
    o[ys0:ys1,xs0:xs1]=g[yd0:yd0+(ys1-ys0),xd0:xd0+(xs1-xs0)]; return o
def _ncount(g):
    m=(g!=0).astype(np.int64); s=np.zeros_like(m)
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            if dy or dx: s+=_shift(m,dy,dx)
    return s.astype(g.dtype)
def _largest(g):
    lab,n=PU.comps(g)
    if n==0: return np.zeros_like(g)
    best=max(range(1,n+1),key=lambda i:(lab==i).sum()); o=np.zeros_like(g); o[lab==best]=g[lab==best]; return o
def ll_apply(op,a,b,p):
    if op==0:   return _shift(a,(p//3)%3-1,(p%3)-1)
    if op==1:   return np.rot90(a,p%4).copy()
    if op==2:   return (a[:,::-1] if p%2 else a[::-1,:]).copy()
    if op==3:   c=p%10; return np.where(a==c,a,0)
    if op==4:   return np.where(a==(p%10),(p//10)%10,a)
    if op==5:   return np.where(a!=0,a,b)
    if op==6:   return np.where((a!=0)&(b!=0),a,0)
    if op==7:   return np.where(b!=0,0,a)
    if op==8:   return _ncount(a)
    if op==9:   return _largest(a)
    return a.copy()

def build_macros(train):
    pool=dict(PU.FIXED)
    if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
    pool.update(H63.NEW)
    cm=PU.infer_cmap(train)
    if cm is not None: pool['cmap']=lambda g,cm=cm: PU.cmap(g,cm)
    return pool

def run_prog(prog, g, macros):
    R=[g.copy()]+[np.zeros_like(g) for _ in range(NR-1)]
    try:
        for step in prog:
            if step[0]=='m':
                r=macros[step[1]](R[0])
                if r is None or not isinstance(r,np.ndarray) or r.size==0: return None
                R[0]=r
                if any(x.shape!=R[0].shape for x in R[1:]): R=[R[0]]+[np.zeros_like(R[0]) for _ in range(NR-1)]
            else:
                _,op,s1,s2,dst,p=step
                r=ll_apply(op,R[s1],R[s2],p)
                if r.shape!=R[0].shape: return None
                R[dst]=r
        return R[0]
    except Exception: return None

def score(prog,demos,macros):
    tot=0.0
    for i,o in demos:
        r=run_prog(prog,i,macros)
        if r is None: return 0.0
        if r.shape==o.shape: tot+=float((r==o).mean())
        else:
            dh=abs(r.shape[0]-o.shape[0]); dw=abs(r.shape[1]-o.shape[1])
            tot+=0.2*max(0.0,1-(dh+dw)/max(1,o.shape[0]+o.shape[1]))
    return tot/len(demos)

def behavior(prog,demos,macros):
    i,o=demos[0]; r=run_prog(prog,i,macros)
    if r is None: return (9,9,9,min(len(prog),LMAX))
    ds=int(np.sign(r.size-i.size))+1
    return (min(len(np.unique(r)),9), min(int((r!=0).mean()*8),8), ds, min(len(prog),LMAX))

def bdist(a,b): return sum((x-y)**2 for x,y in zip(a,b))**0.5

def rand_token(macro_names,wop,wmac):
    if random.random()<0.5 and macro_names:
        return ('m', random.choices(macro_names,weights=[wmac.get(n,1.0) for n in macro_names])[0])
    op=random.choices(range(NLL),weights=[wop.get(o,1.0) for o in range(NLL)])[0]
    return ('l',op,random.randrange(NR),random.randrange(NR),random.randrange(NR),random.randrange(100))

def mutate(prog,macro_names,wop,wmac,big=False):
    p=list(prog); nmut=random.randint(2,4) if big else 1
    for _ in range(nmut):
        k=random.random()
        if k<0.30 and len(p)<LMAX: p.insert(random.randint(0,len(p)),rand_token(macro_names,wop,wmac))
        elif k<0.55 and len(p)>1: del p[random.randrange(len(p))]
        elif k<0.85 and p: p[random.randrange(len(p))]=rand_token(macro_names,wop,wmac)
        else: p.append(rand_token(macro_names,wop,wmac))
    return p[:LMAX]

def evolve(train,macros,macro_names,wop,wmac,budget):
    NISL=4; grids=[{} for _ in range(NISL)]; archive=[]
    best=(0.0,None)
    def insert(isl,prog):
        nonlocal best
        if not prog: return 0.0
        f=score(prog,train,macros); bd=behavior(prog,train,macros)
        g=grids[isl]
        if bd not in g or g[bd][0]<f: g[bd]=(f,prog,bd)
        if f>best[0]: best=(f,prog)
        return f
    for isl in range(NISL):
        for _ in range(20):
            insert(isl,[rand_token(macro_names,wop,wmac) for _ in range(random.randint(1,3))])
    t0=time.time(); it=0
    while time.time()-t0<budget and best[0]<1.0:
        it+=1; isl=it%NISL; g=grids[isl]; vals=list(g.values())
        if not vals: insert(isl,[rand_token(macro_names,wop,wmac)]); continue
        if isl==0:    par=max(vals,key=lambda v:v[0])[1]; ch=mutate(par,macro_names,wop,wmac,big=False)
        elif isl==1:                                   # explorador: novidade (tedio)
            def nov(v): return min([bdist(v[2],a) for a in archive[-60:]],default=9.0)
            par=max(vals,key=nov)[1]; ch=mutate(par,macro_names,wop,wmac,big=False)
            archive.append(behavior(ch,train,macros))
        elif isl==2:  par=random.choice(vals)[1]; ch=mutate(par,macro_names,wop,wmac,big=True)
        else:         par=min(vals,key=lambda v:(len(v[1]) if v[0]>0.5 else 99))[1]; ch=mutate(par,macro_names,wop,wmac,big=False)
        if random.random()<0.2:                        # crossover entre ilhas
            ov=[v for gg in grids for v in gg.values()]
            if ov: oth=random.choice(ov)[1]; cut=random.randint(1,len(ch)); ch=(ch[:cut]+oth)[:LMAX]
        insert(isl,ch)
        if it%40==0:                                   # MIGRACAO (tribunal troca descobertas)
            src=max(range(NISL),key=lambda j:max([v[0] for v in grids[j].values()],default=0))
            sv=max(grids[src].values(),key=lambda v:v[0],default=None)
            if sv: insert((src+1)%NISL,sv[1])
    return best

def main():
    t0=time.time()
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'6.0'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    h89=set(json.load(open(os.path.join(HERE,"h89_evo_solved.json")))) if os.path.exists(os.path.join(HERE,"h89_evo_solved.json")) else set()
    base_union=union|h89                               # 86: o teto atual
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    wop=Counter(); wmac=Counter(); solved=set(); att=0; depths=[]
    for k,fp in enumerate(files):
        name=os.path.basename(fp)
        if name in base_union: continue                # foca no RESIDUO
        t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        macros=build_macros(train); macro_names=list(macros); att+=1
        f,prog=evolve(train,macros,macro_names,wop,wmac,budget)
        if prog is not None and f>=1.0-1e-9:
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(run_prog(prog,i,macros),o) for i,o in train):
                outs=[run_prog(prog,xi,macros) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                    solved.add(name); depths.append(len(prog))
                    for s in prog:
                        if s[0]=='m': wmac[s[1]]+=1.0
                        else: wop[s[1]]+=1.0
        if (k+1)%50==0: print(f"  ... {k+1}/400 (residuo testado={att}, NOVAS={len(solved)})"); sys.stdout.flush()
    json.dump(sorted(solved),open(os.path.join(HERE,"h91_tribunal_solved.json"),"w"))
    print("="*72)
    print(f"H91 Tribunal evolutivo (ilhas+novidade+hibrido) | budget={budget}s | so-CPU")
    print(f"  teto atual (uniao+H89) = {len(base_union)} | residuo testado = {att}")
    print(f"  NOVAS resolvidas no residuo: {len(solved)} {sorted(solved)[:14]}")
    print(f"  novo TETO: {len(base_union|solved)}/400 = {100*len(base_union|solved)/400:.1f}%  (era {100*len(base_union)/400:.1f}%)")
    if depths: print(f"  profundidade media {np.mean(depths):.1f} max {max(depths)}")
    print(f"  macros mais uteis: {[(n,int(w)) for n,w in wmac.most_common(6)]}")
    print(f"  ops baixo-nivel mais uteis: {[(o,int(w)) for o,w in wop.most_common(6)]}")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: se NOVAS>0, o hibrido+tribunal achou primitiva/composicao que o resto nao.")
    print("  Se 0, confirma de novo: a parede e LARGUA de primitiva, imune a engenharia de busca.")

if __name__=="__main__": main()
