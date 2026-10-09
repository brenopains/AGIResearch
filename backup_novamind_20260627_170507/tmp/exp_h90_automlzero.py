"""
H90 — AutoML-Zero para o ARC: evoluir PRIMITIVAS, nao so compor as existentes
==============================================================================
H89 mostrou que profundidade de composicao das primitivas FIXAS rende pouco (+2): o teto e
LARGURA (parede de Solomonoff). Unico caminho evolutivo para largura NOVA: evoluir num nivel
BAIXO — instrucoes minimas sobre registradores de grade — pra que operacoes novas (gravidade,
denoise, simetria, etc.) EMIRJAM da composicao, em vez de serem listadas a mao (Real et al.
2020, AutoML-Zero: algoritmos de ML emergiram de ops escalares/vetoriais cruas).

Substrato (so-CPU; VRAM irrelevante; roda na 3050):
  - NR registradores de grade (R0=entrada; resto=0), mesmo HxW (so tarefas mesmo-shape).
  - Instrucao 3-enderecos: (op, s1, s2, dst, param). Ops minimas: SHIFT, ROT, FLIP, KEEPCOLOR,
    REPLACE, OR(overlay), AND, SUB, NCOUNT(conta vizinhos), LARGEST(maior componente), CMAP, COPY.
  - Saida = R0 apos executar. Fitness graduado (acuracia por celula nos demos). MAP-Elites
    p/ diversidade + priores de op auto-adaptativos (a evolucao aprende quais instrucoes usar).
Gate exato; metrica = NOVAS vs uniao(H85=84). Honesto: espaco de busca grande; com orcamento
de CPU pode rediscover pouco — mas e o teste correto de "gerar codigo/primitiva do zero".
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
NR=4; NOP=12; LMAX=14

def _shift(g,dy,dx):
    o=np.zeros_like(g); H,W=g.shape
    ys0=max(0,dy); ys1=min(H,H+dy); xs0=max(0,dx); xs1=min(W,W+dx)
    yd0=max(0,-dy); xd0=max(0,-dx)
    o[ys0:ys1,xs0:xs1]=g[yd0:yd0+(ys1-ys0),xd0:xd0+(xs1-xs0)]
    return o

def _ncount(g):
    m=(g!=0).astype(np.int64); s=np.zeros_like(m)
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            if dy==0 and dx==0: continue
            s+=_shift(m,dy,dx)
    return s.astype(g.dtype)

def _largest(g):
    lab,n=PU.comps(g)
    if n==0: return np.zeros_like(g)
    best=max(range(1,n+1),key=lambda i:(lab==i).sum()); o=np.zeros_like(g); o[lab==best]=g[lab==best]; return o

def run(prog, g, cm):
    R=[g.copy()]+[np.zeros_like(g) for _ in range(NR-1)]
    try:
        for (op,s1,s2,dst,p) in prog:
            a=R[s1]; b=R[s2]
            if op==0:   r=_shift(a,(p//3)%3-1,(p%3)-1)
            elif op==1: r=np.rot90(a,p%4).copy()
            elif op==2: r=(a[:,::-1] if p%2 else a[::-1,:]).copy()
            elif op==3: c=p%10; r=np.where(a==c,a,0)
            elif op==4: r=np.where(a==(p%10),(p//10)%10,a)
            elif op==5: r=np.where(a!=0,a,b)
            elif op==6: r=np.where((a!=0)&(b!=0),a,0)
            elif op==7: r=np.where(b!=0,0,a)
            elif op==8: r=_ncount(a)
            elif op==9: r=_largest(a)
            elif op==10:r=PU.cmap(a,cm) if cm is not None else a.copy()
            else:       r=a.copy()
            if r.shape!=g.shape: return None
            R[dst]=r
        return R[0]
    except Exception: return None

def rand_instr():
    return (random.randrange(NOP),random.randrange(NR),random.randrange(NR),random.randrange(NR),random.randrange(100))

def score(prog,demos,cm):
    tot=0.0
    for i,o in demos:
        r=run(prog,i,cm)
        if r is None or r.shape!=o.shape: return 0.0
        tot+=float((r==o).mean())
    return tot/len(demos)

def behavior(prog,demos,cm):
    i,o=demos[0]; r=run(prog,i,cm)
    if r is None: return (9,9,min(len(prog),LMAX))
    return (min(len(np.unique(r)),9), min(int((r!=0).mean()*8),8), min(len(prog),LMAX))

def mutate(prog,wop):
    p=list(prog); k=random.random()
    def ins():
        op=random.choices(range(NOP),weights=[wop.get(o,1.0) for o in range(NOP)])[0]
        return (op,random.randrange(NR),random.randrange(NR),random.randrange(NR),random.randrange(100))
    if k<0.30 and len(p)<LMAX: p.insert(random.randint(0,len(p)),ins())
    elif k<0.55 and len(p)>1: del p[random.randrange(len(p))]
    elif k<0.85 and p:
        j=random.randrange(len(p)); op,s1,s2,dst,pa=p[j]
        f=random.randrange(5)
        if f==0: op=random.choices(range(NOP),weights=[wop.get(o,1.0) for o in range(NOP)])[0]
        elif f==1: s1=random.randrange(NR)
        elif f==2: s2=random.randrange(NR)
        elif f==3: dst=random.randrange(NR)
        else: pa=random.randrange(100)
        p[j]=(op,s1,s2,dst,pa)
    else: p.append(ins())
    return p[:LMAX]

def evolve(demos,cm,wop,budget,pop=80):
    grid={}
    def consider(prog):
        if not prog: return 0.0
        f=score(prog,demos,cm); bd=behavior(prog,demos,cm)
        if bd not in grid or grid[bd][0]<f: grid[bd]=(f,prog)
        return f
    best=(0.0,None)
    for _ in range(pop):
        prog=[rand_instr() for _ in range(random.randint(1,3))]
        f=consider(prog)
        if f>best[0]: best=(f,prog)
    t0=time.time()
    while time.time()-t0<budget and best[0]<1.0:
        vals=list(grid.values()); _,par=random.choice(vals)
        ch=mutate(par,wop)
        if random.random()<0.25 and len(vals)>1:
            _,oth=random.choice(vals); cut=random.randint(1,len(ch)); ch=(ch[:cut]+oth)[:LMAX]
        f=consider(ch)
        if f>best[0]: best=(f,ch)
    return best

def main():
    t0=time.time()
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'4.0'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    wop=Counter(); solved=set(); att=0
    for k,fp in enumerate(files):
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue   # so mesmo-shape
        att+=1; cm=PU.infer_cmap(train)
        f,prog=evolve(train,cm,wop,budget)
        if prog is not None and f>=1.0-1e-9:
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(run(prog,i,cm),o) for i,o in train):
                outs=[run(prog,xi,cm) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                    solved.add(name)
                    for (op,*_ ) in prog: wop[op]+=1.0
        if (k+1)%50==0: print(f"  ... {k+1}/400 (mesmo-shape={att}, solved={len(solved)}, NOVAS={len(solved-union)})"); sys.stdout.flush()
    json.dump(sorted(solved),open(os.path.join(HERE,"h90_automlzero_solved.json"),"w"))
    OPN={0:'SHIFT',1:'ROT',2:'FLIP',3:'KEEPCOLOR',4:'REPLACE',5:'OR',6:'AND',7:'SUB',8:'NCOUNT',9:'LARGEST',10:'CMAP',11:'COPY'}
    print("="*72)
    print(f"H90 AutoML-Zero para ARC (evolui primitivas de baixo nivel, so-CPU)")
    print(f"  budget/tarefa={budget}s | mesmo-shape={att}/400 | union(H85)={len(union)}")
    print(f"  resolveu: {len(solved)}/400 | NOVAS vs union: {len(solved-union)} {sorted(solved-union)[:14]}")
    print(f"  nova UNIAO: {len(union|solved)}/400 = {100*len(union|solved)/400:.1f}%")
    print(f"  instrucoes mais uteis: {[(OPN[o],int(w)) for o,w in Counter(wop).most_common(8)]}")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: se NOVAS>0, primitivas NOVAS emergiram de ops cruas (largura nova real).")
    print("  Se 0, o espaco de baixo nivel e grande demais p/ o orcamento de CPU — a largura")
    print("  precisa de bias melhor, nao de mais forca bruta.")

if __name__=="__main__": main()
