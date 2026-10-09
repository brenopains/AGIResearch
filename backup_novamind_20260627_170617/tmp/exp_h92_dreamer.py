"""
H92 — "Jarvis cria Ultron": currículo auto-gerado + LIBRARY LEARNING (DreamCoder-lite)
========================================================================================
A versao HONESTA e real do pedido do usuario ("algoritmo que remodela o ambiente E a propria
arquitetura, gerando tarefas/primitivas novas"). Mecanismo cientifico: DreamCoder (Ellis et al.
2021, "Growing libraries of reusable abstractions") = o unico jeito conhecido de um sistema
CRESCER o proprio vocabulario de primitivas sozinho.

Ciclo (open-ended, so-CPU — busca simbolica nao usa GPU; o gargalo e algoritmo, nao placa):
  1. GERA mundos: tarefas sinteticas = programa aleatorio (sobre a biblioteca ATUAL) aplicado a
     grades aleatorias. Curriculo: a profundidade do gerador cresce a cada geracao (o ambiente
     que o sistema "remodela" pra ficar mais dificil).
  2. EVOLUI pra resolve-las (MAP-Elites sobre a biblioteca atual).
  3. ABSTRACAO (o "Ultron" — remodela a propria arquitetura): os trechos de programa que mais se
     repetem nas solucoes viram PRIMITIVAS NOVAS NOMEADAS, adicionadas a biblioteca. O vocabulario
     CRESCE por compressao do que funcionou (numero -> soma -> multiplicacao -> ...).
  4. TRANSFERENCIA: roda a biblioteca CRESCIDA no ARC REAL (as 314 nao resolvidas). Mede NOVAS.

Tese testada: um curriculo auto-gerado pode fazer emergir uma primitiva que TRANSFERE para o
ARC real? Se sim (NOVAS>0 e uma macro abstraida participa), e a fagulha de descoberta aberta de
primitiva. Se nao, confirma: o gerador so cria dentro do proprio fecho; largura nova exige sinal
de FORA (dados/mundo real), nao auto-jogo — resultado honesto e informativo.
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
LMAX=10

def base_library():
    fns=dict(PU.FIXED); fns.update(PU.square_ops()); fns.update(H63.NEW)
    return fns

def apply_prog(prog, fns, g):
    cur=g
    try:
        for nm in prog:
            cur=fns[nm](cur)
            if cur is None or not isinstance(cur,np.ndarray) or cur.size==0 or cur.ndim!=2: return None
        return cur
    except Exception: return None

def make_macro(seq, fns):
    flist=[fns[n] for n in seq]
    def m(g, flist=flist):
        cur=g
        for f in flist:
            cur=f(cur)
            if cur is None or not isinstance(cur,np.ndarray) or cur.size==0 or cur.ndim!=2: return None
        return cur
    return m

# ---------------- gerador de mundos (tarefas sinteticas) ----------------
def rand_grid():
    H=random.randint(3,11); W=random.randint(3,11); k=random.randint(2,5)
    g=np.zeros((H,W),np.int64)
    for _ in range(random.randint(2,max(3,H*W//4))):
        g[random.randrange(H),random.randrange(W)]=random.randint(1,k)
    return g

def gen_task(names, fns, depth):
    for _ in range(20):
        prog=[random.choice(names) for _ in range(depth)]
        demos=[]
        ok=True
        for _ in range(3):
            gi=rand_grid(); go=apply_prog(prog,fns,gi)
            if go is None or np.array_equal(go,gi) or len(np.unique(go))<=1: ok=False; break
            demos.append((gi,go))
        if ok and len({d[1].tobytes() for d in demos})>1:
            return demos, prog
    return None, None

# ---------------- solucionador evolutivo (MAP-Elites sobre a biblioteca atual) ----------------
def score(prog, fns, demos):
    tot=0.0
    for i,o in demos:
        r=apply_prog(prog,fns,i)
        if r is None: return 0.0
        if r.shape==o.shape: tot+=float((r==o).mean())
        else:
            dh=abs(r.shape[0]-o.shape[0]); dw=abs(r.shape[1]-o.shape[1]); tot+=0.2*max(0.0,1-(dh+dw)/max(1,o.shape[0]+o.shape[1]))
    return tot/len(demos)

def behavior(prog, fns, demos):
    i,o=demos[0]; r=apply_prog(prog,fns,i)
    if r is None: return (9,9,min(len(prog),LMAX))
    return (min(len(np.unique(r)),9), int(np.sign(r.size-i.size))+1, min(len(prog),LMAX))

def evolve(demos, names, fns, weights, budget, pop=48):
    grid={}; best=(0.0,None)
    def consider(prog):
        nonlocal best
        if not prog: return 0.0
        f=score(prog,fns,demos); bd=behavior(prog,fns,demos)
        if bd not in grid or grid[bd][0]<f: grid[bd]=(f,prog)
        if f>best[0]: best=(f,prog)
        return f
    for _ in range(pop):
        consider([random.choices(names,weights=[weights.get(n,1.0) for n in names])[0] for _ in range(random.randint(1,3))])
    t0=time.time()
    while time.time()-t0<budget and best[0]<1.0:
        vals=list(grid.values()); _,par=random.choice(vals); p=list(par); k=random.random()
        def pk(): return random.choices(names,weights=[weights.get(n,1.0) for n in names])[0]
        if k<0.3 and len(p)<LMAX: p.insert(random.randint(0,len(p)),pk())
        elif k<0.55 and len(p)>1: del p[random.randrange(len(p))]
        elif k<0.85 and p: p[random.randrange(len(p))]=pk()
        else: p.append(pk())
        if random.random()<0.25 and len(vals)>1:
            _,oth=random.choice(vals); cut=random.randint(1,len(p)); p=(p[:cut]+oth)[:LMAX]
        consider(p[:LMAX])
    return best

def solve_residue(names, fns, residue_tasks, budget):
    weights=Counter(); solved=set(); used_macro=set()
    for name,train,tests in residue_tasks:
        # injeta cmap por-tarefa (dado-derivado) so aqui
        local=dict(fns); lnames=list(names)
        cm=PU.infer_cmap(train)
        if cm is not None: local['cmap']=lambda g,cm=cm:PU.cmap(g,cm); lnames=lnames+['cmap']
        f,prog=evolve(train,lnames,local,weights,budget)
        if prog is not None and f>=1.0-1e-9:
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(apply_prog(prog,local,i),o) for i,o in train):
                outs=[apply_prog(prog,local,xi) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                    solved.add(name)
                    for nm in prog:
                        weights[nm]+=1.0
                        if nm.startswith('M'): used_macro.add(nm)
    return solved, used_macro

def main():
    t0=time.time()
    gens=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--gens=')),'6'))
    synth=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--synth=')),'40'))
    sbud=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--sbud=')),'0.25'))
    rbud=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--rbud=')),'3.0'))
    absthr=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--absthr=')),'3'))   # limiar de abstracao
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    h89=set(json.load(open(os.path.join(HERE,"h89_evo_solved.json")))) if os.path.exists(os.path.join(HERE,"h89_evo_solved.json")) else set()
    base_union=union|h89
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    residue=[]
    for fp in files:
        name=os.path.basename(fp)
        if name in base_union: continue
        t=json.load(open(fp))
        tr=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        te=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        residue.append((name,tr,te))
    fns=base_library(); names=list(fns); n0=len(names)
    print(f"H92 Dreamer: biblioteca inicial={n0} ops | residuo={len(residue)} | gens={gens}")
    macro_count=0
    for gen in range(gens):
        sub=Counter(); sols=0
        for _ in range(synth):
            depth=3+gen          # curriculo MAIS profundo -> solucoes mais longas -> mais subsequencia compartilhada (forca reuso)
            demos,truep=gen_task(names,fns,depth)
            if demos is None: continue
            f,prog=evolve(demos,names,fns,Counter(),sbud)
            if prog is not None and f>=1.0-1e-9:
                sols+=1
                for L in (2,3):
                    for i in range(len(prog)-L+1): sub[tuple(prog[i:i+L])]+=1
        # ABSTRACAO: trechos frequentes viram primitivas novas
        added=[]
        for seq,c in sub.most_common(20):
            if c>=absthr and macro_count<60:
                nm=f"M{gen}_{macro_count}"; fns[nm]=make_macro(list(seq),fns); names.append(nm); added.append((seq,c)); macro_count+=1
        print(f"  gen {gen}: tarefas resolvidas {sols}/{synth} | macros novos {len(added)} | biblioteca={len(names)}")
        sys.stdout.flush()
    # TRANSFERENCIA: biblioteca crescida no ARC real
    print(f"  -> testando biblioteca crescida ({len(names)} ops, {macro_count} macros) no ARC real...")
    solved, used_macro = solve_residue(names, fns, residue, rbud)
    json.dump(sorted(solved),open(os.path.join(HERE,"h92_dreamer_solved.json"),"w"))
    print("="*72)
    print(f"H92 Dreamer (currículo auto-gerado + library learning) | so-CPU")
    print(f"  biblioteca: {n0} -> {len(names)} ops (+{macro_count} primitivas auto-descobertas)")
    print(f"  teto atual={len(base_union)} | NOVAS no residuo: {len(solved)} {sorted(solved)[:12]}")
    print(f"  macros abstraidos que participaram de solve real: {sorted(used_macro)[:8] if used_macro else 'nenhum'}")
    print(f"  novo TETO: {len(base_union|solved)}/400 = {100*len(base_union|solved)/400:.1f}%  (era {100*len(base_union)/400:.1f}%)")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: se NOVAS>0 via macro auto-descoberto, e fagulha de descoberta ABERTA de")
    print("  primitiva (auto-jogo gerou vocabulario que transferiu). Se 0, o gerador so cria")
    print("  DENTRO do proprio fecho — largura nova exige sinal do mundo real, nao auto-jogo.")

if __name__=="__main__": main()
