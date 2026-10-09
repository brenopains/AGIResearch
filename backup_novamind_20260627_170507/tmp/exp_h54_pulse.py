"""
H54 — "Pulse": solver de ARC SEM TREINO, por minimizacao de energia com ritmo (batimento)
==========================================================================================
Ideia (do usuario + pesquisa recente): a "cola" que integra e um PROCESSO RITMICO
(gating por oscilacao / binding-by-synchrony: Womelsdorf 2014, Palva 2018), e o que
"emerge" dos LLMs de mil-GPUs e COMPUTACAO EM TEMPO DE INFERENCIA (IRED/Du 2024,
FreeDoM, free-energy/Spisak 2025). Atacamos isso DIRETO, sem treino:

  Energia E(programa) = inconsistencia_com_demos  +  lambda * comprimento(programa)
                        (Occam/Solomonoff como escalar; E=0 <=> resolve os demos)

  Dinamica = batimento de dois tempos sobre uma populacao de programas:
    DIASTOLE (gamma/expandir): cada candidato gera mutacoes (anexa/troca operador),
                               com vies de ATENCAO p/ operadores uteis (temperatura alta).
    SISTOLE  (alpha/contrair): ordena por energia, mantem top-K; recozimento (K e temp caem).
    Acoplamento cross-frequencia: a tendencia da melhor energia (lento) regula a
                                  temperatura/precisao do proximo ciclo (rapido).
    AFETO/ritmo (nova_maestro): plato de energia -> "preso" -> amplitude sobe + CONSOLIDA
                                um composto novo (wake-sleep) -> alcanca profundidade 3/4 barato.

Honesto: e uma aposta de EFICIENCIA DE BUSCA (nao quebra a parede de Solomonoff/largura;
nao inventa primitiva fora do conjunto). Roda na CPU em minutos. Reusa o pool do H51.
"""
import os, json, glob, math, random, numpy as np
from collections import deque, Counter
random.seed(0); np.random.seed(0)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

# ---------------- operadores de grade (reusados do H51) ----------------
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
def keep_smallest(g):
    lab,n=comps(g)
    if n==0: return g.copy()
    sm=1+int(np.argmin([(lab==i).sum() for i in range(1,n+1)])); o=np.zeros_like(g); o[lab==sm]=g[lab==sm]; return o
def denoise(g):
    lab,n=comps(g); o=g.copy()
    for i in range(1,n+1):
        if (lab==i).sum()<=1: o[lab==i]=0
    return o
def isolate(g):
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True); k=v[c.argmax()]; o=np.zeros_like(g); o[g==k]=k; return o
def bbox_fill(g):
    lab,n=comps(g); o=g.copy()
    for i in range(1,n+1):
        ys,xs=np.where(lab==i); col=g[ys[0],xs[0]]
        o[ys.min():ys.max()+1,xs.min():xs.max()+1]=col
    return o
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

# ---- primitivas NOVAS de LARGURA (mirando categorias nao-resolvidas do H50) ----
def compress(g):  # remove linhas/colunas totalmente vazias (tamanho muda)
    rows=[r for r in range(g.shape[0]) if g[r].any()]; cols=[c for c in range(g.shape[1]) if g[:,c].any()]
    if not rows or not cols: return g.copy()
    return g[np.ix_(rows,cols)].copy()
def fill_holes(g):  # preenche buracos (0) internos de cada componente com a cor do componente
    o=g.copy(); H,W=g.shape
    lab,n=comps((g==0).astype(g.dtype))  # componentes de fundo
    if n==0: return o
    bg=np.zeros_like(g)
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        touches=ys.min()==0 or xs.min()==0 or ys.max()==H-1 or xs.max()==W-1
        if not touches:  # buraco interno: pinta com cor vizinha dominante
            nb=[g[y+dy,x+dx] for y,x in zip(ys,xs) for dy,dx in((1,0),(-1,0),(0,1),(0,-1))
                if 0<=y+dy<H and 0<=x+dx<W and g[y+dy,x+dx]!=0]
            if nb:
                v,c=np.unique(nb,return_counts=True); o[ys,xs]=v[c.argmax()]
    return o
def hollow(g):  # esvazia o interior de cada componente (so contorno)
    o=g.copy(); lab,n=comps(g)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r,c]!=0:
                inner=all(0<=r+dy<g.shape[0] and 0<=c+dx<g.shape[1] and g[r+dy,c+dx]==g[r,c]
                          for dy,dx in((1,0),(-1,0),(0,1),(0,-1)))
                if inner: o[r,c]=0
    return o
def topcolor(g):  # pinta tudo com a cor nao-zero mais frequente
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True); return np.full_like(g,v[c.argmax()])
def half(g,which):  # metades (saida menor): top/bot/left/right
    H,W=g.shape
    if which=='t': return g[:H//2].copy() if H>=2 else None
    if which=='b': return g[H-H//2:].copy() if H>=2 else None
    if which=='l': return g[:,:W//2].copy() if W>=2 else None
    if which=='r': return g[:,W-W//2:].copy() if W>=2 else None
def overlay(g,axis,mode):  # divide em 2 metades e combina -> mascara 0/1 (cmap mapeia a cor depois)
    H,W=g.shape
    if axis==1:
        if W%2: return None
        a,b=g[:,:W//2],g[:,W//2:]
    else:
        if H%2: return None
        a,b=g[:H//2],g[H//2:]
    A,B=a!=0,b!=0
    if mode=='and': m=A&B
    elif mode=='or': m=A|B
    elif mode=='xor': m=A^B
    else: m=A&~B
    return m.astype(g.dtype)

FIXED={'id':lambda g:g.copy(),'fliplr':lambda g:g[:,::-1].copy(),'flipud':lambda g:g[::-1,:].copy(),'rot180':lambda g:np.rot90(g,2).copy(),
 'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1),
 'grav':grav,'maj':maj,'border':border,'crop':crop,'keep_largest':keep_largest,'keep_smallest':keep_smallest,'denoise':denoise,
 'isolate':isolate,'bbox_fill':bbox_fill,'sym_h':sym_h,'sym_v':sym_v,
 'tile2x2':lambda g:np.tile(g,(2,2)),'tile1x2':lambda g:np.tile(g,(1,2)),'tile2x1':lambda g:np.tile(g,(2,1)),
 'mirror_h':lambda g:np.concatenate([g,g[:,::-1]],1),'mirror_v':lambda g:np.concatenate([g,g[::-1,:]],0),
 'scale2':lambda g:np.kron(g,np.ones((2,2),dtype=g.dtype)),'down2':lambda g:downs(g,2,2),'down3':lambda g:downs(g,3,3),
 'compress':compress,'fill_holes':fill_holes,'hollow':hollow,'topcolor':topcolor,
 'half_t':lambda g:half(g,'t'),'half_b':lambda g:half(g,'b'),'half_l':lambda g:half(g,'l'),'half_r':lambda g:half(g,'r'),
 'ov_h_and':lambda g:overlay(g,1,'and'),'ov_h_or':lambda g:overlay(g,1,'or'),'ov_h_xor':lambda g:overlay(g,1,'xor'),'ov_h_diff':lambda g:overlay(g,1,'diff'),
 'ov_v_and':lambda g:overlay(g,0,'and'),'ov_v_or':lambda g:overlay(g,0,'or'),'ov_v_xor':lambda g:overlay(g,0,'xor'),'ov_v_diff':lambda g:overlay(g,0,'diff')}
def square_ops():
    return {'rot90':lambda x:np.rot90(x,1).copy(),'rot270':lambda x:np.rot90(x,3).copy(),'transpose':lambda x:x.T.copy()}
def gen_ondemand(train):
    gen={}; ratios=set()
    for i,o in train:
        if i.shape[0]==0 or i.shape[1]==0: return gen
        ratios.add((o.shape[0]/i.shape[0],o.shape[1]/i.shape[1]))
    if len(ratios)!=1: return gen
    ra,rb=ratios.pop()
    if ra==int(ra) and rb==int(rb) and ra>=1 and rb>=1 and (ra>1 or rb>1):
        a,b=int(ra),int(rb); gen[f'scale{a}_{b}']=lambda g,a=a,b=b:scale(g,a,b); gen[f'tile{a}_{b}']=lambda g,a=a,b=b:np.tile(g,(a,b))
    if ra>0 and rb>0 and ra<=1 and rb<=1 and (1/ra)==int(1/ra) and (1/rb)==int(1/rb) and (ra<1 or rb<1):
        a,b=int(1/ra),int(1/rb); gen[f'down{a}_{b}']=lambda g,a=a,b=b:downs(g,a,b)
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

# ---------------- avaliacao + energia ----------------
def apply_ops(pool,prog,g):
    for nm in prog:
        g=pool[nm](g)
        if g is None: return None
    return g
def mism(a,b):
    if a is None or a.shape!=b.shape: return 1.0
    return float(np.mean(a!=b))
LAM=0.02   # peso MDL (comprimento): so DESEMPATE no ranking, NUNCA na aceitacao
LAM2=0.01  # prior estrutural (Occam+): penaliza VARIEDADE de operadores distintos
def eval_prog(pool,prog,demos):
    """retorna (mismatch_cru, use_cmap, cmap_dict). mismatch=0 <=> resolve os demos.
    Tenta saida crua e saida+colormap inferido dos demos."""
    outs=[apply_ops(pool,prog,i) for i,_ in demos]
    raw=np.mean([mism(o,t) for o,(_,t) in zip(outs,demos)])
    use_cmap=False; cm=None; best=raw
    if all(o is not None and o.shape==t.shape for o,(_,t) in zip(outs,demos)):
        cm=infer_cmap([(o,t) for o,(_,t) in zip(outs,demos)])
        if cm is not None:
            mapped=np.mean([mism(cmap(o,cm),t) for o,(_,t) in zip(outs,demos)])
            if mapped<best: best=mapped; use_cmap=True
    return best, use_cmap, cm
def energy_of(pool,prog,demos):
    """energia = mismatch + LAM*len + LAM2*distintos (Occam estrutural). Retorna (energia, mismatch)."""
    m,_,_=eval_prog(pool,prog,demos); return m+LAM*len(prog)+LAM2*len(set(prog)), m
def realize(pool,prog,use_cmap,cm):
    if use_cmap and cm is not None:
        return lambda g:(None if apply_ops(pool,prog,g) is None else cmap(apply_ops(pool,prog,g),cm))
    return lambda g:apply_ops(pool,prog,g)

# ---------------- afeto/ritmo (reuso do nova_maestro) ----------------
class Self:
    def __init__(s): s.affect=0.0
    def fail(s): s.affect=min(2.0,s.affect+0.35)
    def succeed(s): s.affect=max(0.0,s.affect*0.6)
    def amp(s): return 1.0+1.5*s.affect          # amplitude do batimento (mutacoes/temp)

# ---------------- batimento: busca por energia com ritmo ----------------
def pulse_search(base_pool,train,max_cycles=14,max_len=4,trace=None):
    pool=dict(base_pool); pool.update(gen_ondemand(train))
    names=list(pool); useful=Counter()
    self=Self()
    def score(p):
        e,m=energy_of(pool,p,train); return (e,m,p)
    # SISTOLE inicial = cobertura EXAUSTIVA de profundidade 1 e 2 (superset do H51);
    # garante que o ritmo so pode ADICIONAR (profundidade 3-4), nunca perder o baseline.
    init=[(a,) for a in names]+[(a,b) for a in names for b in names]
    scored=sorted((score(p) for p in init), key=lambda x:x[0])
    K=40; pop=[s[2] for s in scored[:K]]
    bestE,bestM,bestP=scored[0]
    solvedP=next((s[2] for s in scored if s[1]<=1e-9),None)
    if trace is not None: trace.append(bestM)
    plateau=0; ema=1.0; stall=0; CAP=120
    for cyc in range(max_cycles):
        if solvedP is not None: break
        if stall>=6: break                              # early-stop: plateau de energia
        # cross-frequency: EMA do progresso (lento) regula a temperatura (rapido)
        heat=0.05/(ema+0.05)                            # ~1 estagnado, ~0 progredindo
        temp=max(0.15,0.9*(0.85**cyc)*(0.6+0.8*heat)*self.amp())
        nmut=int(2+3*self.amp())                        # diastole: largura da expansao
        # ATENCAO: distribuicao sobre operadores (vies p/ uteis)
        w=np.array([1.0+useful[n] for n in names],float); w=w**(1.0/max(temp,1e-3)); w/=w.sum()
        cand=set(pop)
        for p in pop[:3]:                               # melhor-primeiro: expande topo com TODOS os ops
            if len(p)<max_len:
                for op in names: cand.add(p+(op,))      # garante cobertura de profundidade nos prefixos bons
        for p in pop:                                   # DIASTOLE estocastica: diversidade (vies de atencao)
            for _ in range(nmut):
                op=names[np.random.choice(len(names),p=w)]
                if len(p)<max_len: cand.add(p+(op,))     # anexa
                if len(p)>=1: cand.add(p[:-1]+(op,))     # troca ultimo
        scored=sorted((score(p) for p in cand), key=lambda x:x[0])
        K=max(8,int(K*0.8))                              # SISTOLE: contrair (recozimento)
        pop=[s[2] for s in scored[:K]]
        newE,newM,newP=scored[0]
        # aceitacao = consistencia exata (mismatch 0), NAO energia
        sol=next((s for s in scored if s[1]<=1e-9), None)
        if sol is not None: solvedP=sol[2]; bestP=sol[2]
        delta=max(0.0,bestE-newE)                        # melhora de energia neste batimento
        ema=0.5*ema+0.5*delta                            # media movel exponencial (lento)
        stall = stall+1 if delta<0.01 else 0             # plateau (<0.01 de melhora)
        if newE<bestE-1e-9:                              # progresso real (energia desce)
            for n in newP: useful[n]+=1                  # reforca atencao
            bestE,bestM,bestP=newE,newM,newP; self.succeed(); plateau=0
        else:
            plateau+=1; self.fail()
        if trace is not None: trace.append(min(bestM,(sol[1] if sol else bestM)))
        # CONSOLIDACAO (wake-sleep) quando preso: abstrai melhor composto -> novo operador
        if solvedP is None and plateau>=2 and len(bestP)>=2 and len(pool)<CAP:
            chain=tuple(bestP); nm='+'.join(chain)
            if nm not in pool:
                pool[nm]=lambda g,chain=chain:apply_ops(pool,chain,g); names.append(nm); useful[nm]=3
                pop.append((nm,))
            plateau=0
    if solvedP is not None:
        m,use_cmap,cm=eval_prog(pool,solvedP,train)
        used_emergent=any('+' in nm for nm in solvedP)
        return realize(pool,solvedP,use_cmap,cm), used_emergent, solvedP
    return None, False, None

# ---------------- harness 400 tarefas ----------------
def main():
    base=dict(FIXED)  # ops dependentes de quadrado entram por tarefa
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    solved=0; emergent_solves=0; traces=[]
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        pool=dict(base)
        if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(square_ops())
        trace=[] if len(traces)<3 else None
        prog,emergent,bestP=pulse_search(pool,train,trace=trace)
        if prog is None: continue
        good=True
        for p in t["test"]:
            xi=np.array(p["input"]); yo=np.array(p["output"])
            try:
                pr=prog(xi)
                if pr is None or pr.shape!=yo.shape or not np.array_equal(pr,yo): good=False; break
            except Exception: good=False; break
        if good:
            solved+=1
            if emergent: emergent_solves+=1
            if trace is not None and len(trace)>1: traces.append((os.path.basename(fp),trace))
    print("="*64)
    print("H54 PULSE — solver SEM treino (energia + ritmo/batimento + wake-sleep)")
    print(f"  RESOLVIDAS: {solved}/400 = {100*solved/400:.1f}%   (baseline H51 = 9.8%)")
    print(f"  Resolvidas via primitiva COMPOSTA auto-inventada (emergencia): {emergent_solves}")
    # artefato de emergencia: traco do batimento (energia x ciclo)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        if traces:
            plt.figure(figsize=(7,4))
            for nm,tr in traces: plt.plot(range(len(tr)),tr,marker='o',label=nm[:14])
            plt.xlabel("ciclo (batimento)"); plt.ylabel("melhor energia"); plt.title("Pulse: energia caindo por batimento")
            plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
            out=os.path.join(os.path.dirname(__file__),"pulse_heartbeat.png"); plt.savefig(out,dpi=110)
            print(f"  Traco do batimento salvo em {out}")
    except Exception as e:
        print(f"  (matplotlib indisponivel: {e})")

if __name__=="__main__": main()
