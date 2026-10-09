"""
H97 — Tem ouro no conectoma? Topologia de cerebro de mamifero como FIACAO do reservatorio
==========================================================================================
O usuario insistiu (com razao): conectoma humano/rato/macaco deve servir pra ALGO. Onde NAO
serve = virar operacao de grade do ARC (numerologia). Onde PODE servir = como TOPOLOGIA da rede
recorrente que computa — porque dinamica recorrente depende da estrutura do grafo. Testamos
honesto: a fiacao de cerebro de mamifero computa MELHOR que aleatoria numa tarefa de MEMORIA
TEMPORAL (dominio natural de reservatorio/echo-state; ha literatura: reservatorios modulares/
small-world tem melhor tradeoff memoria-nao-linearidade).

Condicoes (mesma densidade e MESMO raio espectral p/ ser justo — so muda a TOPOLOGIA):
  RANDOM     : aleatorio denso (controle)
  SMALLWORLD : Watts-Strogatz (alto clustering + caminho curto) — propriedade de cortex
  MODULAR    : comunidades (modulos) com poucas conexoes inter-modulo
  BRAINLIKE  : modular + RICH-CLUB (hubs interligados) + Dale E/I (80% excit, 20% inib) —
               as 3 marcas dos conectomas de mamifero juntas (inclui a expansao inibitoria do MSIF)
Tarefas: (1) CAPACIDADE DE MEMORIA (Jaeger): reconstruir entrada atrasada u(t-k); MC=sum corr^2.
         (2) NARMA-10: tarefa temporal NAO-LINEAR; mede NRMSE (menor=melhor).
Honesto: se BRAINLIKE/MODULAR > RANDOM em MC/NARMA, ha OURO real na topologia do conectoma
(estrutura biologica computa melhor). Se empatar, a topologia nao foi o lever nesta tarefa — medido.
"""
import os, time, numpy as np
rng=np.random.default_rng(0)
N=240; DENS=0.10; RHO=0.95; T=2400; WASH=200; TR=1400

def rescale(W, rho):
    ev=np.max(np.abs(np.linalg.eigvals(W)))
    return W*(rho/ev) if ev>1e-9 else W

def W_random(g):
    M=(g.random((N,N))<DENS).astype(float); W=M*g.normal(0,1,(N,N)); np.fill_diagonal(W,0); return rescale(W,RHO)

def W_smallworld(g):
    k=int(DENS*N); A=np.zeros((N,N))
    for i in range(N):
        for j in range(1,k//2+1): A[i,(i+j)%N]=1; A[i,(i-j)%N]=1
    # rewire 15%
    for i in range(N):
        for j in range(N):
            if A[i,j] and g.random()<0.15: A[i,j]=0; A[i,g.integers(N)]=1
    W=A*g.normal(0,1,(N,N)); np.fill_diagonal(W,0); return rescale(W,RHO)

def W_modular(g, nmod=6):
    A=np.zeros((N,N)); sz=N//nmod
    for m in range(nmod):
        idx=np.arange(m*sz,(m+1)*sz if m<nmod-1 else N)
        for i in idx:
            for j in idx:
                if i!=j and g.random()<DENS*4: A[i,j]=1     # denso dentro do modulo
    inter=(g.random((N,N))<DENS*0.3); A=np.maximum(A,inter)   # esparso entre modulos
    W=A*g.normal(0,1,(N,N)); np.fill_diagonal(W,0); return rescale(W,RHO)

def W_brainlike(g, nmod=6, nhub=18):
    A=np.zeros((N,N)); sz=N//nmod
    for m in range(nmod):
        idx=np.arange(m*sz,(m+1)*sz if m<nmod-1 else N)
        for i in idx:
            for j in idx:
                if i!=j and g.random()<DENS*4: A[i,j]=1
    inter=(g.random((N,N))<DENS*0.2); A=np.maximum(A,inter)
    hubs=g.choice(N,nhub,replace=False)                       # rich-club: hubs densamente interligados
    for i in hubs:
        for j in hubs:
            if i!=j: A[i,j]=1
        A[i,g.choice(N,int(N*DENS*2),replace=False)]=1        # hubs com alto grau
    W=A*np.abs(g.normal(0,1,(N,N)))                            # magnitudes positivas; sinal vem de Dale
    inh=g.choice(N,int(0.2*N),replace=False); sign=np.ones(N); sign[inh]=-1   # Dale E/I 80/20
    W=W*sign[np.newaxis,:]; np.fill_diagonal(W,0); return rescale(W,RHO)

def run_reservoir(W, u):
    Win=rng.normal(0,1,(N,))*0.5
    X=np.zeros((len(u),N)); x=np.zeros(N)
    for t in range(len(u)):
        x=np.tanh(W@x + Win*u[t]); X[t]=x
    return X

def ridge_fit(X,Y,lam=1e-3):
    A=X.T@X+lam*np.eye(X.shape[1]); return np.linalg.solve(A, X.T@Y)

def memory_capacity(W):
    u=rng.uniform(-1,1,T); X=run_reservoir(W,u); mc=0.0
    for k in range(1,41):
        Xtr=X[WASH:TR]; ytr=u[WASH-k:TR-k]; w=ridge_fit(Xtr,ytr)
        Xte=X[TR:T-0]; yte=u[TR-k:T-k]; pred=Xte@w
        c=np.corrcoef(pred,yte)[0,1]; mc+=(c*c if np.isfinite(c) else 0)
    return mc

def narma(W):
    u=rng.uniform(0,0.5,T); y=np.zeros(T)
    for t in range(10,T):
        y[t]=0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-10:t])+1.5*u[t-10]*u[t-1]+0.1
    X=run_reservoir(W,u)
    Xtr=X[WASH:TR]; w=ridge_fit(Xtr,y[WASH:TR]); pred=X[TR:]@w; tgt=y[TR:]
    nrmse=np.sqrt(np.mean((pred-tgt)**2))/(np.std(tgt)+1e-9); return nrmse

def main():
    t0=time.time()
    print("="*74); print(f"H97 Conectoma como fiacao do reservatorio (N={N}, raio={RHO}, mesmo p/ todos)")
    conds=[("RANDOM",W_random),("SMALLWORLD",W_smallworld),("MODULAR",W_modular),("BRAINLIKE(mod+richclub+E/I)",W_brainlike)]
    print(f"\n  {'topologia':<30} {'mem.capacity':>13} {'NARMA NRMSE':>13}  (MC alto=bom; NRMSE baixo=bom)")
    base_mc=None; res={}
    for name,fn in conds:
        mcs=[]; nrs=[]
        for seed in range(3):                                  # 3 sementes p/ media honesta
            g=np.random.default_rng(100+seed); W=fn(g)
            mcs.append(memory_capacity(W)); nrs.append(narma(W))
        mc=np.mean(mcs); nr=np.mean(nrs); res[name]=(mc,nr)
        print(f"  {name:<30} {mc:>13.2f} {nr:>13.3f}")
    print("="*74)
    rmc,rnr=res["RANDOM"]; bmc,bnr=res["BRAINLIKE(mod+richclub+E/I)"]
    print(f"H97 — {time.time()-t0:.0f}s | BRAINLIKE vs RANDOM: dMC {bmc-rmc:+.2f} | dNRMSE {bnr-rnr:+.3f} (negativo=melhor)")
    win = (bmc>rmc*1.05) or (bnr<rnr*0.95)
    print(f"  VEREDITO: {'OURO no conectoma — topologia biologica computa melhor que aleatoria' if win else 'topologia ~ empatou com aleatoria nesta tarefa (medido honesto)'}.")
    print("  Honesto: topologia stats-matched de mamifero (nao a fiacao literal baixada); 3 sementes;")
    print("  mesma densidade e raio espectral p/ isolar SO o efeito da estrutura. Reservatorio computa; nao e AGI.")

if __name__=="__main__": main()
