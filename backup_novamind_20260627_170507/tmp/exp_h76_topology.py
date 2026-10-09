"""
H76 — Tese FlyGM (do Resumo Executivo): topologia estruturada > aleatoria em eficiencia de amostra?
====================================================================================================
O documento ancora-se no FlyGM: uma rede com a TOPOLOGIA do conectoma aprende com MENOS amostras
que grafos aleatorios. Testamos a tese (proxy honesto — sem o grafo exato do FlyWire): uma RNN
recorrente cuja MASCARA de conectividade e (a) ESTRUTURADA (modular + small-world: clusters locais
+ poucos hubs/long-range, estilo conectoma), (b) ALEATORIA (Erdos-Renyi, MESMA densidade), (c) DENSA.
Mesma capacidade/treino. Tarefa: memoria (lembrar a pista por T passos, = H19 supervisionado).
Metrica: EFICIENCIA DE AMOSTRA (acuracia vs nº de updates). CPU. Honesto: proxy da tese FlyGM,
nao o grafo real; mede se ESTRUTURA (vs aleatorio, mesma densidade) ajuda. Nao e ARC, nao e AGI.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cpu"  # pequeno; deixa a GPU p/ os jobs de ARC
N=64; DENS=0.15; T=6

def mask_structured(n,dens):
    """modular (4 comunidades, denso intra) + small-world (anel local) + poucos hubs long-range."""
    m=np.zeros((n,n)); k=n//4
    for c in range(4):                       # comunidades densas
        idx=range(c*k,(c+1)*k)
        for i in idx:
            for j in idx:
                if i!=j and np.random.rand()<0.5: m[i,j]=1
    for i in range(n):                        # anel local (small-world)
        m[i,(i+1)%n]=1; m[i,(i-1)%n]=1
    hubs=np.random.choice(n,4,replace=False)  # rich-club / hubs long-range
    for h in hubs:
        for j in np.random.choice(n,n//5,replace=False): m[h,j]=1; m[j,h]=1
    # ajusta densidade alvo
    return _match_density(m,dens)
def mask_random(n,dens):
    m=(np.random.rand(n,n)<dens).astype(float); np.fill_diagonal(m,0); return m
def _match_density(m,dens):
    cur=m.sum()/(m.shape[0]**2);
    if cur>dens:                              # remove arestas extras aleatorias
        on=np.argwhere(m>0); drop=on[np.random.choice(len(on),int((cur-dens)*m.size),replace=False)]
        for i,j in drop: m[i,j]=0
    return m

class MaskedRNN(nn.Module):
    def __init__(s,mask):
        super().__init__(); s.mask=torch.tensor(mask,dtype=torch.float32,device=DEV)
        s.Wx=nn.Linear(3,N); s.Wh=nn.Linear(N,N,bias=False); s.out=nn.Linear(N,2)
    def forward(s,x):  # x: (B,T,3)
        h=torch.zeros(x.shape[0],N,device=DEV)
        for t in range(x.shape[1]):
            Wh=s.Wh.weight*s.mask                 # aplica a topologia (mascara fixa)
            h=torch.tanh(s.Wx(x[:,t])+h@Wh.t())
        return s.out(h)

def make_batch(B=128):
    cue=np.random.randint(0,2,B)
    x=np.zeros((B,T,3),np.float32)
    x[np.arange(B),0,cue]=1            # pista em t=0 nos canais 0/1
    x[:, 1:,2]=1                        # canal 2 = "espera" (pista some)
    return torch.tensor(x,device=DEV), torch.tensor(cue,device=DEV,dtype=torch.long)

def train_curve(mask, seeds=4, updates=400):
    curves=[]
    for sd in range(seeds):
        torch.manual_seed(sd); net=MaskedRNN(mask).to(DEV); opt=torch.optim.Adam(net.parameters(),5e-3)
        accs=[]
        for u in range(updates):
            x,y=make_batch(); loss=F.cross_entropy(net(x),y); opt.zero_grad(); loss.backward(); opt.step()
            if u%40==0:
                with torch.no_grad():
                    xt,yt=make_batch(512); accs.append((net(xt).argmax(1)==yt).float().mean().item())
        curves.append(accs)
    return np.array(curves).mean(0)

def main():
    print("="*70); print("H76 — tese FlyGM (proxy): topologia ESTRUTURADA vs ALEATORIA vs DENSA")
    print(f"  N={N} neuronios | densidade alvo={DENS} | tarefa memoria T={T} | media de 4 seeds\n")
    ms=mask_structured(N,DENS); mr=mask_random(N,DENS); md=np.ones((N,N))-np.eye(N)
    print(f"  densidades reais: estruturada={ms.mean():.2f} aleatoria={mr.mean():.2f} densa={md.mean():.2f}")
    cs=train_curve(ms); cr=train_curve(mr); cd=train_curve(md)
    xs=list(range(0,400,40))
    print(f"\n  acuracia vs updates (eficiencia de amostra; MAIOR/mais-cedo=melhor):")
    print(f"    update:      "+" ".join(f"{u:>5}" for u in xs))
    print(f"    ESTRUTURADA: "+" ".join(f"{v:>5.2f}" for v in cs))
    print(f"    ALEATORIA:   "+" ".join(f"{v:>5.2f}" for v in cr))
    print(f"    DENSA:       "+" ".join(f"{v:>5.2f}" for v in cd))
    # area sob a curva = eficiencia de amostra agregada
    print(f"\n  AUC (eficiencia agregada): estruturada={cs.mean():.3f} aleatoria={cr.mean():.3f} densa={cd.mean():.3f}")
    win = cs.mean()>cr.mean()+0.02
    print(f"  => {'ESTRUTURA vence aleatorio (suporta FlyGM)' if win else 'sem vantagem clara da estrutura vs aleatorio (mesma densidade)'}.")
    print(f"  Honesto: proxy (sem grafo FlyWire real). Tarefa de memoria, narrow. Nao e ARC, nao e AGI.")

if __name__=="__main__": main()
