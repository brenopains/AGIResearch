"""
H65 — Corpo Cogumelo (MB): expansao aleatoria esparsa contra esquecimento catastrofico
========================================================================================
Hipotese (validada na literatura): os neuronios de Kenyon do MB recebem projecao ALEATORIA
fixa de alta dimensao + inibicao global (APL) -> codigo ESPARSO e DECORRELACIONADO -> novas
memorias nao interferem nas antigas. Refs: Caron 2013 (Nature, convergencia aleatoria),
Lin 2014 / Turner 2008 / Honegger 2011 (sparse coding reduz sobreposicao). = Dasgupta-
Stevens-Navlakha (LSH). Ataca um problema REAL nosso (H1/H10: esquecimento catastrofico).

Teste de APRENDIZADO CONTINUO: aprende 3 grupos de classes em sequencia (A, depois B, depois
C), SEM replay, e mede a retencao do PRIMEIRO grupo (A) ao final. Compara:
  (1) MB: projecao aleatoria FIXA -> top-k WTA (esparso) -> readout linear treinavel;
  (2) MLP denso de capacidade similar (o baseline que esquece).
GPU acelera a expansao (matmul grande). Honesto: vitoria narrow real se MB >> denso; nao e AGI.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

# ---------- dados sinteticos: 12 classes, vetores gaussianos separaveis (input dim=50) ----------
DIN=50; NCLS=12; PERCLS=300
centers=torch.randn(NCLS,DIN)*2.5
def make(cls_list,n=PERCLS):
    X=[]; Y=[]
    for c in cls_list:
        X.append(centers[c]+torch.randn(n,DIN)*1.0); Y+=[c]*n
    return torch.cat(X).to(DEVICE), torch.tensor(Y,device=DEVICE)
GROUPS=[[0,1,2,3],[4,5,6,7],[8,9,10,11]]    # A, B, C (4 classes cada)

# ---------- (1) Mushroom Body: random expansion + top-k WTA ----------
class MB(nn.Module):
    def __init__(s,din=DIN,dhigh=2000,k=40,ncls=NCLS):
        super().__init__()
        s.W=torch.randn(din,dhigh,device=DEVICE)/np.sqrt(din)   # projecao ALEATORIA FIXA (Kenyon)
        s.W.requires_grad=False
        s.k=k; s.readout=nn.Linear(dhigh,ncls).to(DEVICE)        # so o readout aprende
    def expand(s,x):
        h=x@s.W                                                  # expansao
        thr=torch.topk(h,s.k,dim=1).values[:,-1:]                # top-k WTA (inibicao APL)
        return (h*(h>=thr)).clamp(min=0)                         # codigo esparso
    def forward(s,x): return s.readout(s.expand(x))

# ---------- (2) MLP denso (baseline que esquece) ----------
class MLP(nn.Module):
    def __init__(s,din=DIN,h=256,ncls=NCLS):
        super().__init__(); s.net=nn.Sequential(nn.Linear(din,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,ncls)).to(DEVICE)
    def forward(s,x): return s.net(x)

def train_on(model,X,Y,steps=300,lr=1e-2):
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr)
    model.train()
    for _ in range(steps):
        idx=torch.randint(0,X.shape[0],(128,),device=DEVICE)
        loss=F.cross_entropy(model(X[idx]),Y[idx]); opt.zero_grad(); loss.backward(); opt.step()

def acc_on(model,cls_list):
    model.eval(); X,Y=make(cls_list,120)
    with torch.no_grad(): return (model(X).argmax(1)==Y).float().mean().item()

def continual(model,name):
    accA_after=[]
    for gi,grp in enumerate(GROUPS):
        X,Y=make(grp); train_on(model,X,Y)
        accA_after.append(acc_on(model,GROUPS[0]))   # retencao de A apos cada grupo
    final_all=np.mean([acc_on(model,g) for g in GROUPS])
    print(f"  {name:>12}: retencao de A apos [A,B,C] = {[f'{a:.0%}' for a in accA_after]} | media TODAS as classes no fim = {final_all:.0%}")
    return accA_after[-1]

def main():
    t0=time.time()
    print("="*70); print(f"H65 Corpo Cogumelo — anti-esquecimento (aprendizado continuo A->B->C) em {DEVICE}")
    print("  (sem replay; mede quanto do 1o grupo 'A' sobrevive apos aprender B e C)\n")
    rA_mb=continual(MB(),"MB-esparso")
    rA_mlp=continual(MLP(),"MLP-denso")
    vram=torch.cuda.max_memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print("-"*70)
    print(f"  RETENCAO FINAL de A: MB-esparso {rA_mb:.0%}  vs  MLP-denso {rA_mlp:.0%}")
    print(f"  {'MB VENCE' if rA_mb>rA_mlp+0.1 else 'sem diferenca clara'}: expansao aleatoria esparsa {'reduz' if rA_mb>rA_mlp+0.1 else 'nao reduz'} o esquecimento por design.")
    print(f"  tempo {time.time()-t0:.1f}s | VRAM {vram:.0f}MB | Honesto: vitoria narrow (continual learning); NAO e ARC, NAO e AGI.")

if __name__=="__main__": main()
