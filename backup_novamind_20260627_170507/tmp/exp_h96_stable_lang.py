"""
H96 — Iterated Learning ESTABILIZADO: linguagem composicional robusta (platô estável)
======================================================================================
H95 mostrou o sinal (rho 0,10->0,58; zero-shot 5x acaso) mas INSTAVEL e pouco expressivo
(colisoes: so ~6-8 sinais distintos p/ 25 significados), porque a pressao de EXPRESSIVIDADE
(jogo referencial Gumbel) era fraca. Correcoes p/ firmar o plato composicional:
  - vocab MAIOR (V=10) -> 100 sinais possiveis p/ 25 significados (espaco p/ distincao);
  - fase de comunicacao mais LONGA + tau recozido (2.0->0.4) -> sinais discretos limpos e distintos;
  - imitacao mais longa (gargalo bem aprendido); mais geracoes (40) p/ ver o plato;
  - metrica de expressividade alvo ~1,0 (todos distintos) = linguagem REALMENTE expressiva.
A/B mantido: COM gargalo (transmissao parcial) vs SEM. Previsao: COM gargalo -> rho ALTO e
ESTAVEL + zero-shot alto + expressividade ~1,0; SEM -> nada disso. GPU. Honesto: proto-linguagem.
"""
import os, time, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
K=5; V=10; SLOT=2; NMEAN=K*K; GENS=40; HID=64
MEAN=torch.tensor([(a,b) for a in range(K) for b in range(K)],device=DEV)
perm=np.random.default_rng(0).permutation(NMEAN)
HELD=sorted(perm[:5].tolist()); SEEN=sorted(perm[5:].tolist())

def oh(idx,n):
    o=torch.zeros(idx.shape[0],n,device=DEV); o[torch.arange(idx.shape[0],device=DEV),idx]=1; return o
def m_in(idxs): mv=MEAN[idxs]; return torch.cat([oh(mv[:,0],K),oh(mv[:,1],K)],1)

class Speaker(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*K,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,SLOT*V))
    def forward(s,x): return s.net(x).view(-1,SLOT,V)
class Listener(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(SLOT*V,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,SLOT*K))
    def forward(s,x): return s.net(x).view(-1,SLOT,K)

def lang_of(spk):
    with torch.no_grad(): return spk(m_in(torch.arange(NMEAN,device=DEV))).argmax(2)

def imitate(spk, teacher_sig, bidx, steps=400):
    opt=torch.optim.Adam(spk.parameters(),2e-3); bi=torch.tensor(bidx,device=DEV); tgt=teacher_sig[bi]
    for _ in range(steps):
        log=spk(m_in(bi)); loss=F.cross_entropy(log.reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

def communicate(spk, lis, steps=800):
    opt=torch.optim.Adam(list(spk.parameters())+list(lis.parameters()),2e-3)
    idx=torch.arange(NMEAN,device=DEV); mv=MEAN
    for t in range(steps):
        tau=max(0.4,2.0*(1-t/steps))
        log=spk(m_in(idx)); soft=F.gumbel_softmax(log,tau=tau,hard=True,dim=2)
        pred=lis(soft.reshape(NMEAN,SLOT*V)); loss=F.cross_entropy(pred.reshape(-1,K),mv.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

def topo_rho(sig):
    s=sig.cpu().numpy(); m=MEAN.cpu().numpy(); md=[]; sd=[]
    for i,j in itertools.combinations(range(NMEAN),2):
        md.append(abs(m[i,0]-m[j,0])+abs(m[i,1]-m[j,1])); sd.append(int(s[i,0]!=s[j,0])+int(s[i,1]!=s[j,1]))
    return float(spearmanr(md,sd).correlation) if len(set(sd))>1 else 0.0
def expressivity(sig): return len({tuple(r) for r in sig.cpu().tolist()})/NMEAN
def zero_shot(sig):
    lis=Listener().to(DEV); opt=torch.optim.Adam(lis.parameters(),5e-3)
    si=torch.tensor(SEEN,device=DEV); hi=torch.tensor(HELD,device=DEV)
    xin=torch.cat([oh(sig[si,0],V),oh(sig[si,1],V)],1); ytr=MEAN[si]
    for _ in range(400):
        loss=F.cross_entropy(lis(xin).reshape(-1,K),ytr.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh=torch.cat([oh(sig[hi,0],V),oh(sig[hi,1],V)],1); ph=lis(xh).argmax(2)
        return float(((ph[:,0]==MEAN[hi,0])&(ph[:,1]==MEAN[hi,1])).float().mean())

def run(bottleneck, label):
    torch.manual_seed(1); spk=Speaker().to(DEV); sig=lang_of(spk); bn=int(round(bottleneck*len(SEEN))); hist=[]
    for gen in range(GENS):
        bidx=SEEN if bottleneck>=0.999 else list(np.random.default_rng(100+gen).choice(SEEN,size=min(bn,len(SEEN)),replace=False))
        learner=Speaker().to(DEV); imitate(learner,sig,bidx)
        lis=Listener().to(DEV); communicate(learner,lis); sig=lang_of(learner); spk=learner
        if gen%4==0 or gen==GENS-1: hist.append((gen,topo_rho(sig),expressivity(sig),zero_shot(sig)))
    return hist

def main():
    t0=time.time()
    print("="*74); print(f"H96 Iterated Learning ESTABILIZADO em {DEV} | V={V} (vs H95 V=6); GENS={GENS}")
    print(f"  significados={NMEAN}; ocultos(zero-shot)={HELD}; acaso~{1/(K*K):.3f}")
    results={}
    for bn,lab in [(0.6,"COM gargalo"),(1.0,"SEM gargalo (controle)")]:
        h=run(bn,lab); results[lab]=h
        print(f"\n  {lab}:"); print(f"    {'gen':>4} {'rho':>7} {'express':>8} {'zero-shot':>10}")
        for g,r,e,z in h: print(f"    {g:>4} {r:>7.2f} {e:>8.2f} {z:>10.2f}")
    cg=results["COM gargalo"]; last=cg[-len(cg)//3:]
    rho_m=np.mean([r for _,r,_,_ in last]); zs_m=np.mean([z for _,_,_,z in last]); ex_m=np.mean([e for _,_,e,_ in last])
    print("="*74)
    print(f"H96 — {time.time()-t0:.0f}s | COM gargalo, media do TERCO FINAL: rho {rho_m:.2f} | express {ex_m:.2f} | zero-shot {zs_m:.2f}")
    stable = rho_m>0.5 and zs_m>0.2 and ex_m>0.7
    print(f"  VEREDITO: {'PLATO COMPOSICIONAL ESTAVEL — linguagem expressiva que generaliza (fagulha firme)' if stable else 'melhorou mas ainda nao firmou plenamente; ajustar pressao/vocab'}.")
    print("  Honesto: proto-linguagem composicional emergente, medida por zero-shot+rho+expressividade. Nao e linguagem humana.")

if __name__=="__main__": main()
