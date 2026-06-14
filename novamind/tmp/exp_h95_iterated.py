"""
H95 — Iterated Learning: a transmissao cultural FORCA linguagem composicional (Kirby)
======================================================================================
Lição do H94: composicionalidade nao vem da fiacao do cerebro — vem da PRESSAO DE TRANSMISSAO.
Kirby (2008); Ren et al. 2020 ("Compositional languages emerge in a neural iterated learning
model"): quando uma lingua precisa ser REAPRENDIDA por um novo agente que so ve um SUBCONJUNTO
(gargalo) das associacoes significado->sinal, e depois repassada, a selecao cultural favorece
codigos SISTEMATICOS (reutilizaveis) — pois so o que e compressivel/generalizavel sobrevive ao
gargalo. Resultado: a lingua vira composicional ao longo das geracoes e GENERALIZA p/ o nunca-visto.

Mecanismo (gradiente, converge — diferente da evolucao pura do H94):
  Significado = (a,b), a,b in {0..K-1}. Sinal = 2 simbolos de vocab V.
  Cada geracao:
    (A) APRENDIZADO c/ GARGALO: novo falante (rede) imita a lingua do professor, mas SO nos
        significados do gargalo (subconjunto). Aprende por CE. (pressao de aprendibilidade)
    (B) EXPRESSIVIDADE: falante+ouvinte jogam jogo referencial em TODOS os significados (Gumbel-
        Softmax), p/ os sinais continuarem distinguiveis (evita colapso degenerado).
    -> nova lingua = argmax do falante em todos os significados; vira professor da proxima.
  A/B: COM gargalo (transmissao parcial) vs SEM gargalo (ensina tudo). Previsao: so COM gargalo
  a composicionalidade (rho topografico) sobe e a generalizacao ZERO-SHOT aparece.
Mede por geracao: rho topografico, expressividade (#sinais distintos), zero-shot (ouvinte treinado
so nos vistos decodifica os significados OCULTOS?). GPU. Honesto: proto-linguagem; nao e humana.
"""
import os, time, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
K=5; V=6; SLOT=2; NMEAN=K*K; GENS=30; HID=48
MEAN=torch.tensor([(a,b) for a in range(K) for b in range(K)],device=DEV)
rng=np.random.default_rng(0); perm=rng.permutation(NMEAN)
HELD=sorted(perm[:5].tolist()); SEEN=sorted(perm[5:].tolist())   # ocultos p/ zero-shot

def oh(idx,n):
    o=torch.zeros(idx.shape[0],n,device=DEV); o[torch.arange(idx.shape[0],device=DEV),idx]=1; return o
def m_in(idxs):
    mv=MEAN[idxs]; return torch.cat([oh(mv[:,0],K),oh(mv[:,1],K)],1)

class Speaker(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*K,HID),nn.ReLU(),nn.Linear(HID,SLOT*V))
    def forward(s,x): return s.net(x).view(-1,SLOT,V)
class Listener(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(SLOT*V,HID),nn.ReLU(),nn.Linear(HID,SLOT*K))
    def forward(s,x): return s.net(x).view(-1,SLOT,K)

def lang_of(spk):
    with torch.no_grad():
        idx=torch.arange(NMEAN,device=DEV); sig=spk(m_in(idx)).argmax(2)   # (NMEAN,SLOT)
    return sig

def imitate(spk, teacher_sig, bottleneck_idx, steps=200):
    opt=torch.optim.Adam(spk.parameters(),3e-3)
    bi=torch.tensor(bottleneck_idx,device=DEV); tgt=teacher_sig[bi]
    for _ in range(steps):
        log=spk(m_in(bi)); loss=F.cross_entropy(log.reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

def communicate(spk, lis, steps=300, tau=1.0):
    opt=torch.optim.Adam(list(spk.parameters())+list(lis.parameters()),3e-3)
    idx=torch.arange(NMEAN,device=DEV); mv=MEAN
    for t in range(steps):
        log=spk(m_in(idx))                                   # (N,SLOT,V)
        soft=F.gumbel_softmax(log,tau=max(0.5,tau*(1-t/steps)),hard=True,dim=2)  # (N,SLOT,V)
        lin=soft.reshape(NMEAN,SLOT*V); pred=lis(lin)        # (N,SLOT,K)
        loss=F.cross_entropy(pred.reshape(-1,K),mv.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

def topo_rho(sig):
    s=sig.cpu().numpy(); m=MEAN.cpu().numpy(); md=[]; sd=[]
    for i,j in itertools.combinations(range(NMEAN),2):
        md.append(abs(m[i,0]-m[j,0])+abs(m[i,1]-m[j,1]))
        sd.append(int(s[i,0]!=s[j,0])+int(s[i,1]!=s[j,1]))
    return float(spearmanr(md,sd).correlation) if len(set(sd))>1 else 0.0

def expressivity(sig):
    s={tuple(r) for r in sig.cpu().tolist()}; return len(s)/NMEAN

def zero_shot(sig):
    # treina um ouvinte FRESCO so nos sinais dos SEEN; testa decodificar os HELD (nunca vistos)
    lis=Listener().to(DEV); opt=torch.optim.Adam(lis.parameters(),5e-3)
    si=torch.tensor(SEEN,device=DEV); hi=torch.tensor(HELD,device=DEV)
    xin=torch.cat([oh(sig[si,0],V),oh(sig[si,1],V)],1); ytr=MEAN[si]
    for _ in range(300):
        pred=lis(xin); loss=F.cross_entropy(pred.reshape(-1,K),ytr.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh=torch.cat([oh(sig[hi,0],V),oh(sig[hi,1],V)],1); ph=lis(xh).argmax(2)
        acc=float(((ph[:,0]==MEAN[hi,0])&(ph[:,1]==MEAN[hi,1])).float().mean())
    return acc

def run(bottleneck, label):
    torch.manual_seed(1)
    spk=Speaker().to(DEV); sig=lang_of(spk)                  # lingua inicial (aleatoria)
    bn_size=int(round(bottleneck*NMEAN))
    hist=[]
    for gen in range(GENS):
        bidx=SEEN if bottleneck>=0.999 else list(np.random.default_rng(100+gen).choice(SEEN,size=min(bn_size,len(SEEN)),replace=False))
        learner=Speaker().to(DEV)
        imitate(learner,sig,bidx)                            # (A) gargalo
        lis=Listener().to(DEV); communicate(learner,lis)     # (B) expressividade
        sig=lang_of(learner); spk=learner
        if gen%5==0 or gen==GENS-1:
            hist.append((gen,topo_rho(sig),expressivity(sig),zero_shot(sig)))
    return hist

def main():
    t0=time.time()
    print("="*74)
    print(f"H95 Iterated Learning (transmissao cultural; Kirby) em {DEV}")
    print(f"  significados={NMEAN}; ocultos(zero-shot)={HELD}; vocab={V}, slots={SLOT}; acaso zero-shot~{1/(K*K):.3f}")
    for bn,lab in [(0.6,"COM gargalo (ve 60% das assoc.)"),(1.0,"SEM gargalo (ve tudo) — controle")]:
        h=run(bn,lab)
        print(f"\n  {lab}:")
        print(f"    {'gen':>4} {'rho(composic)':>14} {'express.':>9} {'zero-shot':>10}")
        for g,r,e,z in h: print(f"    {g:>4} {r:>14.2f} {e:>9.2f} {z:>10.2f}")
    print("="*74)
    print(f"H95 — {time.time()-t0:.0f}s. Leitura: se COM gargalo o rho SOBE e o zero-shot fica >> acaso,")
    print("  enquanto SEM gargalo NAO, entao a TRANSMISSAO CULTURAL (pressao externa) gerou linguagem")
    print("  composicional que GENERALIZA — o salto 'maior que a fagulha'. Honesto: proto-linguagem, nao humana.")

if __name__=="__main__": main()
