"""
H94 — Estrutura de cerebro real na ALife: a emergencia fica MAIOR que a fagulha?
=================================================================================
A intuicao (certa) do usuario: parar de catalogar primitivas no ARC (esteira infinita) e usar
a ESTRUTURA dos cerebros reais (humano/rato/macaco/mosca) p/ enriquecer o cerebro dos agentes
da ALife — e ver se a emergencia social (linguagem) fica MAIOR.

Medida honesta de "maior que uma fagulha": COMPOSICIONALIDADE + GENERALIZACAO ZERO-SHOT.
Sinalizacao simples (H81) so precisa de um codigo arbitrario. Linguagem DE VERDADE generaliza:
significados = COMBINACOES de atributos (a,b); a linguagem composicional reusa partes -> acerta
combinacoes NUNCA VISTAS. Isso e o salto qualitativo (Kirby; iterated learning).

A/B de ARQUITETURA (principios reais de conectoma, nao copia de fiacao):
  FLAT      : MLP cheio (H=24) — cerebro "entrelacado" (como a tabela do H81, com capacidade).
  BOTTLE    : MLP com GARGALO (H=4) — codificacao eficiente/restricao metabolica (a "compressao"
              do MSIF). Hipotese: compressao FAVORECE codigo composicional (mais compressivel).
  BOTTLE+NM : gargalo + NEUROMODULACAO (temperatura que anneal de alta->baixa) — instabilidade
              controlada / "estado interno difuso" (MSIF/Zealot): explora cedo, consolida depois.
Mede por condicao: acerto (visto), acerto ZERO-SHOT (combinacoes ocultas), e composicionalidade
(rho topografico = corr entre distancia-de-significado e distancia-de-mensagem). Vetorizado, GPU.
Honesto: se BOTTLE/NM > FLAT em zero-shot+rho, estrutura de cerebro AMPLIA a emergencia (sinal
real). Se nao, a estrutura nao foi o gargalo aqui — e medimos isso de forma limpa.
"""
import os, time, numpy as np, torch
from scipy.stats import spearmanr
DEV="cuda" if torch.cuda.is_available() else "cpu"
P=1536; K=5; V=8; SLOT=2; GEN=2500; LOG=250; MUT=0.10; ELITE=P//20
NMEAN=K*K
torch.manual_seed(0)

# significados (a,b); separa held-out p/ zero-shot
allm=torch.tensor([(a,b) for a in range(K) for b in range(K)],device=DEV)
rng=np.random.default_rng(0); perm=rng.permutation(NMEAN)
held=torch.tensor(sorted(perm[:5].tolist()),device=DEV)            # 5 combinacoes ocultas
train_idx=torch.tensor(sorted(perm[5:].tolist()),device=DEV)
heldset=set(held.tolist())

def onehot(idx,n):
    o=torch.zeros(idx.shape[0],n,device=DEV); o[torch.arange(idx.shape[0],device=DEV),idx]=1; return o

def init_pop(H,seed):
    g=torch.Generator(device=DEV).manual_seed(seed)
    # falante: in=2K -> H -> 2V ; ouvinte: in=2V -> H -> 2K
    s={'sW1':torch.randn(P,2*K,H,generator=g,device=DEV)*0.5,'sW2':torch.randn(P,H,SLOT*V,generator=g,device=DEV)*0.5,
       'lW1':torch.randn(P,SLOT*V,H,generator=g,device=DEV)*0.5,'lW2':torch.randn(P,H,SLOT*K,generator=g,device=DEV)*0.5,'H':H}
    return s

def speak(s, mvec, temp=0.0, gen=None):
    a=onehot(mvec[:,0],K); b=onehot(mvec[:,1],K); x=torch.cat([a,b],1)            # (P,2K)
    h=torch.tanh(torch.bmm(x.unsqueeze(1),s['sW1']).squeeze(1))                    # (P,H)
    o=torch.bmm(h.unsqueeze(1),s['sW2']).squeeze(1).view(P,SLOT,V)                 # (P,SLOT,V)
    if temp>0:
        pr=torch.softmax(o/temp,2); sym=torch.multinomial(pr.view(P*SLOT,V),1,generator=gen).view(P,SLOT)
    else: sym=o.argmax(2)
    return sym

def listen(s, sym):
    x=torch.cat([onehot(sym[:,0],V),onehot(sym[:,1],V)],1)                         # (P,2V)
    h=torch.tanh(torch.bmm(x.unsqueeze(1),s['lW1']).squeeze(1))
    o=torch.bmm(h.unsqueeze(1),s['lW2']).squeeze(1).view(P,SLOT,K)
    return o.argmax(2)                                                             # (P,SLOT)

def run(H, neuromod, label, seed=1):
    s=init_pop(H,seed); g=torch.Generator(device=DEV).manual_seed(seed+7)
    fit=torch.zeros(P,device=DEV)
    for gen in range(GEN):
        mi=train_idx[torch.randint(0,len(train_idx),(P,),device=DEV,generator=g)]
        mvec=allm[mi]
        temp=max(0.05,1.0-gen/GEN)*1.0 if neuromod else 0.0
        sym=speak(s,mvec,temp,g)
        partner=torch.randperm(P,device=DEV,generator=g)
        guess=listen({k:(s[k][partner] if k!='H' else s[k]) for k in s},sym)
        correct=((guess[:,0]==mvec[:,0])&(guess[:,1]==mvec[:,1])).float()
        fit=fit*0.9+correct; fit.index_add_(0,partner,correct*0.9)
        # Moran: piores <- filhos mutados dos melhores
        worst=torch.topk(fit,ELITE,largest=False).indices; best=torch.topk(fit,ELITE,largest=True).indices
        par=best[torch.randint(0,ELITE,(ELITE,),device=DEV,generator=g)]
        for w in ('sW1','sW2','lW1','lW2'):
            s[w][worst]=s[w][par]+torch.randn_like(s[w][worst])*MUT
        fit[worst]=0.0
    # avaliacao com o MELHOR agente como falante e populacao como ouvinte (codigo dominante)
    return eval_pop(s,fit,label)

def eval_pop(s, fit, label):
    bi=int(fit.argmax().item())
    one={k:(s[k][bi:bi+1] if k!='H' else s[k]) for k in s}
    def msg_for(mlist):
        out=[]
        for (a,b) in mlist:
            mv=torch.tensor([[a,b]],device=DEV)
            # speaker do melhor (P=1): replica forward com P=1
            x=torch.cat([onehot(mv[:,0],K),onehot(mv[:,1],K)],1)
            h=torch.tanh(x@one['sW1'][0]); o=(h@one['sW2'][0]).view(SLOT,V); out.append(tuple(o.argmax(1).tolist()))
        return out
    # acerto medio da POPULACAO (falante=cada um, ouvinte=parceiro) em vistos e ocultos
    def pop_acc(idxset):
        g=torch.Generator(device=DEV).manual_seed(123)
        mi=idxset[torch.randint(0,len(idxset),(P,),device=DEV,generator=g)]; mvec=allm[mi]
        sym=speak(s,mvec,0.0); partner=torch.randperm(P,device=DEV,generator=g)
        guess=listen({k:(s[k][partner] if k!='H' else s[k]) for k in s},sym)
        return float(((guess[:,0]==mvec[:,0])&(guess[:,1]==mvec[:,1])).float().mean())
    acc_seen=pop_acc(train_idx); acc_zero=pop_acc(held)
    # composicionalidade (rho topografico) com o codigo do melhor falante, nos significados de treino
    tm=[tuple(x) for x in allm[train_idx].tolist()]; msgs=msg_for(tm)
    import itertools
    md=[]; sd=[]
    for i,j in itertools.combinations(range(len(tm)),2):
        md.append(abs(tm[i][0]-tm[j][0])+abs(tm[i][1]-tm[j][1]))       # dist significado (L1)
        sd.append((msgs[i][0]!=msgs[j][0])+(msgs[i][1]!=msgs[j][1]))    # dist mensagem (Hamming)
    rho=float(spearmanr(md,sd).correlation) if len(set(sd))>1 else 0.0
    return {'label':label,'acc_seen':acc_seen,'acc_zero':acc_zero,'rho':rho}

def main():
    t0=time.time()
    print("="*74)
    print(f"H94 Cerebro estruturado na ALife (linguagem COMPOSICIONAL) em {DEV}")
    print(f"  significados={NMEAN} (a,b); ocultos p/ zero-shot={held.tolist()}; vocab={V}, slots={SLOT}; acaso~{1/NMEAN:.3f}")
    res=[]
    res.append(run(24,False,"FLAT (H=24, entrelacado)"))
    res.append(run(4, False,"BOTTLE (H=4, codificacao eficiente)"))
    res.append(run(4, True ,"BOTTLE+NM (gargalo + neuromodulacao)"))
    print(f"\n  {'condicao':<36} {'acerto-visto':>12} {'ZERO-SHOT':>10} {'composic.(rho)':>14}")
    for r in res:
        print(f"  {r['label']:<36} {r['acc_seen']:>12.2f} {r['acc_zero']:>10.2f} {r['rho']:>14.2f}")
    flat=res[0]; best=max(res[1:],key=lambda r:(r['acc_zero'],r['rho']))
    print("="*74)
    gain_zero=best['acc_zero']-flat['acc_zero']; gain_rho=best['rho']-flat['rho']
    print(f"H94 — {time.time()-t0:.0f}s. Melhor estruturado = {best['label']}")
    print(f"  delta vs FLAT: zero-shot {gain_zero:+.2f} | composicionalidade {gain_rho:+.2f}")
    if gain_zero>0.05 or gain_rho>0.1:
        print("  VEREDITO: estrutura de cerebro AMPLIOU a emergencia (linguagem mais composicional/")
        print("  generalizante). Sinal real de que a arquitetura biologica ajuda — alem da fagulha.")
    else:
        print("  VEREDITO: estrutura nao superou o cerebro plano AQUI (medido honesto). A composicio-")
        print("  nalidade depende mais da PRESSAO da tarefa que da arquitetura, neste setup.")
    print("  Honesto: proto-linguagem composicional emergente medida por zero-shot+rho; NAO e")
    print("  linguagem humana. Mas zero-shot>0 ja e generalizacao REAL fora do visto. Nao e AGI.")

if __name__=="__main__": main()
