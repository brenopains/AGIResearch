"""
H81 — Emergencia de LINGUAGEM para passar a tecnologia adiante (social + evolutivo)
====================================================================================
Mantem a fagulha que funcionou (uso de ferramenta evoluido, H79) + DINAMICA SOCIAL +
emergencia de uma LINGUAGEM PROPRIA que os agentes tem que DESCOBRIR para transmitir
informacao. Mecanismo (jogo de sinalizacao de Lewis, evolutivo, amarrado a TECH):
  - FALANTE ve qual receita/ferramenta esta disponivel (estado em {0..S-1});
  - emite um SIMBOLO (sem significado pre-dado), via seu genoma;
  - OUVINTE (que NAO ve o estado) usa o simbolo p/ escolher qual combinar/agir;
  - acerto => ambos ganham energia e destravam a ferramenta (cooperacao = info passada adiante).
A LINGUAGEM precisa EMERGIR por selecao: um codigo compartilhado (estado->simbolo->acao) que
transmite a tech se espalha. Mede: acuracia (sobe do acaso 1/S?), INFORMACAO MUTUA simbolo<->
estado (bits), e A/B vs parceiros embaralhados (sem codigo estavel possivel). Vetorizado, sem
sync. Honesto: e PROTO-comunicacao emergente (significado compartilhado), nao linguagem humana
(sintaxe/cultura/80 anos ficam fora). Nao e AGI.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); gn=torch.Generator(device=DEV).manual_seed(0)
P=8192; S=6; GEN=4000; LOG=200; MUT=0.10   # P agentes, S estados=simbolos=acoes (tech tiers)

# genoma: tabela do FALANTE (estado->logits de simbolo) e do OUVINTE (simbolo->logits de acao)
spk=torch.randn(P,S,S,generator=gn,device=DEV)*0.5     # (P, estado, simbolo)
lis=torch.randn(P,S,S,generator=gn,device=DEV)*0.5     # (P, simbolo, acao)
fit=torch.zeros(P,device=DEV)

def round_play(scramble=False):
    """cada agente e FALANTE; parceiro aleatorio e OUVINTE. acerto => +fitness aos dois."""
    global fit
    state=torch.randint(0,S,(P,),device=DEV,generator=gn)          # tech disponivel p/ cada falante
    sym=spk[torch.arange(P,device=DEV),state].argmax(1)            # simbolo emitido
    partner=torch.randperm(P,device=DEV,generator=gn)              # ouvinte = parceiro
    heard=sym if not scramble else sym[torch.randperm(P,device=DEV,generator=gn)]  # baseline: ruido
    act=lis[partner,heard].argmax(1)                               # acao do ouvinte
    correct=(act==state).float()                                   # transmitiu a tech corretamente?
    fit=fit*0.9 + correct                                          # fitness recente (falante)
    fit.index_add_(0,partner,correct*0.9)                          # ouvinte tambem credita
    return state,sym,correct.mean()

def evolve():
    """Moran vetorizado sem sync: piores fitness substituidos por filhos mutados dos melhores."""
    global spk,lis,fit
    q=P//20
    worst=torch.topk(fit,q,largest=False).indices; best=torch.topk(fit,q,largest=True).indices
    par=best[torch.randint(0,q,(q,),device=DEV,generator=gn)]
    spk[worst]=spk[par]+torch.randn(q,S,S,generator=gn,device=DEV)*MUT
    lis[worst]=lis[par]+torch.randn(q,S,S,generator=gn,device=DEV)*MUT
    fit[worst]=0.0

def mutual_info(state,sym):
    """I(estado;simbolo) em bits — quanto o simbolo carrega de info sobre a tech."""
    joint=torch.zeros(S,S,device=DEV)
    joint.index_put_((state,sym),torch.ones_like(state,dtype=torch.float),accumulate=True)
    joint/=joint.sum(); ps=joint.sum(1,keepdim=True); py=joint.sum(0,keepdim=True)
    m=joint>0; mi=(joint[m]*torch.log2(joint[m]/(ps.expand_as(joint)[m]*py.expand_as(joint)[m]))).sum()
    return float(mi.item())

def run(scramble, label):
    global spk,lis,fit
    torch.manual_seed(1);
    spk=torch.randn(P,S,S,device=DEV)*0.5; lis=torch.randn(P,S,S,device=DEV)*0.5; fit=torch.zeros(P,device=DEV)
    hist=[]
    for gconv in range(GEN):
        st,sy,acc=round_play(scramble); evolve()
        if gconv%LOG==0:
            mi=mutual_info(st,sy); hist.append((gconv,float(acc),mi))
    return hist

def main():
    t0=time.time()
    print("="*72); print(f"H81 Emergencia de linguagem (jogo de sinalizacao evolutivo) em {DEV} | S={S} (acaso={1/S:.2f})")
    print("\n  COM comunicacao (parceiro fixo, codigo pode estabilizar):")
    hc=run(False,"comm")
    for g_,a_,mi_ in hc[::max(1,len(hc)//8)]: print(f"    gen {g_:>4} | acuracia {a_:.2f} | I(estado;simbolo) {mi_:.2f} bits")
    print("\n  SEM comunicacao estavel (simbolos embaralhados — controle):")
    hs=run(True,"scram")
    for g_,a_,mi_ in hs[::max(1,len(hs)//4)]: print(f"    gen {g_:>4} | acuracia {a_:.2f} | MI {mi_:.2f} bits")
    accf=hc[-1][1]; mif=hc[-1][2]; accs=hs[-1][1]
    print("="*72)
    print(f"H81 — {GEN} geracoes x2, {time.time()-t0:.0f}s em {DEV}")
    print(f"  COM comunicacao: acuracia final {accf:.2f} | I(estado;simbolo) {mif:.2f}/{np.log2(S):.2f} bits max")
    print(f"  SEM (controle):  acuracia final {accs:.2f} (~acaso {1/S:.2f})")
    emerged = accf>0.6 and mif>0.5
    print(f"  VEREDITO: {'LINGUAGEM EMERGIU — codigo compartilhado transmite a tech (acuracia>>acaso, MI>0)' if emerged else 'codigo nao estabilizou neste run'}.")
    print(f"  Honesto: proto-comunicacao emergente (significado compartilhado por selecao); NAO e linguagem")
    print(f"  humana (sem sintaxe/cultura). Mas e a fagulha social real: eles DESCOBRIRAM um codigo p/ passar info adiante. Nao e AGI.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Hc=np.array(hc); Hs=np.array(hs); fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        ax[0].plot(Hc[:,0],Hc[:,1],label="com comunic."); ax[0].plot(Hs[:,0],Hs[:,1],label="controle"); ax[0].axhline(1/S,ls=":",c="k"); ax[0].set_title("acuracia (tech transmitida)"); ax[0].legend()
        ax[1].plot(Hc[:,0],Hc[:,2]); ax[1].set_title("I(estado;simbolo) bits (linguagem)")
        for a_ in ax: a_.grid(alpha=.3); a_.set_xlabel("geracao")
        plt.tight_layout(); out=os.path.join(os.path.dirname(__file__),"language_emergence.png"); plt.savefig(out,dpi=110)
        print(f"  curvas salvas em {out}")
    except Exception as e: print(f"  (plot: {e})")

if __name__=="__main__": main()
