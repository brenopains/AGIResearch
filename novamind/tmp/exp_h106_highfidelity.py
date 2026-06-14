"""
H101 — A GRANDE UNIFICACAO: linguagem emergente CARREGA cultura cumulativa + Teoria da Mente
=============================================================================================
Junta os tres sinais positivos do projeto num organismo so:
  - H95 (linguagem composicional emergente) = o CANAL de transmissao;
  - H99 (cultura cumulativa + Teoria da Mente) = o CONTEUDO que se acumula;
  - transmissao com gargalo/turnover darwiniano (H95/H99) = a pressao.
Pergunta-climax: a cultura acumula quando a informacao tem que passar por uma LINGUAGEM QUE OS
PROPRIOS AGENTES INVENTARAM (nao ensino perfeito)? E como humanos: ninguem nasce sabendo, a
lingua tem que emergir E carregar o conhecimento adiante, geracao apos geracao.

Mundo: tech tree cumulativo (tech t exige t-1 e t-2). Cada agente tem: conhecimento (bool[T]),
um FALANTE (tech->2 simbolos) e um OUVINTE (2 simbolos->tech) — a LINGUAGEM, que EVOLUI.
Transmissao com ToM: o professor modela a FRONTEIRA do aluno (o que ele esta pronto p/ aprender),
codifica via SEU falante; o aluno decodifica via SEU ouvinte; so aprende se decodificou certo E
estava pronto. A lingua compartilhada EMERGE por selecao (transmissoes bem-sucedidas dao fitness).
Turnover: morre e renasce sem conhecimento, herdando (mutados) os genes de linguagem.

A/B: UNIFICADO (linguagem emergente + ToM) vs CONTROLE (ouvinte decodifica de parceiro embaralhado
-> nenhum codigo estavel possivel). Mede PROFUNDIDADE tecnologica + taxa de transmissao bem-sucedida.
Sinal de chama: cultura acumula MUITO alem do individuo ATRAVES de uma lingua que ela mesma criou.
Vetorizado GPU. Honesto: proto-organismo cultural; nao e civilizacao nem AGI.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
P=2048; T=60; V=8; SLOT=2; STEPS=6000; LOG=600; TURN=0.02; MUT=0.06
torch.manual_seed(0)

def base_know():
    k=torch.zeros(P,T,dtype=torch.bool,device=DEV); k[:,0]=True; k[:,1]=True; return k
def ready_mask(k):
    r=torch.zeros_like(k); r[:,2:]=k[:,1:-1]&k[:,0:-2]&(~k[:,2:]); return r
def frontier(k):
    r=ready_mask(k); return torch.argmax(r.float(),1), r.any(1)
def depth(k): return (k.float()*torch.arange(T,device=DEV).float()).max(1).values

def run(scramble, label):
    g=torch.Generator(device=DEV).manual_seed(1)
    know=base_know()
    spk=torch.randint(0,V,(P,T,SLOT),device=DEV,generator=g)     # falante: tech->2 simbolos
    lis=torch.randint(0,T,(P,V,V),device=DEV,generator=g)        # ouvinte: (s1,s2)->tech
    fit=torch.zeros(P,device=DEV); hist=[]
    ar=torch.arange(P,device=DEV)
    for s in range(STEPS):
        f,has=frontier(know)                                     # fronteira do APRENDIZ (ToM)
        teacher=torch.randperm(P,device=DEV,generator=g)
        kteach=know[teacher,f]                                    # professor sabe a fronteira do aluno?
        # professor codifica a fronteira do aluno com SEU falante
        msg=spk[teacher,f]                                        # (P,SLOT)
        if scramble:                                             # controle: mensagem de um estranho aleatorio
            msg=msg[torch.randperm(P,device=DEV,generator=g)]
        dec=lis[ar,msg[:,0],msg[:,1]]                            # aluno decodifica com SEU ouvinte
        success=has & kteach & (dec==f)                          # transmissao cultural via lingua deu certo
        know[ar[success],f[success]]=True
        fit=fit*0.92 + success.float()
        fit.index_add_(0,teacher,success.float()*0.92)           # professor tambem credita
        # individual (inovacao lenta) p/ semear o que ensinar
        inv=has & (torch.rand(P,device=DEV,generator=g)<0.01)
        know[ar[inv],f[inv]]=True
        # turnover + Moran nos genes de LINGUAGEM (conhecimento zera ao nascer; lingua herdada+mutada)
        q=P//20
        worst=torch.topk(fit,q,largest=False).indices; best=torch.topk(fit,q,largest=True).indices
        par=best[torch.randint(0,q,(q,),device=DEV,generator=g)]
        sp_new=spk[par].clone(); ls_new=lis[par].clone()
        mmask=torch.rand(q,T,SLOT,device=DEV,generator=g)<MUT
        sp_new[mmask]=torch.randint(0,V,(int(mmask.sum().item()),),device=DEV,generator=g)
        lmask=torch.rand(q,V,V,device=DEV,generator=g)<MUT
        ls_new[lmask]=torch.randint(0,T,(int(lmask.sum().item()),),device=DEV,generator=g)
        spk[worst]=sp_new; lis[worst]=ls_new
        know[worst]=False; know[worst,0]=True; know[worst,1]=True; fit[worst]=0.0
        # turnover aleatorio extra (gargalo)
        die=torch.rand(P,device=DEV,generator=g)<TURN
        know[die]=False; know[die,0]=True; know[die,1]=True
        # INFANCIA (fidelidade): recem-nascidos copiam a lingua de um TUTOR vivo de alto fitness
        if die.any():
            ndie=int(die.sum().item())
            tutor=best[torch.randint(0,q,(ndie,),device=DEV,generator=g)]
            spk[die]=spk[tutor].clone(); lis[die]=lis[tutor].clone()
        if s%LOG==0 or s==STEPS-1:
            d=depth(know); hist.append((s,float(d.max()),float(d.mean()),float(success.float().mean())))
    return hist

def main():
    t0=time.time()
    print("="*74)
    print(f"H106 UNIFICACAO+FIDELIDADE (infancia: recem-nascido alinha lingua com tutor): linguagem emergente carrega cultura cumulativa + ToM em {DEV}")
    print(f"  P={P} agentes, tech cumulativa (cap T={T}), lingua={V} simbolos x{SLOT} slots, turnover {TURN:.0%}")
    for scr,lab in [(False,"UNIFICADO (lingua emergente + ToM)"),(True,"CONTROLE (mensagem embaralhada — sem codigo estavel)")]:
        h=run(scr,lab)
        print(f"\n  {lab}:")
        print(f"    {'passo':>6} {'prof.MAX':>9} {'prof.media':>11} {'transm.ok':>10}")
        for s,mx,me,tr in h[::max(1,len(h)//8)]: print(f"    {s:>6} {mx:>9.0f} {me:>11.1f} {tr:>10.2f}")
        globals()[f"final_{scr}"]=h[-1]
    fu=final_False; fc=final_True
    print("="*74)
    print(f"H106 — {time.time()-t0:.0f}s | UNIFICADO prof.MAX {fu[1]:.0f} (transm {fu[3]:.2f}) | CONTROLE {fc[1]:.0f} (transm {fc[3]:.2f})")
    win = fu[1] > fc[1]*1.5 and fu[1] > 8
    print(f"  VEREDITO: {'CHAMA — a cultura acumulou ATRAVES de uma linguagem que os agentes inventaram (lingua+cultura+ToM, organismo unico)' if win else 'transmissao por lingua emergente nao sustentou ratchet profundo neste setup'}.")
    print("  Honesto: proto-organismo cultural medido (lingua emergente carregando cultura cumulativa).")
    print("  Se a profundidade do UNIFICADO >> controle, os tres sinais positivos viraram um so sistema. Nao e AGI.")

if __name__=="__main__": main()
