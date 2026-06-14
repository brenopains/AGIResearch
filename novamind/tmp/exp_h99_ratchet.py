"""
H99 — Ratchet cultural + Teoria da Mente: a fagulha MAIOR que linguagem (cultura cumulativa)
=============================================================================================
Junta os DOIS unicos sinais positivos do projeto: cultura > genes (H83) + transmissao com gargalo
(H95). A aposta de "algo maior que fagulha": CULTURA CUMULATIVA ABERTA — conhecimento que acumula
ENTRE geracoes ALEM de qualquer individuo (o superpoder do Homo sapiens; Henrich, Boyd-Richerson).

Mundo (tech tree de profundidade ILIMITADA por design, p/ nao saturar como H84):
  - tech 0,1 = base (gratis). tech t (>=2) exige conhecer t-1 E t-2 (cumulativo, encadeado).
  - INOVACAO (individual, lenta): quem esta "pronto" p/ t (sabe t-1,t-2) descobre t com prob baixa.
  - TURNOVER (substrato darwiniano): a cada passo uma fracao MORRE e renasce so com a base ->
    FORCA re-transmissao a cada geracao (o gargalo do H95). Sem cultura, cada um recomeca do zero.
  - TRANSMISSAO (cultura): agentes pareiam; um professor ensina o aluno.
  - TEORIA DA MENTE: o professor MODELA o que o aluno sabe e ensina exatamente a FRONTEIRA do
    aluno (a tech que ele esta pronto p/ aprender). Sem ToM: ensina tech aleatoria (quase sempre
    desperdicada — aluno nao esta pronto ou ja sabe).

A/B (3 condicoes): INDIVIDUAL (sem ensino) | CULTURA (ensino aleatorio) | CULTURA+ToM (ensino na
fronteira). Mede PROFUNDIDADE tecnologica (max e media) ao longo do tempo. Sinal de CHAMA: a
profundidade SOBE geracao apos geracao SEM TETO (so ToM+cultura sustenta o ratchet apesar da morte).
Vetorizado GPU. Honesto: e cultura cumulativa medida, nao civilizacao; mas e qualitativamente alem
da linguagem (acumulo trans-geracional). Nao e AGI.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
P=4096; T=140; STEPS=4000; LOG=400; P_INNOV=0.02; TURNOVER=0.02
torch.manual_seed(0)

def base_know():
    k=torch.zeros(P,T,dtype=torch.bool,device=DEV); k[:,0]=True; k[:,1]=True; return k

def ready_mask(k):
    # ready[:,t] = sabe t-1 e t-2 e NAO sabe t  (t>=2)
    r=torch.zeros_like(k)
    r[:,2:]=k[:,1:-1]&k[:,0:-2]&(~k[:,2:])
    return r

def frontier(k):
    r=ready_mask(k); has=r.any(1)
    f=torch.argmax(r.float(),dim=1)        # primeira tech pronta (0 se nenhuma)
    return f, has

def depth(k):
    idx=(k.float()*torch.arange(T,device=DEV).float()).max(1).values
    return idx

def step(k, mode, g):
    # INOVACAO: descobre a fronteira com prob P_INNOV
    f,has=frontier(k)
    inv=has&(torch.rand(P,device=DEV,generator=g)<P_INNOV)
    k[torch.arange(P,device=DEV)[inv],f[inv]]=True
    # TRANSMISSAO
    if mode!="indiv":
        partner=torch.randperm(P,device=DEV,generator=g)
        f,has=frontier(k)
        if mode=="tom":
            # professor=partner ensina a FRONTEIRA do aluno se ele a conhece
            teach_t=f
            ok=has & k[partner,teach_t]
        else:  # cultura sem ToM: professor oferece tech ALEATORIA que ele sabe; aluno aprende se pronto
            pk=k[partner].float()
            tot=pk.sum(1,keepdim=True).clamp(min=1)
            teach_t=torch.multinomial(pk/tot,1,generator=g).squeeze(1)
            # aluno aprende se esta pronto p/ essa tech (sabe t-1,t-2 e nao sabe)
            tm1=torch.clamp(teach_t-1,0,T-1); tm2=torch.clamp(teach_t-2,0,T-1)
            ready_for=k[torch.arange(P,device=DEV),tm1]&k[torch.arange(P,device=DEV),tm2]&(~k[torch.arange(P,device=DEV),teach_t])&(teach_t>=2)
            ok=ready_for
        idx=torch.arange(P,device=DEV)[ok]
        k[idx,teach_t[ok]]=True
    # TURNOVER darwiniano: morre e renasce so com a base
    die=torch.rand(P,device=DEV,generator=g)<TURNOVER
    if die.any():
        k[die]=False; k[die,0]=True; k[die,1]=True
    return k

def run(mode,label):
    g=torch.Generator(device=DEV).manual_seed(1); k=base_know(); hist=[]
    for s in range(STEPS):
        k=step(k,mode,g)
        if s%LOG==0 or s==STEPS-1:
            d=depth(k); hist.append((s,float(d.max().item()),float(d.mean().item())))
    return hist

def main():
    t0=time.time()
    print("="*74)
    print(f"H99 Ratchet cultural + Teoria da Mente em {DEV} | P={P} agentes, tech ilimitada (cap T={T})")
    print(f"  tech t exige t-1 E t-2 (cumulativo); turnover {TURNOVER:.0%}/passo (gargalo darwiniano); inov {P_INNOV:.0%}")
    res={}
    for mode,lab in [("indiv","INDIVIDUAL (sem ensino)"),("culture","CULTURA (ensino aleatorio)"),("tom","CULTURA+ToM (ensino na fronteira)")]:
        h=run(mode,lab); res[lab]=h
        print(f"\n  {lab}:")
        print(f"    {'passo':>6} {'prof.MAX':>9} {'prof.media':>11}")
        for s,mx,me in h[::max(1,len(h)//8)]: print(f"    {s:>6} {mx:>9.0f} {me:>11.1f}")
    print("="*74)
    fi=res["INDIVIDUAL (sem ensino)"][-1]; fc=res["CULTURA (ensino aleatorio)"][-1]; ft=res["CULTURA+ToM (ensino na fronteira)"][-1]
    print(f"H99 — {time.time()-t0:.0f}s | profundidade MAX final: individual {fi[1]:.0f} | cultura {fc[1]:.0f} | cultura+ToM {ft[1]:.0f}")
    # ainda subindo no fim? (ratchet aberto)
    tom=res["CULTURA+ToM (ensino na fronteira)"]; climbing = tom[-1][1]>tom[-3][1] if len(tom)>=3 else False
    ratchet = ft[1] > fi[1]*1.5
    print(f"  RATCHET? {'SIM — cultura+ToM acumula MUITO alem do individuo' if ratchet else 'fraco'} | ainda subindo no fim? {'SIM (aberto, sem teto)' if climbing else 'platô'}")
    if ratchet and ft[1]>fc[1]*1.2:
        print("  TEORIA DA MENTE acelerou o ratchet (ensinar a fronteira >> ensinar aleatorio).")
    print("  Honesto: cultura cumulativa medida (acumulo trans-geracional apesar da morte); o sinal")
    print("  'maior que linguagem'. Se a profundidade cresce sem teto, e ratchet ABERTO. Nao e AGI.")

if __name__=="__main__": main()
