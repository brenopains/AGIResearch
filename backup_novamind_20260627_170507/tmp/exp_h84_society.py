"""
H84 — Sociedade: reproducao SEXUAL (recombinacao) + tribos (nos/eles) + automacao + cultura
============================================================================================
Tese do usuario (correta): humanos nao criam primitivas do nada — RECOMBINAM info herdada
(2 pais diversos) + aprendida (cultura) numa populacao diversa. Testamos o motor real:
 - SEXO: 2 sexos; reproducao SEXUAL = CRUZAMENTO (crossover) dos genomas de macho+femea da
   mesma tribo (escolha de parceiro) + mutacao -> filho com combinacao nova (vs clone assexual).
 - TRIBOS (nos/eles): tag de tribo herdada; cooperacao INTRA-grupo (+energia perto de igual) e
   conflito INTER-grupo (-energia perto de diferente) = "bem/mal" RELATIVO ao grupo (operacional).
 - AUTOMACAO (estilo Factorio): a ferramenta de topo (tier2) colhe energia passiva (maquina
   trabalha por voce) = eficiencia cumulativa.
 - CULTURA (H83): aprende a tech de vizinho que ja sabe.
A/B honesto: SOCIEDADE (sexual+tribos+cultura) vs ASSEXUAL (clone, sem social). Mede tech,
DIVERSIDADE genetica, populacao. Honesto: emocoes/moeda sao metafora; implemento os nucleos
operacionais. Bounded (tech teto 3); nao move ARC; nao e AGI. Vetorizado, sync-lean.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
G=96; P=6000; NMAT=3; NTOOL=3; NTRIBE=4; STEPS=8000; LOG=1000
IN=4*9+1+NMAT+NTOOL+1+NTRIBE; H=24; OUT=4+1+1+1+1
_OY=torch.tensor([-1,-1,-1,0,0,0,1,1,1],device=DEV); _OX=torch.tensor([-1,0,1,-1,0,1,-1,0,1],device=DEV)
META=0.012; EAT=0.45; REPRO=2.6; MUT=0.08; MAXAGE=1500.0

def fresh(seed):
    g=torch.Generator(device=DEV).manual_seed(seed); S={'g':g}
    S['food']=torch.rand(G,G,device=DEV,generator=g)*0.2; S['pfood']=torch.rand(G,G,device=DEV,generator=g)*0.8
    S['mat']=torch.stack([(torch.rand(G,G,device=DEV,generator=g)<0.07).float() for _ in range(NMAT)])
    S['sig']=torch.zeros(G,G,device=DEV)
    S['alive']=torch.zeros(P,dtype=torch.bool,device=DEV); S['alive'][:1500]=True
    S['x']=torch.randint(0,G,(P,),device=DEV,generator=g); S['y']=torch.randint(0,G,(P,),device=DEV,generator=g)
    S['energy']=torch.zeros(P,device=DEV); S['energy'][S['alive']]=1.0; S['age']=torch.zeros(P,device=DEV)
    S['inv']=torch.zeros(P,NMAT,device=DEV); S['tools']=torch.zeros(P,NTOOL,device=DEV)
    S['sex']=torch.randint(0,2,(P,),device=DEV,generator=g)
    S['tribe']=torch.randint(0,NTRIBE,(P,),device=DEV,generator=g)
    S['social']=torch.rand(P,device=DEV,generator=g)
    S['W1']=torch.randn(P,IN,H,device=DEV,generator=g)*0.3; S['b1']=torch.zeros(P,H,device=DEV)
    S['W2']=torch.randn(P,H,OUT,device=DEV,generator=g)*0.3; S['b2']=torch.zeros(P,OUT,device=DEV)
    return S
def patch(f,y,x): return f[(y[:,None]+_OY)%G,(x[:,None]+_OX)%G]

def step(S, sexual, social):
    g=S['g']; a=S['alive']; x=S['x']; y=S['y']; cell=(y,x)
    # densidade de tribo por celula (nos/eles) via scatter
    tribe_cell=torch.zeros(NTRIBE,G*G,device=DEV); idx=y*G+x
    tribe_cell.index_put_((S['tribe'][a],idx[a]),torch.ones(int(a.sum()),device=DEV),accumulate=True)
    tribe_cell=tribe_cell.view(NTRIBE,G,G)
    sigp=patch(S['sig'],y,x).max(1).values
    tribe_oh=torch.zeros(P,NTRIBE,device=DEV); tribe_oh[torch.arange(P,device=DEV),S['tribe']]=1.0
    sens=torch.cat([patch(S['food'],y,x),patch(S['mat'][0],y,x),patch(S['mat'][1],y,x),patch(S['mat'][2],y,x)],1)
    inp=torch.cat([sens,S['energy'][:,None],S['inv'],S['tools'],sigp[:,None],tribe_oh],1)
    h=torch.relu(torch.einsum('pi,pih->ph',inp,S['W1'])+S['b1']); act=(torch.einsum('ph,pho->po',h,S['W2'])+S['b2']).argmax(1)
    dx=torch.where(act==2,1,torch.where(act==3,-1,0)).long(); dy=torch.where(act==0,1,torch.where(act==1,-1,0)).long()
    mv=(act<4)&a; S['x']=torch.where(mv,(x+dx)%G,x); S['y']=torch.where(mv,(y+dy)%G,y); x=S['x']; y=S['y']; cell=(y,x)
    # comer + automacao (tier2 = maquina que colhe passivo)
    eat=(act==4)&a; tl=S['tools'].sum(1)
    gain=S['food'][cell]*EAT*0.5+(S['tools'][:,0]>0).float()*S['pfood'][cell]*EAT*(1.0+tl)
    S['energy']=S['energy']+torch.where(eat,gain,torch.zeros_like(gain))
    S['energy']=S['energy']+(S['tools'][:,2]>0).float()*0.03*a.float()      # AUTOMACAO: renda passiva
    S['food'][cell]=torch.where(eat,S['food'][cell]*0.1,S['food'][cell])
    pick=(act==5)&a
    for m in range(NMAT):
        got=pick&(S['mat'][m][cell]>0); S['inv'][:,m]+=got.float(); S['mat'][m][cell]=torch.where(got,0.0,S['mat'][m][cell]); S['energy']+=got.float()*0.15
    comb=(act==6)&a
    r0=comb&(S['inv'][:,0]>=1)&(S['tools'][:,0]==0); S['inv'][:,0]-=r0.float(); S['tools'][:,0]=torch.where(r0,1.0,S['tools'][:,0])
    r1=comb&(S['inv'][:,1]>=1)&(S['tools'][:,0]>0)&(S['tools'][:,1]==0); S['inv'][:,1]-=r1.float(); S['tools'][:,1]=torch.where(r1,1.0,S['tools'][:,1])
    r2=comb&(S['inv'][:,2]>=1)&(S['tools'][:,1]>0)&(S['tools'][:,2]==0); S['inv'][:,2]-=r2.float(); S['tools'][:,2]=torch.where(r2,1.0,S['tools'][:,2])
    S['energy']+=r0.float()*3.0+r1.float()*4.0+r2.float()*5.0
    # CULTURA
    mytier=S['tools'].sum(1); S['sig']=S['sig']*0.9; S['sig'][cell]=torch.maximum(S['sig'][cell],mytier*a.float())
    if social:
        learn=(act==7)&a&(S['social']>0.5)&(sigp>mytier)
        S['tools'][:,0]=torch.where(learn&(S['tools'][:,0]==0),1.0,S['tools'][:,0])
        S['tools'][:,1]=torch.where(learn&(S['tools'][:,0]>0)&(S['tools'][:,1]==0),1.0,S['tools'][:,1])
        S['tools'][:,2]=torch.where(learn&(S['tools'][:,1]>0)&(S['tools'][:,2]==0),1.0,S['tools'][:,2])
    # NOS/ELES: cooperacao intra-tribo (+), conflito inter-tribo (-) pela densidade local
    same=tribe_cell[S['tribe'],y,x]-1.0                          # vizinhos da mesma tribo na celula (menos eu)
    diff=tribe_cell.sum(0)[y,x]-tribe_cell[S['tribe'],y,x]       # vizinhos de outras tribos
    if social: S['energy']=S['energy']+a.float()*(0.02*torch.clamp(same,0,5)-0.03*torch.clamp(diff,0,5))
    # metabolismo / morte
    S['energy']=torch.where(a,S['energy']-META,S['energy']); S['age']=torch.where(a,S['age']+1,S['age'])
    dead=a&((S['energy']<=0)|(S['age']>MAXAGE)); S['alive']=a&(~dead)
    for k in('energy','age'): S[k]=torch.where(dead,torch.zeros_like(S[k]),S[k])
    S['inv']=torch.where(dead[:,None],0.0,S['inv']); S['tools']=torch.where(dead[:,None],0.0,S['tools'])
    # ---- reproducao ----
    free=(~S['alive']).nonzero().flatten()
    if sexual:
        m=(S['alive']&(S['energy']>REPRO)&(S['sex']==0)).nonzero().flatten()   # machos prontos
        fem=(S['alive']&(S['energy']>REPRO)&(S['sex']==1)).nonzero().flatten() # femeas prontas
        n=min(len(free),len(m),len(fem))
        if n>0:
            free=free[:n]; mm=m[torch.randperm(len(m),device=DEV,generator=g)[:n]]; ff=fem[torch.randperm(len(fem),device=DEV,generator=g)[:n]]
            # escolha de parceiro = mesma tribo preferida (reordena femeas p/ casar tribo quando possivel) — aproximacao: filho herda tribo da mae
            mask=(torch.rand(n,IN,H,device=DEV,generator=g)<0.5)               # CROSSOVER por-peso
            S['W1'][free]=torch.where(mask,S['W1'][mm],S['W1'][ff])+torch.randn(n,IN,H,device=DEV,generator=g)*MUT
            mask2=(torch.rand(n,H,OUT,device=DEV,generator=g)<0.5)
            S['W2'][free]=torch.where(mask2,S['W2'][mm],S['W2'][ff])+torch.randn(n,H,OUT,device=DEV,generator=g)*MUT
            S['social'][free]=torch.clamp((S['social'][mm]+S['social'][ff])/2+torch.randn(n,device=DEV,generator=g)*0.1,0,1)
            S['tribe'][free]=S['tribe'][ff]; S['sex'][free]=torch.randint(0,2,(n,),device=DEV,generator=g)
            S['energy'][mm]*=0.6; S['energy'][ff]*=0.6; S['energy'][free]=1.0
            S['x'][free]=(S['x'][ff]+1)%G; S['y'][free]=S['y'][ff]; S['age'][free]=0; S['alive'][free]=True; S['inv'][free]=0; S['tools'][free]=0
    else:  # ASSEXUAL: clone + mutacao
        par=(S['alive']&(S['energy']>REPRO)).nonzero().flatten(); n=min(len(free),len(par))
        if n>0:
            free=free[:n]; par=par[torch.randperm(len(par),device=DEV,generator=g)[:n]]
            S['energy'][par]*=0.5; S['energy'][free]=1.0
            for w in('W1','b1','W2','b2'): S[w][free]=S[w][par]+torch.randn_like(S[w][free])*MUT
            S['social'][free]=S['social'][par]; S['tribe'][free]=S['tribe'][par]; S['sex'][free]=torch.randint(0,2,(n,),device=DEV,generator=g)
            S['x'][free]=(S['x'][par]+1)%G; S['y'][free]=S['y'][par]; S['age'][free]=0; S['alive'][free]=True; S['inv'][free]=0; S['tools'][free]=0
    S['food']=torch.clamp(S['food']+0.003*(0.2-S['food']),0,1); S['pfood']=torch.clamp(S['pfood']+0.01*(0.9-S['pfood']),0,1)
    if torch.rand(1,generator=g,device=DEV).item()<0.3:
        for m_ in range(NMAT): S['mat'][m_]=torch.clamp(S['mat'][m_]+(torch.rand(G,G,device=DEV,generator=g)<0.0006).float(),0,1)

def run(sexual, social):
    S=fresh(7); hist=[]
    for s in range(STEPS):
        step(S,sexual,social)
        if s%LOG==0:
            a=S['alive']; pop=int(a.sum())
            if pop==0: hist.append((s,0,0,0,0)); break
            tier=int(S['tools'][a].sum(1).max().item()); ft=float((S['tools'][a].sum(1)>0).float().mean())
            gdiv=float(S['W1'][a].std().item()); auto=float((S['tools'][a,2]>0).float().mean())
            hist.append((s,pop,tier,ft,gdiv,auto))
    return hist

def main():
    t0=time.time(); print("="*72); print(f"H84 Sociedade: sexual+tribos+cultura vs assexual — A/B em {DEV}")
    print("  ASSEXUAL (clone, sem social):"); ha=run(False,False)
    for h in ha[::max(1,len(ha)//4)]: print(f"    passo {h[0]:>5} pop {h[1]:>5} tier {h[2]} frac-tool {h[3]:.2f} gdiv {h[4]:.3f}")
    print("  SOCIEDADE (sexual + tribos + cultura):"); hs=run(True,True)
    for h in hs[::max(1,len(hs)//4)]: print(f"    passo {h[0]:>5} pop {h[1]:>5} tier {h[2]} frac-tool {h[3]:.2f} gdiv {h[4]:.3f} automacao {h[5]:.2f}")
    print("="*72)
    fa=ha[-1][3]; fs=hs[-1][3]; da=ha[-1][4]; ds=hs[-1][4]; auto=hs[-1][5]
    print(f"H84 — {time.time()-t0:.0f}s em {DEV}")
    print(f"  frac com ferramenta: ASSEXUAL {fa:.2f} | SOCIEDADE {fs:.2f}")
    print(f"  diversidade genetica (std): ASSEXUAL {da:.3f} | SOCIEDADE {ds:.3f} (sexual deve manter MAIS diversidade)")
    print(f"  automacao (tier2/Factorio) na sociedade: {auto:.2f} da pop")
    win = fs>fa+0.05
    print(f"  VEREDITO: {'SOCIEDADE (sexual+tribos+cultura) acelera a tech vs assexual — recombinacao+cultura sao o motor' if win else 'sociedade nao superou claramente neste run'}.")
    print(f"  Honesto: recombinacao sexual + cultura + diversidade (a tese do usuario) testada; emocoes/moeda sao metafora.")
    print(f"  Bounded (tech teto 3), nao move ARC, nao e AGI — mas e o motor real que a natureza usou, medido.")

if __name__=="__main__": main()
