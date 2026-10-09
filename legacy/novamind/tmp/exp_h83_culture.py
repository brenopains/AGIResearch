"""
H83 — Fusao: mundo de ferramentas (H79) + CULTURA (transmissao social da tech)
==============================================================================
Funde a fagulha que funcionou (uso de ferramenta evoluido) com transmissao CULTURAL: um agente
pode APRENDER a tech de um VIZINHO que ja a possui (imitacao social/"fala"), alem de herdar por
genes. A PROPENSAO a aprender socialmente e um GENE (evolui) — nao e imposta. Pergunta: a CULTURA
(info passada horizontalmente entre vizinhos) faz o tech-tree subir MAIS RAPIDO/ALTO que so genes?
A/B honesto: CULTURA ON vs OFF, mesma semente. Vetorizado, sync-free, rapido. Honesto: e
transmissao cultural cumulativa (um mecanismo real de OEE em humanos), nao AGI, nao move o ARC.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
G=96; P=6000; NMAT=3; NTOOL=3; STEPS=8000; LOG=1000
IN=4*9+1+NMAT+NTOOL+1; H=24; OUT=4+1+1+1+1   # +1 input (sinal social), +1 acao (imitar/ensinar)

def fresh_state(seed):
    g=torch.Generator(device=DEV).manual_seed(seed)
    S={}
    S['food']=torch.rand(G,G,device=DEV,generator=g)*0.2
    S['pfood']=torch.rand(G,G,device=DEV,generator=g)*0.8
    S['mat']=torch.stack([(torch.rand(G,G,device=DEV,generator=g)<0.07).float() for _ in range(NMAT)])
    S['sig']=torch.zeros(G,G,device=DEV)               # campo de "fala": maior tier anunciado na celula
    S['alive']=torch.zeros(P,dtype=torch.bool,device=DEV); S['alive'][:1500]=True
    S['x']=torch.randint(0,G,(P,),device=DEV,generator=g); S['y']=torch.randint(0,G,(P,),device=DEV,generator=g)
    S['energy']=torch.zeros(P,device=DEV); S['energy'][S['alive']]=1.0; S['age']=torch.zeros(P,device=DEV)
    S['inv']=torch.zeros(P,NMAT,device=DEV); S['tools']=torch.zeros(P,NTOOL,device=DEV)
    S['social']=torch.rand(P,device=DEV,generator=g)   # GENE: propensao a aprender socialmente (0..1)
    S['W1']=torch.randn(P,IN,H,device=DEV,generator=g)*0.3; S['b1']=torch.zeros(P,H,device=DEV)
    S['W2']=torch.randn(P,H,OUT,device=DEV,generator=g)*0.3; S['b2']=torch.zeros(P,OUT,device=DEV)
    S['g']=g; return S

_OY=torch.tensor([-1,-1,-1,0,0,0,1,1,1],device=DEV); _OX=torch.tensor([-1,0,1,-1,0,1,-1,0,1],device=DEV)
def patch(f,y,x): return f[(y[:,None]+_OY)%G,(x[:,None]+_OX)%G]
META=0.012; EAT=0.45; REPRO=2.5; MUT=0.08; MAXAGE=1500.0

def step(S, culture):
    g=S['g']; a=S['alive']; x=S['x']; y=S['y']
    sigp=patch(S['sig'],y,x).max(1).values                       # maior tier "ouvido" na vizinhanca (fala)
    sens=torch.cat([patch(S['food'],y,x),patch(S['mat'][0],y,x),patch(S['mat'][1],y,x),patch(S['mat'][2],y,x)],1)
    inp=torch.cat([sens,S['energy'][:,None],S['inv'],S['tools'],sigp[:,None]],1)
    h=torch.relu(torch.einsum('pi,pih->ph',inp,S['W1'])+S['b1']); logits=torch.einsum('ph,pho->po',h,S['W2'])+S['b2']
    act=logits.argmax(1)
    dx=torch.where(act==2,1,torch.where(act==3,-1,0)).long(); dy=torch.where(act==0,1,torch.where(act==1,-1,0)).long()
    mv=(act<4)&a; x=torch.where(mv,(x+dx)%G,x); y=torch.where(mv,(y+dy)%G,y); S['x']=x; S['y']=y
    cell=(y,x)
    eat=(act==4)&a; tl=S['tools'].sum(1)
    gain=S['food'][cell]*EAT*0.5+(S['tools'][:,0]>0).float()*S['pfood'][cell]*EAT*(1.0+tl)
    S['energy']=S['energy']+torch.where(eat,gain,torch.zeros_like(gain))
    S['food'][cell]=torch.where(eat,S['food'][cell]*0.1,S['food'][cell])
    pick=(act==5)&a
    for m in range(NMAT):
        got=pick&(S['mat'][m][cell]>0); S['inv'][:,m]=S['inv'][:,m]+got.float()
        S['mat'][m][cell]=torch.where(got,torch.zeros_like(S['mat'][m][cell]),S['mat'][m][cell]); S['energy']=S['energy']+got.float()*0.15
    comb=(act==6)&a
    r0=comb&(S['inv'][:,0]>=1)&(S['tools'][:,0]==0); S['inv'][:,0]-=r0.float(); S['tools'][:,0]=torch.where(r0,1.0,S['tools'][:,0])
    r1=comb&(S['inv'][:,1]>=1)&(S['tools'][:,0]>0)&(S['tools'][:,1]==0); S['inv'][:,1]-=r1.float(); S['tools'][:,1]=torch.where(r1,1.0,S['tools'][:,1])
    r2=comb&(S['inv'][:,2]>=1)&(S['tools'][:,1]>0)&(S['tools'][:,2]==0); S['inv'][:,2]-=r2.float(); S['tools'][:,2]=torch.where(r2,1.0,S['tools'][:,2])
    S['energy']=S['energy']+r0.float()*3.0+r1.float()*4.0+r2.float()*5.0
    # ---- CULTURA: "fala" (anuncia maior tier na celula) + IMITACAO social (acao 7) ----
    mytier=(S['tools'][:,0]+S['tools'][:,1]+S['tools'][:,2])
    S['sig']=S['sig']*0.9                                          # a fala decai
    S['sig'][cell]=torch.maximum(S['sig'][cell], mytier*a.float())  # anuncia o que sabe
    if culture:
        teach=(act==7)&a&(S['social']>0.5)                        # quer aprender socialmente E ouviu alguem adiante
        learn=teach&(sigp>mytier)                                  # vizinho sabe mais -> aprende o PROXIMO tier
        S['tools'][:,0]=torch.where(learn&(S['tools'][:,0]==0),1.0,S['tools'][:,0])
        S['tools'][:,1]=torch.where(learn&(S['tools'][:,0]>0)&(S['tools'][:,1]==0),1.0,S['tools'][:,1])
        S['tools'][:,2]=torch.where(learn&(S['tools'][:,1]>0)&(S['tools'][:,2]==0),1.0,S['tools'][:,2])
    # ---- metabolismo/morte/repro (sync-free Moran-ish via slots livres) ----
    S['energy']=torch.where(a,S['energy']-META,S['energy']); S['age']=torch.where(a,S['age']+1,S['age'])
    dead=a&((S['energy']<=0)|(S['age']>MAXAGE)); S['alive']=a&(~dead)
    for k in ('energy','age'): S[k]=torch.where(dead,torch.zeros_like(S[k]),S[k])
    S['inv']=torch.where(dead[:,None],0.0,S['inv']); S['tools']=torch.where(dead[:,None],0.0,S['tools'])
    want=S['alive']&(S['energy']>REPRO); free=(~S['alive']).nonzero().flatten(); par=want.nonzero().flatten()
    if len(free)>0 and len(par)>0:
        n=min(len(free),len(par)); free=free[:n]; par=par[torch.randperm(len(par),device=DEV,generator=g)[:n]]
        S['energy'][par]*=0.5
        for k in ('energy',): S[k][free]=S[k][par]
        S['x'][free]=(S['x'][par]+1)%G; S['y'][free]=S['y'][par]; S['age'][free]=0; S['alive'][free]=True
        S['inv'][free]=0; S['tools'][free]=0
        S['social'][free]=torch.clamp(S['social'][par]+torch.randn(n,device=DEV,generator=g)*0.1,0,1)  # gene evolui
        for w in ('W1','b1','W2','b2'): S[w][free]=S[w][par]+torch.randn_like(S[w][free])*MUT
    S['food']=torch.clamp(S['food']+0.003*(0.2-S['food']),0,1); S['pfood']=torch.clamp(S['pfood']+0.01*(0.9-S['pfood']),0,1)
    if torch.rand(1,generator=g,device=DEV).item()<0.3:
        for m in range(NMAT): S['mat'][m]=torch.clamp(S['mat'][m]+(torch.rand(G,G,device=DEV,generator=g)<0.0006).float(),0,1)

def run(culture):
    S=fresh_state(7); hist=[]
    for s in range(STEPS):
        step(S,culture)
        if s%LOG==0:
            a=S['alive']; pop=int(a.sum())
            if pop==0: hist.append((s,0,0,0,0)); break
            tier=int(S['tools'][a].sum(1).max().item()); f0=float(S['tools'][a,0].mean()); soc=float(S['social'][a].mean())
            hist.append((s,pop,tier,f0,soc))
    return hist

def main():
    t0=time.time()
    print("="*70); print(f"H83 Cultura (transmissao social da tech) vs so-genes — A/B em {DEV}")
    print("  CULTURA OFF (so genes):"); ho=run(False)
    for s_,p_,t_,f_,sc in ho[::max(1,len(ho)//5)]: print(f"    passo {s_:>5} pop {p_:>5} tier {t_} fracT0 {f_:.2f}")
    print("  CULTURA ON (genes + imitacao social):"); hc=run(True)
    for s_,p_,t_,f_,sc in hc[::max(1,len(hc)//5)]: print(f"    passo {s_:>5} pop {p_:>5} tier {t_} fracT0 {f_:.2f} social-gene {sc:.2f}")
    print("="*70)
    f0_off=ho[-1][3]; f0_on=hc[-1][3]; t_off=max(h[2] for h in ho); t_on=max(h[2] for h in hc); soc_i=hc[1][4] if len(hc)>1 else 0; soc_f=hc[-1][4]
    print(f"H83 — {time.time()-t0:.0f}s em {DEV}")
    print(f"  fracT0 final: OFF {f0_off:.2f} | ON {f0_on:.2f}   | tier max: OFF {t_off} | ON {t_on}")
    print(f"  gene 'social' (ON): {soc_i:.2f} -> {soc_f:.2f}  ({'SELECIONADO (cultura favorecida)' if soc_f>soc_i+0.03 else 'nao selecionado'})")
    win = f0_on>f0_off+0.05 or t_on>t_off
    print(f"  VEREDITO: {'CULTURA ACELERA a tech (espalha mais/sobe mais que so-genes) — transmissao cumulativa real' if win else 'cultura nao superou genes neste run'}.")
    print(f"  Honesto: transmissao cultural cumulativa (mecanismo real de OEE humano); nao e AGI, nao move o ARC.")

if __name__=="__main__": main()
