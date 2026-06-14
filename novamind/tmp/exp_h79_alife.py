"""
H79 — Mini-mundo de Vida Artificial (OEE), GPU-vetorizado em ESCALA
====================================================================
Sem padroes prontos: um MUNDO onde agentes com "eu"/sobrevivencia tem que se virar para
desenvolver ferramentas e novidade sozinhos — como a natureza. Tudo VETORIZADO (zero loop
por agente): um passo = punhado de ops de tensor sobre TODA a populacao. Ingredientes que
faltaram ao Tierra/Avida (que pletearam): (1) dinamica DEPENDENTE DO ESTADO (Adams 2016:
recursos regeneram/escasseiam por uso); (2) ambiente aberto; (3) TECH-TREE DESCOBRIVEL
(combinar materiais -> ferramenta -> destrava recurso novo) — inovacao cumulativa, NAO
hard-coded (o agente precisa EVOLUIR o cerebro p/ usar). Mede hallmarks de OEE (novidade,
complexidade, profundidade do tech-tree, diversidade) ao longo de geracoes. Honesto: OEE
sustentada e problema em aberto; reporto crescimento OU plateau, com dado.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
P=8192            # capacidade (slots de agentes) — populacao viva varia dentro disso
G=96              # mundo G x G
NMAT=3            # tipos de material
NTOOL=3           # niveis do tech-tree
IN=4*9+1+NMAT+NTOOL   # patch 3x3 de (food+NMAT) + energia + inventario + ferramentas
H=24; OUT=4+1+1+1     # 4 mover, comer, pegar, combinar
STEPS=80000; LOG=2000
rng=torch.Generator(device=DEV).manual_seed(0)

# ---------------- mundo ----------------
food=torch.rand(G,G,device=DEV)*0.2               # comida livre: ESCASSA/pobre
pfood=torch.rand(G,G,device=DEV)*0.8              # comida RICA: abundante, mas so com tool0+
mat=torch.zeros(NMAT,G,G,device=DEV)
for m in range(NMAT): mat[m]=(torch.rand(G,G,device=DEV)<0.07).float()

# ---------------- agentes (tudo tensor, slots fixos) ----------------
alive=torch.zeros(P,dtype=torch.bool,device=DEV); alive[:1500]=True
x=torch.randint(0,G,(P,),device=DEV); y=torch.randint(0,G,(P,),device=DEV)
energy=torch.zeros(P,device=DEV); energy[alive]=1.0
age=torch.zeros(P,device=DEV)
inv=torch.zeros(P,NMAT,device=DEV)
tools=torch.zeros(P,NTOOL,device=DEV)
# genoma = MLP por-agente
W1=torch.randn(P,IN,H,generator=rng,device=DEV)*0.3; b1=torch.zeros(P,H,device=DEV)
W2=torch.randn(P,H,OUT,generator=rng,device=DEV)*0.3; b2=torch.zeros(P,OUT,device=DEV)
MAXAGE=1500.0; META=0.012; EAT=0.45; REPRO=2.5; MUT=0.08   # escassez real: metabolismo alto -> seleciona
_OY=torch.tensor([-1,-1,-1,0,0,0,1,1,1],device=DEV); _OX=torch.tensor([-1,0,1,-1,0,1,-1,0,1],device=DEV)

def patch(field):  # (P,9) vizinhanca 3x3 (wrap) — um indice avancado, vetorizado
    yy=(y[:,None]+_OY)%G; xx=(x[:,None]+_OX)%G
    return field[yy,xx]

def step():
    global food,pfood,mat,alive,x,y,energy,age,inv,tools
    a=alive
    # ---- percepcao (4 campos x 3x3) ----
    sens=torch.cat([patch(food),patch(mat[0]),patch(mat[1]),patch(mat[2])],1)
    inp=torch.cat([sens,energy[:,None],inv,tools],1)            # (P,IN)
    # ---- cerebro (batched, por-agente) ----
    h=torch.relu(torch.einsum('pi,pih->ph',inp,W1)+b1)
    logits=torch.einsum('ph,pho->po',h,W2)+b2
    act=logits.argmax(1)                                        # (P,)
    # ---- acoes (vetorizadas) ----
    move=act<4
    dx=torch.where(act==2,1,torch.where(act==3,-1,0)).to(torch.long)
    dy=torch.where(act==0,1,torch.where(act==1,-1,0)).to(torch.long)
    x=torch.where(move&a,(x+dx)%G,x); y=torch.where(move&a,(y+dy)%G,y)
    cell=(y,x)
    # comer: comida LIVRE e escassa/pobre; comida RICA so acessivel subindo o tech-tree
    # -> ferramenta vira NECESSARIA p/ sobreviver => selecao real (nao loteria)
    eat=(act==4)&a
    toollvl=tools[:,0]+tools[:,1]+tools[:,2]                     # 0..3
    gain=food[cell]*EAT*0.5 + (tools[:,0]>0).float()*pfood[cell]*EAT*(1.0+toollvl)  # rico escala com o tier
    energy=energy+torch.where(eat,gain,torch.zeros_like(gain))
    food[cell]=torch.where(eat,food[cell]*0.1,food[cell])       # estado-dependente: depleta
    pfood[cell]=torch.where(eat&(tools[:,0]>0),pfood[cell]*0.2,pfood[cell])
    # pegar material — FORRAGEIO ja e adaptativo (degrau/stepping-stone): pequena energia
    pick=(act==5)&a
    for m in range(NMAT):
        got=pick&(mat[m][cell]>0); inv[:,m]=inv[:,m]+got.float()
        mat[m][cell]=torch.where(got,torch.zeros_like(mat[m][cell]),mat[m][cell])
        energy=energy+got.float()*0.15
    # combinar (tech-tree descobrivel, receitas de 1 material = degrau descobrivel; tiers cumulativos)
    comb=(act==6)&a
    r0=comb&(inv[:,0]>=1)&(tools[:,0]==0)                       # tier0: mat0 -> tool0
    inv[:,0]-=r0.float(); tools[:,0]=torch.where(r0,torch.ones_like(tools[:,0]),tools[:,0])
    r1=comb&(inv[:,1]>=1)&(tools[:,0]>0)&(tools[:,1]==0)        # tier1: mat1+tool0 -> tool1
    inv[:,1]-=r1.float(); tools[:,1]=torch.where(r1,torch.ones_like(tools[:,1]),tools[:,1])
    r2=comb&(inv[:,2]>=1)&(tools[:,1]>0)&(tools[:,2]==0)        # tier2: mat2+tool1 -> tool2
    inv[:,2]-=r2.float(); tools[:,2]=torch.where(r2,torch.ones_like(tools[:,2]),tools[:,2])
    # bonus GRANDE por destravar (o descobridor domina a reproducao -> a inovacao se espalha)
    energy=energy+ (r0.float()*3.0 + r1.float()*4.0 + r2.float()*5.0)
    # ---- metabolismo / morte ----
    energy=torch.where(a,energy-META,energy); age=torch.where(a,age+1,age)
    dead=a&((energy<=0)|(age>MAXAGE))
    alive=a&(~dead)
    energy=torch.where(dead,torch.zeros_like(energy),energy)
    # liberar slots mortos
    inv=torch.where(dead[:,None],torch.zeros_like(inv),inv); tools=torch.where(dead[:,None],torch.zeros_like(tools),tools); age=torch.where(dead,torch.zeros_like(age),age)
    # ---- reproducao (preenche slots livres com filhos mutados) ----
    want=alive&(energy>REPRO)
    free=(~alive).nonzero().flatten()
    par=want.nonzero().flatten()
    if len(free)>0 and len(par)>0:
        n=min(len(free),len(par)); free=free[:n]; par=par[torch.randperm(len(par),device=DEV)[:n]]
        energy[par]=energy[par]*0.5
        energy[free]=energy[par]; x[free]=(x[par]+1)%G; y[free]=y[par]; age[free]=0; alive[free]=True
        inv[free]=0; tools[free]=0
        W1[free]=W1[par]+torch.randn_like(W1[free])*MUT; b1[free]=b1[par]+torch.randn_like(b1[free])*MUT
        W2[free]=W2[par]+torch.randn_like(W2[free])*MUT; b2[free]=b2[par]+torch.randn_like(b2[free])*MUT
    # ---- mundo estado-dependente: regenera ----
    food=torch.clamp(food+0.003*(0.2-food),0,1)                # livre: regenera pouco (escassa)
    pfood=torch.clamp(pfood+0.01*(0.9-pfood),0,1)              # rica: regenera bem (recompensa quem destrava)
    if torch.rand(1).item()<0.3:
        for m in range(NMAT):
            spawn=(torch.rand(G,G,device=DEV)<0.0006).float(); mat[m]=torch.clamp(mat[m]+spawn,0,1)
    return act

def main():
    t0=time.time(); hist=[]
    for s in range(STEPS):
        act=step()
        if s%LOG==0:
            a=alive; pop=int(a.sum())
            if pop==0: print(f"  passo {s}: EXTINCAO"); break
            tier=int(tools[a].sum(1).max().item())              # profundidade do tech-tree alcancada
            f0=float(tools[a,0].mean()); f1=float(tools[a,1].mean()); f2=float(tools[a,2].mean())  # FRACAO com cada ferramenta (evolucao vs loteria)
            gdiv=float(W1[a].std().item())
            hist.append((s,pop,tier,f0,f1,f2,gdiv))
            print(f"  passo {s:>6} | pop {pop:>5} | tier {tier} | frac tool0={f0:.2f} tool1={f1:.2f} tool2={f2:.2f} | gstd {gdiv:.3f}")
    dt=time.time()-t0
    print("="*70)
    maxtier=max(h[2] for h in hist) if hist else 0
    f0i,f0f=(hist[1][3],hist[-1][3]) if len(hist)>1 else (0,0)   # fracao tool0 inicio->fim
    f1f=hist[-1][4] if hist else 0; f2f=hist[-1][5] if hist else 0
    print(f"H79 ALife — {STEPS} passos, {dt:.0f}s ({STEPS/dt:.0f} passos/s) em {DEV}")
    print(f"  TECH-TREE max (emergente): tier {maxtier}/{NTOOL}")
    print(f"  FRACAO da pop com ferramenta (evolucao vs loteria): tool0 {f0i:.2f}->{f0f:.2f} | tool1_fim {f1f:.2f} | tool2_fim {f2f:.2f}")
    evolved = f0f>f0i+0.1 or f1f>0.1   # uso de ferramenta SE ESPALHOU pela populacao = selecionado
    print(f"  VEREDITO HONESTO: tech-tree subiu a tier {maxtier}; uso de ferramenta {'SE ESPALHOU (evoluido/selecionado, nao loteria)' if evolved else 'NAO se espalhou (provavel loteria de pop grande, nao evolucao)'}.")
    print(f"  OEE sustentada: {'sinal positivo (frac e tier subindo)' if (evolved and maxtier>=2) else 'NAO confirmada — esperado (problema em aberto da ALife)'}.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        H=np.array(hist); fig,ax=plt.subplots(1,3,figsize=(13,3.5))
        ax[0].plot(H[:,0],H[:,1]); ax[0].set_title("populacao")
        ax[1].plot(H[:,0],H[:,2]); ax[1].set_title("tech-tier max")
        ax[2].plot(H[:,0],H[:,3],label="tool0"); ax[2].plot(H[:,0],H[:,4],label="tool1"); ax[2].plot(H[:,0],H[:,5],label="tool2")
        ax[2].set_title("FRACAO da pop com ferramenta"); ax[2].legend()
        for a_ in ax: a_.grid(alpha=.3); a_.set_xlabel("passo")
        plt.tight_layout(); out=os.path.join(os.path.dirname(__file__),"alife_oee.png"); plt.savefig(out,dpi=110)
        print(f"  curvas OEE salvas em {out}")
    except Exception as e: print(f"  (plot indisponivel: {e})")

if __name__=="__main__": main()
