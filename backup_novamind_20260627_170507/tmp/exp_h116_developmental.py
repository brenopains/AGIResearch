"""
H116 — Poucos agentes, cerebros BASAIS que aprendem DENTRO DA VIDA (plasticidade), sem imitar
==============================================================================================
A refinada (correta) do usuario, que conserta a saturacao do H115:
  - POUCOS agentes (16), nao 2600 -> sem propagacao em massa instantanea.
  - Cerebros BASAIS (pesos pequenos, "nao desenvolvidos") que DESENVOLVEM dentro da vida via
    APRENDIZADO POR RECOMPENSA (REINFORCE online), nao por evolucao entre geracoes.
  - SEM imitacao programada: o aprendizado-pelo-player e INDIRETO — o player muda o mundo
    (provisiona comida/estruturas) e a plasticidade do agente naturalmente passa a explorar isso.
A/B: SEM player vs COM player que provisiona. Mede se os agentes APRENDEM (taxa de comida sobe ao
longo da vida) e se o player acelera INDIRETAMENTE. Honesto: aprendizado embodied medido, nao AGI.
Se validar aqui, justifica a ponte pro Minecraft REAL (Node+Java presentes; mineflayer headless).
"""
import os, sys, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
G=24; N=16; A=5; torch.manual_seed(0)
OY=torch.tensor([-1,1,0,0],device=DEV); OX=torch.tensor([0,0,-1,1],device=DEV)  # mov: N,S,W,E
# sensor: vizinhanca 8 + celula + energia + bias
SY=torch.tensor([-1,-1,-1,0,0,1,1,1,0],device=DEV); SX=torch.tensor([-1,0,1,-1,1,-1,0,1,0],device=DEV)
S=9+1+1  # 9 celulas (8 viz + centro via SY/SX last) ... +energia+bias

def fresh_world(seed):
    g=torch.Generator(device=DEV).manual_seed(seed)
    food=(torch.rand(G,G,device=DEV,generator=g)<0.12).float()
    x=torch.randint(0,G,(N,),device=DEV,generator=g); y=torch.randint(0,G,(N,),device=DEV,generator=g)
    en=torch.ones(N,device=DEV)*5.0                    # vida LONGA p/ a plasticidade ter tempo
    W=torch.randn(N,S,A,device=DEV,generator=g)*0.05   # BASAL: pesos pequenos (nao desenvolvido)
    return g,food,x,y,en,W

def sense(food,x,y,en):
    nb=torch.stack([food[(y+SY[k])%G,(x+SX[k])%G] for k in range(9)],1)  # 8 viz + centro
    return torch.cat([nb,(en.clamp(0,6)/6)[:,None],torch.ones(N,1,device=DEV)],1)  # (N,11)

def run(with_player, steps=8000, seed=1, lr=0.15):
    g,food,x,y,en,W=fresh_world(seed)
    base=torch.zeros(N,device=DEV)  # baseline de recompensa (media movel) p/ REINFORCE
    eaten_hist=[]; eaten_win=0.0
    for t in range(steps):
        st=sense(food,x,y,en)                          # (N,S)
        logits=torch.einsum('ns,nsa->na',st,W)
        p=torch.softmax(logits,1)
        act=torch.multinomial(p,1,generator=g).squeeze(1)
        # executa
        mv=act<4; nx=(x+torch.where(mv,OX[act.clamp(max=3)],torch.zeros_like(x)))%G
        ny=(y+torch.where(mv,OY[act.clamp(max=3)],torch.zeros_like(y)))%G
        x=torch.where(mv,nx,x); y=torch.where(mv,ny,y)
        eat=(act==4); got=eat.float()*food[y,x]
        food[y,x]=torch.where(eat,food[y,x]*0.0,food[y,x])
        reward=got*1.0 - 0.005                           # comer=recompensa; viver custa pouco (vida longa)
        en=en+got*1.0-0.005
        eaten_win+=float(got.sum().item())
        # APRENDIZADO DENTRO DA VIDA (REINFORCE com baseline): reforca acoes que deram recompensa
        if lr>0:
            adv=(reward-base)
            gl=(-p); gl[torch.arange(N,device=DEV),act]+=1.0   # d log pi
            W=W+lr*(adv[:,None,None]*st[:,:,None]*gl[:,None,:])
            W=W.clamp(-3,3)                                     # estabiliza (evita colapso da politica)
            base=0.99*base+0.01*reward
        # morte/renascimento BASAL (cada um aprende na propria vida)
        dead=en<=0
        if dead.any():
            nd=int(dead.sum().item())
            W[dead]=torch.randn(nd,S,A,device=DEV,generator=g)*0.05  # renasce BASAL
            x[dead]=torch.randint(0,G,(nd,),device=DEV,generator=g); y[dead]=torch.randint(0,G,(nd,),device=DEV,generator=g)
            en[dead]=5.0; base[dead]=0.0
        # PLAYER provisiona o mundo (INDIRETO: nao ensina acao, so cria oportunidade)
        if with_player and t%30==0:
            cx=int(torch.randint(0,G,(1,),device=DEV,generator=g).item())
            food[cx,:]=torch.maximum(food[cx,:],(torch.rand(G,device=DEV,generator=g)<0.6).float())  # trilha de comida
        # regen ESTACIONARIO: repoe comida p/ manter densidade ~constante (desconfunde skill x disponibilidade)
        dens=float(food.mean().item())
        if dens<0.12: food=torch.clamp(food+(torch.rand(G,G,device=DEV,generator=g)<(0.12-dens)).float(),0,1)
        if (t+1)%1000==0: eaten_hist.append((t+1,round(eaten_win/1000/N,3))); eaten_win=0.0
    return eaten_hist

def main():
    t0=time.time()
    print("="*68); print(f"H116 Cerebros basais + aprendizado dentro da vida (REINFORCE) | {N} agentes em {DEV}")
    res={}
    for ll,lab in [(0.0,"CONGELADO (sem plasticidade — controle)"),(0.03,"APRENDIZ (plasticidade ON)")]:
        h=run(False,lr=ll); res[lab]=h
        print(f"\n  {lab}  (comida/agente por passo, janelas de 1000):")
        print("    "+" ".join(f"{v}" for _,v in h))
    F=res["CONGELADO (sem plasticidade — controle)"]; L=res["APRENDIZ (plasticidade ON)"]
    print("="*68)
    print(f"H116 — {time.time()-t0:.0f}s | comida estacionaria (skill isolada de disponibilidade)")
    print(f"  CONGELADO: inicio {F[0][1]} -> fim {F[-1][1]}")
    print(f"  APRENDIZ : inicio {L[0][1]} -> fim {L[-1][1]}")
    learned = L[-1][1] > F[-1][1]*1.15 and L[-1][1] > L[0][1]*1.15
    print(f"  VEREDITO: cerebro basal APRENDEU dentro da vida? {'SIM — aprendiz supera congelado e sobe (plasticidade real)' if learned else 'nao claramente neste setup'}")
    print("  Honesto: aprendizado embodied por plasticidade (ontogenia), poucos agentes, sem imitar. Nao e AGI.")
    print("  Se SIM: justifica a ponte pro Minecraft REAL (mineflayer headless; Node+Java ja presentes).")

if __name__=="__main__": main()
