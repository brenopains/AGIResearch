"""
H115 — Os NPCs aprendem com o JOGADOR? (ALife tipo-Minecraft, A/B headless)
============================================================================
A visao do usuario: NPCs vivos num mundo de blocos que sobrevivem, craftam, constroem, evoluem
E APRENDEM COM O PLAYER. O world.py ja e essa ALife (sobrevivencia/mineracao/crafting/construcao/
cultura/evolucao + player observavel sem trapaca). Aqui medimos a parte nova: aprendizado-pelo-
jogador. A/B headless:
  - SEM player: sociedade evolui sozinha.
  - COM player-PROFESSOR: um jogador scriptado DEMONSTRA boas acoes (pegar->craftar->construir->comer);
    agentes por perto observam e copiam (com ego). Mede se a sociedade alcanca tech/construcao/
    sobrevivencia MAIS RAPIDO te observando.
Honesto: emergencia de ALife medida (nao AGI). So-GPU/CPU, roda na 3050.
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"game"))
from world import World, A_EAT,A_PICK,A_CRAFT,A_BUILD

def metrics(W):
    a=W.alive; pop=int(a.sum())
    if pop==0: return (0,0.0,0.0,0)
    maxtier=int(W.tools[a].sum(1).max().item())
    fr_tool=float((W.tools[a].sum(1)>0).float().mean().item())
    blocks=int((W.block>0).sum().item())
    return (pop,maxtier,fr_tool,blocks)

def run(with_player, steps=4000, seed=1):
    W=World(seed=seed)
    if with_player: W.spawn_player()
    # roteiro do professor: demonstra a cadeia de sobrevivencia/tech
    demo=[A_PICK,A_PICK,A_CRAFT,A_BUILD,A_EAT]
    hist=[]
    for t in range(steps):
        if with_player and bool(W.alive[0]):
            W.set_player_action(demo[(t//40)%len(demo)])   # muda a demonstracao a cada 40 ticks
            if t%11==0: W.x[0]=(W.x[0]+1)%110; W.y[0]=(W.y[0]+1)%110   # professor anda p/ espalhar a influencia
        W.step()
        if t%500==0 or t==steps-1:
            pop,mt,fr,bl=metrics(W); hist.append((t,pop,mt,round(fr,2),bl))
    return hist

def main():
    t0=time.time()
    print("="*72)
    print(f"H115 ALife tipo-Minecraft: os NPCs aprendem com o JOGADOR? (A/B headless)")
    for wp,lab in [(False,"SEM player (evolui sozinho)"),(True,"COM player-PROFESSOR (demonstra pegar->craftar->construir->comer)")]:
        h=run(wp,steps=4000)
        print(f"\n  {lab}:")
        print(f"    {'tick':>5} {'pop':>5} {'maxtier':>8} {'frac-tool':>10} {'blocos':>7}")
        for tk,pop,mt,fr,bl in h: print(f"    {tk:>5} {pop:>5} {mt:>8} {fr:>10.2f} {bl:>7}")
        globals()['R_'+str(wp)]=h
    A=R_False[-1]; B=R_True[-1]
    print("="*72)
    print(f"H115 — {time.time()-t0:.0f}s | final: SEM player frac-tool {A[3]} blocos {A[4]} | COM player {B[3]} blocos {B[4]}")
    helped = (B[3]>A[3]+0.03) or (B[4]>A[4]*1.3)
    print(f"  VEREDITO: {'os NPCs APRENDERAM com o jogador (tech/construcao mais rapido te observando)' if helped else 'observar o jogador nao acelerou claramente neste run'}")
    print("  Honesto: emergencia de ALife medida (sobrevivencia/tech/construcao/cultura). Jogavel via tmp/game/server.py. Nao e AGI.")

if __name__=="__main__": main()
