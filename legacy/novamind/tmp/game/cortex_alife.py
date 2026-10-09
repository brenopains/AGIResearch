"""
cortex_alife.py — A LIGACAO: o MAESTRO da AGI governa os NPCs-elite do mundo ALife (tipo-Minecraft).
NAO trapaceia: nao seta acao, nao injeta codigo do jogador. Funciona como CORTEX TOP-DOWN (atencao/
neuromodulacao do global-workspace): a cada N ticks le o estado dos elite, PROPOE um objetivo (vies nos
logits de uma acao), observa o progresso, e APRENDE SOZINHO qual objetivo ajuda — via o loop do maestro:
  PERCEBE (fingerprint do estado) -> MEMORIA (Q por fingerprint, reuso cross-estado) -> PROPOE objetivo
  (bandido dirigido por AFETO: + frustrado -> + exploracao) -> APLICA vies -> recompensa = delta-progresso
  -> atualiza Q + afeto -> DMT (surto de plasticidade nos pesos do elite quando trava e frustrado).
A rede EVOLUIDA do NPC ainda decide (o vies so enviesa, como dopamina). Eu NAO digo qual acao e boa:
o cortex descobre pela retroacao do mundo (sem primitiva minha). A/B mede se acelera tech/construcao.
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from world import World, OUT, DEV, P

GOALS = list(range(OUT))   # candidatos: enviesar p/ cada uma das 12 acoes; o cortex aprende quais ajudam

class Cortex:
    def __init__(s, k=60, every=40, bias=2.5, lr=0.4):
        s.k=k; s.every=every; s.bias=bias; s.lr=lr
        s.frust=0.0; s.mem={}                 # fingerprint -> Q[goal]  (memoria que transfere entre estados)
        s.cur_goal=None; s.cur_fp=None; s.prev_metric=None; s.elite=None; s.bursts=0; s.goal_hist=[]
    def _elite(s, W):
        a=W.alive.clone()
        if W.player: a[0]=False
        idx=a.nonzero().flatten()
        if len(idx)==0: return idx
        score=W.energy[idx]+0.3*W.wealth[idx]
        return idx[torch.topk(score, min(s.k,len(idx))).indices]
    def _fingerprint(s, W, elite):
        if len(elite)==0: return (0,0,0)
        mt=int(W.tools[elite].sum(1).mean().item())
        inv=int(min(5, round(W.inv[elite].sum(1).mean().item())))
        au=int(round((W.tools[elite,3:].sum(1)>0).float().mean().item()*4))
        return (mt, inv, au)
    def _metric(s, W, elite):                  # progresso = tech (peso alto) + construcao + energia
        if len(elite)==0: return 0.0
        return (float(W.tools[elite].sum(1).mean().item())*10.0
                + float((W.block>0).sum().item())*0.002
                + float(W.energy[elite].mean().item()))
    def govern(s, W):
        elite=s._elite(W); m=s._metric(W, elite)
        # 1) RECOMPENSA o objetivo anterior pelo delta-progresso -> atualiza Q + AFETO
        if s.cur_goal is not None and s.prev_metric is not None and s.cur_fp is not None:
            r=m - s.prev_metric
            Q=s.mem.setdefault(s.cur_fp, [0.0]*len(GOALS))
            Q[s.cur_goal]+= s.lr*(r - Q[s.cur_goal])
            s.frust = max(0.0, s.frust-0.15) if r>0.01 else min(1.0, s.frust+0.2)
        # 2) DMT: travado + frustrado -> SURTO DE PLASTICIDADE nos pesos do elite (escapa do mode-collapse)
        burst=False
        if s.frust>0.6 and len(elite)>0:
            with torch.no_grad():
                W.W1[elite]+=torch.randn(len(elite),*W.W1.shape[1:],device=DEV,generator=W.g)*0.05*(W.W1[elite].std()+1e-4)
                W.W2[elite]+=torch.randn(len(elite),*W.W2.shape[1:],device=DEV,generator=W.g)*0.05*(W.W2[elite].std()+1e-4)
            burst=True; s.bursts+=1; s.frust=0.4
        # 3) PROPOE proximo objetivo: bandido dirigido por afeto (+frustrado -> +exploracao). Memoria recupera o Q.
        fp=s._fingerprint(W, elite); Q=s.mem.setdefault(fp, [0.0]*len(GOALS))
        eps=0.15+0.6*s.frust
        if float(torch.rand(1,device=DEV,generator=W.g))<eps:
            goal=int(torch.randint(0,len(GOALS),(1,),device=DEV,generator=W.g))
        else:
            goal=int(np.argmax(Q))
        # 4) APLICA o vies de objetivo nos elite (top-down). Default fora-do-elite = 0 (sem cortex).
        W.goal_bias.zero_(); W.elite_mask.zero_()
        if len(elite)>0:
            W.goal_bias[elite, goal]=s.bias; W.elite_mask[elite]=True
        s.cur_goal=goal; s.cur_fp=fp; s.prev_metric=m; s.elite=elite; s.goal_hist.append(goal)
        return goal, s.frust, burst
    def state(s):
        return {"goal":s.cur_goal,"frust":round(s.frust,2),"bursts":s.bursts,
                "elite":int(len(s.elite)) if s.elite is not None else 0,"mem":len(s.mem)}

A_NAMES={0:"cima",1:"baixo",2:"dir",3:"esq",4:"COMER",5:"PEGAR",6:"CRAFTAR",7:"falar",8:"trocar",9:"atacar",10:"CONSTRUIR",11:"compartilhar"}

def metrics(W):
    a=W.alive; pop=int(a.sum())
    if pop==0: return (0,0,0.0,0)
    return (pop, int(W.tools[a].sum(1).max().item()), float((W.tools[a].sum(1)>0).float().mean().item()), int((W.block>0).sum().item()))

def _hard(W):
    """Mundo DIFICIL: tech/sobrevivencia viram luta (sem isso o mundo satura no tick 500 e nao ha o que acelerar)."""
    W.alive[:]=False; W.alive[:300]=True; W.energy[:]=0.0; W.energy[W.alive]=1.5     # comeca pequeno
    W.META=0.03; W.EAT=0.32; W.REPRO=3.6; W.MAXAGE=1200.0                            # metabolismo duro, repro caro
    W.mat*=0.5                                                                       # materiais escassos
    return W

def run(with_cortex, steps=4000, seed=1, every=40, hard=False):
    W=World(seed=seed)
    if hard: _hard(W)
    cx=Cortex(every=every) if with_cortex else None
    hist=[]; ms={'tier6':None,'tier12':None,'blk500':None}
    for t in range(steps):
        if cx is not None and t % cx.every == 0: cx.govern(W)
        W.step()
        a=W.alive; pop=int(a.sum())
        if pop:
            mt=int(W.tools[a].sum(1).max().item()); bl=int((W.block>0).sum().item())
            if ms['tier6'] is None and mt>=6: ms['tier6']=t
            if ms['tier12'] is None and mt>=12: ms['tier12']=t
            if ms['blk500'] is None and bl>=500: ms['blk500']=t
        if t%500==0 or t==steps-1:
            pop,mt,fr,bl=metrics(W); hist.append((t,pop,mt,round(fr,2),bl))
    return hist, cx, ms

def main():
    t0=time.time(); steps=4000; seeds=[1,2,3]
    print("="*78)
    print(f"LIGACAO: o MAESTRO da AGI governa os NPCs-elite do ALife? (A/B headless HARD, {steps} ticks, seeds {seeds})")
    print(f"  dispositivo: {DEV}")
    def avg_ms(runs,key):
        vs=[m[key] for _,_,m in runs if m[key] is not None]
        return (sum(vs)/len(vs), len(vs))           # (tick medio p/ atingir, quantos seeds atingiram)
    ctrl=[run(False, steps=steps, seed=sd, hard=True) for sd in seeds]
    cort=[run(True,  steps=steps, seed=sd, hard=True) for sd in seeds]
    def fin(runs):
        F=[h[-1] for h,_,_ in runs]
        return (np.mean([f[1] for f in F]), np.mean([f[2] for f in F]), np.mean([f[3] for f in F]), np.mean([f[4] for f in F]))
    ca=fin(ctrl); cb=fin(cort)
    print("="*78)
    print(f"final CONTROLE : pop {ca[0]:.0f}  maxtier {ca[1]:.1f}  frac-tool {ca[2]:.2f}  blocos {ca[3]:.0f}")
    print(f"final C/CORTEX : pop {cb[0]:.0f}  maxtier {cb[1]:.1f}  frac-tool {cb[2]:.2f}  blocos {cb[3]:.0f}")
    print("  tempo-ate-marco (tick medio | nº de seeds que atingiram, de 3):")
    for key,lab in [('tier6','tier>=6'),('tier12','tier12'),('blk500','blocos>=500')]:
        (mc,nc)=avg_ms(ctrl,key) if avg_ms(ctrl,key)[1] else (float('nan'),0)
        (mk,nk)=avg_ms(cort,key) if avg_ms(cort,key)[1] else (float('nan'),0)
        print(f"    {lab:>12}: controle {mc:>7.0f} ({nc}/3) | cortex {mk:>7.0f} ({nk}/3)")
    from collections import Counter
    allg=sum((cx.goal_hist for _,cx,_ in cort if cx),[]); top=Counter(allg).most_common(4)
    bursts=sum(cx.bursts for _,cx,_ in cort if cx); mem=max((len(cx.mem) for _,cx,_ in cort if cx),default=0)
    print(f"  cortex: {bursts} surtos-DMT (total) | memoria ate {mem} fingerprints | objetivos+usados: "
          + ", ".join(f"{A_NAMES[g]}({n})" for g,n in top))
    # marco-chave: tier12 mais rapido OU mais seeds chegando, OU mais blocos/tier final
    t12c=avg_ms(ctrl,'tier12'); t12k=avg_ms(cort,'tier12')
    faster = (t12k[1]>t12c[1]) or (t12k[1]==t12c[1] and t12k[1]>0 and t12k[0]<t12c[0]-20)
    better = (cb[1]>ca[1]+0.3) or (cb[3]>ca[3]*1.15)
    helped = faster or better
    print(f"  VEREDITO: {'o CORTEX ACELEROU/melhorou tech-construcao vs evolucao pura' if helped else 'o cortex NAO superou a evolucao pura neste regime (honesto)'}")
    print(f"  ({time.time()-t0:.0f}s) — honesto: emergencia de ALife; o maestro PROPOE objetivos+DMT, nao trapaceia.")

if __name__=="__main__": main()
