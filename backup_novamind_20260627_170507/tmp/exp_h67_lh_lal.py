"""
H67 — Corno Lateral (LH, valor inato) + Lóbulos Acessórios Laterais (LAL, árbitro de decisão)
=============================================================================================
PDFs do usuário: o LH codifica VALENCIAS INATAS (perigo/comida a priori) e o LAL ARBITRA entre
seguir o plano aprendido (MB) e responder a um instinto imediato (gating pro-objetivo vs
anti-objetivo). Isso e a faculdade de AGENTE que nossos experimentos de curiosidade-do-zero
(H11-13) tentaram aprender e falharam.

Hipotese testavel e honesta: priores de valor inato (LH) + gating (LAL) evitam a CATASTROFE
INICIAL que um aprendiz tabula-rasa so aprende MORRENDO. Mundo de forrageio com predador:
 - comida = objetivo APRENDIDO (Q-learning, analogo MB/DAN-MBON erro de predicao);
 - predador = perigo INATO (LH: valencia negativa a priori, sem precisar aprender);
 - LAL: persegue comida, mas SOBREPOE p/ fugir quando o predador esta perto.
Compara: (1) so-aprendido (precisa aprender o perigo via mortes) vs (2) LH+LAL (aversao inata).
Metrica: MORTES ao longo dos episodios (esp. no inicio) + comida coletada. CPU, segundos.
Honesto: faculdade de agente narrow real (o "ouro": instinto evita erro fatal de exploracao);
NAO e ARC, NAO e AGI.
"""
import numpy as np
rng=np.random.RandomState(0)
G=7; ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]   # cima,baixo,esq,dir

def clamp(p): return (min(max(p[0],0),G-1),min(max(p[1],0),G-1))
def sgn(x): return (x>0)-(x<0)

class World:
    def __init__(s): s.reset()
    def reset(s):
        s.a=(rng.randint(G),rng.randint(G))
        s.f=s._free(); s.p=s._free(far=s.a); return s
    def _free(s,far=None):
        while True:
            q=(rng.randint(G),rng.randint(G))
            if q!=getattr(s,'a',None) and (far is None or abs(q[0]-far[0])+abs(q[1]-far[1])>=3): return q
    def pred_dist(s): return max(abs(s.a[0]-s.p[0]),abs(s.a[1]-s.p[1]))
    def step(s,act):
        s.a=clamp((s.a[0]+ACTIONS[act][0],s.a[1]+ACTIONS[act][1]))
        # perigo EVITAVEL: passeio aleatorio lento (nao persegue) -> politica competente evita
        if rng.rand()<0.25:
            mv=ACTIONS[rng.randint(4)]; s.p=clamp((s.p[0]+mv[0],s.p[1]+mv[1]))
        r=-0.05; dead=False; got=False
        if s.a==s.p: r=-10.0; dead=True
        elif s.a==s.f: r=5.0; got=True; s.f=s._free()
        return r,dead,got
    def safe(s,act):  # a acao leva a uma celula segura (nao sobre o perigo)?
        na=clamp((s.a[0]+ACTIONS[act][0],s.a[1]+ACTIONS[act][1])); return na!=s.p

def state_full(w):  # estado do aprendiz: direcoes p/ comida E predador + perigo perto
    return (sgn(w.f[0]-w.a[0]),sgn(w.f[1]-w.a[1]),sgn(w.p[0]-w.a[0]),sgn(w.p[1]-w.a[1]),int(w.pred_dist()<=1))
def state_food(w):  # estado do forrageador puro: so direcao p/ comida
    return (sgn(w.f[0]-w.a[0]),sgn(w.f[1]-w.a[1]))

def flee_action(w):  # LH/LAL: instinto inato de fuga (afasta do predador)
    dy,dx=w.a[0]-w.p[0],w.a[1]-w.p[1]
    if abs(dy)>=abs(dx): return 1 if dy>0 else 0     # mover no eixo y p/ longe
    return 3 if dx>0 else 2

def run(mode, episodes=4000, maxsteps=40):
    Q={}; eps=0.2; alpha=0.4; gamma=0.9
    def qget(st): return Q.setdefault(st,[0.0,0.0,0.0,0.0])
    deaths=[]; foods=[]
    for ep in range(episodes):
        w=World(); d=0; f=0
        for t in range(maxsteps):
            sf=state_full(w) if mode=='learned' else state_food(w)
            q=qget(sf)
            if rng.rand()<eps: act=rng.randint(4)
            else: act=int(np.argmax(q))
            # LAL: instinto inato (LH) VETA acoes que pisam no perigo; aprendido escolhe entre as seguras
            if mode=='lh_lal' and not w.safe(act):
                safe_acts=[a for a in range(4) if w.safe(a)]
                if safe_acts:
                    act=max(safe_acts,key=lambda a:q[a]) if rng.rand()>eps else rng.choice(safe_acts)
            r,dead,got=w.step(act)
            if got: f+=1
            if dead: d+=1
            ns=state_full(w) if mode=='learned' else state_food(w)
            nq=qget(ns); q[act]+=alpha*(r+gamma*max(nq)-q[act])
            if dead: w.reset()
        deaths.append(d); foods.append(f)
    return np.array(deaths),np.array(foods)

def main():
    print("="*70)
    print("H67 LH (valor inato) + LAL (arbitro) — forrageio com predador")
    print("  hipotese: instinto inato evita a catastrofe inicial que o tabula-rasa aprende morrendo\n")
    dl,fl=run('learned'); dh,fh=run('lh_lal')
    def avg(x,a,b): return x[a:b].mean()
    print(f"  {'fase':>16} | {'mortes/ep (so-aprendido)':>26} | {'mortes/ep (LH+LAL)':>20}")
    print(f"  {'1os 200 eps':>16} | {avg(dl,0,200):>26.2f} | {avg(dh,0,200):>20.2f}")
    print(f"  {'eps 200-1000':>16} | {avg(dl,200,1000):>26.2f} | {avg(dh,200,1000):>20.2f}")
    print(f"  {'ultimos 1000':>16} | {avg(dl,-1000,len(dl)):>26.2f} | {avg(dh,-1000,len(dh)):>20.2f}")
    print(f"\n  MORTES TOTAIS: so-aprendido {dl.sum():.0f}  vs  LH+LAL {dh.sum():.0f}")
    print(f"  COMIDA (ultimos 1000 eps): so-aprendido {fl[-1000:].mean():.2f}  vs  LH+LAL {fh[-1000:].mean():.2f}")
    win = dh.sum()<dl.sum()*0.5
    print(f"\n  => {'LH+LAL VENCE' if win else 'sem vantagem clara'}: o valor inato + gating {'evita' if win else 'nao evita'} a catastrofe")
    print(f"     inicial (o aprendiz puro paga em mortes ate aprender o perigo). 'Ouro': instinto =")
    print(f"     motivacao/seguranca que a curiosidade-do-zero (H11-13) nao conseguiu. NAO e ARC, NAO e AGI.")

if __name__=="__main__": main()
