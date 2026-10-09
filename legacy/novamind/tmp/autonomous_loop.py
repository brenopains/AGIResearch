"""
LOOP AUTONOMO (active-inference + crescimento estrutural) — H188. Fecha a parte de AUTONOMIA: o sistema detecta
a PROPRIA surpresa (residuo que o menu ATIVO nao explica) e RECRUTA sozinho o prior dormente (core-knowledge) que
reduz o residuo — sem ninguem dizer a lei nem qual arquetipo. Contraste direto com H184 (crescer por COMPOSICAO
estagnou 9/400): aqui a 'vida' cresce por RECRUTAR ARQUETIPO -> o poder (cobertura) deve SUBIR.
Pool de priors inatos (Spelke): {LOCAL, GLOBAL, OBJETO}. Menu ATIVO comeca {LOCAL} (GLOBAL e OBJETO dormentes).
Stream de tarefas de 3 tipos. Por tarefa: tenta o menu ativo; se nenhum ajusta (surpresa), marca. Recrutamento:
varre CADA prior dormente contra CADA tarefa-surpresa; recruta os que ajustam (treino-exato + held-out). Mede
cobertura antes/depois. Honesto: criterio de 'resolve' = treino exato E held-out alto (alinhado ao solver de ARC).
"""
import time, numpy as np, torch
from archetype_menu import CNNLocal, GlobalCtx, ObjectNet, L_local, L_global, L_object, make, train, ev, DEV
np.random.seed(1); torch.manual_seed(1)
POOL = {"LOCAL": CNNLocal, "GLOBAL": GlobalCtx, "OBJETO": ObjectNet}
LAWS = [("local", L_local), ("global", L_global), ("objeto", L_object)]
TR_TH, TE_TH = 0.95, 0.8                                    # ajusta treino quase-exato E generaliza held-out

def task(law, H=8, W=8): Xtr, Ytr = make(400, H, W, law); Xte, Yte = make(200, H, W, law); return Xtr, Ytr, Xte, Yte
def fit(cls, Xtr, Ytr, Xte, Yte):
    m = train(cls().to(DEV), Xtr, Ytr, iters=1000); return ev(m, Xtr, Ytr), ev(m, Xte, Yte)
def solves(active, Xtr, Ytr, Xte, Yte):
    for nm in active:
        trf, tef = fit(POOL[nm], Xtr, Ytr, Xte, Yte)
        if trf > TR_TH and tef > TE_TH: return nm
    return None
def coverage(active, stream):
    solved = 0; surpr = []
    for name, law in stream:
        d = task(law)
        if solves(active, *d): solved += 1
        else: surpr.append((name, law, d))
    return solved, surpr

def main():
    t0 = time.time(); stream = [LAWS[i % 3] for i in range(9)]
    print(f"[{DEV}] LOOP AUTONOMO | pool={list(POOL)} | menu ATIVO inicial=[LOCAL] (GLOBAL,OBJETO dormentes)", flush=True)
    active = ["LOCAL"]
    cov0, surpr = coverage(active, stream)
    print(f"  FASE A (menu {active}): cobertura={cov0}/9 | SURPRESA em: {sorted(set(s[0] for s in surpr))}", flush=True)
    # RECRUTAMENTO AUTONOMO: cada prior dormente vs cada tarefa-surpresa; recruta os que reduzem a surpresa
    dormant = [k for k in POOL if k not in active]; recruited = []
    seen_types = set()
    for name, law, d in surpr:
        if name in seen_types: continue                    # 1 sonda por tipo de surpresa basta
        seen_types.add(name)
        for nm in dormant:
            if nm in recruited: continue
            trf, tef = fit(POOL[nm], *d)
            print(f"    surpresa '{name}': sonda prior dormente {nm} -> treino={trf:.0%} held-out={tef:.0%}", flush=True)
            if trf > TR_TH and tef > TE_TH:
                recruited.append(nm); active.append(nm)
                print(f"      -> RECRUTOU '{nm}' (reduziu a surpresa de '{name}').", flush=True); break
    cov1, _ = coverage(active, stream)
    print("=" * 70)
    print(f"  PODER: {cov0}/9 -> {cov1}/9 | menu final={active} | recrutados autonomamente={recruited}")
    if cov1 > cov0:
        print("  Crescer o MENU de priors (recrutar arquetipo) CRESCE o poder — autonomo, guiado por surpresa.")
        print("  CONTRASTA com H184 (composicao: 9->9 estagnado): composicao fica no fecho; recrutar arquetipo sai.")
    else:
        print("  recrutar nao subiu o poder -> menu nao SPANS; precisa INVENTAR arquetipo (frontier).")
    print(f"  ({time.time()-t0:.0f}s)")
    print("  CAVEAT HONESTO: priors vem de um POOL inato (Spelke), nao inventados do zero. Frontier aberta =")
    print("  AUTO-GERAR arquetipo estrutural genuinamente novo quando NENHUM prior do pool reduz a surpresa.")

if __name__ == "__main__":
    main()
