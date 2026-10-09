"""
run_arc_priors.py — TESTE da tese do usuario: exposicao a MULTIPLOS priors (geometria, objetidade, simetria,
contagem, automato, etc.) num world-model+memoria+imaginacao COMPARTILHADOS prepara o cerebro p/ uma tarefa
com REGRAS NOVAS — exatamente o que o ARC cobra (e que reuso de primitiva nao resolve).

Compara, na 7a familia HELD-OUT (gravity/fisica, nunca vista no treino), few-shot (so suporte, sem atualizar pesos):
  - LARGO  : world-model treinado nas 6 familias de priors (multidominio)
  - ESTREITO: world-model treinado em 1 familia so (mesmos passos)
Se LARGO >> ESTREITO no prior novo => a largura de priors da vantagem em regras novas (a tese, com numero).
Metrica estilo ARC: acuracia por celula + ACERTO EXATO da grade (a metrica real do ARC).
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.grid_cognition import GridWorldModel, GridDomains, TRAIN_FAMS, HELDOUT_FAM, C, H, W

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D = GridDomains(DEV)
B, K, Q, STEPS = 64, 4, 4, 4500

def train(model, fams, steps, tag):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    print(f"[{DEV}] treinando {tag} ({'+'.join(fams) if len(fams)<=2 else str(len(fams))+' priors'})...")
    for it in range(steps):
        fam = fams[torch.randint(0, len(fams), (1,)).item()]
        (xs, ys), (xq, yq) = D.episode(fam, B, K, Q)
        logits = model(xq, c=model.context((xs, ys)))
        loss = F.cross_entropy(logits.reshape(B * Q, C, H, W), yq.reshape(B * Q, H, W))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0: print(f"  it{it} loss {loss.item():.3f}")
    return model

@torch.no_grad()
def evaluate(model, fam, n=40):
    cell, exact = [], []
    for _ in range(n):
        (xs, ys), (xq, yq) = D.episode(fam, B, K, Q)
        pred = model(xq, c=model.context((xs, ys))).argmax(2)   # [B,Q,H,W]
        tgt = yq.reshape(B, Q, H, W)
        cell.append((pred == tgt).float().mean().item())
        exact.append((pred == tgt).all(-1).all(-1).float().mean().item())
    return sum(cell) / len(cell), sum(exact) / len(exact)

# baseline trivial: prever a propria entrada (identidade) — mede quao "nao-trivial" e a familia
@torch.no_grad()
def identity_baseline(fam, n=40):
    cell, exact = [], []
    for _ in range(n):
        (_, _), (xq, yq) = D.episode(fam, B, K, Q)
        pred = xq.reshape(B, Q, H, W); tgt = yq.reshape(B, Q, H, W)
        cell.append((pred == tgt).float().mean().item()); exact.append((pred == tgt).all(-1).all(-1).float().mean().item())
    return sum(cell) / len(cell), sum(exact) / len(exact)

broad = train(GridWorldModel().to(DEV), TRAIN_FAMS, STEPS, "LARGO (6 priors)")
narrow = train(GridWorldModel().to(DEV), ["geom"], STEPS, "ESTREITO (so geom)")

print("\n" + "=" * 70 + "\nRESULTADOS — acuracia (celula / ACERTO EXATO da grade, metrica ARC)\n" + "=" * 70)
res = {}
print("\n[A] LARGO domina as familias de treino? (aprendeu os priors)")
for f in TRAIN_FAMS:
    c, e = evaluate(broad, f); res[f"broad_{f}"] = [c, e]; print(f"   {f:10s}  celula {c:5.1%}  exato {e:5.1%}")
ib_c, ib_e = identity_baseline(HELDOUT_FAM)
bg_c, bg_e = evaluate(broad, HELDOUT_FAM); res["broad_gravity"] = [bg_c, bg_e]
ng_c, ng_e = evaluate(narrow, HELDOUT_FAM); res["narrow_gravity"] = [ng_c, ng_e]
res["identity_gravity"] = [ib_c, ib_e]
print(f"\n[B] PRIOR NOVO (held-out: {HELDOUT_FAM}/fisica), few-shot sem treinar nele:")
print(f"   baseline identidade ...... celula {ib_c:5.1%}  exato {ib_e:5.1%}   (quao nao-trivial e a tarefa)")
print(f"   ESTREITO (so geom) ....... celula {ng_c:5.1%}  exato {ng_e:5.1%}")
print(f"   LARGO (6 priors) ......... celula {bg_c:5.1%}  exato {bg_e:5.1%}")
print("\nVEREDITO:")
ok_learn = sum(res[f'broad_{f}'][1] for f in TRAIN_FAMS) / len(TRAIN_FAMS) > 0.5
ok_breadth_cell = bg_c - ng_c > 0.05
ok_breadth_exact = bg_e - ng_e > 0.05
ok_nontrivial = bg_c - ib_c > 0.05
print(f"  - LARGO aprendeu os priors de treino (exato medio > 50%)?    {'SIM' if ok_learn else 'NAO'}")
print(f"  - Largura ajuda no PRIOR NOVO vs estreito (celula)?          {'SIM' if ok_breadth_cell else 'NAO'}  ({bg_c:.1%} vs {ng_c:.1%})")
print(f"  - Largura ajuda no PRIOR NOVO vs estreito (acerto exato)?    {'SIM' if ok_breadth_exact else 'NAO'}  ({bg_e:.1%} vs {ng_e:.1%})")
print(f"  - E nao-trivial (LARGO > identidade)?                        {'SIM' if ok_nontrivial else 'NAO'}")
print(f"  => exposicao multi-prior {'DA vantagem em regras novas (tese sustentada)' if (ok_breadth_cell and ok_nontrivial) else 'NAO mostrou vantagem clara aqui'}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), 'arc_priors_result.json'), 'w'), indent=1)
