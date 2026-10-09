"""
run_arc_integrated.py — NovaMind ARC-AGI Full Integrated Validation
=====================================================================

Pipeline completo:
  1. GridWorldModel (transformer in-context, induz regra dos K pares suporte)
  2. ArchetypeEngine (nomeia padrao reconhecido nos 50k conceitos)
  3. NeuromodulatorState (DA por novidade, CRH por incerteza)
  4. Cerebellum Forward/Inverse models (antecipa erro de transformacao)
  5. TemporalBinder + MultiScaleHDV (episodio gated por saliencia NT)

Uso:
    python run_arc_integrated.py            # full run (GPU se disponivel)
    python run_arc_integrated.py --smoke    # 50 tasks, 50 steps (rapido)
"""

import os, sys, time, json, argparse, random
import io
# Forca UTF-8 no stdout para evitar UnicodeEncodeError no Windows (cp1252)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset
from core.archetypes import ArchetypeEngine
from core.cerebellum import Cerebellum
from core.temporal_binder import TemporalBinder
from core.multiscale_hdv import MultiScaleHDV

try:
    from core.neuromodulator_state import NeuromodulatorState
    _HAS_NT = True
except ImportError:
    _HAS_NT = False
    print("[WARN] NeuromodulatorState nao encontrado, rodando sem modulacao NT.")

ARC_TRAIN_DIR = "data/ARC/data/training"
ARC_EVAL_DIR  = "data/ARC/data/evaluation"

FULL_TRAIN_STEPS  = 300
FULL_EVAL_TASKS   = 400
SMOKE_TRAIN_STEPS = 50
SMOKE_EVAL_TASKS  = 50

MAX_H = MAX_W = 30
PAD_VAL   = 10
N_COLORS  = 11

MODEL_DIM    = 128
MODEL_HEADS  = 4
MODEL_LAYERS = 3
BATCH_SIZE   = 16
K_SUPPORT    = 3
Q_QUERIES    = 1

CEREB_STATE_DIM  = 15
CEREB_ACTION_DIM = 4
NT_SALIENCE_DA   = 0.6
NT_SALIENCE_CRH  = 0.55


def grid_to_state_vec(grid_flat: torch.Tensor) -> np.ndarray:
    g  = grid_flat.cpu().numpy().astype(float)
    hw = len(g)
    h  = w = int(hw ** 0.5)
    hist = np.array([(g == c).sum() / max(1, hw) for c in range(10)])
    g2   = g[:h*w].reshape(h, w)
    nz   = (g2 > 0) & (g2 < 10)
    rows, cols = np.where(nz)
    cx  = cols.mean() / w if len(cols) > 0 else 0.5
    cy  = rows.mean() / h if len(rows) > 0 else 0.5
    den = nz.mean()
    sym_h = np.mean(g2 == np.fliplr(g2))
    sym_v = np.mean(g2 == np.flipud(g2))
    vec = np.concatenate([hist[:6], [cx, cy, den, sym_h, sym_v],
                          [0.0] * (CEREB_STATE_DIM - 11)])
    return vec[:CEREB_STATE_DIM].astype(np.float32)


def grid_delta_action(s_before: np.ndarray, s_after: np.ndarray) -> np.ndarray:
    d = (s_after - s_before)
    a = np.zeros(CEREB_ACTION_DIM, dtype=np.float32)
    a[:min(CEREB_ACTION_DIM, len(d))] = d[:CEREB_ACTION_DIM]
    return a


def nt_from_novelty(arch_hit: bool, pred_err: float) -> Dict:
    da  = 0.75 if arch_hit else 0.3
    crh = min(1.0, pred_err * 2.0)
    na  = 0.5 * (da + crh)
    ach = max(0.2, 0.6 - crh * 0.3)
    sht = max(0.1, 0.4 - crh * 0.2) if not arch_hit else 0.5
    return {"DA": da, "NA": na, "ACh": ach, "5HT": sht, "CRH": crh, "OXT": 0.3}


def archetype_probe(engine: ArchetypeEngine, concepts: List[str]) -> Tuple[bool, str]:
    for c in concepts:
        engine.observe([c])
    arch = engine.archetypes
    if arch:
        top = max(arch.keys(), key=lambda a: engine.archetype_usage.get(a, 0))
        return True, top
    return False, "UNKNOWN"


def grid_to_concepts(grid_flat: torch.Tensor) -> List[str]:
    g2  = grid_flat.cpu().numpy().astype(float)
    hw  = len(g2)
    h   = w = int(hw ** 0.5)
    g2  = g2[:h*w].reshape(h, w)
    nz  = (g2 > 0) & (g2 < 10)
    den = nz.mean()
    concepts = []
    if np.mean(g2 == np.fliplr(g2)) > 0.8: concepts.append("ESPELHO")
    if np.mean(g2 == np.flipud(g2)) > 0.8:  concepts.append("ESPELHO")
    if den < 0.1:   concepts.append("ESCASSO")
    elif den > 0.5: concepts.append("DENSO")
    uniq = len(np.unique(g2[nz])) if nz.any() else 1
    if uniq == 1:  concepts.append("UNIFORME")
    elif uniq > 4: concepts.append("MULTICOLOR")
    top_h = nz[:h//2].mean(); bot_h = nz[h//2:].mean()
    if bot_h > top_h * 1.5: concepts.append("QUEDA")
    if top_h > bot_h * 1.5: concepts.append("SUBIDA")
    if h >= 4 and np.allclose(g2[:h//2], g2[h//2:h//2*2], atol=0): concepts.append("REPETICAO")
    return concepts or ["OBJETO"]


def run(smoke: bool = False):
    print("=" * 65)
    print("   NovaMind — ARC-AGI INTEGRATED VALIDATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_steps = SMOKE_TRAIN_STEPS if smoke else FULL_TRAIN_STEPS
    eval_limit  = SMOKE_EVAL_TASKS  if smoke else FULL_EVAL_TASKS
    label       = "SMOKE TEST" if smoke else "FULL RUN"
    print(f"Mode   : {label}  |  train steps: {train_steps}  |  eval tasks: {eval_limit}\n")

    train_ds = ARCDataset(data_dir=ARC_TRAIN_DIR, max_h=MAX_H, max_w=MAX_W, pad_val=PAD_VAL, dev=device)
    eval_ds  = ARCDataset(data_dir=ARC_EVAL_DIR,  max_h=MAX_H, max_w=MAX_W, pad_val=PAD_VAL, dev=device)

    if not train_ds.tasks:
        print(f"[ERROR] Nenhuma task ARC em {ARC_TRAIN_DIR}")
        sys.exit(1)

    hw = MAX_H * MAX_W

    print("[INIT] GridWorldModel (transformer in-context)...")
    model = GridWorldModel(d=MODEL_DIM, heads=MODEL_HEADS, layers=MODEL_LAYERS,
                           max_hw=hw, colors=N_COLORS).to(device)
    opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print(f"       Parametros: {sum(p.numel() for p in model.parameters()):,}")

    arch_engine = ArchetypeEngine()
    cerebellum  = Cerebellum(learning_rate=0.02, state_dim=CEREB_STATE_DIM, action_dim=CEREB_ACTION_DIM)
    t_binder    = TemporalBinder(seed=42)
    hdv_mem     = MultiScaleHDV()
    nt_state    = NeuromodulatorState() if _HAS_NT else None
    print()

    # ── FASE 1: TREINO ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("FASE 1: TREINO ONLINE NAS 400 TASKS ARC REAIS")
    print("=" * 65)
    model.train()
    cereb_errors = []; arch_hits = arch_total = 0
    t0 = time.time()

    for step in range(1, train_steps + 1):
        (xs, ys), (xq, yq) = train_ds.sample_batch(BATCH_SIZE, K_SUPPORT, Q_QUERIES)

        sb = grid_to_state_vec(xq[0, 0])
        sa = grid_to_state_vec(yq[0, 0])
        av = grid_delta_action(sb, sa)
        pn = cerebellum.predict_consequence(sb, av)
        ce = float(np.mean((pn - sa) ** 2))
        cereb_errors.append(ce)

        concepts  = grid_to_concepts(xq[0, 0])
        ah, aname = archetype_probe(arch_engine, concepts)
        arch_total += 1
        if ah: arch_hits += 1

        nt = nt_from_novelty(ah, ce)
        if nt_state is not None:
            nt_state.update_from_body({"reward": nt["DA"] - 0.5, "pain": nt["CRH"]})

        logits = model(xq, (xs, ys))
        loss   = F.cross_entropy(
            logits.reshape(BATCH_SIZE * Q_QUERIES, N_COLORS, hw),
            yq.reshape(BATCH_SIZE * Q_QUERIES, hw),
            ignore_index=PAD_VAL
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        cerebellum.observe_result(sb, av, sa)

        # Consolida arquétipos a cada 10 steps (descobre padrões recorrentes)
        if step % 10 == 0:
            new_archs = arch_engine.consolidate()
            if new_archs:
                arch_hits += len(new_archs)  # conta arquétipos emergentes

        if nt["DA"] > NT_SALIENCE_DA or nt["CRH"] > NT_SALIENCE_CRH:
            dummy = np.sign(np.random.randn(10000)).astype(np.float32)
            hdv_mem.write(f"step_{step}_{aname}", dummy, nt,
                          metadata={"step": step, "arch": aname, "loss": loss.item()})

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                pred  = logits.argmax(2).reshape(BATCH_SIZE, Q_QUERIES, hw)
                valid = (yq != PAD_VAL)
                px_acc = ((pred == yq) & valid).float().sum() / valid.float().sum().clamp(min=1)
                exact  = ((pred == yq) | ~valid).all(dim=2).float().mean()
                avg_ce = float(np.mean(cereb_errors[-50:]))
                arch_r = arch_hits / max(1, arch_total)
            print(
                f"[Treino] {step:4d}/{train_steps}"
                f" | Loss: {loss.item():.4f}"
                f" | Pixel: {px_acc:.1%}"
                f" | Exact: {exact:.1%}"
                f" | Cereb: {avg_ce:.4f}"
                f" | DA={nt['DA']:.2f} CRH={nt['CRH']:.2f}"
                f" | Arch: {arch_r:.1%}"
                f" | {time.time()-t0:.0f}s"
            )

    train_time = time.time() - t0
    print(f"\nTreino concluido em {train_time:.1f}s")
    print(f"Archetype Hit Rate  : {arch_hits}/{arch_total} = {arch_hits/max(1,arch_total):.1%}")
    print(f"Cerebellum Avg Err  : {float(np.mean(cereb_errors)):.5f}")
    print(f"Memorias L2 salvas  : {len(hdv_mem.L2)}")

    # Salva modelo treinado para nao perder em caso de falha posterior
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arc_model_trained.pt')
    torch.save(model.state_dict(), model_path)
    print(f'Modelo salvo: {model_path}')

    # -- FASE 2: AVALIACAO OOD -----------------------------------------------------
    print()
    print("=" * 65)
    print("FASE 2: AVALIACAO OOD NAS TASKS DE AVALIACAO (nunca vistas)")
    print("=" * 65)
    model.eval()

    all_px = []; all_ex = []; all_cp = []
    all_ce = []; all_da = []; all_crh = []
    ev_hits = ev_total = ev_done = 0
    t0e = time.time()

    with torch.no_grad():
        while ev_done < eval_limit:
            bs = min(32, eval_limit - ev_done)
            (xs, ys), (xq, yq) = eval_ds.sample_batch(bs, K_SUPPORT, Q_QUERIES)

            logits = model(xq, (xs, ys))
            pred   = logits.argmax(2).reshape(bs, Q_QUERIES, hw)
            valid  = (yq != PAD_VAL)

            px = ((pred == yq) & valid).float().sum() / valid.float().sum().clamp(min=1)
            ex = ((pred == yq) | ~valid).all(dim=2).float().mean()
            cp = ((xq == yq) & valid).float().sum() / valid.float().sum().clamp(min=1)
            all_px.append(px.item()); all_ex.append(ex.item()); all_cp.append(cp.item())

            sb = grid_to_state_vec(xq[0, 0]); sa = grid_to_state_vec(yq[0, 0])
            av = grid_delta_action(sb, sa); pn = cerebellum.predict_consequence(sb, av)
            ce = float(np.mean((pn - sa) ** 2)); all_ce.append(ce)

            ah, _ = archetype_probe(arch_engine, grid_to_concepts(xq[0, 0]))
            ev_total += 1
            if ah: ev_hits += 1

            nt = nt_from_novelty(ah, ce)
            all_da.append(nt["DA"]); all_crh.append(nt["CRH"])
            ev_done += bs

    eval_time = time.time() - t0e

    mpx  = float(np.mean(all_px));  mex  = float(np.mean(all_ex))
    mcp  = float(np.mean(all_cp));  mce  = float(np.mean(all_ce))
    mda  = float(np.mean(all_da));  mcrh = float(np.mean(all_crh))
    ehr  = ev_hits / max(1, ev_total)
    chance = 1.0 / (N_COLORS - 1)

    print()
    print("=" * 65)
    print("  RESULTADOS")
    print("=" * 65)
    print(f"  Tasks avaliadas       : {ev_done}")
    print(f"  Tempo de avaliacao    : {eval_time:.1f}s")
    print()
    print("  -- Pixel Accuracy ------------------------------------------")
    print(f"  Acaso (chance)        : {chance:.1%}")
    print(f"  Copiar entrada        : {mcp:.1%}")
    print(f"  NovaMind (integrado)  : {mpx:.1%}")
    print()
    print("  -- Exact Grid Match ----------------------------------------")
    print(f"  NovaMind Exact Match  : {mex:.1%}")
    print()
    print("  -- Modulos Cognitivos --------------------------------------")
    print(f"  Archetype Hit Rate    : {ehr:.1%}  ({ev_hits}/{ev_total})")
    print(f"  Cerebellum Avg Err    : {mce:.5f}")
    print(f"  NT medio DA           : {mda:.3f}  (reconhecimento/recompensa)")
    print(f"  NT medio CRH          : {mcrh:.3f}  (incerteza/ameaca)")
    print()

    if mex > 0.05:
        verdict = "BREAKTHROUGH — Resolveu >5% das tasks corretamente OOD!"
    elif mpx > mcp * 1.15:
        verdict = "SINAL FORTE — Superou copiar-entrada em >15%"
    elif mpx > chance * 2:
        verdict = "APRENDIZADO DETECTADO — Acima do acaso, abaixo do copiar"
    else:
        verdict = "ABAIXO DO LIMIAR — Precisa mais treino ou arquitetura maior"

    print(f"  VEREDITO: {verdict}")
    print("=" * 65)

    results = {
        "mode": label, "device": str(device),
        "train_steps": train_steps, "eval_tasks": ev_done,
        "train_time_s": round(train_time, 1), "eval_time_s": round(eval_time, 1),
        "pixel_accuracy": round(mpx, 4), "exact_match": round(mex, 4),
        "copy_baseline": round(mcp, 4), "chance_baseline": round(chance, 4),
        "archetype_hit_rate": round(ehr, 4), "cerebellum_avg_error": round(mce, 6),
        "nt_avg_da": round(mda, 4), "nt_avg_crh": round(mcrh, 4),
        "verdict": verdict,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_integrated_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados salvos em: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",     action="store_true", help="Smoke test rapido (50 tasks)")
    parser.add_argument("--eval-only", action="store_true", help="Carrega modelo salvo e so avalia (sem re-treinar)")
    args = parser.parse_args()

    if args.eval_only:
        # Carrega modelo e roda apenas avaliacao
        import torch, numpy as np
        from core.grid_cognition import GridWorldModel
        from core.arc_dataset import ARCDataset
        from core.archetypes import ArchetypeEngine
        from core.cerebellum import Cerebellum

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hw = MAX_H * MAX_W
        eval_ds = ARCDataset(data_dir=ARC_EVAL_DIR, max_h=MAX_H, max_w=MAX_W, pad_val=PAD_VAL, dev=device)
        model = GridWorldModel(d=MODEL_DIM, heads=MODEL_HEADS, layers=MODEL_LAYERS, max_hw=hw, colors=N_COLORS).to(device)

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arc_model_trained.pt')
        if not os.path.exists(model_path):
            print(f"[ERRO] Modelo nao encontrado: {model_path}")
            print("       Execute sem --eval-only primeiro para treinar.")
            sys.exit(1)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[OK] Modelo carregado de {model_path}")
        print(f"Device: {device} | Tasks de avaliacao: {len(eval_ds.tasks)}\n")

        arch_engine = ArchetypeEngine()
        cerebellum  = Cerebellum(learning_rate=0.02, state_dim=CEREB_STATE_DIM, action_dim=CEREB_ACTION_DIM)

        all_px=[]; all_ex=[]; all_cp=[]; all_ce=[]; all_da=[]; all_crh=[]
        ev_hits=ev_total=ev_done=0
        t0e=time.time()

        with torch.no_grad():
            while ev_done < FULL_EVAL_TASKS:
                bs = min(32, FULL_EVAL_TASKS - ev_done)
                (xs,ys),(xq,yq) = eval_ds.sample_batch(bs, K_SUPPORT, Q_QUERIES)
                logits = model(xq,(xs,ys))
                pred   = logits.argmax(2).reshape(bs, Q_QUERIES, hw)
                valid  = (yq != PAD_VAL)
                px = ((pred==yq)&valid).float().sum()/valid.float().sum().clamp(min=1)
                ex = ((pred==yq)|~valid).all(dim=2).float().mean()
                cp = ((xq==yq)&valid).float().sum()/valid.float().sum().clamp(min=1)
                all_px.append(px.item()); all_ex.append(ex.item()); all_cp.append(cp.item())
                sb=grid_to_state_vec(xq[0,0]); sa=grid_to_state_vec(yq[0,0])
                av=grid_delta_action(sb,sa); pn=cerebellum.predict_consequence(sb,av)
                ce=float(np.mean((pn-sa)**2)); all_ce.append(ce)
                ah,_=archetype_probe(arch_engine, grid_to_concepts(xq[0,0]))
                ev_total+=1
                if ah: ev_hits+=1
                nt=nt_from_novelty(ah,ce)
                all_da.append(nt["DA"]); all_crh.append(nt["CRH"])
                ev_done+=bs

        eval_time=time.time()-t0e
        mpx=float(np.mean(all_px)); mex=float(np.mean(all_ex)); mcp=float(np.mean(all_cp))
        mce=float(np.mean(all_ce)); mda=float(np.mean(all_da)); mcrh=float(np.mean(all_crh))
        ehr=ev_hits/max(1,ev_total); chance=1.0/(N_COLORS-1)

        print("=" * 65)
        print("  RESULTADOS OOD (modelo treinado carregado do disco)")
        print("=" * 65)
        print(f"  Tasks avaliadas       : {ev_done}")
        print(f"  Tempo de avaliacao    : {eval_time:.1f}s")
        print(f"  Acaso (chance)        : {chance:.1%}")
        print(f"  Copiar entrada        : {mcp:.1%}")
        print(f"  NovaMind Pixel Acc    : {mpx:.1%}")
        print(f"  NovaMind Exact Match  : {mex:.1%}")
        print(f"  Archetype Hit Rate    : {ehr:.1%}  ({ev_hits}/{ev_total})")
        print(f"  Cerebellum Avg Err    : {mce:.5f}")
        print(f"  NT avg DA             : {mda:.3f}")
        print(f"  NT avg CRH            : {mcrh:.3f}")

        if mex > 0.05:    verdict = "BREAKTHROUGH -- Resolveu >5% OOD!"
        elif mpx > mcp*1.15: verdict = "SINAL FORTE -- Superou copy-input em >15%"
        elif mpx > chance*2: verdict = "APRENDIZADO DETECTADO -- Acima do acaso"
        else:             verdict = "ABAIXO DO LIMIAR"
        print(f"  VEREDITO: {verdict}")
        print("=" * 65)

        results = {
            "mode": "EVAL-ONLY (loaded trained model)", "device": str(device),
            "eval_tasks": ev_done, "eval_time_s": round(eval_time,1),
            "pixel_accuracy": round(mpx,4), "exact_match": round(mex,4),
            "copy_baseline": round(mcp,4), "chance_baseline": round(chance,4),
            "archetype_hit_rate": round(ehr,4), "cerebellum_avg_error": round(mce,6),
            "nt_avg_da": round(mda,4), "nt_avg_crh": round(mcrh,4), "verdict": verdict,
        }
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_integrated_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Resultados salvos em: {out}")
    else:
        run(smoke=args.smoke)
