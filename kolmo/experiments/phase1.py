"""Phase 1: real library vs. shuffled vs. empty, on held-out tasks.

Death criterion (from the plan): if the real library doesn't beat the
shuffled one, there's no abstraction being captured -- stop. If it doesn't
beat empty, the parametrization is pure overhead. Both controls run before
any claim is made, in the same script, same commit -- that is the specific
discipline the NovaMind audit found missing everywhere.

Two disjoint task sets, split by (deterministic, sorted) load order:
  - construction tasks: fit independently (Phase 0 style), their converged
    weight vectors are PCA'd into the library basis B.
  - held-out tasks: never seen during construction. Fit three ways (real /
    shuffled / empty library) and compared.

Usage:
    python -m kolmo.experiments.phase1 --construction 40 --holdout 15 --k 8 --steps 2000
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch

from ..fit import fit_task
from ..grids import load_tasks
from ..ledger import append
from ..library import Library

ARC_ROOT = "legacy/novamind/data/ARC"
LEDGER_PATH = "kolmo/ledger.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=15)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    total_needed = args.construction + args.holdout
    all_tasks = load_tasks("training", ARC_ROOT, device=args.device, max_size=15, limit=total_needed)
    ids = list(all_tasks.keys())
    construction_ids = ids[:args.construction]
    holdout_ids = ids[args.construction:args.construction + args.holdout]
    print(f"[phase1] {len(construction_ids)} construction tasks, {len(holdout_ids)} held-out tasks "
          f"(disjoint by construction), k={args.k}, steps={args.steps}")

    # --- Step A/B: fit construction tasks independently, PCA their weights into B
    t0 = time.time()
    thetas = []
    for i, tid in enumerate(construction_ids):
        res = fit_task(all_tasks[tid].train, steps=args.steps, device=args.device, seed=0)
        thetas.append(res.theta)
        append(LEDGER_PATH, {
            "experiment": "phase1_construction", "task_id": tid, "condition": "none", "seed": 0,
            "bits_total": res.bits_total, "bits_recon": res.bits_recon, "bits_weights": res.bits_weights,
            "steps": res.steps, "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
        })
        print(f"  [construct {i+1}/{len(construction_ids)}] {tid}  bits={res.bits_total:8.2f}  ({res.seconds:.1f}s)")

    theta_stack = torch.stack(thetas).to(args.device)  # (N, P)
    library = Library.from_weight_vectors(theta_stack, k=args.k)
    library_shuf = library.shuffled(seed=0)
    library_empty = library.empty()
    print(f"[phase1] library built: B shape {tuple(library.B.shape)}, "
          f"sigma_a={library.sigma_a}, sigma_r={library.sigma_r}")

    # --- Step C/D/E: three conditions on held-out tasks
    conditions = {"real": library, "shuffled": library_shuf, "empty": library_empty}
    results = {c: [] for c in conditions}

    for i, tid in enumerate(holdout_ids):
        per_task = {}
        for cond_name, lib in conditions.items():
            for seed in args.seeds:
                res = fit_task(all_tasks[tid].train, steps=args.steps, device=args.device,
                                seed=seed, library=lib)
                append(LEDGER_PATH, {
                    "experiment": "phase1", "task_id": tid, "condition": cond_name, "seed": seed,
                    "bits_total": res.bits_total, "bits_recon": res.bits_recon,
                    "bits_weights": res.bits_weights, "steps": res.steps,
                    "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
                })
                per_task.setdefault(cond_name, []).append(res.bits_total)
        means = {c: st.mean(v) for c, v in per_task.items()}
        for c, v in means.items():
            results[c].append(v)
        print(f"  [holdout {i+1}/{len(holdout_ids)}] {tid}  "
              f"real={means['real']:8.2f}  shuffled={means['shuffled']:8.2f}  empty={means['empty']:8.2f}")

    total_dt = time.time() - t0
    print(f"\n[phase1] done in {total_dt/60:.1f} min.")
    means_overall = {c: st.mean(v) for c, v in results.items()}
    print(f"  mean bits_total across {len(holdout_ids)} held-out tasks, "
          f"{len(args.seeds)} seeds each:")
    for c in ("real", "shuffled", "empty"):
        print(f"    {c:10s}: {means_overall[c]:8.2f}")

    n_real_beats_shuffled = sum(1 for r, s in zip(results["real"], results["shuffled"]) if r < s)
    n_real_beats_empty = sum(1 for r, e in zip(results["real"], results["empty"]) if r < e)
    print(f"\n  real < shuffled on {n_real_beats_shuffled}/{len(holdout_ids)} held-out tasks (paired)")
    print(f"  real < empty    on {n_real_beats_empty}/{len(holdout_ids)} held-out tasks (paired)")

    verdict_shuf = "PASS" if means_overall["real"] < means_overall["shuffled"] else "FAIL"
    verdict_empty = "PASS" if means_overall["real"] < means_overall["empty"] else "FAIL"
    print(f"  death criterion (real beats shuffled): {verdict_shuf}")
    print(f"  death criterion (real beats empty):    {verdict_empty}")


if __name__ == "__main__":
    main()
