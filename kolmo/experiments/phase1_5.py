"""Phase 1.5: re-run of the Phase 1 death criterion with a fixed library.

Phase 1 FAILED real-vs-shuffled (6/12 tasks, mean 217.23 vs 192.63). The
diagnosis (see the Phase 1.5 plan) found three construction defects that
could produce that tie even if real abstraction exists:
  1. the mean of the fitted weights was discarded (now kept as mu);
  2. library fits started at theta=0, a dead net (now start at theta=mu);
  3. PCA ran over unaligned weight vectors, blind to hidden-channel
     permutation symmetry (now aligned first, kolmo.align).
Plus: k was fixed at 8 with no justification (now swept on a split of the
construction set, never touching the holdout).

The death criterion itself is UNCHANGED (mean real < mean shuffled on
held-out tasks), and the pre-registered decision rule is: if real still
does not beat shuffled on mean AND median after these fixes, the Phase 1
verdict stands -- a linear weight-space library does not capture
abstraction -- and the next step is a function-space library, not more
tuning of this parametrization.

Usage:
    python -m kolmo.experiments.phase1_5 --construction 40 --holdout 15 --steps 2000
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import time

import torch
from scipy.stats import wilcoxon

from ..align import align_thetas
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
    ap.add_argument("--k-sweep", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    # smoke runs MUST point this elsewhere: a short run writes rows under the
    # same experiment name, and they silently contaminate any later average
    # that forgets to filter by steps (this happened -- see the phase2
    # variational M sweep, where two 100-step rows moved M=1/M=2 from ~245
    # to ~14862 bits and would have flipped the chosen M if read naively)
    ap.add_argument("--ledger", type=str, default=LEDGER_PATH)
    args = ap.parse_args()

    total_needed = args.construction + args.holdout
    all_tasks = load_tasks("training", ARC_ROOT, device=args.device, max_size=15, limit=total_needed)
    ids = list(all_tasks.keys())
    construction_ids = ids[:args.construction]
    holdout_ids = ids[args.construction:args.construction + args.holdout]
    print(f"[phase1_5] {len(construction_ids)} construction tasks, {len(holdout_ids)} held-out tasks "
          f"(disjoint by construction), k sweep {args.k_sweep}, steps={args.steps}")

    # --- Step A: fit construction tasks independently (Phase 0 style)
    t0 = time.time()
    thetas, names, shapes = [], None, None
    for i, tid in enumerate(construction_ids):
        res = fit_task(all_tasks[tid].train, steps=args.steps, device=args.device, seed=0)
        thetas.append(res.theta)
        names, shapes = res.param_names, res.param_shapes
        append(args.ledger, {
            "experiment": "phase1_5_construction", "task_id": tid, "condition": "none", "seed": 0,
            "bits_total": res.bits_total, "bits_recon": res.bits_recon, "bits_weights": res.bits_weights,
            "steps": res.steps, "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
        })
        print(f"  [construct {i+1}/{len(construction_ids)}] {tid}  bits={res.bits_total:8.2f}  ({res.seconds:.1f}s)")

    # --- Step B: permutation-align all fits to a common reference (fix 3)
    t_align = time.time()
    theta_stack = torch.stack(thetas)  # cpu, (N, P)
    aligned = align_thetas(theta_stack, names, shapes)
    delta = (aligned - theta_stack).abs().sum(dim=1)
    n_moved = int((delta > 1e-9).sum())
    print(f"[phase1_5] alignment done in {time.time()-t_align:.1f}s: "
          f"{n_moved}/{len(thetas)} fits changed by alignment")
    aligned = aligned.to(args.device)

    # --- Step C: choose k on a split of the construction set (holdout untouched)
    n_val = max(2, len(construction_ids) // 5)
    build, val = aligned[:-n_val], aligned[-n_val:]
    val_ids = construction_ids[-n_val:]
    print(f"[phase1_5] k sweep on {len(build)} build / {n_val} val construction tasks")
    k_scores = {}
    for k in args.k_sweep:
        lib_k = Library.from_weight_vectors(build, k=k)
        bits = []
        for tid in val_ids:
            res = fit_task(all_tasks[tid].train, steps=args.steps, device=args.device,
                           seed=0, library=lib_k)
            bits.append(res.bits_total)
            append(args.ledger, {
                "experiment": "phase1_5_sweep", "task_id": tid, "condition": f"k={k}", "seed": 0,
                "bits_total": res.bits_total, "bits_recon": res.bits_recon,
                "bits_weights": res.bits_weights, "steps": res.steps,
                "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
            })
        k_scores[k] = st.mean(bits)
        print(f"  [sweep] k={k:3d}  mean bits on val: {k_scores[k]:8.2f}")
    k_star = min(k_scores, key=k_scores.get)
    print(f"[phase1_5] chosen k*={k_star}")

    # --- Step D: rebuild library from ALL aligned construction fits at k*
    library = Library.from_weight_vectors(aligned, k=k_star)
    library_shuf = library.shuffled(seed=0)
    library_empty = library.empty()
    print(f"[phase1_5] library built: mu norm {library.mu.norm().item():.2f}, "
          f"B shape {tuple(library.B.shape)}, sigma_a={library.sigma_a}, sigma_r={library.sigma_r}")

    # --- Step E: three conditions on held-out tasks
    conditions = {"real": library, "shuffled": library_shuf, "empty": library_empty}
    results = {c: [] for c in conditions}

    for i, tid in enumerate(holdout_ids):
        per_task = {}
        for cond_name, lib in conditions.items():
            for seed in args.seeds:
                res = fit_task(all_tasks[tid].train, steps=args.steps, device=args.device,
                                seed=seed, library=lib)
                append(args.ledger, {
                    "experiment": "phase1_5", "task_id": tid, "condition": cond_name, "seed": seed,
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
    print(f"\n[phase1_5] done in {total_dt/60:.1f} min.")
    print(f"  per-condition bits_total across {len(holdout_ids)} held-out tasks, "
          f"{len(args.seeds)} seeds each:")
    for c in ("real", "shuffled", "empty"):
        print(f"    {c:10s}: mean {st.mean(results[c]):8.2f}   median {st.median(results[c]):8.2f}")

    n_beats_shuf = sum(1 for r, s in zip(results["real"], results["shuffled"]) if r < s)
    n_beats_empty = sum(1 for r, e in zip(results["real"], results["empty"]) if r < e)
    print(f"\n  real < shuffled on {n_beats_shuf}/{len(holdout_ids)} held-out tasks (paired)")
    print(f"  real < empty    on {n_beats_empty}/{len(holdout_ids)} held-out tasks (paired)")

    try:
        stat, p = wilcoxon(results["real"], results["shuffled"])
        print(f"  Wilcoxon (real vs shuffled, paired): statistic={stat:.1f}, p={p:.4f}")
    except ValueError as e:  # e.g. all differences zero
        print(f"  Wilcoxon (real vs shuffled): not computable ({e})")

    mean_verdict = "PASS" if st.mean(results["real"]) < st.mean(results["shuffled"]) else "FAIL"
    median_verdict = "PASS" if st.median(results["real"]) < st.median(results["shuffled"]) else "FAIL"
    empty_verdict = "PASS" if st.mean(results["real"]) < st.mean(results["empty"]) else "FAIL"
    print(f"  death criterion (real beats shuffled, mean -- official): {mean_verdict}")
    print(f"  death criterion (real beats shuffled, median):           {median_verdict}")
    print(f"  death criterion (real beats empty):                      {empty_verdict}")
    if mean_verdict == "FAIL" and median_verdict == "FAIL":
        print("\n  Pre-registered decision rule: verdict FINAL -- a linear weight-space "
              "library does not capture abstraction here. Next step is a function-space "
              "library, not more tuning of this parametrization.")


if __name__ == "__main__":
    main()
