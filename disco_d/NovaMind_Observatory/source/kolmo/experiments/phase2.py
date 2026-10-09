"""Phase 2, step 1: differentiation. Does splitting the prior into
specialized prototypes pay for its own selection cost?

Where this comes from: Phase 1.5 left a single global prior winning
hugely on 6 of 15 held-out tasks (-389, -133, -124 bits) and losing
slightly on 9 (+33..+95). That is the signature of ONE average prototype
that fits a subpopulation and drags on the rest -- one cell type where
the data asks for several. This run splits it (k-means over the aligned
construction fits, one mu and local basis per cluster) and makes every
task pay log2(M) bits to name the prototype it selected. If the split is
not real, that price alone sinks it.

Conditions, all on the same held-out tasks, same commit, same script:
  empty     -- mu=0, B=0: everything through the residual (capacity floor)
  shuffled  -- mixture with mu/B entries permuted (structure destroyed)
  global    -- the Phase 1.5 single library (the incumbent to beat)
  mixture   -- M prototypes + log2(M) index (the claim)

Death criteria:
  1. mixture < global  (mean AND median) -- differentiation pays for itself
  2. mixture < shuffled (mean AND median) -- it is structure, not capacity
  3. mixture < empty                      -- the parametrization isn't overhead
Failing (1) means the population of tasks does not want to be split at
this M, and Phase 2 should move to promotion/pruning of atoms rather than
of prototypes. Failing (2) kills the mixture outright.

Usage:
    python -m kolmo.experiments.phase2 --construction 40 --holdout 30 --steps 2000
"""
from __future__ import annotations

import argparse
import statistics as st
import time
from collections import Counter

import torch
from scipy.stats import wilcoxon

from ..align import align_thetas
from ..fit import GAUSSIAN, VARIATIONAL, fit_task, fit_task_mixture
from ..grids import load_tasks
from ..ledger import append
from ..library import Library, Mixture

ARC_ROOT = "legacy/novamind/data/ARC"
LEDGER_PATH = "kolmo/ledger.csv"


ENV = {"experiment": "phase2", "ledger": LEDGER_PATH}


def _log(tid, cond, seed, bits, res, extra=None):
    row = {
        "experiment": ENV["experiment"], "task_id": tid, "condition": cond, "seed": seed,
        "bits_total": bits, "bits_recon": res.bits_recon, "bits_weights": res.bits_weights,
        "steps": res.steps, "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
    }
    if extra:
        row.update(extra)
    append(ENV["ledger"], row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=30)
    # M=1 is in the sweep on purpose: the sweep must be able to conclude
    # "this population does not want to be split" on its own evidence.
    ap.add_argument("--m-sweep", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--k", type=int, default=2, help="local basis width (k*=2 from phase1_5)")
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    # smoke runs MUST point this elsewhere: a short run writes rows under the
    # same experiment name, and they silently contaminate any later average
    # that forgets to filter by steps (this happened -- see the phase2
    # variational M sweep, where two 100-step rows moved M=1/M=2 from ~245
    # to ~14862 bits and would have flipped the chosen M if read naively)
    ap.add_argument("--ledger", type=str, default=LEDGER_PATH)
    ap.add_argument("--codec", type=str, default=GAUSSIAN, choices=[GAUSSIAN, VARIATIONAL],
                    help="weight code: gaussian prices norm, variational prices precision")
    args = ap.parse_args()

    fit_kw = dict(steps=args.steps, device=args.device, codec=args.codec)
    ENV["experiment"] = f"phase2_{args.codec}" if args.codec != GAUSSIAN else "phase2"
    ENV["ledger"] = args.ledger
    total_needed = args.construction + args.holdout
    all_tasks = load_tasks("training", ARC_ROOT, device=args.device, max_size=15, limit=total_needed)
    ids = list(all_tasks.keys())
    construction_ids = ids[:args.construction]
    holdout_ids = ids[args.construction:args.construction + args.holdout]
    print(f"[phase2] {len(construction_ids)} construction / {len(holdout_ids)} held-out tasks, "
          f"M sweep {args.m_sweep}, k={args.k}, steps={args.steps}, codec={args.codec}")

    # --- A: independent construction fits (Phase 0 style)
    t0 = time.time()
    thetas, names, shapes = [], None, None
    for i, tid in enumerate(construction_ids):
        res = fit_task(all_tasks[tid].train, seed=0, **fit_kw)
        thetas.append(res.theta)
        names, shapes = res.param_names, res.param_shapes
        _log(tid, "construction", 0, res.bits_total, res)
        print(f"  [construct {i+1}/{len(construction_ids)}] {tid}  bits={res.bits_total:8.2f}  ({res.seconds:.1f}s)")

    # --- B: permutation-align, so clustering compares functions not labelings
    aligned = align_thetas(torch.stack(thetas), names, shapes).to(args.device)
    print(f"[phase2] aligned {len(thetas)} construction fits")

    # --- C: choose M on a split of the construction set (holdout untouched)
    n_val = max(2, len(construction_ids) // 5)
    build, val_ids = aligned[:-n_val], construction_ids[-n_val:]
    print(f"[phase2] M sweep on {len(build)} build / {n_val} val construction tasks")
    sweep = {}
    for m in args.m_sweep:
        mix_m = Mixture.from_weight_vectors(build, m=m, k=args.k, seed=0)
        bits, chosen = [], []
        for tid in val_ids:
            got = fit_task_mixture(all_tasks[tid].train, mix_m, probe_steps=args.probe_steps,
                                   seed=0, **fit_kw)
            bits.append(got.bits_total)
            chosen.append(got.selected)
            _log(tid, f"sweep_M={m}", 0, got.bits_total, got.fit, {"solved": f"sel={got.selected}"})
        sweep[m] = (st.mean(bits), sorted(set(chosen)), len(mix_m.components))
        print(f"  [sweep] M={m} ({sweep[m][2]} live)  mean bits {sweep[m][0]:8.2f}  "
              f"prototypes used: {sweep[m][1]}")
    m_star = min(sweep, key=lambda m: sweep[m][0])
    print(f"[phase2] chosen M*={m_star}")

    # --- D: rebuild on ALL construction fits, then prune by use
    mixture = Mixture.from_weight_vectors(aligned, m=m_star, k=args.k, seed=0)
    n_before = len(mixture.components)
    # "Use it or lose it", measured over the whole construction corpus with
    # the cheap probe -- never on the holdout. Sweep-time selections cannot
    # be reused here: that mixture was clustered on a subset, so its
    # component indices name different prototypes than this one's.
    use = Counter()
    for tid in construction_ids:
        got = fit_task_mixture(all_tasks[tid].train, mixture, probe_steps=args.probe_steps,
                               seed=0, steps=args.probe_steps, device=args.device,
                               codec=args.codec)
        use[got.selected] += 1
    survivors = [i for i in range(n_before) if use[i] > 0]
    print(f"[phase2] prototype use over construction corpus: "
          f"{ {i: use[i] for i in range(n_before)} }")
    if len(survivors) < n_before:
        mixture = mixture.keep(survivors)
        print(f"[phase2] pruned {n_before} -> {len(mixture.components)} unused prototypes; "
              f"index cost now {mixture.bits_index:.2f} bits")
    global_lib = Library.from_weight_vectors(aligned, k=args.k)
    print(f"[phase2] mixture: {len(mixture.components)} prototypes, index {mixture.bits_index:.2f} bits; "
          f"global: B {tuple(global_lib.B.shape)}")

    # --- E: four conditions on held-out tasks
    mix_shuf, mix_empty = mixture.shuffled(seed=0), mixture.empty()
    results = {c: [] for c in ("mixture", "shuffled", "global", "empty")}
    picks = Counter()

    for i, tid in enumerate(holdout_ids):
        per = {c: [] for c in results}
        for seed in args.seeds:
            got = fit_task_mixture(all_tasks[tid].train, mixture, probe_steps=args.probe_steps,
                                   seed=seed, **fit_kw)
            picks[got.selected] += 1
            per["mixture"].append(got.bits_total)
            _log(tid, "mixture", seed, got.bits_total, got.fit, {"solved": f"sel={got.selected}"})

            gs = fit_task_mixture(all_tasks[tid].train, mix_shuf, probe_steps=args.probe_steps,
                                  seed=seed, **fit_kw)
            per["shuffled"].append(gs.bits_total)
            _log(tid, "shuffled", seed, gs.bits_total, gs.fit)

            ge = fit_task_mixture(all_tasks[tid].train, mix_empty, probe_steps=args.probe_steps,
                                  seed=seed, **fit_kw)
            per["empty"].append(ge.bits_total)
            _log(tid, "empty", seed, ge.bits_total, ge.fit)

            gg = fit_task(all_tasks[tid].train, seed=seed, library=global_lib, **fit_kw)
            per["global"].append(gg.bits_total)
            _log(tid, "global", seed, gg.bits_total, gg)

        means = {c: st.mean(v) for c, v in per.items()}
        for c, v in means.items():
            results[c].append(v)
        print(f"  [holdout {i+1}/{len(holdout_ids)}] {tid}  "
              f"mix={means['mixture']:8.2f}  glob={means['global']:8.2f}  "
              f"shuf={means['shuffled']:8.2f}  empty={means['empty']:8.2f}")

    # --- report
    print(f"\n[phase2] done in {(time.time()-t0)/60:.1f} min.")
    print(f"  bits_total across {len(holdout_ids)} held-out tasks, {len(args.seeds)} seeds each:")
    for c in ("mixture", "global", "shuffled", "empty"):
        print(f"    {c:10s}: mean {st.mean(results[c]):8.2f}   median {st.median(results[c]):8.2f}")
    print(f"  prototype selections on holdout: {dict(sorted(picks.items()))}")

    def verdict(a: str, b: str) -> str:
        mean_ok = st.mean(results[a]) < st.mean(results[b])
        med_ok = st.median(results[a]) < st.median(results[b])
        n_win = sum(1 for x, y in zip(results[a], results[b]) if x < y)
        try:
            p = wilcoxon(results[a], results[b]).pvalue
            ptxt = f"p={p:.4f}"
        except ValueError as e:
            ptxt = f"p n/a ({e})"
        tag = "PASS" if (mean_ok and med_ok) else ("MIXED" if (mean_ok or med_ok) else "FAIL")
        return (f"  {a} vs {b}: mean {'<' if mean_ok else '>='} , median {'<' if med_ok else '>='} , "
                f"wins {n_win}/{len(results[a])}, {ptxt}  -> {tag}")

    print("\n  death criteria:")
    print(verdict("mixture", "global"))
    print(verdict("mixture", "shuffled"))
    print(verdict("mixture", "empty"))


if __name__ == "__main__":
    main()
