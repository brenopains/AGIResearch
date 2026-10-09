"""The control that decides whether the cross-domain effect is interesting.

phase3_probe found: a library built only from ARC lowers the bits of
cellular-automata tasks by 14/14 over its shuffled control (p=0.0001) --
and is statistically indistinguishable from a library built from CA
itself (8/14, p=0.068), capturing 108% of the same-domain benefit.

That last number is the problem. If ARC-specific abstraction were what
transfers, a home-grown CA library should win on CA. It does not. The
simple explanation is that NEITHER library carries anything specific to
its domain, and what transfers is a generic "converged grid-net weights"
prior -- which is pretraining-as-initialization, already known, not a
discovery. The shuffled control rules out "any vector of the right scale
works"; it does not rule out "any converged weights work".

This script separates the two by building libraries from ARC tasks whose
input->output relation has been destroyed, while everything else (the
architecture, the optimizer, the number of steps, the fact of reaching
convergence) is held identical:

  scram_random   -- outputs replaced by uniform random colors. No function
                    exists; the fit can only memorize.
  scram_recolor  -- each output recolored by a different random color
                    permutation. Output statistics and layout preserved,
                    but no consistent rule remains.

Read against the source_lib / source_shuf / empty / target_lib numbers
already in the ledger for the SAME held-out CA tasks, same seeds, same
codec (that is why this script does not recompute them).

  If scrambled libraries transfer as well as the real one
      -> the effect is an optimization prior. Real, but known. The
         project's claim has to be restated, and running five more
         domains would have produced the same artifact five more times.
  If they do not
      -> what transfers is structure learned from the source tasks, and
         breadth across domains becomes worth the GPU hours.

Usage:
    python -u -m kolmo.experiments.phase3_control
"""
from __future__ import annotations

import argparse
import hashlib
import statistics as st
import time

import torch
from scipy.stats import wilcoxon

from ..align import align_thetas
from ..domains import ca as ca_domain
from ..fit import GAUSSIAN, VARIATIONAL, fit_task
from ..grids import load_tasks as load_arc
from ..grids import scramble_random_targets, scramble_recolor_targets
from ..ledger import append
from ..library import Library

ARC_ROOT = "legacy/novamind/data/ARC"
LEDGER_PATH = "kolmo/ledger.csv"
PROBE_TAG = "phase3probe_arc2ca_variational"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=15)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ledger", type=str, default=LEDGER_PATH)
    ap.add_argument("--codec", type=str, default=VARIATIONAL, choices=[GAUSSIAN, VARIATIONAL])
    args = ap.parse_args()

    fit_kw = dict(steps=args.steps, device=args.device, codec=args.codec)
    tag = f"phase3control_arc2ca_{args.codec}"

    # identical calls to phase3_probe's, so the held-out CA tasks are the same
    arc = load_arc("training", ARC_ROOT, device=args.device, max_size=15, limit=args.construction)
    ca = ca_domain.load_tasks(args.construction + args.holdout, device=args.device)
    ca_holdout = list(ca)[args.construction:args.construction + args.holdout]

    scramblers = {
        "scram_random": scramble_random_targets,
        "scram_recolor": scramble_recolor_targets,
    }
    print(f"[phase3control] {len(arc)} ARC construction tasks -> {len(ca_holdout)} held-out CA "
          f"tasks, controls: {list(scramblers)}, steps={args.steps}, codec={args.codec}")

    t0 = time.time()
    libs = {}
    for name, scramble in scramblers.items():
        thetas, names, shapes = [], None, None
        for i, tid in enumerate(arc):
            # NOT hash(tid): Python randomizes str hashing per process, so the
            # scramble would differ between runs and nothing would reproduce.
            # This is the exact defect the audit found in NovaMind's
            # typed_graph.py ("hyperbolic" positions from abs(hash(name))).
            pairs = scramble(arc[tid].train, seed=int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16))
            res = fit_task(pairs, seed=0, **fit_kw)
            thetas.append(res.theta)
            names, shapes = res.param_names, res.param_shapes
            append(args.ledger, {
                "experiment": tag, "task_id": tid, "condition": f"construction_{name}", "seed": 0,
                "bits_total": res.bits_total, "bits_recon": res.bits_recon,
                "bits_weights": res.bits_weights, "steps": res.steps,
                "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
            })
            print(f"  [{name} construct {i+1}/{len(arc)}] {tid}  bits={res.bits_total:8.2f}  "
                  f"({res.seconds:.1f}s)")
        aligned = align_thetas(torch.stack(thetas), names, shapes).to(args.device)
        libs[name] = Library.from_weight_vectors(aligned, k=args.k)
    print(f"[phase3control] control libraries built ({(time.time()-t0)/60:.1f} min)")

    results = {c: [] for c in libs}
    for i, tid in enumerate(ca_holdout):
        per = {}
        for name, lib in libs.items():
            for seed in args.seeds:
                res = fit_task(ca[tid].train, seed=seed, library=lib, **fit_kw)
                append(args.ledger, {
                    "experiment": tag, "task_id": tid, "condition": name, "seed": seed,
                    "bits_total": res.bits_total, "bits_recon": res.bits_recon,
                    "bits_weights": res.bits_weights, "steps": res.steps,
                    "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
                })
                per.setdefault(name, []).append(res.bits_total)
        means = {c: st.mean(v) for c, v in per.items()}
        for c, v in means.items():
            results[c].append(v)
        print(f"  [holdout {i+1}/{len(ca_holdout)}] {tid}  "
              + "  ".join(f"{c}={means[c]:8.2f}" for c in libs))

    # join against the real-library numbers already measured on these tasks
    import collections
    import csv
    rows = [r for r in csv.DictReader(open(args.ledger, encoding="utf-8"))
            if r["experiment"] == PROBE_TAG and r["steps"] == str(args.steps)]
    probe = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        probe[r["task_id"]][r["condition"]].append(float(r["bits_total"]))
    joined = [t for t in ca_holdout
              if all(len(probe[t].get(c, [])) == len(args.seeds)
                     for c in ("source_lib", "source_shuf", "empty", "target_lib"))]
    idx = {t: i for i, t in enumerate(ca_holdout)}

    print(f"\n[phase3control] done in {(time.time()-t0)/60:.1f} min. "
          f"{len(joined)}/{len(ca_holdout)} tasks join with the probe run.")
    if not joined:
        print("  no overlap with the probe run -- cannot compare; rerun phase3_probe first")
        return

    series = {c: [st.mean(probe[t][c]) for t in joined]
              for c in ("source_lib", "source_shuf", "empty", "target_lib")}
    for c in results:
        series[c] = [results[c][idx[t]] for t in joined]
    for c in ("source_lib", "scram_recolor", "scram_random", "target_lib", "source_shuf", "empty"):
        print(f"    {c:14s}: mean {st.mean(series[c]):8.2f}   median {st.median(series[c]):8.2f}")

    print("\n  the question: does the REAL library beat the structure-free ones?")
    for c in ("scram_recolor", "scram_random"):
        a, y = series["source_lib"], series[c]
        mo, me = st.mean(a) < st.mean(y), st.median(a) < st.median(y)
        w = sum(1 for x, z in zip(a, y) if x < z)
        p = wilcoxon(a, y).pvalue
        tag_ = "PASS" if (mo and me) else ("MIXED" if (mo or me) else "FAIL")
        print(f"  source_lib vs {c:14s}: mean {'<' if mo else '>='}, median {'<' if me else '>='}, "
              f"wins {w}/{len(a)}, p={p:.4f}  -> {tag_}")
    print("\n  FAIL on both = the transfer is an optimization prior, not learned abstraction.")


if __name__ == "__main__":
    main()
