"""Phase 3, cheap probe: does abstraction cross a domain boundary at all?

The full Phase 3 measures a CURVE (bits on held-out tasks as a function of
how many unrelated-domain tasks have been compressed) and costs ~5 GPU
hours. This script answers the binary question the curve presupposes, for
about an hour: is the cross-domain number different from zero?

Everything measured so far lives inside ARC. A fair critic would say the
30/30 win over the shuffled control shows that ARC puzzles resemble ARC
puzzles -- and they would be right. Cellular automata share no generative
process with ARC: no author, no intent, just a deterministic local update
rule. A library built ONLY from ARC that lowers the bits of CA tasks is
transfer; one that does not is specialization.

Conditions on the same held-out target tasks, same commit, same script:
  source_lib   -- library built from source-domain construction fits only.
                  THE CLAIM. It has never seen a target-domain task.
  source_shuf  -- same library, mu/B entries permuted. If this ties, the
                  effect is capacity, not abstraction.
  empty        -- mu=0, B=0. The parametrization's overhead floor.
  target_lib   -- library built from OTHER target-domain tasks. Not a
                  control: a same-domain REFERENCE, so "transfers" can be
                  reported as a fraction of what staying home is worth.

Death criteria (each needs mean AND median):
  1. source_lib < source_shuf -- cross-domain transfer exists
  2. source_lib < empty       -- and is not just parametrization overhead
Failing 1 means abstraction here is domain-bound, and the project's claim
(§5 of the plan) has to be restated or abandoned. That is the point of
running it before building the expensive curve.

Usage:
    python -u -m kolmo.experiments.phase3_probe --source arc --target ca
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
from scipy.stats import wilcoxon

from ..align import align_thetas
from ..domains import ca as ca_domain
from ..fit import GAUSSIAN, VARIATIONAL, fit_task
from ..grids import load_tasks as load_arc
from ..ledger import append
from ..library import Library

ARC_ROOT = "legacy/novamind/data/ARC"
LEDGER_PATH = "kolmo/ledger.csv"


def load_domain(name: str, n: int, device: str) -> dict:
    if name == "arc":
        return load_arc("training", ARC_ROOT, device=device, max_size=15, limit=n)
    if name == "ca":
        return ca_domain.load_tasks(n, device=device)
    raise ValueError(f"unknown domain: {name}")


def build_library(tasks: dict, ids: list, k: int, fit_kw: dict, ledger: str,
                  tag: str, label: str) -> Library:
    thetas, names, shapes = [], None, None
    for i, tid in enumerate(ids):
        res = fit_task(tasks[tid].train, seed=0, **fit_kw)
        thetas.append(res.theta)
        names, shapes = res.param_names, res.param_shapes
        append(ledger, {
            "experiment": tag, "task_id": tid, "condition": f"construction_{label}", "seed": 0,
            "bits_total": res.bits_total, "bits_recon": res.bits_recon,
            "bits_weights": res.bits_weights, "steps": res.steps,
            "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
        })
        print(f"  [{label} construct {i+1}/{len(ids)}] {tid}  bits={res.bits_total:8.2f}  ({res.seconds:.1f}s)")
    aligned = align_thetas(torch.stack(thetas), names, shapes).to(fit_kw["device"])
    return Library.from_weight_vectors(aligned, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="arc", choices=["arc", "ca"])
    ap.add_argument("--target", type=str, default="ca", choices=["arc", "ca"])
    ap.add_argument("--construction", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=15)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ledger", type=str, default=LEDGER_PATH)
    ap.add_argument("--codec", type=str, default=VARIATIONAL, choices=[GAUSSIAN, VARIATIONAL])
    args = ap.parse_args()

    assert args.source != args.target, "a cross-domain probe needs two different domains"
    fit_kw = dict(steps=args.steps, device=args.device, codec=args.codec)
    tag = f"phase3probe_{args.source}2{args.target}_{args.codec}"

    source = load_domain(args.source, args.construction, args.device)
    target = load_domain(args.target, args.construction + args.holdout, args.device)
    target_ids = list(target)
    target_construction = target_ids[:args.construction]
    target_holdout = target_ids[args.construction:args.construction + args.holdout]
    print(f"[phase3probe] source={args.source} ({len(source)} tasks) -> "
          f"target={args.target} ({len(target_holdout)} held out, disjoint from "
          f"{len(target_construction)} used for the same-domain reference), "
          f"k={args.k}, steps={args.steps}, codec={args.codec}")

    t0 = time.time()
    source_lib = build_library(source, list(source), args.k, fit_kw, args.ledger, tag, args.source)
    target_lib = build_library(target, target_construction, args.k, fit_kw, args.ledger, tag,
                               args.target)
    conditions = {
        "source_lib": source_lib,
        "source_shuf": source_lib.shuffled(seed=0),
        "empty": source_lib.empty(),
        "target_lib": target_lib,
    }
    print(f"[phase3probe] libraries built ({(time.time()-t0)/60:.1f} min)")

    results = {c: [] for c in conditions}
    for i, tid in enumerate(target_holdout):
        per = {}
        for cond, lib in conditions.items():
            for seed in args.seeds:
                res = fit_task(target[tid].train, seed=seed, library=lib, **fit_kw)
                append(args.ledger, {
                    "experiment": tag, "task_id": tid, "condition": cond, "seed": seed,
                    "bits_total": res.bits_total, "bits_recon": res.bits_recon,
                    "bits_weights": res.bits_weights, "steps": res.steps,
                    "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
                })
                per.setdefault(cond, []).append(res.bits_total)
        means = {c: st.mean(v) for c, v in per.items()}
        for c, v in means.items():
            results[c].append(v)
        print(f"  [holdout {i+1}/{len(target_holdout)}] {tid}  "
              + "  ".join(f"{c}={means[c]:8.2f}" for c in conditions))

    print(f"\n[phase3probe] done in {(time.time()-t0)/60:.1f} min.")
    print(f"  bits_total across {len(target_holdout)} held-out {args.target} tasks, "
          f"{len(args.seeds)} seeds each:")
    for c in conditions:
        print(f"    {c:12s}: mean {st.mean(results[c]):8.2f}   median {st.median(results[c]):8.2f}")

    def verdict(a: str, b: str) -> str:
        mo = st.mean(results[a]) < st.mean(results[b])
        me = st.median(results[a]) < st.median(results[b])
        w = sum(1 for x, y in zip(results[a], results[b]) if x < y)
        try:
            ptxt = f"p={wilcoxon(results[a], results[b]).pvalue:.4f}"
        except ValueError as e:
            ptxt = f"p n/a ({e})"
        tag_ = "PASS" if (mo and me) else ("MIXED" if (mo or me) else "FAIL")
        return (f"  {a} vs {b}: mean {'<' if mo else '>='}, median {'<' if me else '>='}, "
                f"wins {w}/{len(results[a])}, {ptxt}  -> {tag_}")

    print("\n  death criteria:")
    print(verdict("source_lib", "source_shuf"))
    print(verdict("source_lib", "empty"))

    # descriptive, not a criterion: how much of the same-domain benefit
    # survives the domain boundary. 0% = no transfer, 100% = the source
    # library is worth as much as a library grown at home.
    e, s, t = (st.mean(results[c]) for c in ("empty", "source_lib", "target_lib"))
    if e > t:
        print(f"\n  cross-domain benefit as a fraction of same-domain: "
              f"{100 * (e - s) / (e - t):.1f}%  (empty {e:.1f}, source_lib {s:.1f}, target_lib {t:.1f})")
    else:
        print(f"\n  same-domain reference did not beat empty ({t:.1f} vs {e:.1f}); "
              "the fraction is undefined and the target domain itself may be the problem")


if __name__ == "__main__":
    main()
