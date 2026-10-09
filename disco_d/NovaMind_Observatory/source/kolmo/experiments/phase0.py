"""Phase 0: does bits(tau) correlate with solving the task?

Death criterion (from the plan): if it doesn't, the objective is wrong --
stop and rethink before building anything on top of it. This script exists
to let that happen, not to produce a nice-looking number.

Two runs came back INCONCLUSIVE (0/25 solved under both the Gaussian and
the variational weight code), so there was nothing to correlate.
--variants exists to unlock that criterion without ASSUMING any symmetry:
each task is fitted under four hypotheses and MDL itself picks. Selection
bits = reconstruction of the original demos + weight bits + the price of
the hypothesis (2 bits for the choice index; +log2(10!) for color_canon,
since the receiver cannot derive the color mapping from data it does not
have yet).

The four hypotheses include the SAME D4 symmetry claim in both ledger
columns it can occupy, which is the point:

  d4_aug   -- augment the demos. The symmetry is paid FOR in weight bits:
              8x the pairs demands precision in 8x more places. Measured
              over 25 tasks under both codecs, it lost every single
              selection while being the only variant that solved anything
              (2 tasks) -- correct accounting of a badly framed question.
  d4_equiv -- constrain the kernels to the D4-invariant subspace. The
              symmetry is paid BY: 3 coefficients per 3x3 kernel instead
              of 9, so a task that really is D4-symmetric costs strictly
              fewer bits. This is the column where MDL can reward it.

Where D4 does not hold, d4_equiv simply cannot fit the demos and its
reconstruction bits blow up, so it loses -- the hypothesis stays priced,
never assumed.

Usage:
    python -m kolmo.experiments.phase0 --tasks 30 --steps 2000
    python -m kolmo.experiments.phase0 --tasks 30 --steps 2000 --variants
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import time

import torch

from ..codec import bits_crossentropy
from ..fit import GAUSSIAN, VARIATIONAL, exact_match, fit_task
from ..grids import apply_color_map, color_canon_map, d4_pairs, load_tasks
from ..ledger import append

ARC_ROOT = "legacy/novamind/data/ARC"
LEDGER_PATH = "kolmo/ledger.csv"

BITS_COLOR_MAP = math.log2(math.factorial(10))  # ~21.79: transmit the color bijection
BITS_VARIANT_CHOICE = math.log2(4.0)            # 2 bits: transmit which variant was used


@torch.no_grad()
def _recon_bits(res, pairs: list, mc_samples: int = 16) -> float:
    """Reconstruction bits of `pairs` under the fitted code.

    Every variant must be scored on the SAME data (the original demos) --
    d4 fits 8x as many pairs, so its own bits_recon is not comparable.
    Under the variational codec the code pays E_q[CE], estimated here by
    sampling weights from the fitted posterior, so the scoring estimator
    matches the one bits_weights was computed against.
    """
    net = res.net
    if res.theta_log_sigma is None:
        return sum(bits_crossentropy(net(p.x), p.y_idx, p.mask).item() for p in pairs)

    mean, sigma = res.theta.to(res.theta_log_sigma.device), torch.exp(res.theta_log_sigma)
    total = 0.0
    for _ in range(mc_samples):
        torch.nn.utils.vector_to_parameters(mean + sigma * torch.randn_like(sigma), net.parameters())
        total += sum(bits_crossentropy(net(p.x), p.y_idx, p.mask).item() for p in pairs)
    torch.nn.utils.vector_to_parameters(mean, net.parameters())  # restore the point estimate
    return total / mc_samples


def _fit_variants(task, steps: int, seed: int, device: str, codec: str) -> dict:
    """Fit the three hypotheses. Returns {name: (selection_bits, solved, res)}.

    selection_bits for every variant describes the SAME data (the original
    demos, canonically relabeled for color_canon -- a bijection, so equal
    cost), which is what makes the three numbers comparable.
    """
    mapping = color_canon_map(task.train)
    canon_train = [apply_color_map(p, mapping) for p in task.train]
    canon_test = [apply_color_map(p, mapping) for p in task.test]
    variants = {
        # name: (train pairs, demos to score, test pairs, extra bits, symmetric net)
        "none": (task.train, task.train, task.test, 0.0, False),
        # the SAME symmetry hypothesis in the two ledger columns it can occupy:
        # d4_aug pays for it in weight bits (fits 8x the pairs), d4_equiv is
        # paid BY it (3 coefficients per 3x3 kernel instead of 9)
        "d4_aug": (d4_pairs(task.train), task.train, task.test, 0.0, False),
        "d4_equiv": (task.train, task.train, task.test, 0.0, True),
        "color_canon": (canon_train, canon_train, canon_test, BITS_COLOR_MAP, False),
    }
    out = {}
    for name, (fit_pairs, score_pairs, test_pairs, extra, sym) in variants.items():
        res = fit_task(fit_pairs, steps=steps, device=device, seed=seed, codec=codec,
                       symmetric=sym)
        sel_bits = _recon_bits(res, score_pairs) + res.bits_weights + extra + BITS_VARIANT_CHOICE
        solved = exact_match(res.net, test_pairs)
        out[name] = (sel_bits, solved, res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    # smoke runs MUST point this elsewhere: a short run writes rows under the
    # same experiment name, and they silently contaminate any later average
    # that forgets to filter by steps (this happened -- see the phase2
    # variational M sweep, where two 100-step rows moved M=1/M=2 from ~245
    # to ~14862 bits and would have flipped the chosen M if read naively)
    ap.add_argument("--ledger", type=str, default=LEDGER_PATH)
    ap.add_argument("--variants", action="store_true",
                    help="fit none/d4/color_canon per task, MDL-select between them")
    ap.add_argument("--codec", type=str, default=GAUSSIAN, choices=[GAUSSIAN, VARIATIONAL],
                    help="weight code: gaussian prices norm, variational prices precision")
    args = ap.parse_args()

    tasks = load_tasks("training", ARC_ROOT, device=args.device, max_size=15, limit=args.tasks)
    mode = "variants" if args.variants else "plain"
    print(f"[phase0] {len(tasks)} shape-preserving tasks loaded, steps={args.steps}, "
          f"seed={args.seed}, mode={mode}, codec={args.codec}")

    rows = []
    t_start = time.time()
    for i, (task_id, task) in enumerate(tasks.items()):
        if not args.variants:
            res = fit_task(task.train, steps=args.steps, device=args.device, seed=args.seed,
                           codec=args.codec)
            solved = exact_match(res.net, task.test)
            bits_sel = res.bits_total
            append(args.ledger, {
                "experiment": f"phase0_{args.codec}", "task_id": task_id, "condition": "none", "seed": args.seed,
                "bits_total": res.bits_total, "bits_recon": res.bits_recon, "bits_weights": res.bits_weights,
                "steps": res.steps, "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb,
                "solved": solved,
            })
            print(f"[{i+1}/{len(tasks)}] {task_id}  bits={res.bits_total:8.2f}  "
                  f"recon={res.bits_recon:6.2f}  w={res.bits_weights:8.2f}  "
                  f"solved={solved}  ({res.seconds:.1f}s)")
        else:
            t_task = time.time()
            fits = _fit_variants(task, steps=args.steps, seed=args.seed, device=args.device,
                                 codec=args.codec)
            for name, (sel_bits, v_solved, res) in fits.items():
                append(args.ledger, {
                    "experiment": f"phase0_variants_{args.codec}", "task_id": task_id, "condition": name,
                    "seed": args.seed, "bits_total": sel_bits, "bits_recon": res.bits_recon,
                    "bits_weights": res.bits_weights, "steps": res.steps,
                    "seconds": res.seconds, "peak_vram_mb": res.peak_vram_mb, "solved": v_solved,
                })
            chosen = min(fits, key=lambda n: fits[n][0])
            bits_sel, solved, _ = fits[chosen]
            append(args.ledger, {
                "experiment": f"phase0_variants_{args.codec}", "task_id": task_id,
                "condition": f"selected:{chosen}", "seed": args.seed,
                "bits_total": bits_sel, "steps": args.steps,
                "seconds": time.time() - t_task, "solved": solved,
            })
            per_v = "  ".join(f"{n}={fits[n][0]:7.1f}{'*' if n == chosen else ' '}" for n in fits)
            print(f"[{i+1}/{len(tasks)}] {task_id}  {per_v}  solved={solved}  "
                  f"({time.time()-t_task:.1f}s)")
        rows.append((task_id, bits_sel, solved))

    total_dt = time.time() - t_start
    solved_bits = [b for _, b, s in rows if s]
    unsolved_bits = [b for _, b, s in rows if not s]
    n_solved = len(solved_bits)

    print(f"\n[phase0] done in {total_dt/60:.1f} min. {n_solved}/{len(rows)} solved exactly on held-out test.")
    if solved_bits:
        print(f"  mean bits_total | solved:   {st.mean(solved_bits):8.2f} (n={len(solved_bits)})")
    if unsolved_bits:
        print(f"  mean bits_total | unsolved: {st.mean(unsolved_bits):8.2f} (n={len(unsolved_bits)})")
    if solved_bits and unsolved_bits:
        verdict = "PASS (solved tasks compress more)" if st.mean(solved_bits) < st.mean(unsolved_bits) else \
                  "FAIL (solved tasks do NOT compress more -- objective is not correlated with solving)"
        print(f"  death criterion: {verdict}")
    else:
        print("  death criterion: INCONCLUSIVE (need both solved and unsolved tasks in the sample)")


if __name__ == "__main__":
    main()
