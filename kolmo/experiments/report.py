"""Read kolmo/ledger.csv and print the death-criterion verdicts.

This exists because the ledger, not a script's stdout, is the source of
truth per the plan (section 7): a claim that can't be re-derived from the
CSV isn't a claim this codebase makes.

Usage:
    python -m kolmo.experiments.report
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict

LEDGER_PATH = "kolmo/ledger.csv"


def load(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report_phase0(rows: list) -> None:
    p0 = [r for r in rows if r["experiment"] == "phase0"]
    if not p0:
        print("[phase0] no rows yet")
        return
    print(f"\n=== Phase 0: bits(tau) vs. held-out exact match ({len(p0)} task fits) ===")
    bits = [float(r["bits_total"]) for r in p0]
    print(f"  mean bits_total: {st.mean(bits):.2f}   min: {min(bits):.2f}   max: {max(bits):.2f}")

    has_solved = all(r.get("solved", "") != "" for r in p0)
    if not has_solved:
        print("  (rows from before the 'solved' column was added -- can't compute the "
          "death criterion from the ledger alone for this run; see the run's stdout log.)")
        return

    solved_bits = [float(r["bits_total"]) for r in p0 if r["solved"] == "True"]
    unsolved_bits = [float(r["bits_total"]) for r in p0 if r["solved"] == "False"]
    print(f"  solved: {len(solved_bits)}/{len(p0)}")
    if solved_bits:
        print(f"  mean bits_total | solved:   {st.mean(solved_bits):8.2f} (n={len(solved_bits)})")
    if unsolved_bits:
        print(f"  mean bits_total | unsolved: {st.mean(unsolved_bits):8.2f} (n={len(unsolved_bits)})")
    if solved_bits and unsolved_bits:
        verdict = "PASS (solved tasks compress more)" if st.mean(solved_bits) < st.mean(unsolved_bits) else \
                  "FAIL (solved tasks do NOT compress more -- objective is not correlated with solving)"
        print(f"  death criterion: {verdict}")
    else:
        print("  death criterion: INCONCLUSIVE (need both solved and unsolved tasks)")


def report_phase1(rows: list) -> None:
    p1 = [r for r in rows if r["experiment"] == "phase1"]
    if not p1:
        print("[phase1] no rows yet")
        return
    by_task_cond = defaultdict(list)
    for r in p1:
        by_task_cond[(r["task_id"], r["condition"])].append(float(r["bits_total"]))
    task_ids = sorted({t for t, _ in by_task_cond})
    means = {(t, c): st.mean(v) for (t, c), v in by_task_cond.items()}

    print(f"\n=== Phase 1: real vs. shuffled vs. empty library ({len(task_ids)} held-out tasks) ===")
    per_cond = defaultdict(list)
    for t in task_ids:
        row = {c: means.get((t, c)) for c in ("real", "shuffled", "empty")}
        if None in row.values():
            continue
        for c, v in row.items():
            per_cond[c].append(v)
        print(f"  {t}:  real={row['real']:8.2f}  shuffled={row['shuffled']:8.2f}  empty={row['empty']:8.2f}")

    if not per_cond["real"]:
        print("  (no task has all three conditions complete yet)")
        return

    overall = {c: st.mean(v) for c, v in per_cond.items()}
    n = len(per_cond["real"])
    n_beats_shuf = sum(1 for r, s in zip(per_cond["real"], per_cond["shuffled"]) if r < s)
    n_beats_empty = sum(1 for r, e in zip(per_cond["real"], per_cond["empty"]) if r < e)

    print(f"\n  mean bits_total:  real={overall['real']:.2f}  shuffled={overall['shuffled']:.2f}  "
          f"empty={overall['empty']:.2f}")
    print(f"  real < shuffled on {n_beats_shuf}/{n} tasks (paired)")
    print(f"  real < empty    on {n_beats_empty}/{n} tasks (paired)")
    print(f"  death criterion (real beats shuffled): "
          f"{'PASS' if overall['real'] < overall['shuffled'] else 'FAIL'}")
    print(f"  death criterion (real beats empty):    "
          f"{'PASS' if overall['real'] < overall['empty'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER_PATH)
    args = ap.parse_args()
    rows = load(args.ledger)
    print(f"[report] {len(rows)} rows in {args.ledger}")
    report_phase0(rows)
    report_phase1(rows)


if __name__ == "__main__":
    main()
