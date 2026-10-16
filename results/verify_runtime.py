#!/usr/bin/env python3
"""Recompute the reported runtime table from the released measurements.

This script is self-contained: it reads only `runtime_matched.json` and
`runtime_dreamrenderer.json`, which sit beside it, and needs nothing beyond the
standard library.

  python results/verify_runtime.py           # print the table
  python results/verify_runtime.py --check   # diff against the reported values

`--check` compares the recomputed means and speedups with the values printed in
the manuscript and exits non-zero on any mismatch.

RefRoute timings come from the matched-protocol run, which times both methods
across the same start and stop points. DreamRenderer* timings come from its own
run under the same protocol: 20 sampling steps, three repeats after a discarded
warm-up, one measurement process per repeat on an idle GPU.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MATCHED = os.path.join(HERE, "runtime_matched.json")
BASELINE = os.path.join(HERE, "runtime_dreamrenderer.json")

# The manuscript reports K = 1, 3 and 16.  The baseline file also carries K = 2
# and K = 7, measured for the sequence-scaling discussion and not tabulated.
REPORTED_K = (1, 3, 16)
REPORTED = {
    1: {"dreamrenderer_star": 29.4, "refroute": 11.7, "speedup": 2.5},
    3: {"dreamrenderer_star": 40.6, "refroute": 13.2, "speedup": 3.1},
    16: {"dreamrenderer_star": 98.9, "refroute": 22.0, "speedup": 4.5},
}


def load() -> tuple[dict[int, float], dict[int, float], dict]:
    matched = json.load(open(MATCHED, encoding="utf-8"))
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    ours = {int(r["reference_count"]): float(r["refroute_total_mean"])
            for r in matched["results"]}
    theirs = {int(r["reference_count"]): float(r["seconds_mean"])
              for r in baseline["results"]}
    return ours, theirs, matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="diff against the values reported in the manuscript")
    args = parser.parse_args()

    ours, theirs, matched = load()
    missing = [k for k in REPORTED_K if k not in ours or k not in theirs]
    if missing:
        print(f"MISSING measurements for K = {missing}")
        return 1

    print(f"{matched['gpu']}, {matched['steps']} steps, "
          f"mean of {matched['repeats']} runs after a discarded warm-up\n")
    print(f"{'Method':<20}" + "".join(f"{'K='+str(k):>10}" for k in REPORTED_K))
    print(f"{'DreamRenderer*':<20}" + "".join(f"{theirs[k]:>10.1f}" for k in REPORTED_K))
    print(f"{'RefRoute':<20}" + "".join(f"{ours[k]:>10.1f}" for k in REPORTED_K))
    print(f"{'Speedup':<20}" + "".join(f"{theirs[k]/ours[k]:>9.1f}x" for k in REPORTED_K))

    extra = sorted(set(theirs) - set(REPORTED_K))
    if extra:
        print(f"\nAlso measured for the baseline but not tabulated: K = "
              f"{', '.join(str(k) for k in extra)}")

    if not args.check:
        return 0

    bad = 0

    def close(label: str, got: float, ref: float, tolerance: float) -> None:
        nonlocal bad
        if abs(got - ref) > tolerance:
            print(f"MISMATCH {label}: {got:.3f} vs {ref:.3f}")
            bad += 1

    for k in REPORTED_K:
        # The table prints one decimal, so a rounding-width tolerance is exact
        # enough to catch a changed measurement without tripping on formatting.
        close(f"K={k}/dreamrenderer_star", theirs[k], REPORTED[k]["dreamrenderer_star"], 0.05)
        close(f"K={k}/refroute", ours[k], REPORTED[k]["refroute"], 0.05)
        close(f"K={k}/speedup", theirs[k] / ours[k], REPORTED[k]["speedup"], 0.05)

    print("\nall runtime values match the reported table" if not bad
          else f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
