#!/usr/bin/env python3
"""Recompute every reported human-evaluation number from the released table."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(HERE, "human_ratings_public.csv")
SUMMARY = os.path.join(HERE, "human_summary_public.json")

N_BOOT = 10000
SEED = 0
CONDITIONS = ("depth", "canny")
METHODS = ("refroute", "dreamrenderer_star")
DEFECTS = (
    "duplicated_major_part",
    "missing_or_malformed_major_part",
    "unnatural_region_boundary",
    "disconnected_or_collaged_structure",
)
DEFECT_LABELS = {
    "duplicated_major_part": "part duplication",
    "missing_or_malformed_major_part": "missing or malformed part",
    "unnatural_region_boundary": "unnatural region boundary",
    "disconnected_or_collaged_structure": "foreign patch",
}
ANIMATE = {"animals", "birds", "butterflies"}


def rows(condition: str) -> list[dict]:
    with open(RATINGS, newline="", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if r["condition"] == condition]


def boot(by_cluster: dict[str, list[float]]) -> tuple[float, float, float]:
    # Sort by bare basename as the frozen analysis did; the domain prefix only disambiguates.
    keys = np.array(sorted(by_cluster, key=lambda value: os.path.basename(value)))
    flat = [v for k in keys for v in by_cluster[k]]
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        picked = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = np.mean([v for k in picked for v in by_cluster[k]])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(flat)), float(lo), float(hi)


def rate(records: list[dict], method: str, field) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = {}
    for r in records:
        if r["method"] != method:
            continue
        grouped.setdefault(r["source_id"], []).append(float(field(r)))
    return boot(grouped)


def paired(records: list[dict], field) -> tuple[float, float, float]:
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for r in records:
        pairs.setdefault((r["rater_id"], r["case_id"]), {})[r["method"]] = float(field(r))
    grouped: dict[str, list[float]] = {}
    source = {(r["rater_id"], r["case_id"]): r["source_id"] for r in records}
    for key, sides in pairs.items():
        if len(sides) != 2:
            continue
        grouped.setdefault(source[key], []).append(
            sides["refroute"] - sides["dreamrenderer_star"])
    return boot(grouped)


def fmt(triple: tuple[float, float, float]) -> str:
    point, lo, hi = triple
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def compute() -> dict:
    report: dict = {}
    for condition in CONDITIONS:
        records = rows(condition)
        raters = sorted({r["rater_id"] for r in records}, key=lambda s: int(s[1:]))
        cases = sorted({r["case_id"] for r in records})
        sources = sorted({r["source_id"] for r in records})

        not_intact = lambda r: int(r["not_intact"])
        severe = lambda r: int(r["severity"] == "2")

        block: dict = {
            "n_raters": len(raters),
            "n_cases": len(cases),
            "n_source_clusters": len(sources),
            "n_ratings_per_method": sum(
                1 for r in records if r["method"] == "refroute"),
            "not_intact": {m: rate(records, m, not_intact) for m in METHODS},
            "severe": {m: rate(records, m, severe) for m in METHODS},
            "paired_not_intact": paired(records, not_intact),
            "paired_severe": paired(records, severe),
            "by_defect": {
                d: {m: rate(records, m, lambda r, d=d: int(r[d])) for m in METHODS}
                for d in DEFECTS
            },
        }

        # Preference is one judgment per (rater, case); count it once.
        seen: dict[tuple[str, str], str] = {}
        for r in records:
            seen.setdefault((r["rater_id"], r["case_id"]), r["preferred_method"])
        total = len(seen)
        block["preference"] = {
            value: (sum(1 for v in seen.values() if v == value),
                    sum(1 for v in seen.values() if v == value) / total)
            for value in ("refroute", "dreamrenderer_star", "none")
        }

        block["by_object_kind"] = {}
        for kind, domains in (("animate", ANIMATE),
                              ("inanimate", None)):
            subset = [r for r in records
                      if (r["domain"] in ANIMATE) == (domains is not None)]
            block["by_object_kind"][kind] = {
                "n_ratings_per_method": sum(
                    1 for r in subset if r["method"] == "refroute"),
                "not_intact": {m: rate(subset, m, not_intact) for m in METHODS},
                "paired": paired(subset, not_intact),
            }

        block["by_rater"] = {}
        for rater in raters:
            subset = [r for r in records if r["rater_id"] == rater]
            block["by_rater"][rater] = {
                m: float(np.mean([int(r["not_intact"])
                                  for r in subset if r["method"] == m]))
                for m in METHODS
            }

        # Duplication flagged by at least two raters, per configuration.
        block["duplication_majority"] = {}
        for method in METHODS:
            flagged = 0
            for case in cases:
                marks = sum(1 for r in records
                            if r["case_id"] == case and r["method"] == method
                            and r["duplicated_major_part"] == "1")
                flagged += int(marks >= 2)
            block["duplication_majority"][method] = (flagged, len(cases))

        report[condition] = block
    return report


def show(report: dict) -> None:
    for condition, b in report.items():
        print("=" * 72)
        print(f"{condition.upper()}   "
              f"{b['n_raters']} raters x {b['n_cases']} cases = "
              f"{b['n_raters'] * b['n_cases']} paired judgments; "
              f"{b['n_ratings_per_method']} ratings per method; "
              f"{b['n_source_clusters']} source clusters")
        print("=" * 72)
        print(f"{'endpoint':<34}{'RefRoute':<26}{'DreamRenderer*'}")
        for name, key in (("Defect Rate (primary)", "not_intact"),
                          ("Severe Defect Rate", "severe")):
            print(f"{name:<34}{fmt(b[key]['refroute']):<26}"
                  f"{fmt(b[key]['dreamrenderer_star'])}")
        for d in DEFECTS:
            print(f"{DEFECT_LABELS[d] + ' (exploratory)':<34}"
                  f"{fmt(b['by_defect'][d]['refroute']):<26}"
                  f"{fmt(b['by_defect'][d]['dreamrenderer_star'])}")
        print()
        print(f"Paired difference (RefRoute - DreamRenderer*)")
        print(f"  Defect Rate         {fmt(b['paired_not_intact'])}")
        print(f"  Severe Defect Rate  {fmt(b['paired_severe'])}")
        rr = b["not_intact"]["refroute"][0]
        dr = b["not_intact"]["dreamrenderer_star"][0]
        sr = b["severe"]["refroute"][0]
        sd = b["severe"]["dreamrenderer_star"][0]
        print(f"  relative reduction  Defect {1 - rr / dr:.1%}   "
              f"Severe {1 - sr / sd:.1%}")
        print()
        pref = b["preference"]
        print("Reference-transfer preference (exploratory): "
              + ", ".join(f"{k} {v[1]:.3f} ({v[0]})" for k, v in pref.items()))
        dup = b["duplication_majority"]
        print("Duplication flagged by >=2 raters: "
              + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in dup.items()))
        print()
        print("By object kind (Defect Rate):")
        for kind, k in b["by_object_kind"].items():
            print(f"  {kind:<11}{fmt(k['not_intact']['refroute']):<26}"
                  f"{fmt(k['not_intact']['dreamrenderer_star']):<26}"
                  f"paired {fmt(k['paired'])}")
        print()
        print("By rater (Defect Rate):")
        for rater, v in b["by_rater"].items():
            print(f"  {rater:<5}RefRoute {v['refroute']:.3f}   "
                  f"DreamRenderer* {v['dreamrenderer_star']:.3f}   "
                  f"diff {v['refroute'] - v['dreamrenderer_star']:+.3f}")
        print()


def check(report: dict) -> int:
    with open(SUMMARY, encoding="utf-8") as handle:
        published = json.load(handle)["conditions"]
    bad = 0
    def close(label: str, got: float, ref: float) -> None:
        nonlocal bad
        if abs(got - ref) > 5e-4:
            print(f"MISMATCH {label}: {got:.6f} vs {ref:.6f}")
            bad += 1

    def triple(label: str, got, ref: dict, point_key: str = "rate") -> None:
        close(f"{label}/{point_key}", got[0], ref[point_key])
        close(f"{label}/ci95-low", got[1], ref["ci95"][0])
        close(f"{label}/ci95-high", got[2], ref["ci95"][1])

    for condition, b in report.items():
        want = published[condition]
        for key in ("n_raters", "n_cases", "n_source_clusters",
                    "n_ratings_per_method"):
            if b[key] != want[key]:
                print(f"MISMATCH {condition}/{key}: {b[key]} vs {want[key]}")
                bad += 1
        for key in ("not_intact", "severe"):
            for method in METHODS:
                triple(f"{condition}/{key}/{method}", b[key][method],
                       want[key][method])
        for key in ("paired_not_intact", "paired_severe"):
            triple(f"{condition}/{key}", b[key], want[key], "diff")
        for defect in DEFECTS:
            for method in METHODS:
                triple(
                    f"{condition}/by_defect/{defect}/{method}",
                    b["by_defect"][defect][method],
                    want["by_defect"][method][defect],
                )
        for value in ("refroute", "dreamrenderer_star", "none"):
            got_n, got_rate = b["preference"][value]
            ref = want["preference"][value]
            if got_n != ref["n"]:
                print(f"MISMATCH {condition}/preference/{value}/n: "
                      f"{got_n} vs {ref['n']}")
                bad += 1
            close(f"{condition}/preference/{value}/rate", got_rate, ref["rate"])
        for kind in ("animate", "inanimate"):
            for method in METHODS:
                triple(
                    f"{condition}/by_object_kind/{kind}/{method}",
                    b["by_object_kind"][kind]["not_intact"][method],
                    want["by_object_kind"][kind]["not_intact"][method],
                )
            triple(
                f"{condition}/by_object_kind/{kind}/paired",
                b["by_object_kind"][kind]["paired"],
                want["by_object_kind"][kind]["paired_not_intact"],
                "diff",
            )
        for rater, methods in b["by_rater"].items():
            for method in METHODS:
                close(
                    f"{condition}/by_rater/{rater}/{method}",
                    methods[method], want["by_rater"][rater]["not_intact"][method]
                )
        for method, (flagged, n_cases) in b["duplication_majority"].items():
            ref = want["duplication_majority"][method]
            if flagged != ref["n_flagged"] or n_cases != ref["n_cases"]:
                print(f"MISMATCH {condition}/duplication_majority/{method}: "
                      f"{flagged}/{n_cases} vs {ref['n_flagged']}/{ref['n_cases']}")
                bad += 1
    print("all endpoints match the published summary" if not bad
          else f"{bad} mismatches")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="diff against human_summary_public.json")
    args = parser.parse_args()
    report = compute()
    show(report)
    return check(report) if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
