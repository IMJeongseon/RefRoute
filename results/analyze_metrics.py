#!/usr/bin/env python3
"""Aggregate and compare the released per-configuration automatic metrics.

Examples
--------
Summarize one method across its appearance/spatial files::

    python results/analyze_metrics.py --summary depth_refroute

Compute a paired comparison with the paper's clustered confidence interval::

    python results/analyze_metrics.py \
      --compare depth_refroute depth_dreamrenderer_fg

List every released run and validate its 60-case coverage::

    python results/analyze_metrics.py --all-runs --check
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALUES = ROOT / "results/per_case_metrics.csv"
DEFAULT_MANIFEST = ROOT / "benchmark/manifest_60.json"
N_BOOT = 10_000
SEED = 0

# name -> higher is better, display scale used in the paper
METRICS = {
    "region_dino": (True, 1.0),
    "dino_routing_accuracy": (True, 1.0),
    "vgg_gram_distance": (False, 1000.0),
    "foreground_sam_iou": (True, 1.0),
    "nonreference_clip_t": (True, 1.0),
    "clip_iqa": (True, 1.0),
}


def finite(value: str) -> bool:
    if value == "":
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_map(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {case.get("id", case.get("case_id")):
            case.get("source", case.get("structure_image"))
            for case in manifest["cases"]}


def resolve_runs(prefix: str, run_names: set[str]) -> list[str]:
    if prefix in run_names:
        return [prefix]
    matches = sorted(name for name in run_names if name.startswith(prefix + "_"))
    if not matches:
        raise ValueError(f"no released run matches {prefix!r}")
    return matches


def values_for(rows: list[dict[str, str]], selected: list[str]) -> dict:
    selected_set = set(selected)
    result: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    provenance: dict[str, str] = {}
    for row in rows:
        if row["run"] not in selected_set:
            continue
        for metric in METRICS:
            if not finite(row[metric]):
                continue
            case = row["case_id"]
            value = float(row[metric])
            if case in result[metric] and not math.isclose(
                result[metric][case], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"conflicting {metric} values for {case} across {selected}")
            result[metric][case] = value
            provenance[metric] = row["run"]
    return {metric: {"values": values, "run": provenance.get(metric)}
            for metric, values in result.items() if values}


def summarize(block: dict) -> dict:
    report = {}
    for metric, entry in block.items():
        values = np.asarray(list(entry["values"].values()), dtype=np.float64)
        report[metric] = {
            "run": entry["run"],
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "paper_display_mean": float(values.mean() * METRICS[metric][1]),
        }
    return report


def clustered_difference(
    first: dict[str, float],
    second: dict[str, float],
    sources: dict[str, str],
) -> dict:
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("comparison has no shared configurations")
    by_source: dict[str, list[float]] = defaultdict(list)
    for case in shared:
        by_source[sources[case]].append(first[case] - second[case])
    # The frozen analysis sorted source-image basenames before drawing cluster
    # indices. Preserve that ordering so the seeded Monte Carlo interval is
    # bit-for-bit reproducible from the public domain-prefixed identifiers.
    keys = np.asarray(sorted(by_source, key=lambda value: Path(value).name))
    flat = [value for key in keys for value in by_source[str(key)]]
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for index in range(N_BOOT):
        picked = rng.choice(keys, size=len(keys), replace=True)
        draws[index] = np.mean(
            [value for key in picked for value in by_source[str(key)]])
    low, high = np.percentile(draws, (2.5, 97.5))
    return {
        "difference": float(np.mean(flat)),
        "ci95": [float(low), float(high)],
        "n_configurations": len(shared),
        "n_structure_clusters": len(keys),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
    }


def compare(first: dict, second: dict, sources: dict[str, str]) -> dict:
    report = {}
    for metric in METRICS:
        if metric not in first or metric not in second:
            continue
        a, b = first[metric]["values"], second[metric]["values"]
        paired = clustered_difference(a, b, sources)
        scale = METRICS[metric][1]
        report[metric] = {
            "first_run": first[metric]["run"],
            "second_run": second[metric]["run"],
            "first_mean": float(np.mean(list(a.values()))),
            "second_mean": float(np.mean(list(b.values()))),
            "higher_is_better": METRICS[metric][0],
            **paired,
            "paper_display": {
                "scale": scale,
                "first_mean": float(np.mean(list(a.values())) * scale),
                "second_mean": float(np.mean(list(b.values())) * scale),
                "difference": paired["difference"] * scale,
                "ci95": [value * scale for value in paired["ci95"]],
            },
        }
    return report


def coverage_check(rows: list[dict[str, str]], cases: set[str]) -> dict:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    problems = []
    for run, records in sorted(grouped.items()):
        ids = [record["case_id"] for record in records]
        if len(ids) != 60 or len(set(ids)) != 60 or set(ids) != cases:
            problems.append({
                "run": run, "rows": len(ids), "unique_cases": len(set(ids)),
                "missing": sorted(cases - set(ids)),
                "unknown": sorted(set(ids) - cases),
            })
    if problems:
        raise ValueError(f"metric coverage check failed: {problems}")
    return {"runs": len(grouped), "rows": len(rows),
            "configurations_per_run": 60}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--summary", metavar="RUN_PREFIX")
    group.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"))
    group.add_argument("--all-runs", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.values)
    sources = source_map(args.manifest)
    run_names = {row["run"] for row in rows}
    report: dict = {
        "policy": {
            "pairing": "per configuration; FIRST minus SECOND",
            "cluster_unit": "source structure image",
            "bootstrap": {"replicates": N_BOOT, "seed": SEED,
                          "interval": "percentile 95%"},
        }
    }
    if args.check:
        report["coverage"] = coverage_check(rows, set(sources))
    if args.summary:
        chosen = resolve_runs(args.summary, run_names)
        report["summary"] = {"requested": args.summary, "runs": chosen,
                             "metrics": summarize(values_for(rows, chosen))}
    elif args.compare:
        first_runs = resolve_runs(args.compare[0], run_names)
        second_runs = resolve_runs(args.compare[1], run_names)
        report["comparison"] = {
            "first": args.compare[0], "second": args.compare[1],
            "first_runs": first_runs, "second_runs": second_runs,
            "metrics": compare(values_for(rows, first_runs),
                               values_for(rows, second_runs), sources),
        }
    elif args.all_runs:
        report["runs"] = {
            run: summarize(values_for(rows, [run]))
            for run in sorted(run_names)
        }
    else:
        comparisons = (
            ("depth_refroute", "depth_dreamrenderer_fg"),
            ("canny_refroute", "canny_dreamrenderer_fg"),
            ("depth_refroute", "depth_global_redux"),
            ("canny_refroute", "canny_global_redux"),
            ("depth_refroute", "depth_abl_full_isolation"),
            ("canny_refroute", "canny_abl_full_isolation"),
        )
        report["reported_comparisons"] = {}
        for first_name, second_name in comparisons:
            first_runs = resolve_runs(first_name, run_names)
            second_runs = resolve_runs(second_name, run_names)
            report["reported_comparisons"][f"{first_name}--{second_name}"] = compare(
                values_for(rows, first_runs), values_for(rows, second_runs), sources)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
