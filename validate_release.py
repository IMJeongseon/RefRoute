#!/usr/bin/env python3
"""Validate the shareable RefRoute release bundle without model weights."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = (
    "requirements.txt",
    "benchmark/manifest_60.json",
    "benchmark/image_inventory.csv",
    "code/src/run.py",
    "code/src/prepare.py",
    "code/experiments/generate_variants.py",
    "code/experiments/manifest_to_plan.py",
    "results/per_case_metrics.csv",
    "results/reported_run_map.json",
    "results/human_ratings_public.csv",
    "results/human_summary_public.json",
    "results/study_codebook_public.json",
    "results/analyze_metrics.py",
    "results/verify_human_eval.py",
    "results/runtime_matched.json",
    "results/runtime_dreamrenderer.json",
    "results/verify_runtime.py",
)

FORBIDDEN = ("results/human_preference_reasons_public.csv",)
METRIC_COLUMNS = (
    "run", "case_id", "domain", "prompt_key", "reference_count", "output_name",
    "region_dino", "dino_routing_accuracy", "vgg_gram_distance",
    "foreground_sam_iou", "nonreference_clip_t", "clip_iqa",
)
RUN_MAP_SECTIONS = (
    "main_comparison",
    "routing_and_connectivity_ablation",
    "dreamrenderer_input_matching",
    "seed_replication",
    "reference_scale_sensitivity",
    "foreground_token_selection",
)


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(*args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / args[0]), *args[1:]],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in strings(child)]
    return []


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing release artifacts: {missing}")
    unexpected = [name for name in FORBIDDEN if (ROOT / name).exists()]
    if unexpected:
        raise FileExistsError(f"unexpected exploratory artifacts: {unexpected}")

    manifest = json.loads((ROOT / "benchmark/manifest_60.json").read_text())
    cases = manifest["cases"]
    case_ids = {case["case_id"] for case in cases}
    assert len(cases) == len(case_ids) == 60
    assert len({case["structure_image"] for case in cases}) == 31
    assert set(Counter(case["domain"] for case in cases).values()) == {10}
    assert set(Counter(case["scene"] for case in cases).values()) == {15}
    assert all(len(case["assignments"]) == 3 for case in cases)

    inventory = read_csv("benchmark/image_inventory.csv")
    assert len(inventory) == 42
    assert all(len(row["sha256"]) == 64 for row in inventory)

    metrics = read_csv("results/per_case_metrics.csv")
    run_counts = Counter(row["run"] for row in metrics)
    assert tuple(metrics[0]) == METRIC_COLUMNS
    run_map = json.loads(
        (ROOT / "results/reported_run_map.json").read_text())
    prefixes = {
        prefix for section in RUN_MAP_SECTIONS for prefix in strings(run_map[section])
    }
    expected_runs = {
        f"{prefix}_{family}" for prefix in prefixes
        for family in ("appearance", "spatial")
    }
    assert len(metrics) == 3_600
    assert set(run_counts) == expected_runs
    assert len(run_counts) == 60 and set(run_counts.values()) == {60}
    assert all(row["case_id"] in case_ids for row in metrics)

    ratings = read_csv("results/human_ratings_public.csv")
    anonymous_ids = {f"R{i}" for i in range(1, 12)}
    assert len(ratings) == 2_640
    assert {row["rater_id"] for row in ratings} == anonymous_ids

    preferences: dict[tuple[str, str, str], str] = {}
    for row in ratings:
        key = (row["condition"], row["rater_id"], row["case_id"])
        previous = preferences.setdefault(key, row["preferred_method"])
        assert previous == row["preferred_method"]
    assert len(preferences) == 1_320

    codebook = json.loads(
        (ROOT / "results/study_codebook_public.json").read_text())
    assert len(codebook["ab_mapping"]) == 120
    assert "preference_reason_columns" not in codebook
    assert "missing_value_rules" in codebook

    pins = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert len(pins) == len(set(pins))
    assert all("==" in pin for pin in pins)

    for relative in (
        "code/src/run.py",
        "code/src/prepare.py",
        "code/experiments/generate_variants.py",
        "code/experiments/manifest_to_plan.py",
        "results/analyze_metrics.py",
        "results/verify_human_eval.py",
        "results/verify_runtime.py",
    ):
        compile((ROOT / relative).read_text(encoding="utf-8"), relative, "exec")

    with tempfile.TemporaryDirectory() as directory:
        plan = Path(directory) / "plan.json"
        run("code/experiments/manifest_to_plan.py", "--out", str(plan))
        rebuilt = json.loads(plan.read_text())
        assert len(rebuilt["cases"]) == 60

    run("results/analyze_metrics.py", "--all-runs", "--check")
    run("results/verify_human_eval.py", "--check")
    run("results/verify_runtime.py", "--check")

    print("release validation passed")
    print("  60 configurations / 31 structures / 42 image digests")
    print("  3,600 reported metric rows across 60 complete runs")
    print("  2,640 output ratings covering 1,320 paired trials")
    print("  human and runtime aggregates match the reported values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
