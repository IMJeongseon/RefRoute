"""Rebuild a runnable plan file from the released benchmark manifest.

`release/benchmark/manifest_60.json` describes the 60 configurations without
carrying any image.  Every path the generators need follows a fixed convention,
so the plan can be reconstructed from the manifest alone; only the images and
masks themselves have to be supplied.

  python experiments/manifest_to_plan.py \
      --manifest release/benchmark/manifest_60.json --out data/prepared/plans/plan_intent.json

  python experiments/manifest_to_plan.py --verify data/prepared/plans/plan_intent.json

`--verify` compares the reconstruction against an existing plan and reports any
field that differs, which is how the manifest is checked for completeness.

The emitted `steps` and `guidance` are the reported protocol: 20 steps and
guidance 10, the Depth value.  Canny runs pass `--guidance 30` on the command
line.  The plan file used for the reported runs carried a stale 28-step default
that every run overrode; `--plan-defaults` reproduces that file exactly and is
implied by `--verify`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The script is shipped both in the repository (experiments/) and inside the
# release bundle (release/code/experiments/), so the manifest sits at a
# different depth in each.  Both are tried before giving up.
MANIFEST_CANDIDATES = (
    ROOT / "release/benchmark/manifest_60.json",
    ROOT.parent / "benchmark/manifest_60.json",
)

ROUTING_POLICY = {
    "N=1": "all three target masks -> R_order[0]",
    "N=2": "M_order[0] -> R_order[0]; remaining two masks -> R_order[1]",
    "N=3": "M_order[i] -> R_order[i] for i=0,1,2",
}


def build_plan(manifest: dict, plan_defaults: bool = False) -> dict:
    cases = []
    for case in manifest["cases"]:
        source = f"data/{case['structure_image']}"
        domain, stem = case["domain"], Path(source).stem
        settings = case["settings"]
        defaults = settings.get("plan_file_defaults", {})
        parts = []
        for part in case["assignments"]:
            donor = f"data/{part['reference_image']}"
            parts.append({
                "name": part["part_name"],
                "target_mask": f"data/{part['target_mask']}",
                "donor": donor,
                "donor_mask": f"data/{part['reference_foreground_mask']}",
            })
        cases.append({
            "id": case["case_id"],
            "domain": domain,
            "source": source,
            "depth_control": f"data/prepared/depth/{domain}/{stem}.png",
            "prompt_key": case["scene"],
            "prompt": case["prompt"],
            "order": case["routing_order"],
            "parts": parts,
            "lam": settings["reference_scale_lambda"],
            "threshold": settings["foreground_token_threshold_tau"],
            "reference_token_budget": settings["reference_token_budget"],
            # The plan file carried a stale 28-step default that every reported
            # run overrode on the command line.  Emit the reported value, so a
            # plan built here reproduces the protocol even when --steps is not
            # passed; --plan-defaults restores the original for verification.
            "steps": (defaults.get("steps", settings["steps"]) if plan_defaults
                      else settings["steps"]),
            "guidance": (defaults.get("guidance", settings["guidance"]["depth"])
                         if plan_defaults else settings["guidance"]["depth"]),
            "seed": settings["seed"],
        })
    return {"routing_policy": manifest.get("routing_policy") or ROUTING_POLICY,
            "cases": cases}


def differences(built: dict, existing: dict) -> list[str]:
    problems = []
    by_id = {case["id"]: case for case in existing["cases"]}
    if len(built["cases"]) != len(existing["cases"]):
        problems.append(f"case count {len(built['cases'])} != {len(existing['cases'])}")
    for case in built["cases"]:
        other = by_id.get(case["id"])
        if other is None:
            problems.append(f"{case['id']}: absent from the reference plan")
            continue
        for key in case:
            if case[key] != other.get(key):
                problems.append(f"{case['id']}.{key}: {case[key]!r} != {other.get(key)!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: the manifest shipped beside this script")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path,
                        help="compare the reconstruction against this plan file")
    parser.add_argument("--plan-defaults", action="store_true",
                        help="emit the original plan file's stale sampling "
                             "defaults instead of the reported protocol; "
                             "implied by --verify")
    args = parser.parse_args()

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = next((c for c in MANIFEST_CANDIDATES if c.is_file()), None)
        if manifest_path is None:
            parser.error("no manifest found; pass --manifest explicitly")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = build_plan(manifest, plan_defaults=args.plan_defaults or bool(args.verify))

    if args.verify:
        problems = differences(plan, json.loads(args.verify.read_text(encoding="utf-8")))
        if problems:
            print(f"{len(problems)} difference(s):")
            print("\n".join(problems[:40]))
            return 1
        print(f"identical: {len(plan['cases'])} configurations reconstructed from the manifest")
        return 0

    if not args.out:
        parser.error("--out is required unless --verify is given")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(plan['cases'])} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
