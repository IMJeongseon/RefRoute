from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import run
from experiments.control_conditions import (
    CONTROL_MODEL_IDS,
    control_model_id,
    control_run_metadata,
    load_control_image,
)
from experiments.prompts import (
    DOMAIN_NOUNS,
    PROMPT_SETTINGS,
    appearance_clause,
    appearance_prompt,
    base_prompt,
)


DEFAULT_PLAN = ROOT / "data/prepared/plans/plan_v2_final.json"
DEFAULT_OUTPUTS = ROOT / "experiments/outputs"
DEFAULT_APPEARANCE_DESCRIPTIONS = ROOT / "experiments/appearance_descriptions.json"
VARIANTS = (
    "refroute",
    "depth_only",
    "global_redux",
    "one_wrong_reference",
    "all_reference_tokens",
    "random_tokens",
    "first_tokens",
    "full_image_block",
    "mismatched_depth",
)


def routing_groups(order: list[int], count: int) -> tuple[tuple[int, ...], ...]:
    if count == len(order):
        return tuple((index,) for index in order)
    if count == 1:
        return (tuple(order),)
    if count == 2:
        return ((order[0],), (order[1], order[2]))
    if count == 3:
        return tuple((index,) for index in order)
    raise ValueError(f"unsupported reference count: {count}")


def grouped_ownership(
    parts: list[dict[str, str]], order: list[int], count: int
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    masks = run.target_ownership(parts)
    groups = routing_groups(order, count)
    ownership = np.stack([masks[list(group)].sum(0) for group in groups])
    return ownership, groups


def soft_grouped_ownership(
    parts: list[dict[str, str]],
    order: list[int],
    count: int,
    radius: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    ownership, groups = grouped_ownership(parts, order, count)
    softened = []
    for layer in ownership:
        raster = Image.fromarray(
            np.uint8(np.clip(layer, 0.0, 1.0).reshape(64, 64) * 255)
        )
        blurred = raster.filter(ImageFilter.GaussianBlur(radius=radius))
        softened.append(
            np.asarray(blurred, dtype=np.float32).reshape(-1) / 255.0
        )
    return np.stack(softened), groups


def instance_text_tokens(pipeline, prompts: list[str], count: int,
                         device) -> list[torch.Tensor]:
    tokens = []
    for prompt in prompts:
        embeds, _, _ = pipeline.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device)
        length = min(count, embeds.shape[1])
        tokens.append(embeds[:, :length, :])
    return tokens


def proportional_budget(capacities: list[int], budget: int) -> list[int]:
    values = np.asarray(capacities, dtype=np.int64)
    total = int(values.sum())
    if budget < 0:
        raise ValueError("reference token budget must be non-negative")
    if budget == 0 or budget >= total:
        return values.tolist()
    active = np.flatnonzero(values > 0)
    if budget < len(active):
        raise ValueError("reference token budget is smaller than donor count")
    exact = values.astype(np.float64) * budget / total
    allocation = np.floor(exact).astype(np.int64)
    allocation[active] = np.maximum(allocation[active], 1)
    allocation = np.minimum(allocation, values)
    while int(allocation.sum()) > budget:
        candidates = [index for index in active if allocation[index] > 1]
        index = min(candidates, key=lambda item: exact[item] - allocation[item])
        allocation[index] -= 1
    while int(allocation.sum()) < budget:
        candidates = [index for index in active
                      if allocation[index] < values[index]]
        index = max(candidates, key=lambda item: exact[item] - allocation[item])
        allocation[index] += 1
    return allocation.tolist()


def random_tokens(tokens: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if count >= tokens.shape[1]:
        return tokens
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    indices = torch.randperm(
        tokens.shape[1], generator=generator, device=tokens.device)[:count]
    return tokens.index_select(1, indices.sort().values)


def deterministic_kmeans(
    tokens: torch.Tensor,
    count: int,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape[0] != 1:
        raise ValueError("token clustering expects batch size one")
    values = tokens[0].float()
    count = max(1, min(int(count), int(values.shape[0])))
    normalized = torch.nn.functional.normalize(values, dim=-1)
    centre = torch.nn.functional.normalize(normalized.mean(0), dim=0)
    first = int(torch.argmax(normalized @ centre).item())
    selected = [first]
    distance = 1.0 - normalized @ normalized[first]
    for _ in range(1, count):
        distance[selected] = -1
        index = int(torch.argmax(distance).item())
        selected.append(index)
        distance = torch.minimum(
            distance, 1.0 - normalized @ normalized[index]
        )
    centroids = values[torch.tensor(selected, device=values.device)].clone()
    assignments = torch.full(
        (values.shape[0],), -1, device=values.device, dtype=torch.long
    )
    for _ in range(iterations):
        directions = torch.nn.functional.normalize(centroids, dim=-1)
        updated_assignments = torch.argmax(normalized @ directions.T, dim=1)
        if torch.equal(assignments, updated_assignments):
            break
        assignments = updated_assignments
        updated = []
        for cluster in range(count):
            members = values[assignments == cluster]
            updated.append(
                centroids[cluster] if not len(members) else members.mean(0)
            )
        centroids = torch.stack(updated)

    cluster_centroids = []
    supports = []
    for cluster in range(count):
        members = values[assignments == cluster]
        if not len(members):
            continue
        centroid = members.mean(0)
        # Averaging shrinks norms; rescale so lambda keeps its meaning.
        typical_norm = members.norm(dim=-1).mean()
        centroid = torch.nn.functional.normalize(centroid, dim=0) * typical_norm
        cluster_centroids.append(centroid)
        supports.append(len(members))
    return (
        torch.stack(cluster_centroids).to(tokens.dtype),
        torch.tensor(supports, device=tokens.device, dtype=torch.long),
    )


def centroid_tokens(
    tokens: torch.Tensor, count: int
) -> tuple[torch.Tensor, list[int]]:
    centroids, supports = deterministic_kmeans(tokens, count)
    return centroids[None], [int(value) for value in supports.tolist()]


def robust_centroid_tokens(
    tokens: torch.Tensor,
    count: int,
    candidate_count: int,
    min_fraction: float,
) -> tuple[torch.Tensor, list[int]]:
    centroids, supports = deterministic_kmeans(tokens, candidate_count)
    minimum = max(2, int(math.ceil(tokens.shape[1] * min_fraction)))
    eligible = [
        index for index, support in enumerate(supports.tolist())
        if int(support) >= minimum
    ]
    if not eligible:
        eligible = [int(torch.argmax(supports).item())]
    eligible.sort(key=lambda index: (-int(supports[index]), index))
    eligible = eligible[:count]
    indices = torch.tensor(eligible, device=centroids.device)
    kept_supports = [int(supports[index]) for index in eligible]
    return centroids.index_select(0, indices)[None], kept_supports


def stable_seed(case_id: str, reference_index: int, seed: int) -> int:
    digest = hashlib.sha256(
        f"{case_id}:{reference_index}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def full_partition_bias(
    text_len: int, reference_sizes: list[int], ownership: np.ndarray
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    bias, routes = run.regional_bias(text_len, reference_sizes, ownership)
    condition_len = text_len + sum(reference_sizes)
    image_start = condition_len
    image_end = condition_len + ownership.shape[1]
    partitions = [layer > 0.5 for layer in ownership]
    partitions.append(np.stack(partitions).sum(0) == 0)
    for partition in partitions:
        query_ids = np.flatnonzero(partition)
        key_ids = np.flatnonzero(partition)
        if not len(query_ids):
            continue
        rows = torch.from_numpy(query_ids).to(run.TEXT_DEV) + image_start
        cols = torch.from_numpy(key_ids).to(run.TEXT_DEV) + image_start
        bias[0, 0, rows[:, None], image_start:image_end] = float("-inf")
        bias[0, 0, rows[:, None], cols[None, :]] = 0
    return bias, routes


def soft_regional_bias(
    text_len: int,
    reference_sizes: list[int],
    ownership: np.ndarray,
    floor: float = 1e-3,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    condition_len = text_len + sum(reference_sizes)
    total_len = condition_len + 64 * 64
    bias = torch.zeros(
        (1, 1, total_len, total_len), device=run.TEXT_DEV, dtype=run.DTYPE
    )
    bias[..., text_len:condition_len] = float("-inf")
    offset = text_len
    routes = []
    for weights, size in zip(ownership, reference_sizes):
        row_ids = np.flatnonzero(weights >= floor)
        rows = torch.from_numpy(row_ids).to(run.TEXT_DEV) + condition_len
        cols = torch.arange(offset, offset + size, device=run.TEXT_DEV)
        penalties = torch.from_numpy(weights[row_ids]).to(
            run.TEXT_DEV, dtype=run.DTYPE
        ).clamp_min(floor).log()
        bias[0, 0, rows[:, None], cols[None, :]] = penalties[:, None]
        routes.append((len(row_ids), size))
        offset += size
    return bias, routes


def one_wrong_reference_bias(
    text_len: int,
    reference_sizes: list[int],
    ownership: np.ndarray,
) -> tuple[
    torch.Tensor,
    list[tuple[int, int]],
    list[dict[str, int]],
]:
    if len(reference_sizes) < 2:
        raise ValueError("one_wrong_reference requires at least two references")
    if len(reference_sizes) != ownership.shape[0]:
        raise ValueError("reference blocks and target-region groups must align")
    bias, routes = run.regional_bias(text_len, reference_sizes, ownership)
    condition_len = text_len + sum(reference_sizes)
    offsets = []
    offset = text_len
    for size in reference_sizes:
        offsets.append(offset)
        offset += size

    wrong_routes: list[dict[str, int]] = []
    for region_index, mask in enumerate(ownership):
        wrong_index = (region_index + 1) % len(reference_sizes)
        row_ids = np.flatnonzero(mask > 0.5)
        rows = torch.from_numpy(row_ids).to(run.TEXT_DEV) + condition_len
        start = offsets[wrong_index]
        size = reference_sizes[wrong_index]
        cols = torch.arange(start, start + size, device=run.TEXT_DEV)
        bias[0, 0, rows[:, None], cols[None, :]] = 0
        wrong_routes.append({
            "query_region": region_index,
            "wrong_reference": wrong_index,
            "query_tokens": len(row_ids),
            "reference_tokens": size,
        })
    return bias, routes, wrong_routes


class AttentionMaskGate:
    def __init__(self, processor: object, enabled: bool) -> None:
        self.processor = processor
        self.enabled = enabled

    def __call__(self, *args: object, **kwargs: object) -> object:
        if not self.enabled:
            # diffusers passes the mask positionally or as a keyword depending on version.
            if len(args) >= 4:
                mutable_args = list(args)
                mutable_args[3] = None
                args = tuple(mutable_args)
            kwargs.pop("attention_mask", None)
        return self.processor(*args, **kwargs)


def processor_location(name: str) -> tuple[str, int] | None:
    match = re.search(r"(^|\.)single_transformer_blocks\.(\d+)\.", name)
    if match:
        return "single", int(match.group(2))
    match = re.search(r"(^|\.)transformer_blocks\.(\d+)\.", name)
    if match:
        return "double", int(match.group(2))
    return None


def configure_mask_scope(pipeline: object, scope: str) -> dict[str, bool]:
    processors = pipeline.transformer.attn_processors
    if scope == "all":
        return {name: True for name in processors}
    locations = {name: processor_location(name) for name in processors}
    maxima: dict[str, int] = {}
    for location in locations.values():
        if location is not None:
            stream, index = location
            maxima[stream] = max(maxima.get(stream, -1), index)
    enabled: dict[str, bool] = {}
    wrapped = {}
    for name, processor in processors.items():
        location = locations[name]
        active = True
        if location is not None and scope != "all":
            stream, index = location
            count = maxima[stream] + 1
            lower = count // 3
            upper = math.ceil(2 * count / 3)
            in_middle = lower <= index < upper
            active = in_middle if scope == "middle" else not in_middle
        enabled[name] = active
        wrapped[name] = AttentionMaskGate(processor, active)
    pipeline.transformer.set_attn_processor(wrapped)
    return enabled


def read_plan(path: Path, ids: list[str] | None) -> list[dict[str, object]]:
    cases = json.loads(path.read_text())["cases"]
    if ids:
        requested = set(ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    return cases


def variant_name(args: argparse.Namespace) -> str:
    if args.output_name:
        return args.output_name
    name = args.variant
    if (
        args.control_type != "depth"
        and not name.endswith(f"_{args.control_type}")
    ):
        name += f"_{args.control_type}"
    if args.budget is not None:
        name += f"_b{args.budget}"
    if args.lam is not None:
        name += f"_l{args.lam:g}"
    if args.threshold is not None:
        name += f"_t{args.threshold:g}"
    if args.allocation != "equal":
        name += f"_{args.allocation}"
    if args.mask_scope != "all" and args.variant != "global_redux":
        name += f"_{args.mask_scope}"
    if args.prompt_key_override is not None:
        name += f"_prompt_{args.prompt_key_override}"
    if args.prompt_mode != "neutral":
        name += f"_{args.prompt_mode}"
    if args.reference_permutation != "none":
        name += f"_perm_{args.reference_permutation}"
    if args.seed_override is not None:
        name += f"_seed_{args.seed_override}"
    if args.steps is not None:
        name += f"_s{args.steps}"
    if args.guidance is not None:
        name += f"_g{args.guidance:g}"
    return name


def rotate(values: list[object], direction: str) -> list[object]:
    if direction == "left":
        return values[1:] + values[:1]
    if direction == "right":
        return values[-1:] + values[:-1]
    return values


def mismatched_depths(cases: list[dict[str, object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for case in cases:
        candidates = [other for other in cases
                      if other["domain"] == case["domain"]
                      and other["source"] != case["source"]]
        if not candidates:
            raise ValueError(f"no mismatched depth candidate for {case['id']}")
        mapping[str(case["id"])] = str(candidates[0]["depth_control"])
    return mapping


def write_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    cases = read_plan(plan_path, args.ids)
    jobs = [(case, count) for case in cases for count in args.n_values]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    label = variant_name(args)
    outputs_base = Path(args.outputs).resolve()
    output_root = outputs_base / label
    control_cache = (
        Path(args.control_cache).resolve()
        if args.control_cache
        else outputs_base / "control_inputs"
    )
    selected_control_model = control_model_id(
        args.control_type, args.control_model
    )
    if args.validate_only:
        required: list[Path] = []
        for case, _ in jobs:
            structural = (
                str(case["depth_control"])
                if args.control_type == "depth"
                else str(case["source"])
            )
            required.append(ROOT / structural)
            for part in case["parts"]:
                required.extend(
                    ROOT / str(part[key])
                    for key in ("target_mask", "donor", "donor_mask")
                )
        missing = sorted({str(path) for path in required if not path.is_file()})
        if missing:
            raise FileNotFoundError("missing inputs: " + ", ".join(missing))
        print(json.dumps({
            "variant": args.variant,
            "label": label,
            "plan": str(plan_path.resolve()),
            "outputs": str(output_root),
            "n_values": list(args.n_values),
            "jobs": len(jobs),
            **control_run_metadata(
                control_type=args.control_type,
                model_id=selected_control_model,
                cache_root=control_cache,
                low_threshold=args.canny_low_threshold,
                high_threshold=args.canny_high_threshold,
                resolution=args.control_resolution,
            ),
        }, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    descriptions = json.loads(
        Path(args.appearance_descriptions).read_text(encoding="utf-8"))
    metadata_path = output_root / "metrics.jsonl"
    run_info_path = output_root / "run.json"
    if args.force and metadata_path.exists():
        metadata_path.unlink()

    torch.set_grad_enabled(False)
    load_started = time.perf_counter()
    original_control_model = run.DEPTH
    run.DEPTH = selected_control_model
    try:
        pipeline, redux = run.load_models()
    finally:
        run.DEPTH = original_control_model
    pipeline.set_progress_bar_config(disable=True)
    scope_map = configure_mask_scope(pipeline, args.mask_scope)
    model_load_seconds = time.perf_counter() - load_started
    run_info = {
        "variant": args.variant,
        "label": label,
        "plan": str(plan_path),
        "n_values": args.n_values,
        **control_run_metadata(
            control_type=args.control_type,
            model_id=selected_control_model,
            cache_root=control_cache,
            low_threshold=args.canny_low_threshold,
            high_threshold=args.canny_high_threshold,
            resolution=args.control_resolution,
        ),
        "budget_override": args.budget,
        "lambda_override": args.lam,
        "threshold_override": args.threshold,
        "instance_condition": args.instance_condition,
        "reference_query_scope": args.reference_query_scope,
        "block_image_to_text": args.block_image_to_text,
        "block_text_scope": args.block_text_scope,
        "pair_output_names": args.pair_output_names,
        "allocation": args.allocation,
        "mask_scope": args.mask_scope,
        "token_reduction": args.token_reduction,
        "token_clusters": args.token_clusters,
        "robust_candidate_clusters": args.robust_candidate_clusters,
        "robust_min_fraction": args.robust_min_fraction,
        "soft_mask_radius": args.soft_mask_radius,
        "prompt_key_override": args.prompt_key_override,
        "prompt_mode": args.prompt_mode,
        "reference_permutation": args.reference_permutation,
        "wrong_reference_policy": (
            "cyclic_next_while_retaining_correct"
            if args.variant == "one_wrong_reference"
            else None
        ),
        "seed_override": args.seed_override,
        "steps_override": args.steps,
        "guidance_override": args.guidance,
        "mask_enabled_processors": sum(scope_map.values()),
        "mask_total_processors": len(scope_map),
        "model_load_seconds": model_load_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    run_info_path.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    token_cache: dict[tuple[str, str, str, float], torch.Tensor] = {}
    mismatch = (
        mismatched_depths(read_plan(plan_path, None))
        if args.variant == "mismatched_depth" else {})
    experiment_started = time.perf_counter()

    for job_index, (case, count) in enumerate(jobs, 1):
        prompt_key = args.prompt_key_override or str(case["prompt_key"])
        if args.pair_output_names:
            source_stem = Path(str(case["source"])).stem
            donor_stems = ",".join(
                Path(str(part["donor"])).stem for part in case["parts"]
            )
            output_id = f"{source_stem}_{donor_stems}"
        else:
            output_id = str(case.get("output_id", case["id"]))
        if Path(output_id).name != output_id:
            raise ValueError(f"invalid output_id: {output_id!r}")
        output = (output_root / str(case["domain"]) / str(count) /
                  prompt_key / f"{output_id}.png")
        if output.exists() and (args.skip_existing or not args.force):
            print(f"[variants] {job_index}/{len(jobs)} skip {output}", flush=True)
            continue
        selected_indices = [int(index) for index in case["order"][:count]]
        selected_parts = [case["parts"][index] for index in selected_indices]
        if count > 1:
            selected_parts = rotate(selected_parts, args.reference_permutation)
            selected_indices = rotate(selected_indices, args.reference_permutation)
        prep_started = time.perf_counter()
        threshold = float(
            case["threshold"] if args.threshold is None else args.threshold)
        references = []
        cache_hits = 0
        skip_references = (args.variant == "depth_only"
                           or args.instance_condition == "text")
        for part in ([] if skip_references else selected_parts):
            mode = "all" if args.variant == "all_reference_tokens" else "foreground"
            key = (mode, str(part["donor"]), str(part["donor_mask"]),
                   threshold)
            if not args.no_cache and key in token_cache:
                tokens = token_cache[key].to(run.TEXT_DEV)
                cache_hits += 1
            else:
                if mode == "all":
                    tokens = run.redux_tokens(redux, str(part["donor"]))
                else:
                    tokens = run.foreground_tokens(
                        redux, str(part["donor"]), str(part["donor_mask"]),
                        threshold)
                if not args.no_cache:
                    token_cache[key] = tokens.cpu()
            references.append(tokens)

        if args.instance_condition == "text":
            descriptions = [
                str(part.get("instance_prompt")
                    or f"{part['name'].replace('_', ' ')}")
                for part in selected_parts
            ]
            references = instance_text_tokens(
                pipeline, descriptions, args.instance_text_tokens, run.TEXT_DEV)
        capacities = [int(tokens.shape[1]) for tokens in references]
        budget = int(
            case.get(
                "reference_token_budget", run.DEFAULT_REFERENCE_TOKEN_BUDGET
            )
            if args.budget is None
            else args.budget
        )
        requested_allocation = (
            run.allocate_equal_budget(capacities, budget)
            if args.allocation == "equal"
            else proportional_budget(capacities, budget)
        )
        selected = []
        cluster_supports: list[list[int]] = []
        for reference_index, (tokens, amount) in enumerate(
                zip(references, requested_allocation)):
            if args.token_reduction == "centroid":
                chosen, supports = centroid_tokens(
                    tokens, min(args.token_clusters, amount)
                )
            elif args.token_reduction == "robust":
                chosen, supports = robust_centroid_tokens(
                    tokens,
                    min(args.token_clusters, amount),
                    min(args.robust_candidate_clusters, amount),
                    args.robust_min_fraction,
                )
            elif args.variant == "random_tokens":
                chosen = random_tokens(
                    tokens, amount,
                    stable_seed(str(case["id"]), reference_index,
                                int(case["seed"])))
                supports = []
            elif args.variant == "first_tokens":
                chosen = tokens[:, :amount]
                supports = []
            else:
                chosen = run.farthest_point_tokens(tokens, amount)
                supports = []
            selected.append(chosen)
            cluster_supports.append(supports)
        allocation = [int(tokens.shape[1]) for tokens in selected]

        prompt = str(case["prompt"])
        if args.prompt_key_override is not None:
            prompt = base_prompt(
                str(case["domain"]), prompt_key, str(case["source"])
            )
        if args.prompt_mode == "aligned":
            prompt = appearance_prompt(
                prompt,
                appearance_clause(selected_parts, selected_parts, descriptions),
            )
        elif args.prompt_mode == "conflicting":
            prompt = appearance_prompt(
                prompt,
                appearance_clause(
                    selected_parts, rotate(selected_parts, "left"), descriptions
                ),
            )
        if not args.no_cache and prompt in text_cache:
            text, pooled = (tensor.to(run.TEXT_DEV)
                            for tensor in text_cache[prompt])
            text_cache_hit = True
        else:
            text, pooled, _ = pipeline.encode_prompt(
                prompt=prompt, prompt_2=prompt, device=run.TEXT_DEV)
            text_cache_hit = False
            if not args.no_cache:
                text_cache[prompt] = (text.cpu(), pooled.cpu())
        # Depth-only has no references, so no lambda is recorded.
        lam = None if args.variant == "depth_only" else float(
            case["lam"] if args.lam is None else args.lam
        )
        scaled_references = (
            [] if lam is None
            else [lam * tokens.to(text.dtype) for tokens in selected]
        )
        condition = torch.cat([text, *scaled_references], dim=1)
        if args.soft_mask_radius > 0:
            ownership, groups = soft_grouped_ownership(
                case["parts"],
                [int(index) for index in case["order"]],
                count,
                args.soft_mask_radius,
            )
        else:
            ownership, groups = grouped_ownership(
                case["parts"], [int(index) for index in case["order"]], count
            )
        wrong_routes: list[dict[str, int]] = []
        if args.variant in {"global_redux", "depth_only"}:
            attention = None
            routes = (
                []
                if args.variant == "depth_only"
                else [
                    (int(mask.sum()), size)
                    for mask, size in zip(ownership, allocation)
                ]
            )
        elif args.variant == "full_image_block":
            attention, routes = full_partition_bias(
                int(text.shape[1]), allocation, ownership)
        elif args.soft_mask_radius > 0:
            attention, routes = soft_regional_bias(
                int(text.shape[1]), allocation, ownership
            )
        elif args.variant == "one_wrong_reference":
            attention, routes, wrong_routes = one_wrong_reference_bias(
                int(text.shape[1]), allocation, ownership
            )
        else:
            attention, routes = run.regional_bias(
                int(text.shape[1]), allocation, ownership)
        if args.reference_query_scope == "own_region":
            # Condition rows see only their own region, as in DreamRenderer.
            if attention is None:
                raise ValueError(
                    "--reference-query-scope applies only to variants that "
                    "carry the routing bias")
            rq_text_len = int(text.shape[1])
            rq_condition_len = int(condition.shape[1])
            offset = rq_text_len
            for layer, size in zip(ownership, allocation):
                rows = torch.arange(offset, offset + size, device=attention.device)
                owned = np.flatnonzero(layer > 0.5)
                attention[0, 0, rows, rq_condition_len:] = float("-inf")
                cols = (torch.from_numpy(owned).to(attention.device)
                        + rq_condition_len)
                attention[0, 0, rows[:, None], cols[None, :]] = 0
                offset += size
        if args.block_image_to_text:
            if attention is None:
                raise ValueError(
                    "--block-image-to-text applies only to variants that "
                    "carry the routing bias")
            block_text_len = int(text.shape[1])
            block_condition_len = int(condition.shape[1])
            if args.block_text_scope == "all":
                attention[..., block_condition_len:, :block_text_len] = (
                    float("-inf"))
            else:
                owned_rows = np.flatnonzero(ownership.sum(0) > 0.5)
                rows = (torch.from_numpy(owned_rows).to(attention.device)
                        + block_condition_len)
                attention[0, 0, rows, :block_text_len] = float("-inf")
        control_case = case
        if args.variant == "mismatched_depth":
            control_case = dict(case)
            control_case["depth_control"] = mismatch[str(case["id"])]
        control_image, control_info = load_control_image(
            control_case,
            control_type=args.control_type,
            root=ROOT,
            cache_root=control_cache,
            low_threshold=args.canny_low_threshold,
            high_threshold=args.canny_high_threshold,
            resolution=args.control_resolution,
        )
        preparation_seconds = time.perf_counter() - prep_started

        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        synthesis_started = time.perf_counter()
        pipeline_args = {
            "prompt_embeds": condition,
            "pooled_prompt_embeds": pooled,
            "control_image": control_image,
            "image": Image.new("RGB", (1024, 1024), "white"),
            "strength": 1.0,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": int(
                case["steps"] if args.steps is None else args.steps
            ),
            "guidance_scale": float(
                case["guidance"] if args.guidance is None else args.guidance
            ),
            "generator": torch.Generator(run.TEXT_DEV).manual_seed(
                int(case["seed"] if args.seed_override is None
                    else args.seed_override)),
        }
        if attention is not None:
            pipeline_args["joint_attention_kwargs"] = {
                "attention_mask": attention}
        image = pipeline(**pipeline_args).images[0]
        torch.cuda.synchronize()
        synthesis_seconds = time.perf_counter() - synthesis_started
        peak_memory = torch.cuda.max_memory_allocated()
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)

        record = {
            "case_id": case["id"],
            "output_id": output_id,
            "domain": case["domain"],
            "prompt_key": prompt_key,
            "prompt": prompt,
            "prompt_mode": args.prompt_mode,
            "reference_permutation": args.reference_permutation,
            "selected_reference_part_indices": selected_indices,
            "reference_count": count,
            "variant": args.variant,
            "output": str(output.relative_to(ROOT)),
            "groups": groups,
            "routes": routes,
            "wrong_reference_routes": wrong_routes,
            "reference_capacities": capacities,
            "requested_reference_allocation": requested_allocation,
            "reference_allocation": allocation,
            "reference_cluster_supports": cluster_supports,
            "token_reduction": args.token_reduction,
            "soft_mask_radius": args.soft_mask_radius,
            "instance_condition": args.instance_condition,
        "reference_query_scope": args.reference_query_scope,
            "block_image_to_text": args.block_image_to_text,
            "block_text_scope": args.block_text_scope,
            "reference_token_budget": budget,
            "condition_tokens": int(condition.shape[1]),
            "lambda": lam,
            "threshold": threshold,
            **control_info,
            "control_model": selected_control_model,
            "depth_control": (
                control_info["control_image"]
                if args.control_type == "depth"
                else None
            ),
            "reference_cache_hits": cache_hits,
            "text_cache_hit": text_cache_hit,
            "preparation_seconds": preparation_seconds,
            "synthesis_seconds": synthesis_seconds,
            "memory_before_gib": memory_before / 2**30,
            "peak_memory_gib": peak_memory / 2**30,
            "peak_increment_gib": (peak_memory - memory_before) / 2**30,
            "steps": int(case["steps"] if args.steps is None else args.steps),
            "guidance": float(
                case["guidance"] if args.guidance is None else args.guidance
            ),
            "seed": int(case["seed"] if args.seed_override is None
                        else args.seed_override),
        }
        write_jsonl(metadata_path, record)
        elapsed = time.perf_counter() - experiment_started
        print(
            f"[variants:{args.control_type}] {job_index}/{len(jobs)} {label} K={count} "
            f"{case['id']} refs={capacities}->{allocation} "
            f"prep={preparation_seconds:.2f}s synth={synthesis_seconds:.2f}s "
            f"peak={peak_memory / 2**30:.2f}GiB elapsed={elapsed:.1f}s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--appearance-descriptions",
                        default=str(DEFAULT_APPEARANCE_DESCRIPTIONS))
    parser.add_argument("--output-name")
    parser.add_argument("--n-values", nargs="+", type=int, default=(3,),
                        help="reference counts to run; the benchmark uses 1-3, "
                             "multi-object layouts may declare more")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--ids-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--budget", type=int,
        help="optional total reference-token cap; 0 disables the cap")
    parser.add_argument("--lam", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument(
        "--control-type", choices=tuple(CONTROL_MODEL_IDS), default="depth"
    )
    parser.add_argument(
        "--control-model",
        help="checkpoint id or local path; defaults to the official model for the control type",
    )
    parser.add_argument(
        "--control-cache",
        help="shared directory for deterministic Canny inputs (default: OUTPUTS/control_inputs)",
    )
    parser.add_argument("--control-resolution", type=int, default=1024)
    parser.add_argument("--canny-low-threshold", type=int, default=50)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--allocation", choices=("equal", "proportional"),
                        default="equal")
    parser.add_argument("--mask-scope", choices=("all", "middle", "early_late"),
                        default="all")
    parser.add_argument(
        "--token-reduction",
        choices=("fps", "centroid", "robust"),
        default="fps",
    )
    parser.add_argument("--token-clusters", type=int, default=16)
    parser.add_argument("--robust-candidate-clusters", type=int, default=32)
    parser.add_argument("--robust-min-fraction", type=float, default=0.03)
    parser.add_argument("--soft-mask-radius", type=float, default=0.0)
    parser.add_argument("--prompt-key-override", choices=tuple(PROMPT_SETTINGS))
    parser.add_argument("--prompt-mode",
                        choices=("neutral", "aligned", "conflicting"),
                        default="neutral")
    parser.add_argument("--reference-permutation",
                        choices=("none", "left", "right"), default="none")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument(
        "--instance-condition", choices=("image", "text"), default="image",
        help="drive each region with reference-image tokens or its own text")
    parser.add_argument("--instance-text-tokens", type=int, default=64)
    parser.add_argument(
        "--reference-query-scope", choices=("global", "own_region"),
        default="global",
        help="reference token rows attend to the whole image or only their region")
    parser.add_argument(
        "--block-image-to-text", action="store_true",
        help="block image queries from attending to text keys in joint attention",
    )
    parser.add_argument(
        "--block-text-scope", choices=("all", "regions"), default="all",
        help="scope of the block: all=every image row, regions=target-region "
             "rows only, leaving the background with text access",
    )
    parser.add_argument(
        "--pair-output-names", action="store_true",
        help="save as {source}_{reference0},{reference1},{reference2}.png",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip cases whose output image already exists",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()
    if args.skip_existing and args.force:
        parser.error("--skip-existing and --force cannot be used together")
    if args.token_clusters <= 0 or args.robust_candidate_clusters <= 0:
        parser.error("token cluster counts must be positive")
    if not 0.0 <= args.robust_min_fraction < 1.0:
        parser.error("--robust-min-fraction must be in [0, 1)")
    if args.soft_mask_radius < 0:
        parser.error("--soft-mask-radius must be non-negative")
    if args.budget is not None and args.budget < 0:
        parser.error("--budget must be non-negative (0 disables the cap)")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.guidance is not None and args.guidance < 0:
        parser.error("--guidance must be non-negative")
    if args.control_resolution <= 0:
        parser.error("--control-resolution must be positive")
    if not (
        0 <= args.canny_low_threshold < args.canny_high_threshold <= 255
    ):
        parser.error("Canny thresholds require 0 <= low < high <= 255")
    if args.control_type == "canny" and args.variant == "mismatched_depth":
        parser.error("mismatched_depth is defined only for depth control")
    if args.variant == "one_wrong_reference" and 1 in args.n_values:
        parser.error("one_wrong_reference requires K >= 2")
    if (
        args.variant == "one_wrong_reference"
        and args.reference_permutation != "none"
    ):
        parser.error(
            "one_wrong_reference cannot be combined with reference permutation"
        )
    if args.soft_mask_radius > 0 and args.variant != "refroute":
        parser.error("soft target-mask blending is supported only for refroute")
    if args.token_reduction != "fps" and args.variant in {
        "random_tokens", "first_tokens"
    }:
        parser.error("token reduction cannot be combined with token-selection variants")
    if args.ids_file:
        file_ids = [line.strip() for line in Path(args.ids_file).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        args.ids = list(dict.fromkeys([*(args.ids or []), *file_ids]))
    return args


if __name__ == "__main__":
    os.chdir(ROOT)
    generate(parse_args())
