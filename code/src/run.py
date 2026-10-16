from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpus", default=None)
_gpus = _pre.parse_known_args()[0].gpus
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

import numpy as np
import torch
import yaml
from accelerate import dispatch_model, infer_auto_device_map
from diffusers import FluxControlImg2ImgPipeline, FluxPriorReduxPipeline
from diffusers import FluxTransformer2DModel
from PIL import Image


FLUX = "black-forest-labs/FLUX.1-dev"
DEPTH = "black-forest-labs/FLUX.1-Depth-dev"
CANNY = "black-forest-labs/FLUX.1-Canny-dev"
REDUX = "black-forest-labs/FLUX.1-Redux-dev"
CONTROL_MODELS = {"depth": DEPTH, "canny": CANNY}
DEFAULT_GUIDANCE = {"depth": 10.0, "canny": 30.0}
IMAGE_SIZE = 1024
DTYPE = torch.bfloat16
DEFAULT_LAMBDA = 3.0
DEFAULT_THRESHOLD = 0.3
DEFAULT_REFERENCE_TOKEN_BUDGET = 0  # 0 means no cap.
DEFAULT_STEPS = 20
DEFAULT_SEED = 0
DEFAULT_CANNY_LOW_THRESHOLD = 50
DEFAULT_CANNY_HIGH_THRESHOLD = 200
GPU_COUNT = max(1, torch.cuda.device_count())
SHARD_GPUS = [0] if GPU_COUNT == 1 else [0, 1]
TEXT_DEV = f"cuda:{min(2, GPU_COUNT - 1)}"
PREP_DEV = f"cuda:{GPU_COUNT - 1}"


def load_config(path):
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"config file not found: {path}")
    with config_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    config = {str(key).replace("-", "_"): value
              for key, value in loaded.items()}
    allowed = {
        "source", "depth_control", "parts", "prompt", "lam", "threshold",
        "reference_token_budget", "steps", "guidance", "seed",
        "control_type", "canny_control", "canny_guidance",
        "canny_low_threshold", "canny_high_threshold",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    required = {"source", "parts", "prompt"}
    if not required <= set(config):
        raise ValueError(f"config requires {sorted(required)}")
    control_type = str(config.get("control_type", "depth")).lower()
    if control_type not in CONTROL_MODELS:
        raise ValueError(
            f"control_type must be one of {sorted(CONTROL_MODELS)}")
    config["control_type"] = control_type
    if not isinstance(config["parts"], list) or not config["parts"]:
        raise ValueError("config parts must be a non-empty list")
    names = set()
    for index, part in enumerate(config["parts"]):
        if not isinstance(part, dict):
            raise ValueError(f"parts[{index}] must be a mapping")
        if set(part) != {"name", "target_mask", "donor", "donor_mask"}:
            raise ValueError(
                f"parts[{index}] requires name, target_mask, donor, donor_mask")
        if part["name"] in names:
            raise ValueError("part names must be unique")
        names.add(part["name"])
    return config


def shard_budget():
    budget = {}
    for index in SHARD_GPUS:
        free = torch.cuda.mem_get_info(index)[0] / 2**30
        reserve = 12.0 if f"cuda:{index}" in (TEXT_DEV, PREP_DEV) else 7.0
        budget[index] = f"{max(2, int(free - reserve))}GiB"
    return budget


def load_redux():
    redux = FluxPriorReduxPipeline.from_pretrained(
        REDUX, torch_dtype=DTYPE)
    redux.image_encoder.to(TEXT_DEV, DTYPE)
    redux.image_embedder.to(TEXT_DEV, DTYPE)
    return redux


def load_models(control_type="depth"):
    if control_type not in CONTROL_MODELS:
        raise ValueError(f"unsupported control type: {control_type}")
    control_model = DEPTH if control_type == "depth" else CANNY
    transformer = FluxTransformer2DModel.from_pretrained(
        control_model, subfolder="transformer", torch_dtype=DTYPE)
    device_map = infer_auto_device_map(
        transformer, max_memory=shard_budget(),
        no_split_module_classes=[
            "FluxTransformerBlock", "FluxSingleTransformerBlock"])
    transformer = dispatch_model(transformer, device_map=device_map)
    forward_device = next(transformer.x_embedder.parameters()).device
    original_forward = transformer.forward

    def sharded_forward(**kwargs):
        caller_device = kwargs["hidden_states"].device
        kwargs = {key: value.to(forward_device) if torch.is_tensor(value)
                  else value for key, value in kwargs.items()}
        output = original_forward(**kwargs)
        if isinstance(output, tuple):
            return tuple(value.to(caller_device) if torch.is_tensor(value)
                         else value for value in output)
        return output.to(caller_device)

    transformer.forward = sharded_forward

    class Pipeline(FluxControlImg2ImgPipeline):
        @property
        def _execution_device(self):
            return torch.device(TEXT_DEV)

    pipeline = Pipeline.from_pretrained(
        FLUX, transformer=transformer, torch_dtype=DTYPE)
    pipeline.text_encoder.to(TEXT_DEV)
    pipeline.text_encoder_2.to(TEXT_DEV)
    pipeline.vae.to(TEXT_DEV)
    return pipeline, load_redux()


def make_canny_control(
    image,
    low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
    high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
):
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("Canny thresholds require 0 <= low < high <= 255")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for Canny preprocessing") from error
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    rgb = np.asarray(resized, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray, threshold1=low_threshold, threshold2=high_threshold,
        apertureSize=3, L2gradient=False)
    return Image.fromarray(np.repeat(edges[..., None], 3, axis=2), mode="RGB")


def load_control_image(config, control_type, override_path=None,
                       low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
                       high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD):
    configured_path = (
        config.get("depth_control") if control_type == "depth"
        else config.get("canny_control")
    )
    path = override_path or configured_path
    if path is not None:
        image = Image.open(path).convert("RGB")
        if control_type == "canny" and image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return image, str(path)
    if control_type == "depth":
        raise ValueError(
            "depth control requires depth_control in the config or "
            "--control-image")
    source = config["source"]
    return (
        make_canny_control(
            Image.open(source), low_threshold=low_threshold,
            high_threshold=high_threshold),
        str(source),
    )


def effective_guidance(config, control_type, override=None):
    if override is not None:
        return float(override)
    specific = config.get(f"{control_type}_guidance")
    if specific is not None:
        return float(specific)
    if config.get("control_type", "depth") == control_type:
        return float(config.get("guidance", DEFAULT_GUIDANCE[control_type]))
    return DEFAULT_GUIDANCE[control_type]


def redux_tokens(redux, image_path):
    image = Image.open(image_path).convert("RGB")
    pixels = redux.feature_extractor(
        images=image, return_tensors="pt").pixel_values.to(TEXT_DEV, DTYPE)
    with torch.inference_mode():
        encoded = redux.image_encoder(pixels).last_hidden_state
        return redux.image_embedder(encoded).image_embeds


def foreground_tokens(redux, donor, mask_path, threshold):
    tokens = redux_tokens(redux, donor)
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise ValueError("Redux token sequence is not a square spatial grid")
    mask = np.asarray(Image.open(mask_path).convert("L").resize(
        (side, side), Image.Resampling.BILINEAR), np.float32) / 255.0
    indices = np.flatnonzero(mask.reshape(-1) >= threshold)
    if len(indices) == 0:
        raise ValueError(f"donor mask selects no tokens: {mask_path}")
    keep = torch.from_numpy(indices).long().to(tokens.device)
    return tokens.index_select(1, keep)


def target_ownership(parts):
    layers = []
    for part in parts:
        mask = Image.open(part["target_mask"]).convert("L").resize(
            (64, 64), Image.Resampling.NEAREST)
        layer = (np.asarray(mask, np.uint8).reshape(-1) >= 128).astype(np.float32)
        if not layer.any():
            raise ValueError(f"target mask is empty: {part['target_mask']}")
        layers.append(layer)
    ownership = np.stack(layers)
    if np.any(ownership.sum(0) > 1):
        raise ValueError("target part masks overlap at the latent grid")
    return ownership


def allocate_equal_budget(capacities, budget):
    capacities = np.asarray(capacities, dtype=np.int64)
    total = int(capacities.sum())
    if budget < 0:
        raise ValueError("reference token budget must be non-negative")
    if budget == 0 or budget >= total:
        return capacities.tolist()
    active = np.flatnonzero(capacities > 0)
    if budget < len(active):
        raise ValueError("reference token budget is smaller than donor count")
    allocation = np.zeros_like(capacities)
    remaining = budget
    while remaining:
        for index in active:
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                remaining -= 1
                if remaining == 0:
                    break
    return allocation.tolist()


def farthest_point_tokens(tokens, count):
    if count <= 0:
        raise ValueError("token count must be positive")
    if count >= tokens.shape[1]:
        return tokens
    values = torch.nn.functional.normalize(tokens[0].float(), dim=-1)
    centre = torch.nn.functional.normalize(values.mean(0), dim=0)
    index = int(torch.argmax(values @ centre).item())
    selected = [index]
    chosen = torch.zeros(values.shape[0], device=values.device, dtype=torch.bool)
    chosen[index] = True
    distance = 1.0 - values @ values[index]
    for _ in range(1, count):
        distance[chosen] = -1
        index = int(torch.argmax(distance).item())
        selected.append(index)
        chosen[index] = True
        distance = torch.minimum(distance, 1.0 - values @ values[index])
    indices = torch.tensor(sorted(selected), device=tokens.device)
    return tokens.index_select(1, indices)


def regional_bias(text_len, reference_sizes, ownership):
    condition_len = text_len + sum(reference_sizes)
    total_len = condition_len + 64 * 64
    bias = torch.zeros(
        (1, 1, total_len, total_len), device=TEXT_DEV, dtype=DTYPE)
    bias[..., text_len:condition_len] = float("-inf")
    offset = text_len
    routes = []
    for mask, size in zip(ownership, reference_sizes):
        row_ids = np.flatnonzero(mask > 0.5)
        rows = torch.from_numpy(row_ids).to(TEXT_DEV) + condition_len
        cols = torch.arange(offset, offset + size, device=TEXT_DEV)
        bias[0, 0, rows[:, None], cols[None, :]] = 0
        routes.append((len(row_ids), size))
        offset += size
    return bias, routes


def self_check():
    assert DEFAULT_LAMBDA == 3.0
    assert DEFAULT_THRESHOLD == 0.3
    assert DEFAULT_REFERENCE_TOKEN_BUDGET == 0
    assert DEFAULT_STEPS == 20
    assert DEFAULT_SEED == 0
    assert allocate_equal_budget([2, 20, 20], 0) == [2, 20, 20]
    assert allocate_equal_budget([20, 20, 20], 10) == [4, 3, 3]
    assert allocate_equal_budget([2, 20, 20], 10) == [2, 4, 4]
    tokens = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]])
    assert farthest_point_tokens(tokens, 3).shape == (1, 3, 2)
    assert effective_guidance(
        {"control_type": "depth", "guidance": 10}, "canny") == 30.0
    assert effective_guidance({}, "depth") == 10.0
    assert effective_guidance({}, "canny") == 30.0
    assert effective_guidance(
        {"control_type": "canny", "guidance": 25}, "canny") == 25.0
    print("[run] self-check OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="single-case configuration; the reported runs are driven from a "
             "plan file through experiments/generate_variants.py instead")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--control-type", choices=tuple(CONTROL_MODELS), default=None,
        help="structural control type; defaults to the config value or depth")
    parser.add_argument(
        "--control-image",
        help="precomputed depth or Canny control; Canny is derived from source "
             "when omitted")
    parser.add_argument("--canny-low-threshold", type=int, default=None)
    parser.add_argument("--canny-high-threshold", type=int, default=None)
    parser.add_argument(
        "--reference-token-budget", type=int, default=None,
        help="optional total reference-token cap; 0 disables the cap (default)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    try:
        if not args.config:
            parser.error("--config is required unless --self-check is given")
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    control_type = args.control_type or config["control_type"]
    low_threshold = int(
        config.get("canny_low_threshold", DEFAULT_CANNY_LOW_THRESHOLD)
        if args.canny_low_threshold is None else args.canny_low_threshold)
    high_threshold = int(
        config.get("canny_high_threshold", DEFAULT_CANNY_HIGH_THRESHOLD)
        if args.canny_high_threshold is None else args.canny_high_threshold)
    if not 0 <= low_threshold < high_threshold <= 255:
        parser.error("Canny thresholds require 0 <= low < high <= 255")
    configured_budget = int(
        config.get("reference_token_budget", DEFAULT_REFERENCE_TOKEN_BUDGET)
    )
    if configured_budget < 0:
        parser.error("reference_token_budget must be non-negative")
    if (
        args.reference_token_budget is not None
        and args.reference_token_budget < 0
    ):
        parser.error(
            "--reference-token-budget must be non-negative "
            "(0 disables the cap)"
        )
    configured_control = (
        config.get("depth_control") if control_type == "depth"
        else config.get("canny_control")
    )
    control_path = args.control_image or configured_control
    if control_type == "depth" and control_path is None:
        parser.error(
            "depth control requires depth_control in the config or "
            "--control-image")
    paths = [config["source"]]
    if control_path is not None:
        paths.append(control_path)
    for part in config["parts"]:
        paths.extend([part["target_mask"], part["donor"], part["donor_mask"]])
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        parser.error("file not found: " + ", ".join(missing))
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    ownership = target_ownership(config["parts"])
    torch.set_grad_enabled(False)
    pipeline, redux = load_models(control_type)
    try:
        control_image, control_source = load_control_image(
            config, control_type, args.control_image,
            low_threshold=low_threshold, high_threshold=high_threshold)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    text, pooled, _ = pipeline.encode_prompt(
        prompt=config["prompt"], prompt_2=config["prompt"], device=TEXT_DEV)
    threshold = float(
        config.get("threshold", DEFAULT_THRESHOLD)
        if args.threshold is None else args.threshold)
    references = [
        foreground_tokens(
            redux, part["donor"], part["donor_mask"], threshold)
        for part in config["parts"]
    ]
    capacities = [int(tokens.shape[1]) for tokens in references]
    budget = int(
        configured_budget
        if args.reference_token_budget is None else args.reference_token_budget)
    allocation = allocate_equal_budget(capacities, budget)
    references = [farthest_point_tokens(tokens, count)
                  for tokens, count in zip(references, allocation)]
    lam = float(
        config.get("lam", DEFAULT_LAMBDA) if args.lam is None else args.lam)
    if lam < 0:
        parser.error("lambda must be non-negative")
    if not 0 <= threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    condition = torch.cat(
        [text, *[lam * tokens.to(text.dtype) for tokens in references]], dim=1)
    attention, routes = regional_bias(
        int(text.shape[1]), allocation, ownership)

    steps = int(
        config.get("steps", DEFAULT_STEPS) if args.steps is None else args.steps)
    if steps <= 0:
        parser.error("steps must be positive")
    guidance = effective_guidance(config, control_type, args.guidance)
    if guidance < 0:
        parser.error("guidance must be non-negative")
    seed = int(
        config.get("seed", DEFAULT_SEED) if args.seed is None else args.seed)
    started = time.time()
    image = pipeline(
        prompt_embeds=condition,
        pooled_prompt_embeds=pooled,
        control_image=control_image,
        image=Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white"),
        strength=1.0,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(TEXT_DEV).manual_seed(seed),
        joint_attention_kwargs={"attention_mask": attention},
    ).images[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{Path(config['source']).stem}.png"
    image.save(output)
    print(
        f"[run] control={control_type}:{control_source} routes={routes} "
        f"reference_tokens={capacities}->{allocation} "
        f"condition={condition.shape[1]} steps={steps} guidance={guidance:g} "
        f"elapsed={time.time()-started:.1f}s")
    print(f"[run] {output}", flush=True)


if __name__ == "__main__":
    main()
