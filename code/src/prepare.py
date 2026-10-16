from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpus", default=None)
_gpus = _pre.parse_known_args()[0].gpus
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


RGB = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}
DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def safe_stem(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def relative_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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
    required = {"source", "source_overlay", "prompt", "parts"}
    if not required <= set(config):
        raise ValueError(f"config requires {sorted(required)}")
    if not isinstance(config["parts"], list) or not config["parts"]:
        raise ValueError("config parts must be a non-empty list")
    parts = []
    for index, raw in enumerate(config["parts"]):
        if not isinstance(raw, dict):
            raise ValueError(f"parts[{index}] must be a mapping")
        part = {str(key).replace("-", "_"): value
                for key, value in raw.items()}
        if not {"name", "donor", "color"} <= set(part):
            raise ValueError(f"parts[{index}] requires name, donor, and color")
        colour = str(part["color"]).lower()
        donor_colour = part.get("donor_mask_color")
        if colour not in RGB:
            raise ValueError(f"unsupported target mask color: {colour}")
        if donor_colour and str(donor_colour).lower() not in RGB:
            raise ValueError(f"unsupported donor mask color: {donor_colour}")
        if donor_colour and not part.get("donor_mask"):
            raise ValueError("donor_mask_color requires donor_mask")
        parts.append(part)
    config["parts"] = parts
    return config


def clean_mask(binary):
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if len(areas) else 0
    minimum = max(96, binary.size // 2000, largest // 100)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            keep[labels == index] = 1
    # Pad with a guaranteed exterior background so hole filling remains valid
    # even when foreground touches the top-left corner or any image boundary.
    padded = np.pad(keep, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (padded == 0) & (flood == 0)
    keep[holes[1:-1, 1:-1]] = 1
    return keep


def extract_target_masks(source_path, overlay_path, parts, threshold, out_dir):
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), np.float32)
    height, width = overlay.shape[:2]
    source = np.asarray(Image.open(source_path).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS), np.float32)
    delta = overlay - source
    scores = np.stack((
        delta[..., 0] - 0.5 * (delta[..., 1] + delta[..., 2]),
        delta[..., 2] - 0.5 * (delta[..., 0] + delta[..., 1]),
        delta[..., 1] - 0.5 * (delta[..., 0] + delta[..., 2]),
    ), axis=-1)
    labels = scores.argmax(axis=-1)
    active = scores.max(axis=-1) >= threshold
    layers = [clean_mask(((labels == index) & active).astype(np.uint8))
              for index in range(3)]
    if np.any(np.stack(layers).sum(0) > 1):
        raise ValueError("target part masks overlap")
    outputs = []
    for index, part in enumerate(parts, 1):
        colour_index = {"red": 0, "blue": 1, "green": 2}[
            str(part["color"]).lower()]
        mask = layers[colour_index]
        if not mask.any():
            raise ValueError(f"target mask is empty: {part['name']}")
        output = out_dir / f"target_{index:02d}_{safe_stem(part['name'])}.png"
        Image.fromarray(mask * 255).resize(
            (1024, 1024), Image.Resampling.NEAREST).save(output)
        outputs.append(output)
    return outputs


def make_depth(source_path, output_path, model_size, device, force):
    if output_path.exists() and not force:
        return
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_name = DEPTH_MODELS[model_size]
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    image = Image.open(source_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        values = model(**inputs).predicted_depth[0]
    values = torch.nn.functional.interpolate(
        values[None, None], (1024, 1024), mode="bilinear",
        align_corners=False)[0, 0].float().cpu().numpy()
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    depth = np.stack([np.uint8(values * 255)] * 3, axis=-1)
    Image.fromarray(depth).save(output_path)
    del model
    torch.cuda.empty_cache()


def make_foreground_masks(requests, device):
    if not requests:
        return
    import torchvision.transforms as transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True,
        local_files_only=True).to(device).eval()
    dtype = next(model.parameters()).dtype
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    for image_path, output_path in requests:
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            logits = model(transform(image).unsqueeze(0).to(
                device, dtype=dtype))[-1]
        probability = torch.sigmoid(logits.float().cpu()[0, 0]).numpy()
        Image.fromarray(np.uint8(probability * 255)).resize(
            image.size).save(output_path)
    del model
    torch.cuda.empty_cache()


def prepare_reference_masks(parts, out_dir, device, force):
    outputs = []
    automatic = []
    for index, part in enumerate(parts, 1):
        donor = Path(part["donor"])
        output = out_dir / f"reference_{index:02d}_{safe_stem(donor.stem)}.png"
        outputs.append(output)
        if output.exists() and not force:
            continue
        if part.get("donor_mask"):
            source = Image.open(part["donor_mask"])
            if part.get("donor_mask_color"):
                labels = np.asarray(source.convert("RGB"))
                colour = RGB[str(part["donor_mask_color"]).lower()]
                source = Image.fromarray(
                    np.uint8(np.all(labels == colour, axis=-1)) * 255)
            source.convert("L").resize(
                Image.open(donor).size, Image.Resampling.NEAREST).save(output)
        else:
            automatic.append((donor, output))
    make_foreground_masks(automatic, device)
    return outputs


def write_manifest(config, depth_path, target_masks, reference_masks, output):
    parts = []
    for raw, target_mask, reference_mask in zip(
            config["parts"], target_masks, reference_masks):
        parts.append({
            "name": raw["name"],
            "target_mask": relative_path(target_mask),
            "donor": relative_path(raw["donor"]),
            "donor_mask": relative_path(reference_mask),
        })
    manifest = {
        "source": relative_path(config["source"]),
        "depth_control": relative_path(depth_path),
        "parts": parts,
        "prompt": config["prompt"],
        "lam": float(config.get("lam", 1.0)),
        "threshold": float(config.get("threshold", 0.3)),
        "reference_token_budget": int(
            config.get("reference_token_budget", 0)),
        "steps": int(config.get("steps", 28)),
        "guidance": float(config.get("guidance", 10.0)),
        "seed": int(config.get("seed", 0)),
    }
    with output.open("w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)


def self_check():
    binary = np.zeros((64, 64), np.uint8)
    binary[8:56, 8:56] = 1
    assert clean_mask(binary).sum() >= binary.sum()
    corner = np.zeros((64, 64), np.uint8)
    corner[:20, :20] = 1
    assert np.array_equal(clean_mask(corner), corner)
    assert safe_stem("head & neck") == "head___neck"
    print("[prepare] self-check OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/default.yaml")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    paths = [config["source"], config["source_overlay"]]
    for part in config["parts"]:
        paths.append(part["donor"])
        if part.get("donor_mask"):
            paths.append(part["donor_mask"])
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        parser.error("file not found: " + ", ".join(missing))
    if not torch.cuda.is_available():
        parser.error("CUDA is required for depth and foreground preparation")

    source_stem = safe_stem(Path(config["source"]).stem)
    out_dir = Path(args.out_dir or f"data/prepared/{source_stem}_v6")
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_path = out_dir / "depth.png"
    target_masks = extract_target_masks(
        config["source"], config["source_overlay"], config["parts"],
        float(config.get("overlay_threshold", 60.0)), out_dir)
    make_depth(
        config["source"], depth_path, config.get("depth", "small"),
        args.device, args.force)
    reference_masks = prepare_reference_masks(
        config["parts"], out_dir, args.device, args.force)
    manifest = out_dir / "config.yaml"
    write_manifest(config, depth_path, target_masks, reference_masks, manifest)
    print(f"[prepare] parts={len(config['parts'])} -> {manifest}")


if __name__ == "__main__":
    main()
