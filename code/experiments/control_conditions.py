"""Shared depth/Canny control handling, so every method gets the same edge map."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CONTROL_MODEL_IDS = {
    "depth": "black-forest-labs/FLUX.1-Depth-dev",
    "canny": "black-forest-labs/FLUX.1-Canny-dev",
}


def control_model_id(control_type: str, override: str | None = None) -> str:
    if control_type not in CONTROL_MODEL_IDS:
        raise ValueError(f"unsupported control type: {control_type}")
    return override or CONTROL_MODEL_IDS[control_type]


def _rooted(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _source_cache_path(
    source: Path,
    root: Path,
    cache_root: Path,
    low_threshold: int,
    high_threshold: int,
    resolution: int,
) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    try:
        relative = source.resolve().relative_to((root / "data").resolve())
    except ValueError:
        relative = Path(source.parent.name) / source.name
    name = f"{relative.stem}_{digest}.png"
    return (
        cache_root
        / f"canny_l{low_threshold}_h{high_threshold}_r{resolution}"
        / relative.parent
        / name
    )


def make_canny_image(
    source: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 200,
    resolution: int = 1024,
) -> Image.Image:
    if resolution <= 0:
        raise ValueError("control resolution must be positive")
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("Canny thresholds require 0 <= low < high <= 255")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for Canny preprocessing") from error

    resized = source.convert("RGB").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    rgb = np.asarray(resized, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=3,
        L2gradient=False,
    )
    return Image.fromarray(np.repeat(edges[..., None], 3, axis=2), mode="RGB")


def load_control_image(
    case: dict[str, Any],
    *,
    control_type: str,
    root: Path,
    cache_root: Path,
    low_threshold: int = 50,
    high_threshold: int = 200,
    resolution: int = 1024,
) -> tuple[Image.Image, dict[str, Any]]:
    if control_type == "depth":
        if "depth_control" not in case:
            raise ValueError(f"case {case.get('id')} has no depth_control")
        path = _rooted(str(case["depth_control"]), root)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        return image, {
            "control_type": "depth",
            "control_source": _display_path(path, root),
            "control_image": _display_path(path, root),
            "control_preprocessor": "precomputed plan depth map",
            "control_resolution": list(image.size),
        }

    if control_type != "canny":
        raise ValueError(f"unsupported control type: {control_type}")
    # A shipped edge map is used as is; Canny on it would find edges of edges.
    if "canny_control" in case:
        path = _rooted(str(case["canny_control"]), root)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.LANCZOS)
        return image, {
            "control_type": "canny",
            "control_source": _display_path(path, root),
            "control_image": _display_path(path, root),
            "control_preprocessor": "precomputed edge map supplied by the case",
            "control_resolution": [resolution, resolution],
        }
    if "source" not in case:
        raise ValueError(f"case {case.get('id')} has no source image")
    source = _rooted(str(case["source"]), root)
    if not source.is_file():
        raise FileNotFoundError(source)
    cached = _source_cache_path(
        source,
        root,
        cache_root,
        low_threshold,
        high_threshold,
        resolution,
    )
    if cached.is_file():
        image = Image.open(cached).convert("RGB")
        if image.size != (resolution, resolution):
            raise ValueError(
                f"cached Canny control has unexpected size {image.size}: {cached}"
            )
    else:
        image = make_canny_image(
            Image.open(source),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            resolution=resolution,
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        image.save(cached, optimize=True)
    return image, {
        "control_type": "canny",
        "control_source": _display_path(source, root),
        "control_image": _display_path(cached, root),
        "control_preprocessor": "opencv.Canny after square Lanczos resize",
        "control_resolution": [resolution, resolution],
        "canny_low_threshold": low_threshold,
        "canny_high_threshold": high_threshold,
        "canny_aperture_size": 3,
        "canny_l2_gradient": False,
    }


def control_run_metadata(
    *,
    control_type: str,
    model_id: str,
    cache_root: Path,
    low_threshold: int,
    high_threshold: int,
    resolution: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "control_type": control_type,
        "control_model": model_id,
        "control_cache": str(cache_root.resolve()),
        "control_resolution": resolution,
    }
    if control_type == "canny":
        record.update({
            "control_preprocessor": "opencv.Canny after square Lanczos resize",
            "canny_low_threshold": low_threshold,
            "canny_high_threshold": high_threshold,
            "canny_aperture_size": 3,
            "canny_l2_gradient": False,
        })
    else:
        record["control_preprocessor"] = "precomputed plan depth map"
    return record
