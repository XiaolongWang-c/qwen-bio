from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
GRAYSCALE_CHANNEL_GAP = 6.0
SINGLE_CHANNEL_RATIO = 1.65

BASE_EDIT_INSTRUCTION = """Edit this microscopy image using the input image as a style reference.

- Keep the microscopy modality, acquisition style, crop, zoom, perspective, background appearance, illumination, blur, noise level, contrast, and imaging artifacts unchanged.
- Keep any scale bar, panel letter, bounding box, arrow, text label, border, and overlay exactly unchanged.
- Modify only the biological foreground region and generate a new plausible specimen appearance that is visibly different from the reference foreground.
- The new foreground should still belong to the same experiment domain and the same microscopy style.
- Do not replace the specimen region with empty smooth background.
- Do not redraw the background or annotations.
- Do not create repeated patterns, cartoon edges, synthetic textures, or non-biological structures.
"""

STYLE_PROMPTS = {
    "grayscale": """- Preserve the original grayscale or brightfield appearance exactly.
- Keep the same tone range and soft microscopy blur.
- Keep faint low-contrast specimen texture visible inside the edited foreground.
- Keep the brightfield background unchanged.
""",
    "fluorescence_multichannel": """- Preserve the exact fluorescence color mapping and channel composition.
- Keep the black background and the same multi-channel visual balance.
""",
    "fluorescence_single_channel": """- Preserve the same fluorescence color family and black background.
- Keep the same single-channel dominant appearance and intensity style.
""",
}

CONTENT_PROMPTS = {
    "few_objects": """- Create clearly different biological objects in the foreground region.
- You may change object count, size, shape, and relative placement naturally.
- Keep the result sparse if the input is sparse.
""",
    "many_objects": """- Create a clearly different dense foreground pattern in the edited region.
- You may change local density, clustering, and small-object arrangement naturally.
- Keep the same overall microscopy domain and texture scale.
""",
    "mixed_cluster": """- Create a clearly different clustered biological foreground.
- You may change cluster contour, internal texture, and local protrusions naturally.
- Keep the same cluster-like scene type and imaging style.
""",
    "large_region": """- Create a clearly different continuous biological foreground region.
- You may change boundary waviness, thickness, and local internal texture naturally.
- Keep the same tissue-like or large-area scene type and imaging style.
""",
}

NEGATIVE_PROMPT = """background replacement, changed annotation, missing scale bar, changed overlay,
changed crop, changed zoom, changed magnification, empty background, blank field, removed specimen,
deleted cell, deleted organoid, smooth erased specimen region, cartoon, illustration, painterly,
sharp outlines, halo artifacts, grid pattern, repeated pattern, mosaic artifact, over-sharpening,
ultra detailed, non-microscopy texture, false color mapping, inverted contrast, text artifact
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 3000 microscopy references and generate similar AI-edited images with Qwen image edit."
    )
    parser.add_argument("--micro-root", type=Path, default=Path("micro"))
    parser.add_argument("--output-root", type=Path, default=Path("micro/generated_qwen_3000"))
    parser.add_argument("--subset-train-count", type=int, default=2400)
    parser.add_argument("--subset-test-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN_IMAGE_EDIT_MODEL", "/sda/wangxl/ai-bio/qwen-image-edit"),
        help="Qwen image edit model path. Can also be set by QWEN_IMAGE_EDIT_MODEL.",
    )
    parser.add_argument("--device-map", type=str, default="balanced")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--true-cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--blend-strength", type=float, default=0.42)
    parser.add_argument("--detail-sigma", type=float, default=1.1)
    parser.add_argument("--mask-quantile", type=float, default=94.0)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.002)
    parser.add_argument("--selection-stride", type=int, default=3)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--roi-padding-ratio", type=float, default=0.18)
    parser.add_argument("--roi-min-padding", type=int, default=20)
    parser.add_argument("--foreground-dilation", type=int, default=6)
    parser.add_argument("--annotation-border-ratio", type=float, default=0.12)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-background-diff", type=float, default=2.8)
    parser.add_argument("--max-annotation-diff", type=float, default=1.5)
    parser.add_argument("--min-foreground-diff", type=float, default=7.5)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.45)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-source-copy", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )


def prepare_outputs(output_root: Path, overwrite: bool) -> dict[str, Path]:
    outputs = {
        "train_generated": output_root / "train" / "Micro" / "image",
        "test_generated": output_root / "test" / "Micro" / "image",
        "train_raw_generated": output_root / "raw_model_output" / "train" / "Micro" / "image",
        "test_raw_generated": output_root / "raw_model_output" / "test" / "Micro" / "image",
        "train_source_subset": output_root / "source_subset" / "train" / "Micro" / "image",
        "test_source_subset": output_root / "source_subset" / "test" / "Micro" / "image",
        "records": output_root / "records",
        "preview": output_root / "records" / "preview",
    }
    if overwrite:
        reset_dir(output_root)
    for path in outputs.values():
        ensure_dir(path)
    return outputs


def load_existing_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def classify_style_from_array(array: np.ndarray) -> str:
    channel_gap = (
        np.abs(array[:, :, 0] - array[:, :, 1]).mean()
        + np.abs(array[:, :, 1] - array[:, :, 2]).mean()
        + np.abs(array[:, :, 0] - array[:, :, 2]).mean()
    ) / 3.0
    if channel_gap < GRAYSCALE_CHANNEL_GAP:
        return "grayscale"

    means = array.mean(axis=(0, 1))
    strongest = float(np.max(means))
    weakest = float(np.partition(means, -2)[-2])
    if weakest <= 0:
        weakest = 1.0
    if strongest / weakest >= SINGLE_CHANNEL_RATIO:
        return "fluorescence_single_channel"
    return "fluorescence_multichannel"


def classify_style(image_path: Path) -> str:
    with Image.open(image_path) as image:
        rgb = ImageOps.contain(image.convert("RGB"), (128, 128))
        array = np.asarray(rgb, dtype=np.float32)
    return classify_style_from_array(array)


def select_paths_by_stride(paths: list[Path], stride: int, offset: int) -> list[Path]:
    if stride < 1:
        raise ValueError("--selection-stride must be at least 1.")
    if not 0 <= offset < stride:
        raise ValueError("--selection-offset must satisfy 0 <= offset < --selection-stride.")
    return [path for index, path in enumerate(paths) if index % stride == offset]


def build_manifest(split: str, source_dir: Path, seed: int, selection_stride: int, selection_offset: int) -> dict[str, Any]:
    paths = list_images(source_dir)
    selected_paths = select_paths_by_stride(paths, selection_stride, selection_offset)
    total_bucket_counts: Counter[str] = Counter()
    sampled_bucket_counts: Counter[str] = Counter()
    for path in paths:
        total_bucket_counts[classify_style(path)] += 1
    rng = random.Random(seed)
    sampled_entries: list[dict[str, Any]] = []

    for global_index, path in enumerate(selected_paths):
        bucket = classify_style(path)
        sampled_bucket_counts[bucket] += 1
        sampled_entries.append(
            {
                "name": path.name,
                "source_path": str(path),
                "style_bucket": bucket,
                "seed": rng.randint(0, 2**31 - 1),
                "global_pick_index": global_index,
            }
        )

    sampled_entries.sort(key=lambda item: item["name"])
    return {
        "split": split,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "total_available": len(paths),
        "selection_stride": selection_stride,
        "selection_offset": selection_offset,
        "sample_count": len(sampled_entries),
        "bucket_counts_total": dict(total_bucket_counts),
        "bucket_counts_sampled": dict(sampled_bucket_counts),
        "files": sampled_entries,
    }


def get_split_manifest(
    split: str,
    source_dir: Path,
    seed: int,
    selection_stride: int,
    selection_offset: int,
    manifest_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        existing = load_existing_manifest(manifest_path)
        if existing is not None:
            return existing
    manifest = build_manifest(split, source_dir, seed, selection_stride, selection_offset)
    write_json(manifest_path, manifest)
    return manifest


def resolve_torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_pipeline(model_path: str, torch_dtype_name: str, device_map: str) -> Any:
    from modelscope import QwenImageEditPipeline

    return QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(torch_dtype_name),
        device_map=device_map,
    )


def build_prompt(style_bucket: str, content_type: str) -> str:
    return BASE_EDIT_INSTRUCTION + "\n" + STYLE_PROMPTS[style_bucket] + "\n" + CONTENT_PROMPTS[content_type]


def select_shard(items: list[dict[str, Any]], shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count.")
    return [item for idx, item in enumerate(items) if idx % shard_count == shard_index]


def maybe_copy_source(source_path: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    destination = destination_dir / source_path.name
    if destination.exists():
        return
    shutil.copy2(source_path, destination)


def pil_to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def float_rgb_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def resize_like_source(source: Image.Image, generated: Image.Image) -> Image.Image:
    source_size = source.size
    if generated.size != source_size:
        return generated.resize(source_size, resample=Image.Resampling.LANCZOS)
    return generated


def match_channel_statistics(source_rgb: np.ndarray, generated_rgb: np.ndarray) -> np.ndarray:
    matched = generated_rgb.copy()
    for channel_index in range(source_rgb.shape[2]):
        src = source_rgb[:, :, channel_index]
        gen = generated_rgb[:, :, channel_index]
        src_mean = float(src.mean())
        gen_mean = float(gen.mean())
        src_std = float(src.std())
        gen_std = float(gen.std())
        if gen_std < 1e-5:
            matched[:, :, channel_index] = src_mean
            continue
        scale = src_std / max(gen_std, 1e-5)
        matched[:, :, channel_index] = (gen - gen_mean) * scale + src_mean
    return np.clip(matched, 0, 255)


def component_touches_border(component_mask: np.ndarray) -> bool:
    ys, xs = np.where(component_mask)
    if ys.size == 0:
        return False
    height, width = component_mask.shape
    return ys.min() <= 1 or xs.min() <= 1 or ys.max() >= height - 2 or xs.max() >= width - 2


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    return binary_dilation(mask.astype(bool), iterations=iterations)


def detect_foreground_mask(image: Image.Image, args: argparse.Namespace) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    sigma = max(2.0, min(height, width) / 28.0)
    background = gaussian_filter(gray, sigma=sigma)
    dark_response = np.clip(background - gray, 0, None)
    light_response = np.clip(gray - background, 0, None)
    dark_score = float(np.percentile(dark_response, 97))
    light_score = float(np.percentile(light_response, 97))
    detail = dark_response if dark_score >= light_score else light_response

    threshold = max(
        float(np.percentile(detail, args.mask_quantile)),
        float(detail.mean() + 0.45 * detail.std()),
    )
    mask = detail > threshold
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask)

    labeled, component_count = label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    min_area = max(24, int(height * width * args.min_component_area_ratio))
    max_area = int(height * width * 0.55)

    for component_index in range(1, component_count + 1):
        component_mask = labeled == component_index
        area = int(component_mask.sum())
        if area < min_area or area > max_area:
            continue
        if component_touches_border(component_mask):
            continue
        keep |= component_mask

    if keep.any():
        keep = binary_dilation(keep, iterations=2)
    return keep.astype(np.float32)


def detect_annotation_mask(image: Image.Image, args: argparse.Namespace) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    border = max(6, int(min(height, width) * args.annotation_border_ratio))

    edge_zone = np.zeros_like(gray, dtype=bool)
    edge_zone[:border, :] = True
    edge_zone[-border:, :] = True
    edge_zone[:, :border] = True
    edge_zone[:, -border:] = True

    extreme_gray = (gray <= 22) | (gray >= 233)
    red_overlay = (rgb[:, :, 0] >= 150) & (rgb[:, :, 0] >= rgb[:, :, 1] + 50) & (rgb[:, :, 0] >= rgb[:, :, 2] + 50)
    candidates = (extreme_gray & edge_zone) | red_overlay

    labeled, component_count = label(candidates)
    keep = np.zeros_like(candidates, dtype=bool)
    max_area = max(16, int(height * width * 0.08))

    for component_index in range(1, component_count + 1):
        component_mask = labeled == component_index
        area = int(component_mask.sum())
        if area < 3 or area > max_area:
            continue
        ys, xs = np.where(component_mask)
        if ys.size == 0:
            continue
        near_border = (
            ys.min() < border
            or xs.min() < border
            or ys.max() >= height - border
            or xs.max() >= width - border
        )
        if near_border or bool((component_mask & red_overlay).any()):
            keep |= component_mask

    if keep.any():
        keep = binary_dilation(keep, iterations=1)
    return keep.astype(np.float32)


def classify_content_type_from_mask(mask: np.ndarray) -> str:
    binary = mask > 0
    coverage = float(binary.mean())
    if coverage <= 0.01:
        return "few_objects"

    labeled, component_count = label(binary)
    component_areas = [int((labeled == index).sum()) for index in range(1, component_count + 1)]
    component_areas = [area for area in component_areas if area > 0]
    if not component_areas:
        return "few_objects"

    component_areas.sort(reverse=True)
    largest_ratio = component_areas[0] / max(sum(component_areas), 1)
    large_components = sum(area >= component_areas[0] * 0.2 for area in component_areas)

    if coverage >= 0.28 or largest_ratio >= 0.78:
        return "large_region"
    if largest_ratio >= 0.42 and large_components <= 4:
        return "mixed_cluster"
    if len(component_areas) >= 28:
        return "many_objects"
    if len(component_areas) <= 6:
        return "few_objects"
    return "many_objects" if coverage < 0.12 else "mixed_cluster"


def low_high_frequency_decompose(rgb: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.stack([gaussian_filter(rgb[:, :, channel], sigma=sigma) for channel in range(rgb.shape[2])], axis=2)
    high = rgb - low
    return low, high


def extract_edit_bbox(mask: np.ndarray, args: argparse.Namespace) -> tuple[int, int, int, int]:
    binary = mask > 0
    height, width = binary.shape
    ys, xs = np.where(binary)
    if ys.size == 0:
        return (0, 0, width, height)

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    pad = max(args.roi_min_padding, int(max(y1 - y0, x1 - x0) * args.roi_padding_ratio))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def masked_mean_abs_diff(source_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> float:
    binary = mask > 0
    if not binary.any():
        return 0.0
    diff = np.abs(source_rgb - target_rgb).mean(axis=2)
    return float(diff[binary].mean())


def build_soft_alpha(mask: np.ndarray) -> np.ndarray:
    sigma = max(1.0, min(mask.shape) / 40.0)
    soft = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    max_value = float(soft.max())
    if max_value > 1e-6:
        soft = soft / max_value
    return np.clip(soft, 0.0, 1.0)


def composite_generated_roi(
    source_image: Image.Image,
    generated_roi_image: Image.Image,
    foreground_mask: np.ndarray,
    annotation_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    source_rgb = pil_to_float_rgb(source_image)
    source_roi_rgb = source_rgb[y0:y1, x0:x1, :]
    generated_roi = resize_like_source(Image.fromarray(source_roi_rgb.astype(np.uint8)), generated_roi_image)
    generated_roi_rgb = pil_to_float_rgb(generated_roi)
    generated_roi_rgb = match_channel_statistics(source_roi_rgb, generated_roi_rgb)

    foreground_roi = foreground_mask[y0:y1, x0:x1] > 0
    foreground_roi = dilate_mask(foreground_roi, args.foreground_dilation)
    annotation_roi = annotation_mask[y0:y1, x0:x1] > 0
    soft_alpha = build_soft_alpha(foreground_roi.astype(np.float32))
    soft_alpha = np.where(annotation_roi, 0.0, soft_alpha)

    composed_roi = source_roi_rgb * (1.0 - soft_alpha[:, :, None]) + generated_roi_rgb * soft_alpha[:, :, None]
    composed_roi = np.where(annotation_roi[:, :, None], source_roi_rgb, composed_roi)

    composed_full = source_rgb.copy()
    composed_full[y0:y1, x0:x1, :] = composed_roi
    return float_rgb_to_pil(composed_full)


def evaluate_generation_quality(
    source_image: Image.Image,
    generated_image: Image.Image,
    source_mask: np.ndarray,
    annotation_mask: np.ndarray,
    expected_style_bucket: str,
    args: argparse.Namespace,
) -> dict[str, float | bool | str]:
    source_rgb = pil_to_float_rgb(source_image)
    generated_rgb = pil_to_float_rgb(generated_image)
    protected_mask = dilate_mask((source_mask > 0) | (annotation_mask > 0), 2)
    background_mask = (~protected_mask).astype(np.float32)

    foreground_diff = masked_mean_abs_diff(source_rgb, generated_rgb, source_mask)
    background_diff = masked_mean_abs_diff(source_rgb, generated_rgb, background_mask)
    annotation_diff = masked_mean_abs_diff(source_rgb, generated_rgb, annotation_mask)

    generated_mask = detect_foreground_mask(generated_image, args)
    source_coverage = float((source_mask > 0).mean())
    generated_coverage = float((generated_mask > 0).mean())
    coverage_ratio = generated_coverage / max(source_coverage, 1e-6)
    output_style_bucket = classify_style_from_array(np.asarray(ImageOps.contain(generated_image.convert("RGB"), (128, 128)), dtype=np.float32))
    style_consistent = output_style_bucket == expected_style_bucket
    coverage_ok = source_coverage <= 0.005 or coverage_ratio >= args.min_coverage_ratio

    quality_pass = (
        foreground_diff >= args.min_foreground_diff
        and background_diff <= args.max_background_diff
        and annotation_diff <= args.max_annotation_diff
        and coverage_ok
        and style_consistent
    )

    return {
        "foreground_diff": foreground_diff,
        "background_diff": background_diff,
        "annotation_diff": annotation_diff,
        "source_coverage": source_coverage,
        "generated_coverage": generated_coverage,
        "coverage_ratio": coverage_ratio,
        "output_style_bucket": output_style_bucket,
        "style_consistent": style_consistent,
        "quality_pass": quality_pass,
    }


def save_preview_pairs(pairs: list[tuple[Path, Path]], preview_path: Path) -> None:
    if not pairs:
        return

    thumbs: list[Image.Image] = []
    for source_path, generated_path in pairs:
        with Image.open(source_path) as source:
            source_thumb = ImageOps.contain(source.convert("RGB"), (224, 224))
        with Image.open(generated_path) as generated:
            generated_thumb = ImageOps.contain(generated.convert("RGB"), (224, 224))

        canvas = Image.new("RGB", (480, 248), (248, 248, 248))
        canvas.paste(source_thumb, ((240 - source_thumb.width) // 2, (248 - source_thumb.height) // 2))
        canvas.paste(generated_thumb, (240 + (240 - generated_thumb.width) // 2, (248 - generated_thumb.height) // 2))
        thumbs.append(canvas)

    rows = len(thumbs)
    sheet = Image.new("RGB", (480, rows * 248), (245, 245, 245))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, index * 248))
    sheet.save(preview_path)


def process_split(
    split: str,
    items: list[dict[str, Any]],
    generated_dir: Path,
    raw_generated_dir: Path,
    source_subset_dir: Path,
    records_dir: Path,
    copy_source: bool,
    pipeline: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ensure_dir(generated_dir)
    ensure_dir(raw_generated_dir)
    ensure_dir(records_dir)
    if copy_source:
        ensure_dir(source_subset_dir)

    processed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    preview_pairs: list[tuple[Path, Path]] = []
    postprocess_metrics: list[dict[str, Any]] = []

    if pipeline is None:
        summary = {
            "split": split,
            "assigned": len(items),
            "generated": 0,
            "skipped_existing": 0,
            "failures": 0,
            "preview_path": None,
            "style_buckets": dict(Counter(item["style_bucket"] for item in items)),
        }
        write_json(records_dir / f"{split}_run_summary.json", summary)
        return summary

    import torch

    for index, item in enumerate(items, 1):
        source_path = Path(item["source_path"])
        output_path = generated_dir / item["name"]
        raw_output_path = raw_generated_dir / item["name"]

        if copy_source:
            maybe_copy_source(source_path, source_subset_dir)

        if output_path.exists() and not args.overwrite:
            skipped += 1
            if len(preview_pairs) < args.preview_count:
                preview_pairs.append((source_path, output_path))
            continue

        try:
            with Image.open(source_path) as image:
                rgb = image.convert("RGB")
                source_mask = detect_foreground_mask(rgb, args)
                annotation_mask = detect_annotation_mask(rgb, args)
                content_type = classify_content_type_from_mask(source_mask)
                edit_mask = dilate_mask(source_mask > 0, args.foreground_dilation).astype(np.float32)
                bbox = extract_edit_bbox(edit_mask, args)
                x0, y0, x1, y1 = bbox
                roi_image = rgb.crop(bbox)
                prompt = build_prompt(item["style_bucket"], content_type)

                final_image = None
                final_metrics: dict[str, Any] | None = None
                last_error: str | None = None

                for retry_index in range(args.max_retries + 1):
                    attempt_seed = int(item["seed"]) + retry_index
                    inputs = {
                        "image": roi_image,
                        "prompt": prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "generator": torch.Generator(device="cpu").manual_seed(attempt_seed),
                        "true_cfg_scale": args.true_cfg_scale,
                        "num_inference_steps": args.num_inference_steps,
                    }
                    with torch.inference_mode():
                        output = pipeline(**inputs)

                    generated_roi = output.images[0].convert("RGB")
                    candidate_image = composite_generated_roi(
                        source_image=rgb,
                        generated_roi_image=generated_roi,
                        foreground_mask=edit_mask,
                        annotation_mask=annotation_mask,
                        bbox=bbox,
                        args=args,
                    )
                    candidate_metrics = evaluate_generation_quality(
                        source_image=rgb,
                        generated_image=candidate_image,
                        source_mask=edit_mask,
                        annotation_mask=annotation_mask,
                        expected_style_bucket=item["style_bucket"],
                        args=args,
                    )
                    candidate_metrics.update(
                        {
                            "content_type": content_type,
                            "retry_index": retry_index,
                            "bbox": [x0, y0, x1, y1],
                        }
                    )

                    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_image.save(raw_output_path)

                    if bool(candidate_metrics["quality_pass"]):
                        final_image = candidate_image
                        final_metrics = candidate_metrics
                        break

                    last_error = (
                        f"quality gate failed: fg={candidate_metrics['foreground_diff']:.3f}, "
                        f"bg={candidate_metrics['background_diff']:.3f}, "
                        f"ann={candidate_metrics['annotation_diff']:.3f}, "
                        f"coverage={candidate_metrics['coverage_ratio']:.3f}, "
                        f"style={candidate_metrics['output_style_bucket']}"
                    )
                    final_image = candidate_image
                    final_metrics = candidate_metrics

                if final_image is None or final_metrics is None:
                    raise RuntimeError(last_error or "generation did not produce an output")

                final_image.save(output_path)
                postprocess_metrics.append(
                    {
                        "split": split,
                        "name": item["name"],
                        "style_bucket": item["style_bucket"],
                        **final_metrics,
                    }
                )

            processed += 1
            if len(preview_pairs) < args.preview_count:
                preview_pairs.append((source_path, output_path))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "split": split,
                    "name": item["name"],
                    "source_path": str(source_path),
                    "style_bucket": item["style_bucket"],
                    "error": str(exc),
                }
            )

        if index % args.print_every == 0 or index == len(items):
            print(
                f"{split}: {index}/{len(items)} processed, "
                f"generated={processed}, skipped={skipped}, failures={len(failures)}"
            )

    preview_path = records_dir / f"{split}_preview.png"
    if pipeline is not None:
        save_preview_pairs(preview_pairs, preview_path)

    if failures:
        write_jsonl(records_dir / f"{split}_failures.jsonl", failures)
    if postprocess_metrics:
        write_jsonl(records_dir / f"{split}_fusion_metrics.jsonl", postprocess_metrics)

    summary = {
        "split": split,
        "assigned": len(items),
        "generated": processed,
        "skipped_existing": skipped,
        "failures": len(failures),
        "preview_path": str(preview_path) if preview_pairs and pipeline is not None else None,
        "style_buckets": dict(Counter(item["style_bucket"] for item in items)),
        "raw_generated_dir": str(raw_generated_dir),
    }
    write_json(records_dir / f"{split}_run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()

    train_source = args.micro_root / "train" / "Micro" / "image"
    test_source = args.micro_root / "test" / "Micro" / "image"
    if not train_source.exists() or not test_source.exists():
        raise FileNotFoundError("Expected micro/train/Micro/image and micro/test/Micro/image to exist.")

    outputs = prepare_outputs(args.output_root, args.overwrite)
    records_dir = outputs["records"]

    train_manifest = get_split_manifest(
        split="train",
        source_dir=train_source,
        seed=args.seed,
        selection_stride=args.selection_stride,
        selection_offset=args.selection_offset,
        manifest_path=records_dir / "train_manifest.json",
        overwrite=args.overwrite,
    )
    test_manifest = get_split_manifest(
        split="test",
        source_dir=test_source,
        seed=args.seed + 1,
        selection_stride=args.selection_stride,
        selection_offset=args.selection_offset,
        manifest_path=records_dir / "test_manifest.json",
        overwrite=args.overwrite,
    )

    train_items = train_manifest["files"]
    test_items = test_manifest["files"]

    if args.limit is not None:
        train_items = train_items[: min(args.limit, len(train_items))]
        test_items = test_items[: min(args.limit, len(test_items))]

    train_items = select_shard(train_items, args.shard_count, args.shard_index)
    test_items = select_shard(test_items, args.shard_count, args.shard_index)

    pipeline = None
    if not args.dry_run:
        pipeline = load_pipeline(
            model_path=args.model_path,
            torch_dtype_name=args.torch_dtype,
            device_map=args.device_map,
        )
        pipeline.set_progress_bar_config(disable=None)
        print("pipeline loaded")

    train_summary = process_split(
        split="train",
        items=train_items,
        generated_dir=outputs["train_generated"],
        raw_generated_dir=outputs["train_raw_generated"],
        source_subset_dir=outputs["train_source_subset"],
        records_dir=records_dir,
        copy_source=not args.skip_source_copy,
        pipeline=pipeline,
        args=args,
    )
    test_summary = process_split(
        split="test",
        items=test_items,
        generated_dir=outputs["test_generated"],
        raw_generated_dir=outputs["test_raw_generated"],
        source_subset_dir=outputs["test_source_subset"],
        records_dir=records_dir,
        copy_source=not args.skip_source_copy,
        pipeline=pipeline,
        args=args,
    )

    final_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "model_path": args.model_path if not args.dry_run else None,
        "seed": args.seed,
        "selection_stride": args.selection_stride,
        "selection_offset": args.selection_offset,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "train": train_summary,
        "test": test_summary,
        "overall_assigned": train_summary["assigned"] + test_summary["assigned"],
        "overall_generated": train_summary["generated"] + test_summary["generated"],
        "overall_failures": train_summary["failures"] + test_summary["failures"],
    }
    write_json(records_dir / "generation_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
