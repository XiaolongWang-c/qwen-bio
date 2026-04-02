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

BASE_EDIT_INSTRUCTION = """Edit this microscopy image using the input image as a direct structural reference.

- Keep every visible specimen present in the image. Do not erase, remove, or replace any existing cell, organoid, colony, or tissue region.
- Keep the specimen count, global arrangement, major contours, object sizes, relative positions, crop, zoom, and perspective nearly unchanged.
- Keep the background, blur, noise level, illumination, contrast, and imaging artifacts unchanged.
- Keep any scale bar, panel letter, bounding box, arrow, text label, border, and overlay exactly unchanged.
- Produce a new plausible capture of the same experiment, not a pixel-identical copy.
- Only allow subtle local biological variation in boundary softness, internal texture, and intensity distribution.
- Preserve the outer contour of each visible specimen and keep each specimen centroid in the same place.
- Do not add or remove large structures.
- Do not move specimens noticeably.
- Do not create repeated patterns or synthetic-looking details.
"""

STYLE_PROMPTS = {
    "grayscale": """- Preserve the original grayscale or brightfield appearance exactly.
- Keep the same tone range and soft microscopy blur.
- Keep faint low-contrast specimen texture visible.
- Never turn specimen regions into empty smooth background.
""",
    "fluorescence_multichannel": """- Preserve the exact fluorescence color mapping and channel composition.
- Keep the black background and the same multi-channel visual balance.
""",
    "fluorescence_single_channel": """- Preserve the same fluorescence color family and black background.
- Keep the same single-channel dominant appearance and intensity style.
""",
}

NEGATIVE_PROMPT = """large structural change, different specimen count, moved objects, new region, missing region,
changed crop, changed zoom, changed magnification, changed annotation, missing scale bar, changed overlay,
false color mapping, inverted contrast, cartoon, illustration, painterly, sharp outlines, halo artifacts,
grid pattern, repeated pattern, mosaic artifact, over-sharpening, ultra detailed, non-microscopy texture,
empty background, blank field, removed cells, deleted organoid, missing specimen, smooth erased specimen region
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


def classify_style(image_path: Path) -> str:
    with Image.open(image_path) as image:
        rgb = ImageOps.contain(image.convert("RGB"), (128, 128))
        array = np.asarray(rgb, dtype=np.float32)

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


def proportional_quotas(group_sizes: dict[str, int], total_count: int) -> dict[str, int]:
    total_available = sum(group_sizes.values())
    if total_count > total_available:
        raise ValueError(f"Requested {total_count} files but only found {total_available}.")

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0

    for bucket, size in group_sizes.items():
        raw = total_count * size / total_available
        quota = min(size, math.floor(raw))
        quotas[bucket] = quota
        assigned += quota
        remainders.append((raw - quota, bucket))

    remaining = total_count - assigned
    for _, bucket in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        if quotas[bucket] >= group_sizes[bucket]:
            continue
        quotas[bucket] += 1
        remaining -= 1

    if sum(quotas.values()) != total_count:
        raise RuntimeError("Failed to allocate exact sample quotas.")
    return quotas


def build_manifest(split: str, source_dir: Path, sample_count: int, seed: int) -> dict[str, Any]:
    paths = list_images(source_dir)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[classify_style(path)].append(path)

    quotas = proportional_quotas({bucket: len(items) for bucket, items in grouped.items()}, sample_count)
    rng = random.Random(seed)
    sampled_entries: list[dict[str, Any]] = []

    for bucket, bucket_paths in grouped.items():
        chosen = rng.sample(bucket_paths, quotas[bucket])
        chosen.sort(key=lambda item: item.name)
        for path in chosen:
            sampled_entries.append(
                {
                    "name": path.name,
                    "source_path": str(path),
                    "style_bucket": bucket,
                    "seed": rng.randint(0, 2**31 - 1),
                }
            )

    sampled_entries.sort(key=lambda item: item["name"])
    return {
        "split": split,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "total_available": len(paths),
        "sample_count": len(sampled_entries),
        "bucket_counts_total": {bucket: len(bucket_paths) for bucket, bucket_paths in grouped.items()},
        "bucket_counts_sampled": dict(Counter(item["style_bucket"] for item in sampled_entries)),
        "files": sampled_entries,
    }


def get_split_manifest(
    split: str,
    source_dir: Path,
    sample_count: int,
    seed: int,
    manifest_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        existing = load_existing_manifest(manifest_path)
        if existing is not None:
            return existing
    manifest = build_manifest(split, source_dir, sample_count, seed)
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


def build_prompt(style_bucket: str) -> str:
    return BASE_EDIT_INSTRUCTION + "\n" + STYLE_PROMPTS[style_bucket]


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


def low_high_frequency_decompose(rgb: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.stack([gaussian_filter(rgb[:, :, channel], sigma=sigma) for channel in range(rgb.shape[2])], axis=2)
    high = rgb - low
    return low, high


def fuse_generated_with_source(
    source_image: Image.Image,
    generated_image: Image.Image,
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, float | bool]]:
    resized_generated = resize_like_source(source_image, generated_image)
    source_rgb = pil_to_float_rgb(source_image)
    generated_rgb = pil_to_float_rgb(resized_generated)
    generated_rgb = match_channel_statistics(source_rgb, generated_rgb)

    source_mask = detect_foreground_mask(source_image, args)
    generated_mask = detect_foreground_mask(float_rgb_to_pil(generated_rgb), args)

    source_low, source_high = low_high_frequency_decompose(source_rgb, sigma=args.detail_sigma)
    generated_low, _ = low_high_frequency_decompose(generated_rgb, sigma=args.detail_sigma)
    candidate = np.clip(generated_low + source_high, 0, 255)

    source_coverage = float(source_mask.mean())
    generated_coverage = float(generated_mask.mean())
    collapse_detected = source_coverage > 0.01 and generated_coverage < source_coverage * 0.45

    blend_strength = args.blend_strength
    if collapse_detected:
        blend_strength = min(blend_strength, 0.28)

    soft_mask = gaussian_filter(source_mask, sigma=max(1.0, min(source_mask.shape) / 70.0))
    alpha = np.clip(soft_mask[:, :, None] * blend_strength, 0.0, 1.0)
    fused = source_rgb * (1.0 - alpha) + candidate * alpha
    fused = np.where(source_mask[:, :, None] > 0, fused, source_rgb)

    metrics: dict[str, float | bool] = {
        "source_coverage": source_coverage,
        "generated_coverage": generated_coverage,
        "collapse_detected": collapse_detected,
        "blend_strength_used": float(blend_strength),
    }
    return float_rgb_to_pil(fused), metrics


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

        prompt = build_prompt(item["style_bucket"])

        try:
            with Image.open(source_path) as image:
                rgb = image.convert("RGB")
                inputs = {
                    "image": rgb,
                    "prompt": prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "generator": torch.Generator(device="cpu").manual_seed(int(item["seed"])),
                    "true_cfg_scale": args.true_cfg_scale,
                    "num_inference_steps": args.num_inference_steps,
                }
                with torch.inference_mode():
                    output = pipeline(**inputs)
                raw_image = output.images[0].convert("RGB")
                raw_image.save(raw_output_path)
                fused_image, fusion_metrics = fuse_generated_with_source(rgb, raw_image, args)
                fused_image.save(output_path)
                postprocess_metrics.append(
                    {
                        "split": split,
                        "name": item["name"],
                        "style_bucket": item["style_bucket"],
                        **fusion_metrics,
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
        sample_count=args.subset_train_count,
        seed=args.seed,
        manifest_path=records_dir / "train_manifest.json",
        overwrite=args.overwrite,
    )
    test_manifest = get_split_manifest(
        split="test",
        source_dir=test_source,
        sample_count=args.subset_test_count,
        seed=args.seed + 1,
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
        "subset_train_count": args.subset_train_count,
        "subset_test_count": args.subset_test_count,
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
