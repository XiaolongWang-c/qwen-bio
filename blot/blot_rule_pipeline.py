from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Band:
    y_center: float
    height: float
    width: float
    intensity: float
    blur: float


@dataclass
class Lane:
    x_center: float
    width: float
    bands: list[Band]


@dataclass
class Metadata:
    image_name: str
    split: str
    source_path: str
    image_size: list[int]
    polarity: str
    background: dict[str, float | str]
    blot_region: dict[str, int]
    lanes: list[Lane]
    artifacts: dict[str, float | bool]
    quality: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blot subset and generate rule-based synthetic variants."
    )
    parser.add_argument(
        "--blot-root",
        type=Path,
        default=Path("blot"),
        help="Root blot directory containing train/image and test/image.",
    )
    parser.add_argument(
        "--subset-train-count",
        type=int,
        default=2400,
        help="Number of train images to sample into the subset.",
    )
    parser.add_argument(
        "--subset-test-count",
        type=int,
        default=600,
        help="Number of test images to sample into the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260323,
        help="Random seed used for subset sampling and generation jitter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many sampled images to process per split for debugging.",
    )
    parser.add_argument(
        "--skip-subset-copy",
        action="store_true",
        help="Process the sampled manifest but do not copy images into subset directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite subset, metadata, sketch, generated, and records outputs if they already exist.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def sample_split(paths: list[Path], sample_count: int, seed: int) -> list[Path]:
    if sample_count > len(paths):
        raise ValueError(f"Requested {sample_count} images from {len(paths)} available files.")
    rng = random.Random(seed)
    sampled = rng.sample(paths, sample_count)
    sampled.sort(key=lambda path: path.name)
    return sampled


def write_manifest(
    manifest_path: Path,
    split: str,
    sampled_paths: list[Path],
    total_available: int,
    seed: int,
) -> None:
    payload = {
        "split": split,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_available": total_available,
        "sample_count": len(sampled_paths),
        "files": [path.name for path in sampled_paths],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_subset(sampled_paths: list[Path], destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for source in sampled_paths:
        shutil.copy2(source, destination_dir / source.name)


def to_gray(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        return np.array(gray, dtype=np.uint8)


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append((start, idx))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


def robust_mean(values: np.ndarray, lower_q: float = 10.0, upper_q: float = 90.0) -> float:
    if values.size == 0:
        return 0.0
    low, high = np.percentile(values, [lower_q, upper_q])
    trimmed = values[(values >= low) & (values <= high)]
    if trimmed.size == 0:
        return float(values.mean())
    return float(trimmed.mean())


def estimate_background(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    kernel_w = max(15, (width // 12) | 1)
    kernel_h = max(15, (height // 4) | 1)
    return cv2.GaussianBlur(gray, (kernel_w, kernel_h), 0)


def detect_polarity(gray: np.ndarray, background: np.ndarray) -> str:
    dark_response = background.astype(np.float32) - gray.astype(np.float32)
    light_response = gray.astype(np.float32) - background.astype(np.float32)
    dark_score = float(np.percentile(np.clip(dark_response, 0, None), 97))
    light_score = float(np.percentile(np.clip(light_response, 0, None), 97))
    return "dark_bands" if dark_score >= light_score else "light_bands"


def build_foreground_strength(gray: np.ndarray, background: np.ndarray, polarity: str) -> np.ndarray:
    if polarity == "dark_bands":
        response = background.astype(np.float32) - gray.astype(np.float32)
    else:
        response = gray.astype(np.float32) - background.astype(np.float32)
    return np.clip(response, 0, None)


def refine_background(gray: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    threshold = max(
        float(np.percentile(foreground, 80)),
        float(foreground.mean() + 0.35 * foreground.std()),
    )
    mask = foreground > threshold
    dilate_w = max(3, (gray.shape[1] // 80) | 1)
    dilate_h = max(3, (gray.shape[0] // 10) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_w, dilate_h))
    mask_uint8 = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1)
    inpainted = cv2.inpaint(gray, mask_uint8, 3, cv2.INPAINT_TELEA)
    return cv2.GaussianBlur(inpainted, (0, 0), sigmaX=1.2, sigmaY=0.8)


def detect_blot_region(foreground: np.ndarray) -> dict[str, int]:
    col_signal = gaussian_filter1d(foreground.mean(axis=0), sigma=3)
    row_signal = gaussian_filter1d(foreground.mean(axis=1), sigma=2)

    col_threshold = float(np.percentile(col_signal, 50))
    row_threshold = float(np.percentile(row_signal, 50))

    col_mask = col_signal > col_threshold
    row_mask = row_signal > row_threshold

    col_regions = contiguous_regions(col_mask)
    row_regions = contiguous_regions(row_mask)

    width = foreground.shape[1]
    height = foreground.shape[0]

    if col_regions:
        min_col_width = max(3, int(width * 0.01))
        col_regions = [region for region in col_regions if region[1] - region[0] >= min_col_width] or col_regions
        x0 = min(start for start, _ in col_regions)
        x1 = max(end for _, end in col_regions)
    else:
        x0, x1 = 0, width

    if row_regions:
        min_row_height = max(2, int(height * 0.05))
        row_regions = [region for region in row_regions if region[1] - region[0] >= min_row_height] or row_regions
        y0 = min(start for start, _ in row_regions)
        y1 = max(end for _, end in row_regions)
    else:
        y0, y1 = 0, height

    x_pad = max(2, int(width * 0.01))
    y_pad = max(1, int(height * 0.03))
    return {
        "x0": max(0, x0 - x_pad),
        "x1": min(width, x1 + x_pad),
        "y0": max(0, y0 - y_pad),
        "y1": min(height, y1 + y_pad),
    }


def detect_lanes(foreground: np.ndarray, blot_region: dict[str, int]) -> list[tuple[int, int]]:
    x0, x1 = blot_region["x0"], blot_region["x1"]
    y0, y1 = blot_region["y0"], blot_region["y1"]
    roi = foreground[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    column_profile = gaussian_filter1d(roi.mean(axis=0), sigma=2)
    min_lane_width = max(6, int(roi.shape[1] * 0.025))
    threshold = float(np.percentile(column_profile, 48))
    mask = column_profile > threshold
    close_width = max(5, int(roi.shape[1] * 0.025))
    closed = cv2.morphologyEx(
        (mask[np.newaxis, :].astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1)),
    )[0] > 0
    regions = [(x0 + start, x0 + end) for start, end in contiguous_regions(closed) if end - start >= min_lane_width]
    if regions:
        return merge_lane_regions(regions)

    min_distance = max(10, int(roi.shape[1] * 0.08))
    prominence = max(1.2, float(column_profile.std() * 0.14))
    height = float(np.percentile(column_profile, 58))
    peaks, _ = find_peaks(column_profile, distance=min_distance, prominence=prominence, height=height)
    if len(peaks) > 0:
        lanes = peak_regions_from_profile(peaks, column_profile, min_lane_width)
        lanes = [(x0 + start, x0 + end) for start, end in lanes]
        return merge_lane_regions(lanes)

    fallback_lane_count = max(1, min(8, roi.shape[1] // max(20, roi.shape[0] // 2 + 1)))
    fallback_width = max(roi.shape[1] // max(fallback_lane_count * 2, 1), 6)
    spacing = roi.shape[1] / max(fallback_lane_count + 1, 1)
    fallback_lanes = []
    for idx in range(fallback_lane_count):
        center = int((idx + 1) * spacing)
        start = max(0, center - fallback_width // 2)
        end = min(roi.shape[1], center + fallback_width // 2)
        fallback_lanes.append((x0 + start, x0 + end))
    return fallback_lanes


def merge_close_regions(regions: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def merge_lane_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        prev_width = prev_end - prev_start
        curr_width = end - start
        gap = start - prev_end
        merge_gap = max(3, int(min(prev_width, curr_width) * 0.6))
        if gap <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def peak_regions_from_profile(
    peaks: np.ndarray,
    profile: np.ndarray,
    min_width: int,
) -> list[tuple[int, int]]:
    if len(peaks) == 0:
        return []

    regions: list[tuple[int, int]] = []
    peaks = np.sort(peaks)
    if len(peaks) > 1:
        spacing = np.diff(peaks)
        default_half_width = max(min_width // 2, int(np.median(spacing) * 0.35))
    else:
        default_half_width = max(min_width // 2, int(len(profile) * 0.04))

    baseline = float(np.percentile(profile, 35))
    for idx, peak in enumerate(peaks):
        peak_value = float(profile[peak])
        local_floor = baseline + 0.35 * max(0.0, peak_value - baseline)
        left = int(peak)
        right = int(peak)

        while left > 0 and profile[left] > local_floor:
            left -= 1
        while right < len(profile) - 1 and profile[right] > local_floor:
            right += 1

        if idx > 0:
            left = max(left, int((peaks[idx - 1] + peak) / 2 - default_half_width * 0.15))
        if idx < len(peaks) - 1:
            right = min(right, int((peaks[idx + 1] + peak) / 2 + default_half_width * 0.15))

        if right - left < min_width:
            left = max(0, int(peak - default_half_width))
            right = min(len(profile), int(peak + default_half_width))

        regions.append((max(0, left), min(len(profile), right)))

    return regions


def detect_bands(
    foreground: np.ndarray,
    lane_regions: list[tuple[int, int]],
    blot_region: dict[str, int],
) -> list[Lane]:
    lanes: list[Lane] = []
    y0, y1 = blot_region["y0"], blot_region["y1"]
    blot_height = max(1, y1 - y0)

    for lane_start, lane_end in lane_regions:
        lane_roi = foreground[y0:y1, lane_start:lane_end]
        if lane_roi.size == 0:
            continue

        row_profile = gaussian_filter1d(lane_roi.mean(axis=1), sigma=1.2)
        min_band_height = max(2, int(blot_height * 0.02))
        min_distance = max(3, int(blot_height * 0.08))
        prominence = max(1.0, float(row_profile.std() * 0.18))
        height_threshold = float(np.percentile(row_profile, 55))
        peaks, _ = find_peaks(row_profile, distance=min_distance, prominence=prominence, height=height_threshold)

        bands: list[Band] = []
        band_regions = peak_regions_from_profile(peaks, row_profile, min_band_height)
        if not band_regions:
            threshold = float(np.percentile(row_profile, 65))
            mask = row_profile > threshold
            band_regions = contiguous_regions(mask)

        for band_start, band_end in band_regions:
            band_height = int(band_end - band_start)
            if band_height < min_band_height:
                continue

            band_roi = lane_roi[band_start:band_end, :]
            band_profile = gaussian_filter1d(band_roi.mean(axis=0), sigma=1)
            band_threshold = float(np.percentile(band_profile, 42))
            band_mask = band_profile > band_threshold
            band_regions = contiguous_regions(band_mask)
            if band_regions:
                band_x0, band_x1 = max(band_regions, key=lambda item: item[1] - item[0])
                width = max(float(band_x1 - band_x0), float(lane_end - lane_start) * 0.72)
            else:
                width = max(2.0, float(lane_end - lane_start) * 0.85)

            intensity = float(np.clip(np.percentile(band_roi, 95) / 255.0, 0.1, 1.0))
            blur = float(np.clip(band_height / 3.5, 0.8, 3.5))
            bands.append(
                Band(
                    y_center=float(y0 + (band_start + band_end) / 2.0),
                    height=float(band_height),
                    width=width,
                    intensity=intensity,
                    blur=blur,
                )
            )

        edge_margin = max(2.0, blot_height * 0.1)
        filtered_bands = [
            band
            for band in bands
            if (y0 + edge_margin) <= band.y_center <= (y1 - edge_margin)
        ]
        if filtered_bands:
            bands = filtered_bands

        if not bands:
            inferred_center = float((y0 + y1) / 2.0)
            bands.append(
                Band(
                    y_center=inferred_center,
                    height=float(max(3, blot_height * 0.06)),
                    width=float(max(4, lane_end - lane_start - 2)),
                    intensity=0.55,
                    blur=1.5,
                )
            )

        lanes.append(
            Lane(
                x_center=float((lane_start + lane_end) / 2.0),
                width=float(lane_end - lane_start),
                bands=bands,
            )
        )

    return lanes


def estimate_artifacts(gray: np.ndarray, background: np.ndarray, foreground: np.ndarray) -> dict[str, float | bool]:
    residual = gray.astype(np.float32) - background.astype(np.float32)
    noise_level = float(np.std(residual) / 255.0)
    smear_score = float(np.percentile(gaussian_filter1d(foreground.mean(axis=0), sigma=4), 85))
    return {
        "smear": bool(smear_score > foreground.mean() * 1.5),
        "noise_level": noise_level,
        "compression_like": bool(noise_level > 0.08),
    }


def metadata_to_dict(metadata: Metadata) -> dict[str, Any]:
    payload = asdict(metadata)
    return payload


def extract_metadata(image_path: Path, split: str) -> tuple[Metadata, np.ndarray, np.ndarray, np.ndarray]:
    gray = to_gray(image_path)
    coarse_background = estimate_background(gray)
    polarity = detect_polarity(gray, coarse_background)
    coarse_foreground = build_foreground_strength(gray, coarse_background, polarity)
    background = refine_background(gray, coarse_foreground)
    foreground = build_foreground_strength(gray, background, polarity)
    blot_region = detect_blot_region(foreground)
    lane_regions = detect_lanes(foreground, blot_region)
    lanes = detect_bands(foreground, lane_regions, blot_region)
    artifacts = estimate_artifacts(gray, background, foreground)

    residual = gray.astype(np.float32) - background.astype(np.float32)
    background_mask = foreground < np.percentile(foreground, 60)
    background_pixels = gray[background_mask]
    quality = {
        "lane_count": len(lanes),
        "band_count": int(sum(len(lane.bands) for lane in lanes)),
        "foreground_mean": float(foreground.mean()),
        "foreground_std": float(foreground.std()),
    }
    metadata = Metadata(
        image_name=image_path.name,
        split=split,
        source_path=str(image_path),
        image_size=[int(gray.shape[1]), int(gray.shape[0])],
        polarity=polarity,
        background={
            "mean": robust_mean(background_pixels.astype(np.float32)),
            "std": float(background_pixels.std()) if background_pixels.size else float(gray.std()),
            "gradient": classify_gradient(background),
            "texture_level": float(np.std(residual) / 255.0),
        },
        blot_region=blot_region,
        lanes=lanes,
        artifacts=artifacts,
        quality=quality,
    )
    return metadata, gray, background, foreground


def classify_gradient(background: np.ndarray) -> str:
    left = float(background[:, : max(1, background.shape[1] // 5)].mean())
    right = float(background[:, -max(1, background.shape[1] // 5) :].mean())
    top = float(background[: max(1, background.shape[0] // 5), :].mean())
    bottom = float(background[-max(1, background.shape[0] // 5) :, :].mean())
    magnitude = max(abs(left - right), abs(top - bottom))
    if magnitude < 5:
        return "weak"
    if magnitude < 15:
        return "moderate"
    return "strong"


def assign_rows_to_lane(
    bands: list[Band],
    row_centers: list[float],
) -> dict[int, Band]:
    assignments: dict[int, Band] = {}
    if not row_centers:
        return assignments
    for band in bands:
        distances = [abs(band.y_center - center) for center in row_centers]
        row_idx = int(np.argmin(distances))
        current = assignments.get(row_idx)
        if current is None or abs(current.y_center - row_centers[row_idx]) > abs(band.y_center - row_centers[row_idx]):
            assignments[row_idx] = band
    return assignments


def cluster_band_rows(metadata: Metadata) -> list[dict[str, Any]]:
    all_bands = []
    for lane_idx, lane in enumerate(sorted(metadata.lanes, key=lambda item: item.x_center)):
        for band in lane.bands:
            all_bands.append((lane_idx, band))

    if not all_bands:
        return []

    all_bands.sort(key=lambda item: item[1].y_center)
    cluster_threshold = max(3.0, metadata.image_size[1] * 0.035)
    clusters: list[list[tuple[int, Band]]] = []
    for item in all_bands:
        if not clusters:
            clusters.append([item])
            continue
        cluster_center = float(np.mean([band.y_center for _, band in clusters[-1]]))
        if abs(item[1].y_center - cluster_center) <= cluster_threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        centers = [band.y_center for _, band in cluster]
        heights = [band.height for _, band in cluster]
        widths = [band.width for _, band in cluster]
        intensities = [band.intensity for _, band in cluster]
        blurs = [band.blur for _, band in cluster]
        lane_indices = sorted({lane_idx for lane_idx, _ in cluster})
        rows.append(
            {
                "y_center": float(np.mean(centers)),
                "height": float(np.mean(heights)),
                "width": float(np.mean(widths)),
                "intensity": float(np.mean(intensities)),
                "blur": float(np.mean(blurs)),
                "lane_indices": lane_indices,
            }
        )
    return rows


def build_band_plan(metadata: Metadata, rng: random.Random) -> list[dict[str, Any]]:
    sorted_lanes = sorted(metadata.lanes, key=lambda item: item.x_center)
    row_templates = cluster_band_rows(metadata)
    row_centers = [row["y_center"] for row in row_templates]
    lane_assignments = [assign_rows_to_lane(lane.bands, row_centers) for lane in sorted_lanes]

    lane_layouts = []
    for lane in sorted_lanes:
        lane_layouts.append(
            {
                "x_center": float(np.clip(lane.x_center + rng.uniform(-0.015, 0.015) * max(8.0, lane.width), 1, metadata.image_size[0] - 2)),
                "width": float(np.clip(lane.width * rng.uniform(0.96, 1.04), 4.0, metadata.image_size[0] * 0.25)),
            }
        )

    blot_y0 = metadata.blot_region["y0"]
    blot_y1 = metadata.blot_region["y1"]
    band_plan: list[dict[str, Any]] = []

    for row_idx, row in enumerate(row_templates):
        row_shift = rng.uniform(-1.2, 1.2) * max(1.0, row["height"] * 0.18)
        target_y = float(np.clip(row["y_center"] + row_shift, blot_y0 + 1, blot_y1 - 1))
        for lane_idx, lane in enumerate(sorted_lanes):
            reference_band = lane_assignments[lane_idx].get(row_idx)
            if reference_band is None:
                continue

            lane_layout = lane_layouts[lane_idx]
            width_ratio = reference_band.width / max(1.0, lane.width)
            new_width = lane_layout["width"] * width_ratio * rng.uniform(0.94, 1.06)
            new_width = float(np.clip(new_width, lane_layout["width"] * 0.72, lane_layout["width"] * 1.02))
            new_height = float(np.clip(reference_band.height * rng.uniform(0.92, 1.12), 2.0, metadata.image_size[1] * 0.18))
            local_shift = rng.uniform(-1.0, 1.0) * max(0.6, reference_band.height * 0.12)
            y_center = float(np.clip(target_y + local_shift, blot_y0 + 1, blot_y1 - 1))
            intensity = float(np.clip(reference_band.intensity * rng.uniform(0.88, 1.12), 0.12, 1.0))
            blur = float(np.clip(reference_band.blur * rng.uniform(0.92, 1.08), 0.8, 4.5))

            band_plan.append(
                {
                    "lane_idx": lane_idx,
                    "x_center": lane_layout["x_center"],
                    "lane_width": lane_layout["width"],
                    "y_center": y_center,
                    "width": new_width,
                    "height": new_height,
                    "intensity": intensity,
                    "blur": blur,
                    "row_idx": row_idx,
                }
            )

    return band_plan


def generate_sketch(
    metadata: Metadata,
    background: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    width, height = metadata.image_size
    base_value = int(np.clip(metadata.background["mean"], 20, 235))
    sketch = np.full((height, width), base_value, dtype=np.uint8)

    band_specs = build_band_plan(metadata, rng)
    polarity = metadata.polarity
    lane_fill_delta = -10 if polarity == "dark_bands" else 10

    lane_layouts: dict[int, tuple[float, float]] = {}
    for spec in band_specs:
        lane_layouts[spec["lane_idx"]] = (float(spec["x_center"]), float(spec["lane_width"]))

    for lane_idx, (lane_center, lane_width) in lane_layouts.items():
        lane_x0 = int(np.clip(lane_center - lane_width / 2.0, 0, width - 1))
        lane_x1 = int(np.clip(lane_center + lane_width / 2.0, lane_x0 + 1, width))
        sketch[:, lane_x0:lane_x1] = np.clip(sketch[:, lane_x0:lane_x1].astype(np.int16) + lane_fill_delta, 0, 255)

    for spec in band_specs:
        band_y0 = int(np.clip(round(spec["y_center"] - spec["height"] / 2.0), 0, height - 1))
        band_y1 = int(np.clip(round(spec["y_center"] + spec["height"] / 2.0), band_y0 + 1, height))
        band_x0 = int(np.clip(round(spec["x_center"] - spec["width"] / 2.0), 0, width - 1))
        band_x1 = int(np.clip(round(spec["x_center"] + spec["width"] / 2.0), band_x0 + 1, width))
        if polarity == "dark_bands":
            band_value = int(np.clip(base_value - 80 * spec["intensity"], 0, 255))
        else:
            band_value = int(np.clip(base_value + 80 * spec["intensity"], 0, 255))
        center = ((band_x0 + band_x1) // 2, (band_y0 + band_y1) // 2)
        axes = (max(1, (band_x1 - band_x0) // 2), max(1, (band_y1 - band_y0) // 2))
        cv2.ellipse(sketch, center, axes, 0, 0, 360, color=band_value, thickness=-1)
        spec["x0"] = band_x0
        spec["x1"] = band_x1
        spec["y0"] = band_y0
        spec["y1"] = band_y1
        spec["value"] = band_value

    if band_specs:
        sigma = max(0.8, float(np.mean([spec["blur"] for spec in band_specs]) * 0.55))
        sketch = cv2.GaussianBlur(sketch, (0, 0), sigmaX=sigma, sigmaY=max(0.8, sigma * 0.7))

    return sketch, band_specs


def make_soft_band_mask(
    patch_h: int,
    patch_w: int,
    blur_sigma: float,
    rng: random.Random,
) -> np.ndarray:
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    center = (patch_w // 2, patch_h // 2)
    axis_x = max(1, int(patch_w * rng.uniform(0.34, 0.46)))
    axis_y = max(1, int(patch_h * rng.uniform(0.28, 0.38)))
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, color=1.0, thickness=-1)

    yy, xx = np.mgrid[0:patch_h, 0:patch_w].astype(np.float32)
    center_x = (patch_w - 1) / 2.0
    center_y = (patch_h - 1) / 2.0
    ridge = np.exp(
        -(
            ((yy - center_y) ** 2) / (2.0 * max(1.0, (patch_h * 0.22) ** 2))
            + ((xx - center_x) ** 2) / (2.0 * max(1.0, (patch_w * 0.4) ** 2))
        )
    )
    mask = np.maximum(mask, ridge.astype(np.float32))

    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(1.0, 0.08, size=(patch_h, patch_w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(0.8, patch_w / 18.0), sigmaY=max(0.8, patch_h / 8.0))
    mask *= np.clip(noise, 0.82, 1.18)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.2, blur_sigma), sigmaY=max(0.8, blur_sigma * 0.7))
    return np.clip(mask, 0.0, 1.0)


def make_band_texture(
    patch_h: int,
    patch_w: int,
    local_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    yy, xx = np.mgrid[0:patch_h, 0:patch_w].astype(np.float32)
    center_y = (patch_h - 1) / 2.0
    center_x = (patch_w - 1) / 2.0

    vertical_profile = np.exp(-((yy - center_y) ** 2) / (2.0 * max(1.0, (patch_h * 0.28) ** 2)))
    horizontal_profile = np.exp(-((xx - center_x) ** 2) / (2.0 * max(1.0, (patch_w * 0.55) ** 2)))

    low_freq = np.random.default_rng(rng.randint(0, 10**9)).normal(1.0, 0.10, size=(patch_h, patch_w)).astype(np.float32)
    low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=max(1.4, patch_w / 12.0), sigmaY=max(0.8, patch_h / 4.0))

    row_jitter = np.random.default_rng(rng.randint(0, 10**9)).normal(1.0, 0.06, size=(patch_h, 1)).astype(np.float32)
    row_jitter = cv2.GaussianBlur(row_jitter, (0, 0), sigmaX=0.1, sigmaY=max(0.8, patch_h / 6.0))

    texture = vertical_profile * horizontal_profile
    texture *= low_freq
    texture *= row_jitter
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=max(1.0, patch_w / 20.0), sigmaY=max(0.8, patch_h / 6.0))
    texture *= local_mask
    return np.clip(texture, 0.0, 1.0)


def render_band_field(
    image_shape: tuple[int, int],
    band_specs: list[dict[str, Any]],
    polarity: str,
    rng: random.Random,
) -> np.ndarray:
    height, width = image_shape
    field = np.zeros((height, width), dtype=np.float32)

    for spec in band_specs:
        sigma_x = max(2.0, spec["width"] / 2.8)
        sigma_y = max(1.2, spec["height"] / 2.4)
        pad_x = int(math.ceil(sigma_x * 3.5))
        pad_y = int(math.ceil(sigma_y * 3.0))

        x0 = max(0, int(round(spec["x_center"])) - pad_x)
        x1 = min(width, int(round(spec["x_center"])) + pad_x + 1)
        y0 = max(0, int(round(spec["y_center"])) - pad_y)
        y1 = min(height, int(round(spec["y_center"])) + pad_y + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - float(spec["x_center"])
        dy = yy - float(spec["y_center"])

        gaussian_core = np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
        shoulder = np.exp(-0.5 * ((dx / max(2.0, sigma_x * 1.22)) ** 2 + (dy / max(1.2, sigma_y * 0.92)) ** 2))
        band = 0.78 * gaussian_core + 0.32 * shoulder

        waviness = 1.0 + 0.06 * np.sin(dx / max(2.0, sigma_x * 0.9) + rng.uniform(0, np.pi))
        waviness += 0.04 * np.sin(dy / max(1.2, sigma_y * 0.8) + rng.uniform(0, np.pi))
        waviness = np.clip(waviness, 0.88, 1.14)

        texture_noise = np.random.default_rng(rng.randint(0, 10**9)).normal(1.0, 0.05, size=band.shape).astype(np.float32)
        texture_noise = cv2.GaussianBlur(texture_noise, (0, 0), sigmaX=max(0.8, sigma_x / 2.2), sigmaY=max(0.8, sigma_y / 1.8))
        band *= waviness * np.clip(texture_noise, 0.9, 1.1)

        amplitude = (28.0 + 95.0 * float(spec["intensity"])) * (1.0 if polarity == "light_bands" else -1.0)
        field[y0:y1, x0:x1] += amplitude * band.astype(np.float32)

    return field


def synthesize_image(
    gray: np.ndarray,
    background: np.ndarray,
    metadata: Metadata,
    sketch: np.ndarray,
    band_specs: list[dict[str, Any]],
    rng: random.Random,
) -> np.ndarray:
    generated = background.astype(np.float32).copy()

    low_freq_noise = cv2.GaussianBlur(
        np.random.default_rng(rng.randint(0, 10**9)).normal(0, 1, size=gray.shape).astype(np.float32),
        (0, 0),
        sigmaX=max(3.0, gray.shape[1] / 40.0),
        sigmaY=max(2.0, gray.shape[0] / 6.0),
    )
    noise_scale = max(2.0, metadata.artifacts["noise_level"] * 60.0)
    generated += low_freq_noise * noise_scale

    if band_specs:
        band_field = render_band_field(gray.shape, band_specs, metadata.polarity, rng)
        generated += band_field

    sketch_residual = sketch.astype(np.float32) - float(metadata.background["mean"])
    generated += sketch_residual * 0.035

    grain = np.random.default_rng(rng.randint(0, 10**9)).normal(
        loc=0.0,
        scale=max(0.5, metadata.artifacts["noise_level"] * 18.0),
        size=gray.shape,
    ).astype(np.float32)
    generated += grain

    if metadata.artifacts["smear"] and band_specs:
        smear_axis = rng.choice(["horizontal", "vertical"])
        blur_size = (9, 1) if smear_axis == "horizontal" else (1, 7)
        smear = cv2.GaussianBlur(generated, blur_size, sigmaX=0, sigmaY=0)
        generated = 0.88 * generated + 0.12 * smear

    generated = np.clip(generated, 0, 255).astype(np.uint8)
    return generated


def validate_generated(image: np.ndarray, metadata: Metadata) -> dict[str, Any]:
    quality = {
        "mean": float(image.mean()),
        "std": float(image.std()),
        "min": int(image.min()),
        "max": int(image.max()),
        "blank_like": bool(image.std() < 4.0),
        "too_dark": bool(image.mean() < 20),
        "too_bright": bool(image.mean() > 245),
        "lane_count": len(metadata.lanes),
        "band_count": int(sum(len(lane.bands) for lane in metadata.lanes)),
    }
    quality["valid"] = not (
        quality["blank_like"] or quality["too_dark"] or quality["too_bright"] or quality["band_count"] == 0
    )
    return quality


def save_gray_image(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(array, mode="L").convert("RGB")
    image.save(path)


def per_image_rng(seed: int, split: str, image_name: str) -> random.Random:
    composite = f"{seed}:{split}:{image_name}"
    value = 0
    for char in composite:
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return random.Random(value)


def process_split(
    split: str,
    sampled_paths: list[Path],
    subset_dir: Path,
    metadata_dir: Path,
    sketch_dir: Path,
    generated_dir: Path,
    seed: int,
    skip_subset_copy: bool,
) -> dict[str, Any]:
    ensure_dir(metadata_dir)
    ensure_dir(sketch_dir)
    ensure_dir(generated_dir)
    if not skip_subset_copy:
        ensure_dir(subset_dir)

    summary = {
        "split": split,
        "processed": 0,
        "valid_generated": 0,
        "invalid_generated": 0,
        "lane_counts": [],
        "band_counts": [],
        "image_names": [],
    }

    for image_path in sampled_paths:
        if not skip_subset_copy:
            shutil.copy2(image_path, subset_dir / image_path.name)

        metadata, gray, background, _ = extract_metadata(image_path, split)
        rng = per_image_rng(seed, split, image_path.name)
        sketch, band_specs = generate_sketch(metadata, background, rng)
        generated = synthesize_image(gray, background, metadata, sketch, band_specs, rng)
        generated_quality = validate_generated(generated, metadata)

        metadata.quality["generated"] = generated_quality

        metadata_path = metadata_dir / f"{image_path.stem}.json"
        sketch_path = sketch_dir / image_path.name
        generated_path = generated_dir / image_path.name

        metadata_path.write_text(
            json.dumps(metadata_to_dict(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        save_gray_image(sketch, sketch_path)
        save_gray_image(generated, generated_path)

        summary["processed"] += 1
        summary["valid_generated"] += int(generated_quality["valid"])
        summary["invalid_generated"] += int(not generated_quality["valid"])
        summary["lane_counts"].append(len(metadata.lanes))
        summary["band_counts"].append(int(sum(len(lane.bands) for lane in metadata.lanes)))
        summary["image_names"].append(image_path.name)

    return summary


def aggregate_summary(train_summary: dict[str, Any], test_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    all_lane_counts = train_summary["lane_counts"] + test_summary["lane_counts"]
    all_band_counts = train_summary["band_counts"] + test_summary["band_counts"]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "subset_train_count": args.subset_train_count,
        "subset_test_count": args.subset_test_count,
        "limit": args.limit,
        "train": summarize_counts(train_summary),
        "test": summarize_counts(test_summary),
        "overall": {
            "processed": train_summary["processed"] + test_summary["processed"],
            "valid_generated": train_summary["valid_generated"] + test_summary["valid_generated"],
            "invalid_generated": train_summary["invalid_generated"] + test_summary["invalid_generated"],
            "lane_count_mean": float(np.mean(all_lane_counts)) if all_lane_counts else 0.0,
            "band_count_mean": float(np.mean(all_band_counts)) if all_band_counts else 0.0,
        },
    }


def summarize_counts(summary: dict[str, Any]) -> dict[str, Any]:
    lane_counts = summary["lane_counts"]
    band_counts = summary["band_counts"]
    return {
        "processed": summary["processed"],
        "valid_generated": summary["valid_generated"],
        "invalid_generated": summary["invalid_generated"],
        "lane_count_mean": float(np.mean(lane_counts)) if lane_counts else 0.0,
        "lane_count_min": int(np.min(lane_counts)) if lane_counts else 0,
        "lane_count_max": int(np.max(lane_counts)) if lane_counts else 0,
        "band_count_mean": float(np.mean(band_counts)) if band_counts else 0.0,
        "band_count_min": int(np.min(band_counts)) if band_counts else 0,
        "band_count_max": int(np.max(band_counts)) if band_counts else 0,
    }


def prepare_output_dirs(blot_root: Path, overwrite: bool) -> dict[str, Path]:
    outputs = {
        "subset_train": blot_root / "subset" / "train" / "image",
        "subset_test": blot_root / "subset" / "test" / "image",
        "metadata_train": blot_root / "metadata" / "train",
        "metadata_test": blot_root / "metadata" / "test",
        "sketch_train": blot_root / "sketch" / "train",
        "sketch_test": blot_root / "sketch" / "test",
        "generated_train": blot_root / "generated" / "train",
        "generated_test": blot_root / "generated" / "test",
        "records": blot_root / "split_records",
    }
    if overwrite:
        for key, path in outputs.items():
            if key == "records":
                reset_dir(path)
            else:
                reset_dir(path)
    else:
        for path in outputs.values():
            ensure_dir(path)
    return outputs


def main() -> None:
    args = parse_args()
    blot_root = args.blot_root
    train_source = blot_root / "train" / "image"
    test_source = blot_root / "test" / "image"

    if not train_source.exists() or not test_source.exists():
        raise FileNotFoundError(f"Expected {train_source} and {test_source} to exist.")

    train_paths = list_images(train_source)
    test_paths = list_images(test_source)

    sampled_train = sample_split(train_paths, args.subset_train_count, args.seed)
    sampled_test = sample_split(test_paths, args.subset_test_count, args.seed + 1)

    if args.limit is not None:
        sampled_train = sampled_train[: min(args.limit, len(sampled_train))]
        sampled_test = sampled_test[: min(args.limit, len(sampled_test))]

    outputs = prepare_output_dirs(blot_root, args.overwrite)

    write_manifest(outputs["records"] / "train_manifest.json", "train", sampled_train, len(train_paths), args.seed)
    write_manifest(outputs["records"] / "test_manifest.json", "test", sampled_test, len(test_paths), args.seed + 1)

    train_summary = process_split(
        split="train",
        sampled_paths=sampled_train,
        subset_dir=outputs["subset_train"],
        metadata_dir=outputs["metadata_train"],
        sketch_dir=outputs["sketch_train"],
        generated_dir=outputs["generated_train"],
        seed=args.seed,
        skip_subset_copy=args.skip_subset_copy,
    )
    test_summary = process_split(
        split="test",
        sampled_paths=sampled_test,
        subset_dir=outputs["subset_test"],
        metadata_dir=outputs["metadata_test"],
        sketch_dir=outputs["sketch_test"],
        generated_dir=outputs["generated_test"],
        seed=args.seed + 1,
        skip_subset_copy=args.skip_subset_copy,
    )

    summary = aggregate_summary(train_summary, test_summary, args)
    summary_path = outputs["records"] / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
