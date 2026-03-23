from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.signal import find_peaks, peak_widths


@dataclass
class ReferenceStats:
    image_name: str
    split: str
    source_path: str
    width: int
    height: int
    lane_positions: list[int]
    lane_count: int
    lane_width: int
    lane_spacing: float
    band_positions: list[list[int]]
    band_mean: float
    band_y_pool: list[int]
    intensity_mean: float
    thickness_mean: float
    thickness_std: float
    thin_band_ratio: float
    background_mean: float
    background_noise: float
    blur_sigma: float
    sharpness_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a temp-style small batch blot generation test.")
    parser.add_argument("--blot-root", type=Path, default=Path("blot"))
    parser.add_argument("--refs-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--output-root", type=Path, default=Path("blot/temp_style_test"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def extract_reference_parameters(image_path: Path, split: str) -> ReferenceStats:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {image_path}")

    height, width = img.shape
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_norm = img_blur.astype(np.float32) / 255.0

    vertical_profile = img_norm.mean(axis=0)
    peak_distance = max(12, width // 12)
    peaks, _ = find_peaks(-vertical_profile, distance=peak_distance, prominence=0.008)
    lane_positions = peaks.tolist()
    if len(lane_positions) < 2:
        fallback_count = 4
        lane_positions = np.linspace(width * 0.12, width * 0.88, fallback_count).astype(int).tolist()

    lane_spacing_values = [lane_positions[i + 1] - lane_positions[i] for i in range(len(lane_positions) - 1)]
    lane_spacing = float(np.mean(lane_spacing_values)) if lane_spacing_values else max(width / 5.0, 20.0)
    lane_width = max(8, int(lane_spacing * 0.6))

    band_positions: list[list[int]] = []
    band_intensity: list[float] = []
    band_thickness: list[float] = []
    band_y_pool: list[int] = []

    for x in lane_positions:
        x1 = max(0, x - lane_width // 2)
        x2 = min(width, x + lane_width // 2)
        lane = img_norm[:, x1:x2]
        if lane.size == 0:
            band_positions.append([])
            continue

        profile = lane.mean(axis=1)
        row_distance = max(6, height // 10)
        band_peaks, props = find_peaks(-profile, distance=row_distance, prominence=0.015)
        widths = peak_widths(-profile, band_peaks, rel_height=0.6)[0] if len(band_peaks) > 0 else []
        lane_bands: list[int] = []

        for idx, y in enumerate(band_peaks):
            y1 = max(0, y - 4)
            y2 = min(height, y + 4)
            local_mean = float(lane[y1:y2].mean()) if y2 > y1 else float(profile[y])
            prominence = float(props["prominences"][idx]) if "prominences" in props else 0.03
            lane_bands.append(int(y))
            band_y_pool.append(int(y))
            band_intensity.append(float(np.clip(np.quantile(img_norm, 0.85) - local_mean, 0.06, 0.55)))
            estimated_thickness = float(widths[idx]) if idx < len(widths) else (3.5 + prominence * 90.0)
            band_thickness.append(float(np.clip(estimated_thickness, 2.0, 12.0)))

        band_positions.append(lane_bands)

    bright_pixels = img_norm[img_norm >= np.quantile(img_norm, 0.55)]
    background_mean = float(bright_pixels.mean()) if bright_pixels.size else float(img_norm.mean())
    background_noise = float(np.std(img_norm - cv2.GaussianBlur(img_norm, (0, 0), 3)))
    laplacian_var = float(cv2.Laplacian(img_blur, cv2.CV_32F).var())
    sharpness_score = float(np.clip(laplacian_var / 120.0, 0.2, 3.0))
    blur_sigma = float(np.clip(1.55 - 0.28 * sharpness_score + background_noise * 6.0, 0.7, 1.45))
    band_mean = float(np.mean([len(bands) for bands in band_positions])) if band_positions else 2.0
    intensity_mean = float(np.mean(band_intensity)) if band_intensity else 0.22
    thickness_mean = float(np.mean(band_thickness)) if band_thickness else max(3.5, height * 0.06)
    thickness_std = float(np.std(band_thickness)) if len(band_thickness) > 1 else 1.0
    thin_band_ratio = (
        float(sum(value <= max(3.5, thickness_mean * 0.75) for value in band_thickness)) / len(band_thickness)
        if band_thickness
        else 0.35
    )

    return ReferenceStats(
        image_name=image_path.name,
        split=split,
        source_path=str(image_path),
        width=width,
        height=height,
        lane_positions=lane_positions,
        lane_count=len(lane_positions),
        lane_width=lane_width,
        lane_spacing=lane_spacing,
        band_positions=band_positions,
        band_mean=band_mean,
        band_y_pool=sorted(set(band_y_pool)),
        intensity_mean=intensity_mean,
        thickness_mean=thickness_mean,
        thickness_std=thickness_std,
        thin_band_ratio=thin_band_ratio,
        background_mean=background_mean,
        background_noise=background_noise,
        blur_sigma=blur_sigma,
        sharpness_score=sharpness_score,
    )


def sample_parameters(stats: ReferenceStats, rng: random.Random) -> dict[str, float | int | list[int]]:
    lane_count = max(2, int(round(rng.gauss(stats.lane_count, 0.6))))
    lane_width = max(8, int(round(rng.gauss(stats.lane_width, 1.5))))
    lane_spacing = max(lane_width + 2, int(round(rng.gauss(stats.lane_spacing, 2.0))))
    bands_per_lane = [max(1, int(round(rng.gauss(stats.band_mean, 0.7)))) for _ in range(lane_count)]

    return {
        "width": stats.width,
        "height": stats.height,
        "lanes": lane_count,
        "lane_width": lane_width,
        "lane_spacing": lane_spacing,
        "bands_per_lane": bands_per_lane,
        "intensity_mean": float(np.clip(rng.gauss(stats.intensity_mean, 0.03), 0.08, 0.6)),
        "thickness_mean": float(np.clip(rng.gauss(stats.thickness_mean, max(0.4, stats.thickness_std * 0.35)), 2.5, 12.0)),
        "thickness_std": float(np.clip(stats.thickness_std, 0.6, 3.0)),
        "thin_band_ratio": float(np.clip(stats.thin_band_ratio, 0.15, 0.8)),
        "background": float(np.clip(rng.gauss(stats.background_mean, 0.02), 0.55, 0.88)),
        "noise": float(np.clip(rng.gauss(stats.background_noise, 0.004), 0.01, 0.05)),
        "blur": float(np.clip(rng.gauss(stats.blur_sigma, 0.08), 0.65, 1.55)),
        "sharpness_score": stats.sharpness_score,
        "band_y_pool": stats.band_y_pool,
        "smear_probability": 0.25,
    }


def generate_background(params: dict[str, float | int | list[int]], rng: random.Random) -> np.ndarray:
    height = int(params["height"])
    width = int(params["width"])
    bg = np.ones((height, width), dtype=np.float32) * float(params["background"])
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, float(params["noise"]), (height, width)).astype(np.float32)
    bg += noise

    grad_y = np.linspace(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), height, dtype=np.float32).reshape(-1, 1)
    grad_x = np.linspace(rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), width, dtype=np.float32).reshape(1, -1)
    bg += grad_y + grad_x

    # Keep borders clean: use a very soft illumination field without dark edge vignette.
    low_freq = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 1, (height, width)).astype(np.float32)
    low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=max(12.0, width / 8.0), sigmaY=max(8.0, height / 3.0))
    bg += low_freq * 0.01
    return np.clip(bg, 0, 1)


def add_smear(band: np.ndarray, rng: random.Random) -> np.ndarray:
    smear_len = rng.randint(4, 12)
    result = band.copy()
    for i in range(1, smear_len + 1):
        shifted = np.roll(band, i, axis=0)
        result += shifted * (0.03 * (smear_len - i + 1))
    return result


def draw_band(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    thickness: int,
    intensity: float,
    rng: random.Random,
    smear_probability: float,
    blur_strength: float,
) -> np.ndarray:
    band = np.zeros_like(image, dtype=np.float32)
    pad_x = max(6, int(width * 0.18))
    pad_y = max(4, int(thickness * 1.8))
    x1 = max(0, int(round(x - width / 2)) - pad_x)
    x2 = min(image.shape[1] - 1, int(round(x + width / 2)) + pad_x)
    y1 = max(0, int(round(y - thickness / 2)) - pad_y)
    y2 = min(image.shape[0] - 1, int(round(y + thickness / 2)) + pad_y)

    patch_w = x2 - x1 + 1
    patch_h = y2 - y1 + 1
    if patch_w <= 1 or patch_h <= 1:
        return image

    xx = np.arange(x1, x2 + 1, dtype=np.float32)
    yy = np.arange(y1, y2 + 1, dtype=np.float32)[:, None]
    center_rel = (xx - x) / max(width / 2.0, 1.0)

    # Flat long band body with only mild taper near the ends.
    horizontal_envelope = np.exp(-(np.abs(center_rel) ** 4.6) * 0.72)
    end_softness = np.exp(-(np.abs(center_rel) ** 7.0) * 0.42)

    # Very slight curvature only; keep the band visually close to straight.
    phase = rng.uniform(0.0, np.pi)
    curve_amp = rng.uniform(0.0, 0.09) * max(1.0, thickness)
    tilt = rng.uniform(-0.02, 0.02) * thickness
    curve = curve_amp * np.sin(center_rel * np.pi + phase) + tilt * center_rel

    # Thickness stays stable across the band with only tiny local variation.
    local_thickness = thickness * (0.88 + 0.12 * horizontal_envelope)
    local_thickness *= 0.985 + 0.02 * np.sin(center_rel * np.pi * 1.2 + rng.uniform(0.0, np.pi))
    local_thickness = np.clip(local_thickness, max(2.0, thickness * 0.8), thickness * 1.05)

    centerline = y + curve
    sigma_y = np.maximum(1.0, local_thickness / 2.3)
    vertical_profile = np.exp(-0.5 * ((yy - centerline[None, :]) / sigma_y[None, :]) ** 2)
    vertical_core = np.exp(-0.5 * ((yy - centerline[None, :]) / np.maximum(0.9, sigma_y[None, :] * 0.7)) ** 2)

    # Keep texture very light so the band remains continuous.
    texture = np.random.default_rng(rng.randint(0, 10**9)).normal(1.0, 0.015, size=(patch_h, patch_w)).astype(np.float32)
    texture = cv2.GaussianBlur(
        texture,
        (0, 0),
        sigmaX=max(0.8, width / 14.0),
        sigmaY=max(0.8, thickness / 2.8),
    )

    mask_patch = intensity * (0.72 * vertical_profile + 0.42 * vertical_core) * horizontal_envelope[None, :] * end_softness[None, :]
    mask_patch *= np.clip(texture, 0.97, 1.03)
    mask_patch = cv2.GaussianBlur(
        mask_patch.astype(np.float32),
        (0, 0),
        sigmaX=np.clip(rng.uniform(0.68, 1.18) * blur_strength, 0.58, 1.65),
        sigmaY=np.clip(rng.uniform(0.68, 0.98) * blur_strength, 0.56, 1.28),
    )
    band[y1 : y2 + 1, x1 : x2 + 1] = np.maximum(band[y1 : y2 + 1, x1 : x2 + 1], mask_patch)

    if rng.random() < smear_probability:
        band = add_smear(band, rng)

    image -= band
    return image


def choose_band_rows(pool: list[int], count: int, height: int, rng: random.Random) -> list[int]:
    if pool:
        unique_pool = sorted(set(pool))
        if count <= len(unique_pool):
            rows = rng.sample(unique_pool, count)
        else:
            rows = unique_pool[:]
            while len(rows) < count:
                rows.append(rng.choice(unique_pool))
        rows = sorted(rows)
        return [int(np.clip(row + rng.randint(-2, 2), 12, height - 12)) for row in rows]
    return sorted(rng.sample(range(12, max(13, height - 12)), count))


def generate_gel(params: dict[str, float | int | list[int]], rng: random.Random) -> np.ndarray:
    image = generate_background(params, rng)
    width = int(params["width"])
    height = int(params["height"])
    lane_count = int(params["lanes"])
    lane_spacing = int(params["lane_spacing"])
    lane_width = int(params["lane_width"])
    total_span = (lane_count - 1) * lane_spacing
    start_x = max(lane_width // 2 + 6, int(round((width - total_span) / 2.0)))

    for lane_idx in range(lane_count):
        lane_x = start_x + lane_idx * lane_spacing
        rows = choose_band_rows(list(params["band_y_pool"]), int(params["bands_per_lane"][lane_idx]), height, rng)
        for row in rows:
            if rng.random() < float(params.get("thin_band_ratio", 0.35)):
                base_thickness = float(params["thickness_mean"]) * rng.uniform(0.45, 0.8)
                thickness = max(2, int(round(rng.gauss(base_thickness, 0.45))))
            else:
                spread = max(0.5, float(params.get("thickness_std", 1.0)) * 0.55)
                thickness = max(3, int(round(rng.gauss(float(params["thickness_mean"]), spread))))
            thickness = int(np.clip(thickness, 2, 12))
            intensity = float(np.clip(rng.gauss(float(params["intensity_mean"]) * 1.12, 0.035), 0.08, 0.78))
            x_jitter = rng.randint(-2, 2)
            sharpness_score = float(params.get("sharpness_score", 1.0))
            band_blur = float(np.clip(1.3 - 0.18 * sharpness_score, 0.7, 1.15))
            band_width = max(8, int(round(lane_width * rng.uniform(1.04, 1.12))))
            image = draw_band(
                image=image,
                x=lane_x + x_jitter,
                y=row,
                width=band_width,
                thickness=thickness,
                intensity=intensity,
                rng=rng,
                smear_probability=float(params["smear_probability"]),
                blur_strength=band_blur,
            )

    image = cv2.GaussianBlur(image, (0, 0), sigmaX=float(params["blur"]), sigmaY=max(0.65, float(params["blur"]) * 0.85))
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def save_gray_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="L").save(path)


def build_montage(pairs: list[tuple[Path, Path]], output_path: Path) -> None:
    canvas = Image.new("RGB", (1050, 220 * max(1, len(pairs))), "white")
    y = 0
    for ref_path, gen_path in pairs:
        ref = ImageOps.contain(Image.open(ref_path).convert("L"), (480, 170)).convert("RGB")
        gen = ImageOps.contain(Image.open(gen_path).convert("L"), (480, 170)).convert("RGB")
        canvas.paste(ref, (20, y + 25))
        canvas.paste(gen, (540, y + 25))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, y), f"ref: {ref_path.name}", fill="black")
        draw.text((540, y), f"generated: {gen_path.name}", fill="black")
        y += 220
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.overwrite:
        reset_dir(args.output_root)
    else:
        ensure_dir(args.output_root)

    refs_root = args.output_root / "references"
    gen_root = args.output_root / "generated"
    meta_root = args.output_root / "metadata"
    for path in [refs_root, gen_root, meta_root]:
        ensure_dir(path)

    rng = random.Random(args.seed)
    pairs: list[tuple[Path, Path]] = []
    summary: list[dict[str, object]] = []

    for split in ["train", "test"]:
        source_dir = args.blot_root / split / "image"
        refs = list_images(source_dir)[: args.refs_per_split]
        ensure_dir(refs_root / split)
        ensure_dir(gen_root / split)
        ensure_dir(meta_root / split)

        for ref_path in refs:
            stats = extract_reference_parameters(ref_path, split)
            params = sample_parameters(stats, rng)
            generated = generate_gel(params, rng)

            ref_copy = refs_root / split / ref_path.name
            gen_path = gen_root / split / ref_path.name
            meta_path = meta_root / split / f"{ref_path.stem}.json"

            shutil.copy2(ref_path, ref_copy)
            save_gray_image(generated, gen_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "reference": asdict(stats),
                        "sampled_parameters": params,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            pairs.append((ref_copy, gen_path))
            summary.append(
                {
                    "split": split,
                    "reference": ref_path.name,
                    "generated": gen_path.name,
                    "lane_count": stats.lane_count,
                    "band_mean": stats.band_mean,
                }
            )

    montage_path = args.output_root / "montage.png"
    build_montage(pairs, montage_path)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root), "montage": str(montage_path), "samples": len(summary)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
