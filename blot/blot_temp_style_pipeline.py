from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


def load_temp_style_module(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("temp_style_small_batch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-batch blot generation using the temp-style direct rendering pipeline.")
    parser.add_argument("--blot-root", type=Path, default=Path("blot"))
    parser.add_argument("--subset-train-count", type=int, default=2400)
    parser.add_argument("--subset-test-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-count", type=int, default=4)
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


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if count > len(paths):
        raise ValueError(f"Requested {count} files but only found {len(paths)}")
    rng = random.Random(seed)
    sampled = rng.sample(paths, count)
    sampled.sort(key=lambda item: item.name)
    return sampled


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_outputs(blot_root: Path, overwrite: bool) -> dict[str, Path]:
    outputs = {
        "subset_train": blot_root / "subset" / "train" / "image",
        "subset_test": blot_root / "subset" / "test" / "image",
        "metadata_train": blot_root / "metadata" / "train",
        "metadata_test": blot_root / "metadata" / "test",
        "generated_train": blot_root / "generated" / "train",
        "generated_test": blot_root / "generated" / "test",
        "records": blot_root / "split_records",
    }
    if overwrite:
        for path in outputs.values():
            reset_dir(path)
    else:
        for path in outputs.values():
            ensure_dir(path)
    return outputs


def build_manifest(split: str, sampled_paths: list[Path], seed: int, total_available: int) -> dict[str, object]:
    return {
        "split": split,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_available": total_available,
        "sample_count": len(sampled_paths),
        "files": [path.name for path in sampled_paths],
    }


def process_split(
    split: str,
    sampled_paths: list[Path],
    subset_dir: Path,
    metadata_dir: Path,
    generated_dir: Path,
    pipeline_module: ModuleType,
    seed: int,
) -> tuple[list[tuple[Path, Path]], dict[str, object]]:
    ensure_dir(subset_dir)
    ensure_dir(metadata_dir)
    ensure_dir(generated_dir)

    rng = random.Random(seed)
    preview_pairs: list[tuple[Path, Path]] = []
    summary = {
        "split": split,
        "processed": 0,
        "lane_count_values": [],
        "band_mean_values": [],
        "thickness_mean_values": [],
        "thin_band_ratio_values": [],
        "blur_values": [],
    }

    for index, source_path in enumerate(sampled_paths, 1):
        subset_path = subset_dir / source_path.name
        generated_path = generated_dir / source_path.name
        metadata_path = metadata_dir / f"{source_path.stem}.json"

        shutil.copy2(source_path, subset_path)

        stats = pipeline_module.extract_reference_parameters(source_path, split)
        params = pipeline_module.sample_parameters(stats, rng)
        generated = pipeline_module.generate_gel(params, rng)
        pipeline_module.save_gray_image(generated, generated_path)

        write_json(
            metadata_path,
            {
                "reference": pipeline_module.asdict(stats),
                "sampled_parameters": params,
            },
        )

        if len(preview_pairs) < 8:
            preview_pairs.append((subset_path, generated_path))

        summary["processed"] += 1
        summary["lane_count_values"].append(stats.lane_count)
        summary["band_mean_values"].append(stats.band_mean)
        summary["thickness_mean_values"].append(stats.thickness_mean)
        summary["thin_band_ratio_values"].append(stats.thin_band_ratio)
        summary["blur_values"].append(params["blur"])

        if index % 200 == 0:
            print(f"{split}: processed {index}/{len(sampled_paths)}")

    return preview_pairs, summary


def summarize_metrics(summary: dict[str, object]) -> dict[str, object]:
    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "processed": summary["processed"],
        "lane_count_mean": mean(summary["lane_count_values"]),
        "band_mean": mean(summary["band_mean_values"]),
        "thickness_mean": mean(summary["thickness_mean_values"]),
        "thin_band_ratio_mean": mean(summary["thin_band_ratio_values"]),
        "blur_mean": mean(summary["blur_values"]),
    }


def main() -> None:
    args = parse_args()
    blot_root = args.blot_root
    train_source = blot_root / "train" / "image"
    test_source = blot_root / "test" / "image"
    module_path = Path("blot/temp_style_small_batch.py")

    if not train_source.exists() or not test_source.exists():
        raise FileNotFoundError("Expected blot/train/image and blot/test/image to exist.")
    if not module_path.exists():
        raise FileNotFoundError(f"Expected generator module at {module_path}")

    pipeline_module = load_temp_style_module(module_path)
    train_paths = list_images(train_source)
    test_paths = list_images(test_source)

    sampled_train = sample_paths(train_paths, args.subset_train_count, args.seed)
    sampled_test = sample_paths(test_paths, args.subset_test_count, args.seed + 1)

    outputs = prepare_outputs(blot_root, args.overwrite)

    train_manifest = build_manifest("train", sampled_train, args.seed, len(train_paths))
    test_manifest = build_manifest("test", sampled_test, args.seed + 1, len(test_paths))
    write_json(outputs["records"] / "train_manifest.json", train_manifest)
    write_json(outputs["records"] / "test_manifest.json", test_manifest)

    train_preview, train_summary = process_split(
        split="train",
        sampled_paths=sampled_train,
        subset_dir=outputs["subset_train"],
        metadata_dir=outputs["metadata_train"],
        generated_dir=outputs["generated_train"],
        pipeline_module=pipeline_module,
        seed=args.seed,
    )
    test_preview, test_summary = process_split(
        split="test",
        sampled_paths=sampled_test,
        subset_dir=outputs["subset_test"],
        metadata_dir=outputs["metadata_test"],
        generated_dir=outputs["generated_test"],
        pipeline_module=pipeline_module,
        seed=args.seed + 1,
    )

    preview_pairs = (train_preview[: args.preview_count] + test_preview[: args.preview_count])[: args.preview_count * 2]
    preview_path = outputs["records"] / "preview_montage.png"
    pipeline_module.build_montage(preview_pairs, preview_path)

    generation_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "subset_train_count": args.subset_train_count,
        "subset_test_count": args.subset_test_count,
        "train": summarize_metrics(train_summary),
        "test": summarize_metrics(test_summary),
        "overall": {
            "processed": train_summary["processed"] + test_summary["processed"],
        },
        "preview_montage": str(preview_path),
    }
    write_json(outputs["records"] / "generation_summary.json", generation_summary)
    print(json.dumps(generation_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
