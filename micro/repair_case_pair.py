from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from micro.micro_qwen_dataset_pipeline import (
    fuse_generated_with_source,
)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair an existing source/generated microscopy pair with structure-preserving fusion.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-strength", type=float, default=0.42)
    parser.add_argument("--detail-sigma", type=float, default=1.1)
    parser.add_argument("--mask-quantile", type=float, default=94.0)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    with Image.open(args.source) as source_image:
        source_rgb = source_image.convert("RGB")
    with Image.open(args.generated) as generated_image:
        generated_rgb = generated_image.convert("RGB")

    fused_image, metrics = fuse_generated_with_source(source_rgb, generated_rgb, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused_image.save(args.output)
    print(f"saved: {args.output}")
    print(metrics)


if __name__ == "__main__":
    main()
