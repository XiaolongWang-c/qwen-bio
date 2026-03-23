
import os
from PIL import Image
import torch

from modelscope import QwenImageEditPipeline
model_name = "/sda/wangxl/ai-bio/qwen-image-edit"
type_all = ["micro","macro"]
img_type = type_all[0]

pipeline = QwenImageEditPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    )
print("pipeline loaded")

pipeline.set_progress_bar_config(disable=None)
image = Image.open(f"ai-bio/input/{img_type}/020104.png").convert("RGB")

# prompt = """
# Edit this SDS-PAGE protein gel image:
# - Keep all background elements, overall visual style, light gray tone, contrast, soft blur effect, and noise pattern completely unchanged.
# - Only modify the protein bands: adjust their number, width, and size (thickness/vertical length) .
# - Ensure all modified bands match the original dark gray color and blurred edge appearance.
# - Do not alter any non-band regions of the gel."""Edit this SDS-PAGE protein gel image:
# grayscale 
prompt = """Edit this microscopy image:

- Keep the original grayscale tone, background brightness, illumination, and overall microscopy imaging style unchanged.

- Generate a new realistic cell distribution:
  - Change the number of cells (more  than the original).
  - Vary cell sizes (some larger).
  - Randomize positions naturally, avoiding regular patterns.

"""

# prompt = """Edit this microscopy image:

# - Keep the original grayscale tone, background brightness, illumination, and overall microscopy imaging style unchanged.

# - Generate a new realistic cell distribution:
#   - Change the number of cells to 0.8×–1.5× of the original count (uniformly sampled).
#   - Scale each cell size by a factor sampled from 0.7–1.3 (uniform distribution).
#   - Randomize positions naturally, avoiding regular patterns.

# """

# - Modify cell morphology:
#   - Avoid extreme or exaggerated shapes.

# - IMPORTANT: Make the cells appear realistic under microscopy:
#   - Cell boundaries must be soft, diffuse, and slightly blurred.
#   - Do NOT create sharp edges, outlines, or contour-like borders.
#   - Avoid any artificial edge enhancement or halo effects.

# - Internal structure:
#   - Use low-frequency, smooth grayscale variations.
#   - Avoid strong granular noise or high-frequency texture.
#   - Keep details subtle and slightly blurred.

# - The overall image should look slightly out-of-focus and low-resolution, like a real experimental microscopy capture.

# - Keep the background clean and uniform.
# - Do not introduce any artificial patterns, text, or non-biological structures.


negative_prompt = """
sharp edges, strong outlines, contour lines, halo artifacts, edge enhancement,
high-frequency texture, excessive details, over-sharpening,
artificial patterns, mosaic artifacts, block artifacts,
cartoon, illustration, high-definition, ultra detailed
"""
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 6,
    "negative_prompt": negative_prompt,
    "num_inference_steps": 30,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(f"ai-bio/result/{img_type}/output_image_edit7.png")
    print("image saved at", os.path.abspath("result/output_image_edit7.png"))
