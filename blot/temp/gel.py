import os
import json
import random
import uuid
import numpy as np
import cv2

OUTPUT_DIR = "gel_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

IMG_WIDTH = 512
IMG_HEIGHT = 512


# -----------------------------
# 参数采样
# -----------------------------
def sample_parameters():

    params = {}

    params["lanes"] = random.randint(6, 14)

    params["lane_width"] = random.randint(25, 40)

    params["lane_gap"] = random.randint(8, 20)

    params["lane_offset"] = random.randint(-5, 5)

    params["background_mean"] = random.uniform(0.6, 0.8)

    params["background_noise"] = random.uniform(0.01, 0.05)

    params["blur_sigma"] = random.uniform(0.5, 1.8)

    params["bands_per_lane"] = []

    for _ in range(params["lanes"]):
        params["bands_per_lane"].append(random.randint(1, 8))

    params["seed"] = random.randint(0, 99999999)

    return params


# -----------------------------
# 生成背景
# -----------------------------
def generate_background(params):

    bg = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    bg *= params["background_mean"]

    noise = np.random.normal(
        0,
        params["background_noise"],
        (IMG_HEIGHT, IMG_WIDTH)
    )

    bg += noise

    gradient = np.linspace(0, random.uniform(-0.1, 0.1), IMG_HEIGHT)
    gradient = gradient.reshape(-1, 1)

    bg += gradient

    return np.clip(bg, 0, 1)


# -----------------------------
# 画条带
# -----------------------------
def draw_band(image, x_center, y_center, width, thickness, intensity):

    band = np.zeros_like(image)

    x1 = int(x_center - width / 2)
    x2 = int(x_center + width / 2)

    y1 = int(y_center - thickness / 2)
    y2 = int(y_center + thickness / 2)

    x1 = max(0, x1)
    x2 = min(IMG_WIDTH - 1, x2)
    y1 = max(0, y1)
    y2 = min(IMG_HEIGHT - 1, y2)

    band[y1:y2, x1:x2] = intensity

    band = cv2.GaussianBlur(band, (0, 0), random.uniform(0.8, 2))

    image -= band

    return image


# -----------------------------
# 生成 mock gel
# -----------------------------
def generate_mock_gel(params):

    random.seed(params["seed"])
    np.random.seed(params["seed"])

    img = generate_background(params)

    lane_width = params["lane_width"]
    lane_gap = params["lane_gap"]

    start_x = 50 + params["lane_offset"]

    lane_positions = []

    for i in range(params["lanes"]):

        lane_x = start_x + i * (lane_width + lane_gap)

        lane_positions.append(lane_x)

        band_count = params["bands_per_lane"][i]

        for _ in range(band_count):

            y_pos = random.randint(30, IMG_HEIGHT - 30)

            thickness = random.randint(3, 12)

            intensity = random.uniform(0.2, 0.7)

            img = draw_band(
                img,
                lane_x,
                y_pos,
                lane_width,
                thickness,
                intensity
            )

    img = cv2.GaussianBlur(img, (0, 0), params["blur_sigma"])

    img = np.clip(img, 0, 1)

    img = (img * 255).astype(np.uint8)

    return img


# -----------------------------
# 保存
# -----------------------------
def save_sample(index):

    params = sample_parameters()

    img = generate_mock_gel(params)

    img_id = f"gel_{index:05d}"

    img_path = os.path.join(IMG_DIR, img_id + ".png")

    meta_path = os.path.join(META_DIR, img_id + ".json")

    cv2.imwrite(img_path, img)

    metadata = {
        "id": img_id,
        "synthetic": True,
        "generator": "mock_gel_generator",
        "parameters": params
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------
# 批量生成
# -----------------------------
def generate_dataset(count=3000):

    for i in range(count):

        save_sample(i)

        if i % 100 == 0:
            print("generated:", i)


if __name__ == "__main__":

    generate_dataset(3000)