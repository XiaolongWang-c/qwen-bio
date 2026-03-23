import os
import cv2
import json
import random
import numpy as np
from scipy.signal import find_peaks

# -----------------------------
# 输出目录
# -----------------------------

DATASET_DIR = "gel_dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
META_DIR = os.path.join(DATASET_DIR, "metadata")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


# -----------------------------
# 提取参考图参数
# -----------------------------

def extract_reference_parameters(image_path):

    img = cv2.imread(image_path, 0)

    height, width = img.shape

    img_blur = cv2.GaussianBlur(img,(5,5),0)

    img_norm = img_blur/255.0

    # -------- lane detection --------
    vertical_profile = img_norm.mean(axis=0)

    peaks,_ = find_peaks(
        -vertical_profile,
        distance=25,
        prominence=0.01
    )

    lane_positions = peaks.tolist()

    # lane spacing
    lane_spacing = []

    for i in range(len(lane_positions)-1):
        lane_spacing.append(
            lane_positions[i+1] - lane_positions[i]
        )

    mean_spacing = np.mean(lane_spacing) if lane_spacing else 40

    lane_width = int(mean_spacing*0.6)

    band_positions = []
    band_thickness = []
    band_intensity = []

    for x in lane_positions:

        x1 = max(0,x-int(lane_width/2))
        x2 = min(width,x+int(lane_width/2))

        lane = img_norm[:,x1:x2]

        profile = lane.mean(axis=1)

        peaks,_ = find_peaks(
            -profile,
            distance=15,
            prominence=0.02
        )

        lane_bands = []

        for y in peaks:

            y1=max(0,y-4)
            y2=min(height,y+4)

            intensity = lane[y1:y2].mean()

            lane_bands.append(int(y))

            band_intensity.append(float(intensity))

            band_thickness.append(random.randint(6,12))

        band_positions.append(lane_bands)

    params = {

        "image_width": width,
        "image_height": height,

        "lane_positions": lane_positions,

        "lane_count": len(lane_positions),

        "lane_width": lane_width,

        "lane_spacing": mean_spacing,

        "band_positions": band_positions,

        "band_intensity": band_intensity,

        "band_thickness": band_thickness

    }

    return params


# -----------------------------
# 建立统计模型
# -----------------------------

def build_model(params):

    bands_per_lane = [len(x) for x in params["band_positions"]]

    model = {

        "width": params["image_width"],
        "height": params["image_height"],

        "lane_mean": params["lane_count"],

        "lane_width": params["lane_width"],

        "lane_spacing": params["lane_spacing"],

        "band_mean": np.mean(bands_per_lane) if bands_per_lane else 2,

        "intensity_mean": np.mean(params["band_intensity"]) if params["band_intensity"] else 0.5,

        "thickness_mean": np.mean(params["band_thickness"]) if params["band_thickness"] else 8

    }

    return model


# -----------------------------
# 参数采样
# -----------------------------

def sample_parameters(model):

    lane_count = max(2,int(np.random.normal(model["lane_mean"],1)))

    bands_per_lane=[]

    for _ in range(lane_count):

        bands=max(1,int(np.random.normal(model["band_mean"],1)))

        bands_per_lane.append(bands)

    params={

        "width":model["width"],
        "height":model["height"],

        "lanes":lane_count,

        "lane_width":int(np.random.normal(model["lane_width"],2)),

        "lane_spacing":int(np.random.normal(model["lane_spacing"],3)),

        "bands_per_lane":bands_per_lane,

        "intensity_mean":model["intensity_mean"],

        "thickness_mean":model["thickness_mean"],

        "background":random.uniform(0.6,0.8),

        "noise":random.uniform(0.01,0.04),

        "blur":random.uniform(0.6,1.5)

    }

    return params


# -----------------------------
# 背景生成
# -----------------------------

def generate_background(params):

    h=params["height"]
    w=params["width"]

    bg=np.ones((h,w),dtype=np.float32)

    bg*=params["background"]

    noise=np.random.normal(
        0,
        params["noise"],
        (h,w)
    )

    bg+=noise

    gradient=np.linspace(
        0,
        random.uniform(-0.08,0.08),
        h
    )

    bg+=gradient.reshape(-1,1)

    return np.clip(bg,0,1)


# -----------------------------
# 画条带
# -----------------------------

def draw_band(img,x,y,width,thickness,intensity):

    x1=int(x-width/2)
    x2=int(x+width/2)

    y1=int(y-thickness/2)
    y2=int(y+thickness/2)

    x1=max(0,x1)
    x2=min(img.shape[1]-1,x2)
    y1=max(0,y1)
    y2=min(img.shape[0]-1,y2)

    band=np.zeros_like(img)

    band[y1:y2,x1:x2]=intensity

    band=cv2.GaussianBlur(band,(0,0),1.5)

    img-=band

    return img


# -----------------------------
# 生成凝胶图
# -----------------------------

def generate_gel(params):

    img=generate_background(params)

    start_x=30

    for i in range(params["lanes"]):

        lane_x=start_x+i*params["lane_spacing"]

        for _ in range(params["bands_per_lane"][i]):

            y=random.randint(20,params["height"]-20)

            thickness=max(
                int(params["thickness_mean"]*0.7),
                int(np.random.normal(params["thickness_mean"],2))
            )

            intensity=np.random.normal(
                params["intensity_mean"],
                0.08
            )

            img=draw_band(
                img,
                lane_x,
                y,
                params["lane_width"],
                thickness,
                intensity
            )

    img=cv2.GaussianBlur(img,(0,0),params["blur"])

    img=np.clip(img,0,1)

    img=(img*255).astype(np.uint8)

    return img


# -----------------------------
# 保存
# -----------------------------

def save_sample(i,model):

    params=sample_parameters(model)

    img=generate_gel(params)

    name=f"gel_{i:05d}"

    cv2.imwrite(
        os.path.join(IMAGE_DIR,name+".png"),
        img
    )

    with open(
        os.path.join(META_DIR,name+".json"),
        "w"
    ) as f:

        json.dump(
            {
                "id":name,
                "synthetic":True,
                "parameters":params
            },
            f,
            indent=2
        )


# -----------------------------
# 生成数据集
# -----------------------------

def generate_dataset(reference_image,count=3000):

    print("reading reference...")

    ref_params=extract_reference_parameters(reference_image)

    model=build_model(ref_params)

    print("start generation")

    for i in range(count):

        save_sample(i,model)

        if i%100==0:
            print("generated",i)


# -----------------------------
# main
# -----------------------------

if __name__=="__main__":

    reference="input/input.png"

    generate_dataset(reference,3000)