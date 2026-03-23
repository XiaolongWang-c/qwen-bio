import os
import cv2
import json
import random
import numpy as np
from scipy.signal import find_peaks

DATASET_DIR = "gel_dataset"
IMG_DIR = os.path.join(DATASET_DIR,"images")
META_DIR = os.path.join(DATASET_DIR,"metadata")

os.makedirs(IMG_DIR,exist_ok=True)
os.makedirs(META_DIR,exist_ok=True)


# ------------------------------------------------
# 参考图参数提取
# ------------------------------------------------

def extract_reference_parameters(image_path):

    img=cv2.imread(image_path,0)
    h,w=img.shape

    img_blur=cv2.GaussianBlur(img,(5,5),0)
    img_norm=img_blur/255.0

    vertical_profile=img_norm.mean(axis=0)

    peaks,_=find_peaks(-vertical_profile,distance=25,prominence=0.01)

    lane_positions=peaks.tolist()

    spacing=[]
    for i in range(len(lane_positions)-1):
        spacing.append(lane_positions[i+1]-lane_positions[i])

    lane_spacing=np.mean(spacing) if spacing else 40
    lane_width=int(lane_spacing*0.6)

    band_positions=[]
    band_intensity=[]
    band_thickness=[]

    for x in lane_positions:

        x1=max(0,x-int(lane_width/2))
        x2=min(w,x+int(lane_width/2))

        lane=img_norm[:,x1:x2]

        profile=lane.mean(axis=1)

        peaks,_=find_peaks(-profile,distance=15,prominence=0.02)

        lane_bands=[]

        for y in peaks:

            y1=max(0,y-4)
            y2=min(h,y+4)

            lane_bands.append(int(y))

            band_intensity.append(lane[y1:y2].mean())
            band_thickness.append(random.randint(6,12))

        band_positions.append(lane_bands)

    return {
        "width":w,
        "height":h,
        "lane_count":len(lane_positions),
        "lane_width":lane_width,
        "lane_spacing":lane_spacing,
        "band_positions":band_positions,
        "band_intensity":band_intensity,
        "band_thickness":band_thickness
    }


# ------------------------------------------------
# 构建统计模型
# ------------------------------------------------

def build_model(p):

    bands=[len(x) for x in p["band_positions"]]

    return {
        "width":p["width"],
        "height":p["height"],
        "lane_mean":p["lane_count"],
        "lane_width":p["lane_width"],
        "lane_spacing":p["lane_spacing"],
        "band_mean":np.mean(bands) if bands else 2,
        "intensity_mean":np.mean(p["band_intensity"]) if p["band_intensity"] else 0.5,
        "thickness_mean":np.mean(p["band_thickness"]) if p["band_thickness"] else 8
    }


# ------------------------------------------------
# 参数采样
# ------------------------------------------------

def sample_parameters(model):

    lane_count=max(2,int(np.random.normal(model["lane_mean"],1)))

    bands=[]
    for _ in range(lane_count):
        bands.append(max(1,int(np.random.normal(model["band_mean"],1))))

    return {
        "width":model["width"],
        "height":model["height"],
        "lanes":lane_count,
        "lane_width":int(np.random.normal(model["lane_width"],2)),
        "lane_spacing":int(np.random.normal(model["lane_spacing"],3)),
        "bands_per_lane":bands,
        "intensity_mean":model["intensity_mean"],
        "thickness_mean":model["thickness_mean"],
        "background":random.uniform(0.6,0.8),
        "noise":random.uniform(0.01,0.04),
        "blur":random.uniform(0.6,1.5)
    }


# ------------------------------------------------
# 背景模拟
# ------------------------------------------------

def generate_background(p):

    h=p["height"]
    w=p["width"]

    bg=np.ones((h,w),dtype=np.float32)
    bg*=p["background"]

    noise=np.random.normal(0,p["noise"],(h,w))
    bg+=noise

    # illumination gradient
    gradient=np.linspace(0,random.uniform(-0.08,0.08),h)
    bg+=gradient.reshape(-1,1)

    # vignette
    x=np.linspace(-1,1,w)
    y=np.linspace(-1,1,h)

    X,Y=np.meshgrid(x,y)

    vignette=1-0.3*(X**2+Y**2)

    bg*=vignette

    return np.clip(bg,0,1)


# ------------------------------------------------
# 条带拖尾
# ------------------------------------------------

def add_smear(band):

    h,w=band.shape

    smear_len=random.randint(5,20)

    for i in range(smear_len):

        shifted=np.roll(band,i,axis=0)

        band+=shifted*(0.05*(smear_len-i))

    return band


# ------------------------------------------------
# 画条带
# ------------------------------------------------

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

    # 横向扩散
    band=cv2.GaussianBlur(band,(0,0),random.uniform(1,3))

    # 条带拖尾
    if random.random()<0.4:
        band=add_smear(band)

    img-=band

    return img


# ------------------------------------------------
# 泳道弯曲
# ------------------------------------------------

def lane_warp(x,y):

    offset=int(3*np.sin(y/50))

    return x+offset


# ------------------------------------------------
# 生成凝胶
# ------------------------------------------------

def generate_gel(p):

    img=generate_background(p)

    start_x=30

    for i in range(p["lanes"]):

        lane_x=start_x+i*p["lane_spacing"]

        for _ in range(p["bands_per_lane"][i]):

            y=random.randint(20,p["height"]-20)

            x=lane_warp(lane_x,y)

            thickness=max(
                int(p["thickness_mean"]*0.7),
                int(np.random.normal(p["thickness_mean"],2))
            )

            intensity=np.random.normal(
                p["intensity_mean"],
                0.08
            )

            img=draw_band(
                img,
                x,
                y,
                p["lane_width"],
                thickness,
                intensity
            )

    img=cv2.GaussianBlur(img,(0,0),p["blur"])

    img=np.clip(img,0,1)

    img=(img*255).astype(np.uint8)

    return img


# ------------------------------------------------
# 保存
# ------------------------------------------------

def save_sample(i,model):

    params=sample_parameters(model)

    img=generate_gel(params)

    name=f"gel_{i:05d}"

    cv2.imwrite(
        os.path.join(IMG_DIR,name+".png"),
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


# ------------------------------------------------
# 生成数据集
# ------------------------------------------------

def generate_dataset(reference_image,count=3000):

    ref=extract_reference_parameters(reference_image)

    model=build_model(ref)

    for i in range(count):

        save_sample(i,model)

        if i%100==0:
            print("generated",i)


# ------------------------------------------------
# main
# ------------------------------------------------

if __name__=="__main__":

    reference="input/input.png"

    generate_dataset(reference,3000)