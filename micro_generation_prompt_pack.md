# Micro 3 图测试最终 Prompt 包

本文件给出首轮 3 张测试图可直接用于 `qwen-image-edit` 的 prompt 草案。

原则：

- 每张图单独描述
- 不使用统一大类模板
- 所有 prompt 都必须满足以下约束：
- 与参考图属于同一显微成像类型
- 保留相同背景、照明方式、色调和成像风格
- 保留相同层级的模糊程度、噪声水平和对比度
- 只改变实验结果相关内容
- 新图必须明显不同于参考图，但不能跳出同一成像域

## 1. 通用负向 Prompt

```text
sharp contour, hard edge, halo artifact, oversharpening, cartoon texture,
synthetic pattern, repeated clone artifacts, duplicated objects, grid artifact,
mosaic artifact, pasted appearance, artificial edges, watermark, text, labels,
ruler overlay, non-biological objects, wrong microscopy modality,
dramatic color shift, unrealistic contrast, background replacement,
illumination mismatch, copied object layout
```

## 2. 测试图 1

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1+020100-JC.jpg](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1+020100-JC.jpg)

### 最终 Prompt

```text
Edit this microscopy image based on the reference.

Keep the same grayscale brightfield microscopy modality, the same pale gray background, the same soft diffuse illumination, the same low-contrast appearance, the same slight blur, and the same low-noise imaging style as the reference image.

Preserve the same background brightness, the same smooth background texture, and the same soft blurry boundaries of the biological objects.

Generate a new realistic result in the same image domain:
- keep the same type of round soft cell-like clusters
- clearly change the experimental result by increasing the number of visible clusters from 3 to about 5-7
- vary cluster sizes moderately within a realistic range
- rearrange cluster positions into a new sparse non-regular layout
- make some clusters slightly more round and some slightly more irregular
- keep all object edges soft and slightly blurred
- make the generated image obviously different from the reference image, not a near-copy

Do not change the imaging modality, background style, illumination, tone, blur level, noise level, or contrast level.
Do not add sharp outlines, artificial texture, fluorescent color, text, or non-biological objects.
```

### 建议

- 变化强度：中到强
- 主要风险：改动太小，或者边缘变硬

## 3. 测试图 2

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1003+080000-NA-BC-JC.jpg](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1003+080000-NA-BC-JC.jpg)

### 最终 Prompt

```text
Edit this microscopy image based on the reference.

Keep the same fluorescence microscopy modality, the same near-black background, the same magenta-blue fluorescent signal style, the same high-contrast dark-field appearance, and the same soft glowing edges as the reference image.

Preserve the same dark background, the same sparse composition, the same fluorescent color relationship, and the same overall signal sharpness and blur balance.

Generate a new realistic result in the same image domain:
- keep the same type of fluorescent round cell-like objects
- clearly change the experimental result by changing the number of visible fluorescent objects from 3 to about 4-6
- vary object sizes slightly within a realistic range
- rearrange all fluorescent objects into a new sparse layout
- keep object shapes rounded with soft glow and subtle morphology differences
- make the generated image obviously different from the reference image, not a near-copy

Do not change the imaging modality, background style, color channels, illumination style, blur level, noise level, or contrast level.
Do not add brightfield texture, new colors, hard neon edges, text, or artificial fluorescent artifacts.
```

### 建议

- 变化强度：中
- 主要风险：风格漂移到 brightfield，或者荧光边缘过硬

## 4. 测试图 3

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1016+090000-NONE.png](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1016+090000-NONE.png)

### 最终 Prompt

```text
Edit this microscopy image based on the reference.

Keep the same stained histology microscopy modality, the same light slide background, the same brown-blue staining character, the same dense tissue-section appearance, the same brightfield illumination, and the same overall texture scale as the reference image.

Preserve the same tissue background, the same lumen-like and cellular microstructure style, the same stain intensity range, and the same moderate focus softness.

Generate a new realistic result in the same image domain:
- keep the same histology tissue-section style
- clearly change the experimental result by changing local stained-cell distribution and local stained-region arrangement
- vary local cell density, nuclear distribution, and small tissue morphology in a realistic way
- keep the overall tissue architecture consistent with the reference image domain
- make the generated image obviously different from the reference image, not a near-copy

Do not change the imaging modality, tissue-section background, staining style, illumination, blur level, noise level, or contrast level.
Do not create isolated cells on an empty background, do not add fluorescent colors, text, markers, cartoon texture, or artificial contour lines.
```

### 建议

- 变化强度：中偏弱
- 主要风险：破坏组织切片背景，或者改动不足

## 5. 首轮测试建议

建议先固定这 3 个条件不动：

- 每张参考图只生成 1 张
- 通用负向 Prompt 先保持一致
- 先只比较 prompt 本身是否足够稳定

建议重点观察：

- 图 1 是否容易出现硬边和伪轮廓
- 图 2 是否容易跑偏成非荧光风格
- 图 3 是否容易破坏组织背景，或者变化不明显

## 6. 如果首轮结果不好，优先调整顺序

1. 先调正向 prompt 里的“must keep”部分，使风格锁定更强。
2. 再调“变化强度”描述，解决改动太小或太大的问题。
3. 最后再加强负向 prompt，抑制边缘、复制感和风格漂移。
