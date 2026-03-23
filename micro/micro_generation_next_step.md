# Micro 生成任务执行规范与 3 图测试草案

## 1. 最终筛选规范

### 1.1 数据范围

- 输入目录：`train/Micro/image`
- 数据用途：训练集参考图筛选
- 生成关系：`1` 张参考图对应生成 `1` 张新图

### 1.2 全局筛选规则

- 按当前文件顺序遍历全部图像
- 采用全局固定步长 `3` 取样
- 保留全局位置为 `1, 4, 7, 10, ...` 的图像

按当前数据统计：

- 原始图像数：`8514`
- 全局每 `3` 个选 `1` 个后的参考图数：`2838`

### 1.3 质量控制规则

质量控制优先于数量。

保留标准：

- 背景和照明稳定，没有明显曝光异常
- 主体结构清楚，主体不是几乎空白
- 图像模糊程度符合真实显微采集，不是严重失焦
- 没有大面积文字、水印、遮挡
- 压缩伪影和块效应不过重

剔除标准：

- 大面积空白
- 明显采集失败
- 严重失焦
- 主体被裁断过多
- 压缩噪声或伪影明显破坏结构

### 1.4 筛选结果清单字段

建议输出表字段：

- `ref_path`
- `global_index`
- `selected`
- `quality_flag`
- `keep_reason`
- `drop_reason`
- `note`

### 1.5 执行顺序

1. 对 `train/Micro/image` 建立全局顺序列表。
2. 取全局位置 `1, 4, 7, 10, ...` 的图像。
3. 对筛出的 `2838` 张做质量过滤。
4. 得到首轮参考图集。
5. 每张参考图生成 `1` 张新图。

## 2. 单图动态描述规范

当前不采用先分大类再套模板的方案，而是每张图单独解析、单独生成 prompt。

### 2.1 每张图都必须先产出结构化描述

建议 JSON 结构：

```json
{
  "reference_summary": {
    "imaging_type": "",
    "color_mode": "",
    "background_tone": "",
    "background_texture": "",
    "illumination": "",
    "focus_level": "",
    "contrast_level": "",
    "noise_level": "",
    "magnification_feel": ""
  },
  "content_summary": {
    "object_type": "",
    "object_count_estimate": "",
    "size_range": "",
    "density_level": "",
    "spatial_pattern": "",
    "shape_profile": "",
    "overlap_level": ""
  },
  "edit_plan": {
    "count_change": "",
    "count_delta_range": "",
    "size_delta_range": "",
    "layout_change_strength": "",
    "morphology_change_strength": "",
    "must_keep": [],
    "must_avoid": []
  }
}
```

### 2.2 每张图都必须满足的硬约束

- 与参考图属于同一显微成像类型
- 保留相同背景、照明方式、色调和成像风格
- 保留相同层级的模糊程度、噪声水平和对比度
- 只改变实验结果相关内容，即细胞数量、大小、位置、局部形态和局部聚集关系
- 新图必须明显不同于参考图，但不能跳出同一成像域

### 2.3 通用动态 Prompt 骨架

```text
Edit this microscopy image based on the reference.

Keep the same microscopy modality, acquisition style, magnification feel, background tone, illumination pattern, blur level, contrast level, noise character, and overall visual style as the reference image.

Preserve these fixed attributes from the reference:
- background: {background_tone_and_texture}
- illumination: {illumination}
- color and tone: {color_mode}
- focus and blur: {focus_level}
- noise and contrast: {noise_contrast_profile}
- biological object type: {object_type}

Generate a new experimental result within the same image domain:
- clearly change the number of visible objects from the reference
- change object sizes within a realistic range for this image type
- rearrange object positions with a natural, non-regular spatial distribution
- change local clustering pattern and local density
- introduce realistic local morphology variation while staying consistent with this imaging type
- make the output obviously different from the reference image, not a near-copy

Do not change the background style or imaging modality.
Do not add text, markers, artificial edges, strong outlines, or non-biological objects.
```

### 2.4 通用负向 Prompt

```text
sharp contour, hard edge, halo artifact, oversharpening, cartoon texture,
synthetic pattern, repeated clone artifacts, grid artifact, mosaic artifact,
text, watermark, labels, ruler overlay, non-biological objects,
wrong microscopy modality, dramatic color shift, unrealistic contrast,
background replacement, illumination mismatch, copied object layout
```

## 3. 首轮 3 图测试草案

## 3.1 测试图 1

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1+020100-JC.jpg](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1+020100-JC.jpg)

### 结构化描述草案

```json
{
  "reference_summary": {
    "imaging_type": "grayscale brightfield microscopy",
    "color_mode": "light gray monochrome",
    "background_tone": "uniform pale gray",
    "background_texture": "smooth with very mild low-frequency variation",
    "illumination": "soft diffuse illumination",
    "focus_level": "slightly soft and mildly out of focus",
    "contrast_level": "low to medium",
    "noise_level": "low",
    "magnification_feel": "medium magnification with sparse objects"
  },
  "content_summary": {
    "object_type": "soft round cell-like clusters",
    "object_count_estimate": 3,
    "size_range": "medium with mild size variation",
    "density_level": "sparse",
    "spatial_pattern": "three separated objects on a clean background",
    "shape_profile": "round to irregularly rounded with soft blurry boundaries",
    "overlap_level": "none"
  },
  "edit_plan": {
    "count_change": "increase",
    "count_delta_range": "1.5-2.3x",
    "size_delta_range": "0.8-1.2x",
    "layout_change_strength": "strong",
    "morphology_change_strength": "medium",
    "must_keep": [
      "same grayscale brightfield style",
      "same pale gray background",
      "same soft diffuse lighting",
      "same low-detail blur level",
      "same low-noise appearance"
    ],
    "must_avoid": [
      "sharp cell boundaries",
      "dark harsh outlines",
      "new textures in background",
      "fluorescent colors",
      "copying the original three-object layout"
    ]
  }
}
```

### Prompt 草案

```text
Edit this microscopy image based on the reference.

Keep the same grayscale brightfield microscopy style, the same pale gray background, the same diffuse illumination, the same slight soft-focus blur, the same low-noise appearance, and the same overall low-contrast visual style as the reference image.

Preserve these fixed attributes from the reference:
- background: smooth pale gray background with very mild illumination variation
- illumination: soft diffuse illumination without strong directional shadows
- color and tone: light gray monochrome
- focus and blur: slightly soft and mildly out of focus
- noise and contrast: low noise and low-to-medium contrast
- biological object type: soft round cell-like clusters

Generate a new experimental result within the same image domain:
- increase the number of visible cell-like clusters clearly beyond the reference image
- create about 5 to 7 visible objects instead of 3
- vary object sizes moderately within a realistic range
- rearrange the objects naturally across the field with a non-regular sparse distribution
- change local morphology subtly so some objects are more round and some are slightly irregular
- make the output obviously different from the reference image, not a near-copy

Do not change the background style or imaging modality.
Do not add sharp outlines, artificial textures, text, markers, or non-biological objects.
```

## 3.2 测试图 2

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1003+080000-NA-BC-JC.jpg](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1003+080000-NA-BC-JC.jpg)

### 结构化描述草案

```json
{
  "reference_summary": {
    "imaging_type": "fluorescence microscopy on dark background",
    "color_mode": "magenta-blue fluorescence on black background",
    "background_tone": "near-black",
    "background_texture": "uniform dark field",
    "illumination": "self-emissive fluorescence without visible background illumination",
    "focus_level": "moderately focused with soft fluorescent edges",
    "contrast_level": "high",
    "noise_level": "low",
    "magnification_feel": "high magnification with very few objects"
  },
  "content_summary": {
    "object_type": "fluorescent round cell-like objects with merged magenta and blue signal",
    "object_count_estimate": 3,
    "size_range": "small to medium with mild variation",
    "density_level": "very sparse",
    "spatial_pattern": "three separated bright objects on dark background",
    "shape_profile": "rounded spots with soft glowing edges",
    "overlap_level": "low"
  },
  "edit_plan": {
    "count_change": "increase",
    "count_delta_range": "1.3-2.0x",
    "size_delta_range": "0.85-1.15x",
    "layout_change_strength": "strong",
    "morphology_change_strength": "subtle",
    "must_keep": [
      "same black fluorescence background",
      "same magenta-blue signal style",
      "same high-contrast dark-field appearance",
      "same soft fluorescent edges",
      "same sparse composition"
    ],
    "must_avoid": [
      "gray brightfield background",
      "new color channels",
      "hard neon edges",
      "large diffuse haze",
      "copying the original triangular layout"
    ]
  }
}
```

### Prompt 草案

```text
Edit this microscopy image based on the reference.

Keep the same fluorescence microscopy modality, the same black background, the same magenta-blue fluorescent color relationship, the same high-contrast dark-field appearance, and the same soft glowing object edges as the reference image.

Preserve these fixed attributes from the reference:
- background: uniform near-black dark field
- illumination: fluorescence-like emissive signal without visible brightfield illumination
- color and tone: magenta and blue fluorescent signal on black background
- focus and blur: moderately focused objects with soft glowing boundaries
- noise and contrast: low noise with high contrast
- biological object type: round fluorescent cell-like objects

Generate a new experimental result within the same image domain:
- change the number of visible fluorescent objects clearly from the reference image
- create about 4 to 6 objects instead of 3
- vary object sizes slightly within a realistic range
- redistribute the fluorescent objects into a new sparse arrangement
- keep the objects realistically rounded with subtle morphology differences
- make the output obviously different from the reference image, not a near-copy

Do not change the imaging modality or background style.
Do not introduce new colors, brightfield texture, hard outlines, text, or artificial fluorescent artifacts.
```

## 3.3 测试图 3

参考图：
[+home+Userlist+fangjiaqi+data+true+Microscope+1016+090000-NONE.png](/Users/wangbei/PycharmProjects/AI-Image/bio-image/qwen-bio-prompt/train/Micro/image/+home+Userlist+fangjiaqi+data+true+Microscope+1016+090000-NONE.png)

### 结构化描述草案

```json
{
  "reference_summary": {
    "imaging_type": "stained histology microscopy",
    "color_mode": "brown and blue tissue staining on light background",
    "background_tone": "light cream-white tissue slide background",
    "background_texture": "dense tissue architecture with tubular and cellular structure",
    "illumination": "even brightfield slide illumination",
    "focus_level": "moderately focused with slight softness",
    "contrast_level": "medium",
    "noise_level": "low",
    "magnification_feel": "medium magnification tissue field"
  },
  "content_summary": {
    "object_type": "stained cells and tissue structures within a histology section",
    "object_count_estimate": "dense field, not discrete single-cell counting",
    "size_range": "small nuclei and medium structural cavities",
    "density_level": "dense",
    "spatial_pattern": "continuous tissue section with irregular local cell density",
    "shape_profile": "mixed tissue microstructures, nuclei, and lumen-like spaces",
    "overlap_level": "high by tissue packing"
  },
  "edit_plan": {
    "count_change": "similar_but_rearranged",
    "count_delta_range": "local density redistribution",
    "size_delta_range": "subtle structural variation only",
    "layout_change_strength": "medium",
    "morphology_change_strength": "subtle",
    "must_keep": [
      "same stained histology modality",
      "same brown-blue staining character",
      "same tissue-section appearance",
      "same brightfield slide illumination",
      "same medium-detail texture level"
    ],
    "must_avoid": [
      "switching to isolated cells on blank background",
      "fluorescent colors",
      "cartoon tissue texture",
      "sharp synthetic contours",
      "destroying the tissue-section appearance"
    ]
  }
}
```

### Prompt 草案

```text
Edit this microscopy image based on the reference.

Keep the same stained histology microscopy style, the same light slide background, the same brown-blue staining character, the same tissue-section appearance, the same brightfield illumination, and the same overall texture scale as the reference image.

Preserve these fixed attributes from the reference:
- background: light cream-white slide background with dense tissue architecture
- illumination: even brightfield slide illumination
- color and tone: brown and blue histology staining
- focus and blur: moderately focused with slight softness
- noise and contrast: low noise and medium contrast
- biological object type: stained cells and tissue structures within a histology section

Generate a new experimental result within the same image domain:
- keep the tissue-section style and the same general structural background
- change the local distribution of stained cells and stained regions clearly from the reference
- vary local cell density and local arrangement inside the tissue
- introduce realistic small changes in nuclear distribution, stained-cell clustering, and local morphology
- keep the result obviously different from the reference image, not a near-copy

Do not change the imaging modality or replace the tissue background.
Do not create isolated cells on an empty background, do not add fluorescent colors, text, markers, or artificial contour lines.
```

## 4. 3 图测试的判定标准

每张测试图生成后，至少人工检查以下 5 项：

1. 成像类型是否保持不变。
2. 背景、照明、色调、模糊和噪声是否与参考图一致。
3. 主体变化是否足够明显。
4. 是否出现复制感、拼贴感、重复结构。
5. 是否出现风格漂移或显著伪影。

建议记录三个标签：

- `pass`
- `change_too_small`
- `style_drift`
- `artifact`

## 5. 下一步

如果这份草案通过确认，下一步就可以继续：

1. 定义最终 JSON 字段枚举和填写规则。
2. 为首轮 3 张测试图形成最终可直接投喂的 prompt/negative prompt。
3. 再决定是否需要从单图动态 prompt 进一步抽象出若干子模板。
