# Micro 数据集生成增强方案

## 1. 目标

基于 `train/Micro/image` 中的训练集显微图数据，挑选约 3000 张参考图，使用 `qwen-image-edit` 生成与参考图属于同一显微成像类型、保持相同背景和成像风格、但在细胞数量、大小、位置和局部形态上明显不同的新图。

本方案只讨论数据筛选、描述结构、prompt 设计和质控流程，不修改现有代码。

## 2. 数据集现状与关键观察

基于当前仓库内数据的快速统计：

- 图像总数：`8514`
- 文件位置：`train/Micro/image`
- 图像格式：`4362` 张 `.png`，`4152` 张 `.jpg`
- 类块数量：`748`
- 同类样本在文件排序后是连续块，平均每类约 `11.38` 张
- 每类样本数量范围：`1` 到 `76`
- 如果按全局顺序“每三个选一个”取样，可得到 `2838` 张参考图

从文件名可稳定拆出两层信息：

- `Microscope+<class_id>+<code-stain>.ext`
- 其中 `<class_id>` 可视为类块 ID
- `<code>` 的前两位可作为显微成像风格或采集子类型的代理字段
- `stain` 部分描述染色/通道组合，如 `NA`、`MB`、`BC`、`JC`、`NONE`

补充说明：

- “类内”原本指的是：按连续出现的同一 `class_id` 视为一个类块，在该类块内部做取样
- 例如某一类块有 9 张图，类内“每三个选一个”就是取第 `1, 4, 7` 张
- 但你当前已经明确不采用这个口径，而是采用“全局取样”

当前数据的一个重要特征是：

- 单一 `class_id` 内经常混有多个 `<code>` 前缀
- 因此 `class_id` 更像“同结构类别块”
- `<code>` 前缀更像“成像/实验子类型”

虽然数据在顺序上存在类块结构，但当前方案按你的要求，直接采用全局顺序下采样。

## 3. 推荐筛选策略

### 3.1 筛选目标

筛选结果应同时满足：

- 保留尽可能多的类块覆盖，避免只集中在头部大类
- 保留主要显微成像风格的分布
- 每个类块内部优先选择“最适合作为生成参考图”的样本
- 尽量减少极差样本、严重伪影样本和信息量过低样本

### 3.2 当前确认的筛选规则

你当前确认的规则是：

- 这部分数据属于训练集
- 筛选时按全局顺序取样
- 整个训练集统一采用“每三个选一个”
- 不再额外按类块或大类做分层筛选

具体执行方式：

- 先对 `train/Micro/image` 中全部图像按当前文件顺序排列
- 从全局列表中固定选择位置 `1, 4, 7, 10, ...`
- 即整批数据统一按全局步长 `3` 下采样

按当前训练集统计：

- 全局“每三个选一个”后，可得到 `2838` 张参考图
- 后续任务定义为“每个参考图生成 1 张新图”
- 因此首轮生成规模对应为 `2838 -> 2838`

### 3.3 全局选图后的质量规则

对全局采样后得到的参考图，再做质量过滤：

1. 先按全局顺序执行“每三个选一个”
2. 再在被选中的图中做质量过滤
3. 如果某张图质量明显不达标，则剔除
4. 是否补选由后续执行阶段再决定

质量控制是第一规则。优先保留：

- 背景均匀、亮度不过曝不过暗
- 主体结构完整，没有大面积裁切
- 焦平面相对稳定，不过度模糊
- 没有明显文字、水印、比例尺遮挡主体
- 细胞/组织区域占画面比例合适，不是几乎空白，也不是完全堆叠

建议剔除：

- 大面积空白背景图
- 严重失焦图
- 强压缩噪声或块效应明显的图
- 颜色或亮度异常、疑似采集失败的图
- 主体被边缘裁断过多的图

### 3.4 推荐输出清单格式

建议最终产出一个参考图清单表，至少包含：

- `ref_path`
- `class_id`
- `style_prefix`
- `stain_tag`
- `global_pick_index`
- `quality_flag`
- `note`

建议额外补两个字段：

- `keep_reason`
- `drop_reason`

这样后面做批量生成、失败重跑、按类型回查都会方便很多。

## 4. 单图到结构化描述与生成 prompt 的任务定义

### 4.1 输入

- 一张参考显微镜图

### 4.2 输出

对每张参考图，输出两部分：

- 面向程序的结构化描述
- 面向生成模型的最终 prompt

目标约束：

- 与参考图属于同一显微成像类型
- 保留相同背景、照明方式、色调和成像风格
- 保留相同层级的模糊程度、噪声水平和对比度
- 只改变实验结果相关内容，即细胞数量、大小、位置、局部形态和局部聚集关系
- 新图必须明显不同于参考图，但不能跳出同一成像域

## 5. 结构化描述设计

当前方案不再依赖“大类模板优先”的方式，而是直接精确到每张图像本身。

推荐把结构化描述拆成两层：

- 固定属性
- 可变属性

也就是说：

- 不先假设某张图一定属于哪套 prompt 模板
- 每张图都先单独解析视觉特征
- prompt 直接由这张图的结构化描述动态生成

### 5.1 固定属性

这些属性在生成图中应尽量保持不变：

- `imaging_type`：brightfield / fluorescence / stained histology / phase-contrast / low-light microscopy 等
- `color_mode`：grayscale / blue-fluorescence / brown-blue staining / mixed
- `background_tone`：亮背景、灰背景、暗背景
- `background_texture`：均匀、轻微颗粒、纤维状、组织底纹
- `illumination`：是否中心亮、边缘暗、照明方向偏置
- `focus_level`：清晰、轻微失焦、明显柔化
- `contrast_level`
- `noise_level`
- `magnification_feel`：高倍稀疏、高倍密集、低倍组织视野
- `border_artifacts`：是否有黑边、边缘阴影、局部遮挡

### 5.2 可变属性

这些属性允许变化，但要在合理范围内：

- `object_type`：细胞团、单细胞、细胞核、组织腔体、纤维束、条带状结构等
- `object_count_estimate`
- `size_range`
- `density_level`
- `spatial_pattern`：随机散布、局部聚团、沿边缘分布、中央稀疏边缘密集等
- `shape_profile`：圆形、椭圆、拉长、不规则、带伪足、空泡感、核深染等
- `overlap_level`
- `local_variation`：允许某些区域更密、某些区域更稀

### 5.3 变化控制字段

建议额外输出一组显式控制字段，避免模型只做微小改动：

- `count_change`: `increase` / `decrease` / `similar_but_rearranged`
- `count_delta_range`: 例如 `0.7-1.4x`
- `size_delta_range`: 例如 `0.8-1.25x`
- `layout_change_strength`: `medium` / `strong`
- `morphology_change_strength`: `subtle` / `medium`
- `must_keep`: 背景、照明、色调、模糊、噪声、成像域
- `must_avoid`: 文本、人工轮廓、锐化边缘、卡通感、跨模态变化

### 5.4 每图必须满足的硬约束

每张图生成时都必须满足以下规则：

- 与参考图属于同一显微成像类型
- 保留相同背景、照明方式、色调和成像风格
- 保留相同层级的模糊程度、噪声水平和对比度
- 只改变实验结果相关内容，即细胞数量、大小、位置、局部形态和局部聚集关系
- 新图必须明显不同于参考图，但不能跳出同一成像域

## 6. 推荐的结构化输出格式

建议输出 JSON 或 JSON-like 字典。示例：

```json
{
  "reference_summary": {
    "imaging_type": "brightfield microscopy",
    "color_mode": "light gray background with faint gray cell bodies",
    "background_tone": "uniform pale gray",
    "background_texture": "smooth with slight low-frequency illumination variation",
    "illumination": "soft diffuse illumination",
    "focus_level": "slightly out of focus",
    "contrast_level": "low to medium",
    "noise_level": "low",
    "magnification_feel": "medium magnification, sparse field"
  },
  "content_summary": {
    "object_type": "rounded adherent cells",
    "object_count_estimate": 18,
    "size_range": "small to medium, mildly varied",
    "density_level": "sparse",
    "spatial_pattern": "random scatter with a small local cluster in the upper-right region",
    "shape_profile": "mostly round to oval, soft blurry boundaries",
    "overlap_level": "low"
  },
  "edit_plan": {
    "count_change": "increase",
    "count_delta_range": "1.2-1.6x",
    "size_delta_range": "0.85-1.2x",
    "layout_change_strength": "strong",
    "morphology_change_strength": "medium",
    "must_keep": [
      "same imaging modality",
      "same background brightness",
      "same grayscale tone",
      "same blur level",
      "same noise character"
    ],
    "must_avoid": [
      "sharp outlines",
      "artificial textures",
      "cartoon-like edges",
      "new non-biological structures",
      "changing microscopy modality"
    ]
  }
}
```

## 7. Prompt 生成策略

当前 prompt 策略改为“单图精确描述驱动”：

1. 不先套大类模板
2. 先解析当前参考图的具体成像属性
3. 再基于这张图的属性拼装动态 prompt

推荐把最终 prompt 拆成四段：

1. 当前单图的成像域锁定
2. 当前单图的背景与风格锁定
3. 当前单图的主体与变化指令
4. 禁止项

### 7.1 动态 Prompt 模板

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

### 7.2 更适合显微图的负向 prompt

```text
sharp contour, hard edge, halo artifact, oversharpening, cartoon texture,
synthetic pattern, repeated clone artifacts, grid artifact, mosaic artifact,
text, watermark, labels, ruler overlay, non-biological objects,
wrong microscopy modality, dramatic color shift, unrealistic contrast,
background replacement, illumination mismatch, copied object layout
```

### 7.3 动态调整原则

每张图的 prompt 都应动态调整以下字段：

- `background_tone_and_texture`
- `illumination`
- `color_mode`
- `focus_level`
- `noise_contrast_profile`
- `object_type`
- `object_count_estimate`
- `density_level`
- `spatial_pattern`
- `shape_profile`
- `count_delta_range`
- `size_delta_range`
- `layout_change_strength`
- `morphology_change_strength`

也就是说：

- 每张图都要动态调整
- 但所有动态变化都必须服从第 5.4 节的硬约束

## 8. 单图处理建议流程

推荐按两阶段做，而不是直接让模型只吃一句 prompt。

### 阶段 A：视觉解析

输入参考图，先得到结构化描述：

- 判定当前图像的具体显微成像类型
- 判定背景和成像风格
- 估计主体数量、密度、尺度和分布
- 给出允许变化范围

### 阶段 B：条件生成

把“参考图 + 结构化描述 + 动态 prompt + negative prompt”一起送入生成模型。

这样做的好处是：

- prompt 更稳定
- 更容易控制“哪些不变，哪些可变”
- 便于后续做批量规则和失败重试

## 9. 生成结果质控建议

每张生成图至少检查四项：

- 是否仍属于同一显微成像类型
- 是否保留了原背景、照明、色调和成像风格
- 是否保留了同层级的模糊、噪声和对比度
- 是否与参考图有明显差异，而不是轻微平移或局部复制
- 是否出现伪影，如重复克隆、边缘发光、过锐化、异常纹理

建议把失败样本分成三类：

- `style_drift`：跑偏到其他成像域
- `change_too_small`：改动太弱
- `artifact`：伪影过多

这样后面方便分别调 prompt、负向 prompt 和 CFG/step。

## 10. 当前推荐落地版本

如果现在先做第一版，我建议直接采用下面的最小可行方案：

### 数据筛选

- 使用 `train/Micro/image`
- 全局按 `1, 4, 7, 10, ...` 取样
- 先得到约 `2838` 张参考图
- 质量控制优先，如果某张图质量差则先剔除

### 单图生成

- 先为每张参考图输出一份结构化描述 JSON
- 直接基于单图结构化描述生成动态 prompt
- prompt 重点写“保持成像域不变，只改变结果形态”
- 每张图都必须满足第 5.4 节的硬约束

### 评估

- 先做 `3` 张代表图试生成
- 3 张图都按单图动态 prompt 生成
- 先人工判断哪些图最容易风格漂移、哪些图最容易改动不足
- 再决定是否需要在后续阶段引入子模板

## 11. 需要确认的关键点

在正式执行前，建议先确认四件事：

1. 筛选是否正式固定为：训练集内全局“每三个选一个”
2. 是否接受首轮参考图数量约为 `2838`
3. 3 图测试是否直接按“单图动态 prompt”执行，而不先分大类模板
4. 后续结构化描述是否固定成机器可读 JSON，以便批量跑

如果这四点成立，这个方案就可以进入下一步：给出最终筛选规范和首轮 3 图动态 prompt 草案。
