# SVF-Panel：语义速度场与分镜叙事关键帧选择

SVF-Panel 是一个 training-free 的长视频关键帧方法。它复用
`preprocess/extract.py` 生成的 query-frame 相关性和视觉特征，但不使用小波：
把帧特征看成语义空间中的轨迹，以语义速度、方向曲率和 query 相关变化寻找事件边界，
再用固定预算下的漫画 closure（沟壑补全）优化关键帧间的叙事连续性。

## 1. 完善后的核心定义

为避免术语混淆，本实现固定使用以下三个层级：

- **panel**：一个被选中的关键帧；
- **page**：时间连续、语义相近的一组 panel；
- **collage**：推理侧可选的像素拼贴图，本模块暂不生成。

给定第 `t` 个采样帧的视觉特征 `f_t` 和 query 相关性 `s_t`，先对特征做
L2 归一化与局部平滑，得到语义轨迹 `x_t`：

```text
v_t     = x_t - x_(t-1)
speed_t = ||v_t||_2
curv_t  = (1 - cos(v_t, v_(t-1))) / 2
grad_t  = |r_t - r_(t-1)|
gate_t  = gate_floor + (1 - gate_floor) * r_t
energy_t = (alpha * speed_t + beta * curv_t) * gate_t + gamma * grad_t
```

其中 `r_t` 是鲁棒归一化到 `[0, 1]` 的 query 相关性。`gate_floor` 保留少量
与 query 弱相关但对叙事连续性必要的场景边界，避免硬门控把上下文完全切掉。

所有场都在帧维度对齐：`speed_t` 表示从 `t-1` 到 `t` 的变化，检测到的峰 `t`
因此被解释为新片段的起点。输入中若包含 token/spatial 维度，会先沿中间维池化，
保留最后一维作为 embedding。

## 2. 关键帧选择

1. 对 `energy_t` 使用 median/MAD 高度阈值、prominence 和最小峰间距寻找边界。
2. 边界把视频切成半开区间 `[start, end)`。
3. 按片段时长、平均相关性、最大相关性和平均边界能量计算片段重要性。
4. 用容量受限的整数分配保证总预算精确，预算足够时每个片段至少获得一帧。
5. 片段内使用 MMR，帧效用综合相关性、边界能量、速度和曲率，多样性由视觉
   embedding 余弦相似度衡量。

这比“把所有峰都当必选帧”更稳健：当事件数超过帧预算时，高 query 价值片段优先；
当事件数较少时，剩余预算自然分配给高价值且内部多样的片段。

## 3. 固定预算下的 gutter 优化

相邻 panel `k_i, k_(i+1)` 的 gutter 定义为归一化语义弧长：

```text
gutter(i, i+1) = sum(speed_t, t=k_i+1..k_(i+1)) / sum(speed_t, whole video)
```

若整段视频语义速度近似为零，则退化为归一化时间距离。对 `B` 帧预算，理想平均
gutter 为 `1/(B-1)`；低阈值和高阈值由该目标乘以系数得到，因此会随视频和预算
自适应。

- gutter 过宽：在弧长中点附近加入 bridge panel；
- gutter 过窄：优先删除低效用、冗余的 panel；
- 每次插入都伴随一次删除，任何时候都保持帧预算不变；
- 交换次数受 `refinement_steps` 限制，并用历史状态防止振荡。

## 4. Narrative 输出

最终 JSON 保留原有 `keyframe_indices`，所以当前 lmms-eval keyframe 路径可以直接
读取。额外的 `narrative` 包含：

```json
{
  "method": "svf_panel_v1",
  "field_source": "visual_features",
  "panels": [
    {"position": 0, "sampled_index": 12, "frame_index": 300,
     "relevance": 0.91, "boundary_energy": 0.43}
  ],
  "pages": [[0, 1, 2], [3, 4]],
  "transitions": [
    {"from": 0, "to": 1, "from_frame": 300, "to_frame": 450,
     "type": "action_to_action", "gutter": 0.08}
  ],
  "emphasis": {"3": 1.82}
}
```

转场类型是由速度、曲率、端点视觉差异和相关性连续度得到的**可操作代理标签**，
不是对 McCloud 六类转场的语义真值识别。它适合用于消融、prompt 提示或后续
collage 布局，但实验报告中不应将其表述为训练过的转场分类器。

## 5. 运行

预处理与 WFS 相同。已有特征后可运行：

```bash
python3 -m svf.pipeline \
  --benchmark videomme \
  --feature_model blip2 \
  --max_frames 16
```

输出默认写到 `outputs/<benchmark>/SVF_Panel_<benchmark>_<model>_<N>f.json`。
使用 `--no_narrative` 可只保留兼容字段；使用 `--no_visual_features` 会进入
`query_proxy` 降级模式，该结果不能作为完整 SVF 方法的正式实验结果。

## 6. 当前边界与下一阶段

当前版本完成“弱耦合”和结构元数据：选帧结果无需修改 LVLM 即可生效，narrative
可供后续推理侧读取。它尚未把多个 panel 合成为一张 collage，也未按 emphasis
动态分配视觉像素。实现 collage 前应先做三组消融：

1. `WFS` 对比 `SVF selection only`；
2. `SVF selection only` 对比 `SVF + gutter refinement`；
3. 在前两项成立后，再比较原生 video 输入与 collage 输入。

这样可以区分收益来自选帧、时序交换还是像素布局，避免把三个变量绑在一次实验里。
