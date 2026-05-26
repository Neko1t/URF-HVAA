# Architecture v1 — Original Refine Pipeline (`refine_with_tag.py`)

> 版本：v1 | 日期：2025-03 | 状态：基线

## Pipeline Flow

```
video_pre_caption.py [VLM]
    → 每 16 帧抽一帧，生成 coarse caption

llm_anomaly_scorer.py [LLM]
    → 对每个 caption 打分 (0-1)

score_filter.py
    → 滑窗（window = max(total_frames/10, 300)）找最高分和最低分区间
    → 输出 highest_avg_score, lowest_avg_score, std

summarize_window.py [VLM]
    → 用眼睛看最高分区间的实际视频画面
    → 输出可疑行为标签 (tag)，如 "physical assault, robbery"

refine_with_tag.py [LLM]
    → Score Gate 判断 + Tag 注入 + 全帧重打分
```

## Score Gate 机制

```
threshold_margin = std²
不确定区间 = [0.5 - std², 0.5 + std²]

if highest_avg_score ∈ 区间:
    触发 refine（LLM 不确定有没有异常）
else:
    跳过（LLM 已有把握，refine 无意义）
```

**核心思想**：`highest_avg_score` 是 LLM 对"视频有异常"的置信度代理指标。
- 太高（> upper）：LLM 确信有异常，不需要 refine
- 太低（< lower）：LLM 确信没异常，不需要 refine
- 在 0.5 附近：LLM 不确定，需要 VLM 标签辅助

**为什么用 std²？** 方差大 → 分数分散 → LLM 判断一致性差 → 区间放宽 → 更多视频被 refine。

## Tag 注入

```python
system_prompt += f"\n[Potentially reported suspicious activities: {tag}]"
```

Tag 被注入到**每一帧**的评分 prompt 中，作为全局语义锚点，引导 LLM 以一致的视角重新评估所有帧。

## 两个跳过条件（视频级别）

1. Score Gate 不过：`highest_avg_score` 在不确定区间外
2. 无 tag：`suspicious_phrases` 中无该视频的 tag

两条都通过才执行带 tag 的全帧重打分。

## 为什么有效

| 要素 | 作用 |
|---|---|
| VLM 视觉验证 | Tag 来自真实视觉信号，非文本推理（Grounded） |
| Score Gate | 只在 LLM 不确信时干预，避免过度修正 |
| Tag 语义锚点 | 给 LLM 统一的观察视角，消除评分不一致性 |
| 全帧重打分 | Tag 作为先验一致作用于所有帧，而非只修正局部 |
