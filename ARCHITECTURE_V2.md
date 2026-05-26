# Architecture v2 — 非对称双通路反射 + 对抗性重审视

> 版本：v2 | 日期：2026-05-26 | 状态：设计阶段（未实现）

## 与 v1 的核心差异

| 维度 | v1（原 refine） | v2（优化方案） |
|---|---|---|
| VLM 重检目标 | 最高分区间 | 分数连续性区间 + 冲突检测区间（双源） |
| VLM 输出 | 单一 tag | (tag, confidence) + (negative_tag, confidence) |
| LLM 评分 | tag 注入（单极） | 正负双向 tag 注入 |
| Score Gate | 单阈值 std² | 双阈值 + 高分帧密度约束 |
| 时间窗口 | N/A（看整个最高分区间） | 自适应邻域（基于分数连续性） |
| 闭环验证 | 无 | 条件式二次反射 |

---

## Pipeline Flow（5 阶段）

```
Stage A [VLM]: Coarse captioning, interval=16, 单 prompt（保持极简）
    → captions/phase1_coarse/{video}.json

Stage B [LLM]: 初始评分
    → scores/phase1_initial/{video}.json

--- Score Gate 前置（新增）---
    ↓
条件 A: Max_Score < 0.3 且 Var < 0.05 → 直接输出为正常
条件 B: Max_Score > 0.85 且 high_score_frames/total > 10% → 直接输出为异常
其他 → 触发反射回路（Stage C → D → E）
    ↓

Stage C [LLM]: 异常区间定位 + 自适应时间窗口
    → 基于分数连续性计算邻域（不固定 k，不固定超参数）
    → 对每个候选区间生成简要场景描述
    → reflection/phase3_candidates/{video}.json

Stage D [VLM]: 对抗性重审视（双视角验证）
    → 对每个候选区间做正反向 VLM 提问
    → 输出: (caption_refined, positive_tag, confidence, negative_tag, confidence)
    → captions/phase4_fine/{video}.json

Stage E [LLM]: 双向 Tag 锚点评分
    → 注入正负 tag + 置信度
    → LLM 自行权衡，全帧重打分
    → scores/final/{video}.json

--- 条件式二次反射（可选）---
    ↓
final_avg_score ∈ [0.35, 0.65]? → 回到 Stage D（最多 2 轮）
```

---

## 1. Dual-Threshold Score Gate（双阈值门控）

```
条件 A（极度正常）: Max_Score < 0.3 且 Variance < 0.05
    → 跳过反射，输出 [全正常] 分数
条件 B（极度异常）: Max_Score > 0.85 且 high_score_frames/total > 10%
    → 跳过反射，保留 Phase 1 原始高分
    → 需高分帧有足够的空间密度，防止 VLM 幻觉导致的单帧尖峰误判
其他 → 触发 Stage C/D/E 反射回路
```

**为什么条件 B 需要密度约束**：Shooting 案例中，VLM 幻觉让帧 64 得了 1.0 分，如果只看 Max_Score 就直接判异常，假阳性永远无法纠正。

---

## 2. 自适应时间动量采样（无固定超参数）

```
对于候选异常帧 t:
   向前搜索 scores[t-i] > score_percentile(70) 的最远连续帧 → t_left
   向后搜索 scores[t+i] > score_percentile(70) 的最远连续帧 → t_right
   VLM 检视区间 = [t_left, t_right]
```

- 孤立分数尖峰 → 窗口窄（只看附近），自动过滤幻觉
- 持续高分区间 → 窗口宽，VLM 看到完整事件过程
- threshold 用每视频自己的 `score_percentile(70)` 动态决定

---

## 3. 对抗性重审视（Stage D 核心升级）

VLM 对每个候选区间做双视角验证：

```
视角 A（正向询问）:
    "Describe what is actually happening in this video segment.
     Report only observable facts. Do not speculate."

视角 B（反向挑战）:
    "Argue why the events in this segment could be part of a normal,
     non-suspicious situation. What innocent explanations exist?"
```

**输出结构**：
```json
{
    "frame": 1488,
    "caption_refined": "A person approaches police officer near patrol car...",
    "positive_tag": "police officer physically attacked",
    "positive_confidence": 0.72,
    "negative_tag": "officer tripped on uneven pavement, passerby approached to help",
    "negative_confidence": 0.84
}
```

**为什么在 Stage D 加双视角不浪费计算**：VLM 已经加载，多一个视角只是一次 forward pass（~1-2s），而非重新加载模型（~30s）。Stage D 只对 flagged 区间执行，在全局算力中占比极小。

---

## 4. 双向 Tag 锚点评分（Stage E）

LLM 评分 prompt 同时包含正负信号：

```
[潜在异常 — VLM 视觉置信度 0.72]: police officer attacked
[可能正常解释 — VLM 视觉置信度 0.84]: pedestrian approached to ask directions,
    officer fell accidentally

[评分指南]:
  - 两个判断的置信度仅供你参考
  - 请根据 caption 的客观内容自行判断权重
  - 不要盲从任一标签
```

**设计意图**：LLM 看到对立解释 → 被迫做权衡 → 不会盲从单极 tag 放大假阳性。

---

## 5. 条件式二次反射（可选，极端模糊视频触发）

```
Stage E 输出
    ↓
final_avg_score ∈ [0.35, 0.65]?
    ├── NO  → 输出最终分数
    └── YES → 用 Stage E 的高分帧更新 flagged 区间，回到 Stage D（最多 2 轮）
```

只对最难判断的视频触发，计算开销可控。

---

## 算力账

| 阶段 | v1 原架构 | v2 优化后 | 变化 |
|---|---|---|---|
| Stage A VLM | interval=16，单 prompt | 不变 | 0 |
| Stage B LLM | 全量评分 | 不变 | 0 |
| Score Gate | N/A | LLM forward 忽略不计 | ~0 |
| Stage C LLM | 全遍历冲突检测 | 区间定位（更轻） | ↓ |
| Stage D VLM | flagged 帧，单 prompt | flagged 区间，双 prompt | +1 forward/区间 |
| Stage E LLM | 上下文评分 | 双向 tag 评分 | 相同 forward 数 |

## 理论闭环

> Phase 1 允许犯错（算力优先）→ Score Gate 过滤确定案例 → Stage C 自适应定位异常区间 → Stage D 对抗性重审视（双向验证 + 置信度解耦）→ Stage E 双向 tag 锚点（LLM 自主权衡）→ 极端案例可选二次反射
