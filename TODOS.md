# TODO — URF-HVAA 改进计划

## 1. 移除 Phase 2 场景常态构建的硬上限 + 时间均匀采样

### 问题

当前 `context_memory.py` 的 `_generate_context()` 用 `max_captions_per_context`（默认 30，quick_test 中为 10）硬截断正常帧 caption 列表。这导致：

1. **信息瓶颈**：窗口内底部 30% 的正常 caption 可能有 30-40 个，只取前 K 个丢弃了大量正常行为样本，场景常态描述不完整
2. **时间分布不均**：取 `captions[:K]` 意味着只取窗口前段的 caption（正常帧按时间排序），后段信息完全丢失
3. **与设计原则矛盾**：CLAUDE.md 明确写了"Dynamic percentile threshold — not hardcoded"，但 caption 数量却被硬编码截断

### 方案

- 移除 `max_captions_per_context` 硬截断
- 改为 token-budget 感知的**时间均匀采样**：在窗口内等间隔选取正常 caption，在 token 预算内尽可能覆盖整个窗口的时间跨度
- 同时提高 Stage C 的 `max_seq_len` 到 4096 或更高，给输入留足够空间

### 相关文件

- `src/reflection/context_memory.py` — `__init__`, `_generate_context()`
- `src/pipeline/stage_c_context_reflect.py` — `max_seq_len` 参数
- `main.py` — `--max-captions-per-context` 参数

### 状态

已讨论，方案已对齐，待实现。`main.py` 已预留 `--max-captions-per-context` 参数（默认 30，0=无限制）。

---

## 2. 🗸 修复 quick_test.py 的 GPU OOM + contexts[0] 错误

已完成（2026-05-26）：
- `main()` 中 LLM 卸载移到 VLM 探活之前，避免 OOM
- `run_stage_d()` 中 `contexts[0]` 改为 `_find_context()` 按时间戳匹配正确窗口
- 随后统一重构到 `main.py` 入口

---

## 3. 🗸 统一入口 main.py + 管线重构

已完成（2026-05-26）：
- 所有 5 个管线 stage 重构为 `run()` 函数（`main()` 仍可用作 CLI）
- 新增 `main.py` 统一入口（`--quick-test` / 全量）
- Stage E 支持 context-aware scoring prompt
- Stage D 增加 per-frame tqdm 进度条
- Stage B/C 增加显式 GPU 清理
- 删除 `scripts/quick_test.py`（逻辑已迁移到管线 + main.py）
- 更新 `scripts/QUICK_TEST_README.md`
