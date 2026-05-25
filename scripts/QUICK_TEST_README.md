## 快速预实验测试 — 运行指南

### 文件位置
脚本已生成在 `scripts/quick_test.py`

### 在 AutoDL 服务器上运行

#### 1. 激活环境 + 验证依赖
```bash
conda activate VAA
cd /path/to/URF-HVAA

# 验证关键包
python -c "
import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import numpy; print('numpy:', numpy.__version__)
import scipy; print('scipy:', scipy.__version__)
import sklearn; print('sklearn:', sklearn.__version__)
import fairscale; print('fairscale: OK')
import tqdm; print('tqdm: OK')
print('All packages OK')
"

# 可选但建议安装（提升 drift 检测精度）
pip install sentence-transformers
```

#### 2. 确认模型文件存在
```bash
ls -lh libs/llama/llama3.1-8b/consolidated.00.pth
ls -lh libs/llama/llama3.1-8b/tokenizer.model
```

#### 3. 确认测试数据存在
```bash
# 5个测试视频的 captions + scores 应该已有
for v in Abuse028_x264 Arrest001_x264 Arson016_x264 Burglary021_x264 Shooting015_x264; do
  echo -n "$v: captions="
  [ -f "data/ucf_crime/captions/video_llama3_json_results/${v}.json" ] && echo -n "OK" || echo -n "MISSING"
  echo -n " scores="
  [ -f "data/ucf_crime/scores/videollama3/${v}.json" ] && echo "OK" || echo "MISSING"
done
```

#### 4. (可选) 确认视频文件 — 决定能否跑 Stage D
```bash
# 如果视频文件存在，可以跑完整的 VLM 靶向验证
for v in Abuse028_x264 Arrest001_x264 Arson016_x264 Burglary021_x264 Shooting015_x264; do
  echo -n "$v: "
  [ -f "data/ucf_crime/videos/${v}.mp4" ] && echo "OK" || echo "NOT FOUND"
done
```

#### 5. 运行测试
```bash
# 完整模式（如果有 .mp4 视频文件）
python scripts/quick_test.py

# 或者强制跳过 Stage D（纯文本模式）
python scripts/quick_test.py --skip-stage-d
```

#### 6. 查看结果
```bash
# 查看 AUC 对比
cat data/ucf_crime/quick_test_results/stage_e_metrics/comparison.txt

# 查看定性分析（flagged frames 是否命中 anomaly）
cat data/ucf_crime/quick_test_results/qualitative_analysis.json

# 查看 Stage C 冲突检测输出
ls data/ucf_crime/quick_test_results/stage_c_flagged/
cat data/ucf_crime/quick_test_results/stage_c_flagged/Abuse028_x264.json
```

### 各 Stage 完成的内容

| Stage | 模型 | 耗时(估) | 输入 | 输出 | 做什么 |
|---|---|---|---|---|---|
| **Pre-flight** | — | 10s | — | — | 检查所有依赖、模型文件、数据文件、GPU |
| **C** | LLM | 5-8min | Phase 1 captions + scores | context windows + flagged list | ①动态百分位阈值找正常帧 ②时间滑动窗口生成场景上下文(Phase 2) ③全遍历逻辑冲突检测,找出"与场景违和"的帧(Phase 3) |
| **定性分析** | — | 1s | flagged frames + GT | precision report | 检查 flagged frames 是否命中 ground truth 异常区间 |
| **D** | VLM | 10-20min | flagged frames + scene context + .mp4 | refined captions | ①仅对 flagged frames 做细粒度重采样(interval=4) ②用反幻觉中立提示词做视觉验证 ③如果没有视频文件则跳过 |
| **E** | LLM | 5-8min | original scores + refined captions | final scores + AUC | ①LLM 对 refined captions 重新打分 ②替换 flagged 位置的分数 ③全局高斯平滑(sigma=2) ④计算 ROC-AUC/PR-AUC ⑤与原始管线 refined_scores 对比 |

### 两种运行模式

| 模式 | 条件 | 能测什么 | 不能测什么 |
|---|---|---|---|
| **FULL** | 有 .mp4 视频文件 + transformers | 完整 5-stage 新管线 | — |
| **TEXT-ONLY** | 无视频文件 | Stage C 冲突检测 + 定性分析 + conflict-aware LLM 重打分 | VLM 靶向重采样(Phase 4a) |

TEXT-ONLY 模式下，Stage E 会将 Stage C 的冲突信息（`suspicious_element` / `alternative_explanation`）注入 LLM 打分 Prompt，测试**冲突感知是否能提升 LLM 评分准确度**。

### 结果解读

- **ROC-AUC / PR-AUC delta > 0** → 新管线优于原始管线
- **Flagged precision 高** → 冲突检测能准确找到真实异常帧
- **Flagged 帧占比低**（<5%）→ VLM 计算量显著节省
