## 快速预实验测试 — 运行指南

### 统一入口

预实验和全量实验现在共用同一套管线代码，通过 `main.py` 统一入口：

```bash
cd /path/to/URF-HVAA

# 预实验（5 个代表性视频）
python main.py --quick-test

# 全量实验（所有视频）
python main.py

# 强制跳过 Stage D（纯文本模式）
python main.py --quick-test --skip-stage-d

# 切换数据集
python main.py --dataset xd_violence
```

`--quick-test` 和全量实验使用**完全相同的管线代码**（`src/pipeline/stage_*.py`），区别仅在于处理 5 个视频还是全量视频。

### 各 Stage 管线也可独立调用

```bash
# Stage A: VLM 粗粒度盲描述
python src/pipeline/stage_a_coarse_caption.py \
    --video_folder data/ucf_crime/videos \
    --index_file data/ucf_crime/annotations/test.txt \
    --output_dir data/ucf_crime/captions/phase1_coarse

# Stage B: LLM 初步打分
python src/pipeline/stage_b_initial_scoring.py \
    --root_path data/ucf_crime \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse \
    --output_dir data/ucf_crime/scores/phase1_initial \
    --ckpt_dir libs/llama/llama3.1-8b \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model

# Stage C: 场景上下文 + 冲突检测
python src/pipeline/stage_c_context_reflect.py \
    --root_path data/ucf_crime \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse \
    --scores_dir data/ucf_crime/scores/phase1_initial \
    --context_output data/ucf_crime/context/phase2 \
    --flagged_output data/ucf_crime/reflection/phase3_flagged \
    --ckpt_dir libs/llama/llama3.1-8b \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model

# Stage D: VLM 靶向验证
python src/pipeline/stage_d_targeted_verify.py \
    --flagged_dir data/ucf_crime/reflection/phase3_flagged \
    --context_dir data/ucf_crime/context/phase2 \
    --video_folder data/ucf_crime/videos \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --output_dir data/ucf_crime/captions/phase4_fine \
    --root_path data/ucf_crime

# Stage E: 最终打分 + 合并 + 评估
python src/pipeline/stage_e_final_merge.py \
    --root_path data/ucf_crime \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --original_scores_dir data/ucf_crime/scores/phase1_initial \
    --refined_captions_dir data/ucf_crime/captions/phase4_fine \
    --output_dir data/ucf_crime/scores/final \
    --ckpt_dir libs/llama/llama3.1-8b \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model \
    --context_dir data/ucf_crime/context/phase2 \
    --flagged_dir data/ucf_crime/reflection/phase3_flagged \
    --run_eval \
    --temporal_annotation_file data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

### GPU 生命周期

单 RTX 3090 (24GB) 上 VLM (~15GB) 和 LLM (~16GB) 串行加载：

```
[VLM load] → Stage A → [VLM unload]
[LLM load] → Stage B → [LLM del + empty_cache]
[LLM load] → Stage C → [LLM del + empty_cache]
[VLM load] → Stage D → [VLM unload]    (conditional)
[LLM load] → Stage E → [LLM del + empty_cache]
```

每个 Stage 内部自行管理模型加载和释放，`main.py` 只负责按顺序调用。

### 5 个预实验视频

| Anomaly Type | Video | Frames | Duration |
|---|---|---|---|
| Abuse | Abuse028_x264 | 1412 | 47s |
| Arrest | Arrest001_x264 | 2374 | 79s |
| Arson | Arson016_x264 | 1795 | 60s |
| Burglary | Burglary021_x264 | 1537 | 51s |
| Shooting | Shooting015_x264 | 1713 | 57s |
