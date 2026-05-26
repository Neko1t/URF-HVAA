# URF-HVAA — Asymmetric Dual-Pass Reflection for Video Anomaly Detection

> A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis (NeurIPS 2025)

[[Project Page](https://rathgrith.github.io/Unified_Frame_VAA/)] [[Paper](https://openreview.net/pdf?id=Qla5PqFL0s)]

---

## Overview

This repository implements the **Asymmetric Dual-Pass Reflection** architecture — a training-free, zero-shot pipeline for Video Anomaly Detection (VAD), Localisation (VAL), and Understanding (VAU).

**Core idea**: A lightweight LLM performs full-traversal reasoning to detect local-global contradictions, then guides an expensive VLM to re-examine only the suspicious frames. Like human cognition: when told "there may be a fire nearby," your eyes become sensitive to brightness, smoke, and red — rather than dismissing them as neon lights.

### Models

| Model | Role | VRAM |
|-------|------|------|
| VideoLLaMA3-7B (`DAMO-NLP-SG/VideoLLaMA3-7B`) | Visual perception (VLM) | ~15 GB |
| Llama 3.1 8B Instruct | Text reasoning (LLM) | ~16 GB |

GPU requirement: single RTX 3090/4090 (24 GB). VLM and LLM are **never loaded simultaneously**.

---

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate VAA
pip install -r requirements.txt
pip install sentence-transformers   # optional — for drift detection
```

### 2. Models

Place model files under `libs/`:

```
libs/
├── llama/
│   ├── llama/             # Llama model code
│   └── llama3.1-8b/       # consolidated.00.pth, params.json, tokenizer.model
├── VideoLLaMA3-7B/        # VideoLLaMA3-7B checkpoint (HuggingFace format)
└── embedder/              # sentence-transformers model (optional)
```

Download Llama 3.1 8B checkpoint:
- [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main/original)
- [ModelScope](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/files)
- SHA256 of `consolidated.00.pth`: `ab33d910f405204e5d388bc3521503584800461dc96808e287821dd451c1edac`

### 3. Data

```
data/{dataset}/
├── annotations/
│   ├── test.txt                                    # video index
│   └── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
├── videos/           # .mp4 files (optional for text-only mode)
├── frames/           # extracted .jpg sequences (optional)
├── captions/         # pre-computed captions (legacy baseline)
│   └── video_llama3_json_results/
└── scores/           # pre-computed scores (legacy baseline)
    └── videollama3/
```

Pre-computed captions and scores for UCF-Crime are available on [Google Drive](https://drive.google.com/file/d/1jULt7PKZDTronu4eqiMwCqteKRjjVlmn/view?usp=sharing).

### 4. Run

```bash
# Pre-experiment — 5 videos, ~25 min (starts from Stage C, uses pre-computed data)
python main.py --quick-test

# Pre-experiment — full pipeline including VLM (starts from Stage A)
python main.py --quick-test --resume-from A

# Full experiment — all videos
python main.py

# Force text-only mode (skip VLM Stage D)
python main.py --quick-test --skip-stage-d
```

---

## New Pipeline: Asymmetric Dual-Pass Reflection

### Architecture (5 Stages)

```
Stage A [VLM ~15GB]   Coarse blind captioning (interval=16, max_frames=8)
         ↓            captions/phase1_coarse/{video}.json
Stage B [LLM ~16GB]   Initial anomaly scoring (0–1)
         ↓            scores/phase1_initial/{video}.json
Stage C [LLM ~16GB]   Phase 2: Sliding-window scene context memory
                      Phase 3: Full-traversal logical conflict detection
         ↓            context/phase2/{video}_windows.json
         ↓            reflection/phase3_flagged/{video}.json
Stage D [VLM ~15GB]   Targeted fine-grained verification (interval=4, max_frames=16)
         ↓            Anti-hallucination neutral prompts
         ↓            captions/phase4_fine/{video}.json
Stage E [LLM ~16GB]   Context-aware re-scoring + score merge + Gaussian smoothing
                      AUC evaluation (ROC-AUC / PR-AUC)
                      scores/final/{video}.json
```

### GPU Lifecycle

Each stage explicitly cleans up GPU before returning. No two models are ever co-resident:

```
[VLM load] → Stage A → engine.unload() → empty_cache()
[LLM load] → Stage B → scorer.cleanup() → del + empty_cache()
[LLM load] → Stage C → del generator → empty_cache()
[VLM load] → Stage D → engine.unload() → empty_cache()
[LLM load] → Stage E → del generator → empty_cache()
```

Peak VRAM: ~16 GB. RTX 3090/4090 (24 GB) is fully sufficient.

### Design Principles

1. **Serial model loading** — GPU hosts only one model at a time
2. **Asymmetric compute** — LLM does full traversal (cheap); VLM is on-demand (expensive)
3. **Anti-hallucination** — Phase 4 prompts are neutral verification queries
4. **Coarse-to-fine sampling** — interval=16 for patrol, interval=4 for focus
5. **Context-aware final scoring** — Stage E injects scene baseline + conflict notes into LLM prompts

---

## v2 Optimizations (experimental)

The v2 architecture (`ARCHITECTURE_V2.md`) adds three pluggable optimizations on top of the core pipeline. All are **opt-in** — the pipeline behaves identically to v1 when not enabled.

### Score Gate (`--score-gate`)

A dual-threshold gate between Stage B and C that skips the reflection loop (C/D/E) for videos the LLM is already confident about:

| Condition | Criteria | Action |
|-----------|----------|--------|
| Extremely Normal | Max_Score < 0.3 AND Variance < 0.05 | Skip reflection, output all-low scores |
| Extremely Anomalous | Max_Score > 0.85 AND high-density > 10% | Skip reflection, keep Phase 1 raw scores |
| Ambiguous | Everything else | Trigger Stage C → D → E |

The density constraint on the Anomalous condition prevents a single VLM hallucination spike from skipping reflection.

### Adversarial Verification (`--adversarial`)

Upgrades Stage D from single-perspective to dual-perspective VLM verification:

```
Pass 1 (Positive): "What anomalous activity exists?"
Pass 2 (Negative): "Why could this be a normal situation?"
```

Outputs `(positive_tag, confidence)` and `(negative_tag, confidence)` — both injected into Stage E's scoring prompt so the LLM weighs competing interpretations rather than blindly trusting one.

### Interval-based Conflict Detection (`--detection-mode intervals`)

Replaces full-traversal LLM conflict detection with score-continuity-based candidate intervals. Only frames within high-score regions are sent to the LLM for conflict analysis, reducing Stage C LLM calls.

### Usage

```bash
# v1 baseline (no v2 features enabled)
python main.py --quick-test

# Enable Score Gate only
python main.py --quick-test --score-gate

# Enable Adversarial VLM verification
python main.py --quick-test --adversarial

# Full v2 (all optimizations)
python main.py --quick-test --score-gate --adversarial

# Ablation: Score Gate + intervals mode, no adversarial
python main.py --quick-test --score-gate --detection-mode intervals

# Full experiment with v2
python main.py --score-gate --adversarial
```

---

## `main.py` — Unified Entry Point

### Usage

| Command | What it does | Est. time |
|---------|-------------|-----------|
| `python main.py --quick-test` | 5 videos, **starts from Stage C**, uses pre-computed data | ~25 min |
| `python main.py --quick-test --resume-from A` | 5 videos, **full pipeline** from VLM captioning | ~40–50 min |
| `python main.py --quick-test --resume-from B` | 5 videos, from LLM scoring (needs Stage A output) | ~30 min |
| `python main.py` | **Full experiment** — all videos, from Stage A | hours |
| `python main.py --resume-from C` | Full experiment, skip A+B (use pre-computed data) | hours |
| `python main.py --dataset xd_violence` | Switch dataset | — |

### All CLI Arguments

```
--quick-test              Run on 5 representative videos instead of all
--resume-from {A,B,C,D,E} Stage to start from (default: A for full, C for quick-test)
--dataset NAME            Dataset under data/ (default: ucf_crime)
--output-base PATH        Override base output directory
--skip-stage-d            Force skip VLM Stage D (text-only mode)
--no-eval                 Skip final AUC evaluation
--max-captions-per-context N   Max normal captions per scene context window (default: 30)

v2 (opt-in):
--score-gate              Enable dual-threshold Score Gate between Stage B and C
--adversarial             Enable dual-perspective adversarial VLM in Stage D
--detection-mode {intervals,full}  Stage C mode: intervals (v2, lighter) or full (v1)
```

### Quick-test Videos

| Type | Video | Frames | Duration |
|------|-------|--------|----------|
| Abuse | Abuse028_x264 | 1,412 | 47s |
| Arrest | Arrest001_x264 | 2,374 | 79s |
| Arson | Arson016_x264 | 1,795 | 60s |
| Burglary | Burglary021_x264 | 1,537 | 51s |
| Shooting | Shooting015_x264 | 1,713 | 57s |

---

## Individual Stage CLI

Each stage can also run independently. All accept standard argparse arguments.

### Stage A — Coarse Blind Captioning [VLM]

```bash
python src/pipeline/stage_a_coarse_caption.py \
    --video_folder data/ucf_crime/videos \
    --index_file data/ucf_crime/annotations/test.txt \
    --output_dir data/ucf_crime/captions/phase1_coarse \
    --frame_interval 16 --mode coarse
```

### Stage B — Initial Scoring [LLM]

```bash
python src/pipeline/stage_b_initial_scoring.py \
    --root_path data/ucf_crime \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse \
    --output_dir data/ucf_crime/scores/phase1_initial \
    --ckpt_dir libs/llama/llama3.1-8b \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model
```

### Stage C — Context Memory + Conflict Detection [LLM]

```bash
python src/pipeline/stage_c_context_reflect.py \
    --root_path data/ucf_crime \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse \
    --scores_dir data/ucf_crime/scores/phase1_initial \
    --video_folder data/ucf_crime/videos \
    --context_output data/ucf_crime/context/phase2 \
    --flagged_output data/ucf_crime/reflection/phase3_flagged \
    --ckpt_dir libs/llama/llama3.1-8b \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model \
    --max_seq_len 4096 --max_gen_len 2048
```

Key parameters: `--window_seconds 60 --stride_seconds 30 --normality_percentile 30 --cap_max_flags 20`

### Stage D — Targeted Visual Verification [VLM]

```bash
python src/pipeline/stage_d_targeted_verify.py \
    --flagged_dir data/ucf_crime/reflection/phase3_flagged \
    --context_dir data/ucf_crime/context/phase2 \
    --video_folder data/ucf_crime/videos \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --output_dir data/ucf_crime/captions/phase4_fine \
    --root_path data/ucf_crime
```

Skip this stage if video `.mp4` files are unavailable.

### Stage E — Final Scoring + Merge + Evaluation [LLM]

```bash
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
    --video_folder data/ucf_crime/videos \
    --run_eval \
    --temporal_annotation_file data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

When `--context_dir` and `--flagged_dir` are provided, Stage E uses the enriched scoring prompt with scene baseline + conflict notes. Without them, it falls back to simple scoring (backward compatible).

---

## New Pipeline Output Structure

```
data/{dataset}/
├── captions/
│   ├── phase1_coarse/         # Stage A output: blind captions
│   └── phase4_fine/           # Stage D output: refined captions (flagged frames only)
├── scores/
│   ├── phase1_initial/        # Stage B output: first-round scores
│   └── final/                 # Stage E output: merged + smoothed scores
│       └── metrics/           # ROC-AUC, PR-AUC, optimal threshold
├── context/
│   └── phase2/                # Stage C output: scene context windows
└── reflection/
    └── phase3_flagged/        # Stage C output: flagged frame lists
```

---

## Original Baseline Pipeline

The original pipeline scripts are preserved under `src/` for reference and backward compatibility:

```bash
# 1. VLM pre-captioning (every 16 frames)
python src/video_pre_caption.py \
    --video_folder data/{dataset}/videos \
    --index_file data/{dataset}/annotations/test.txt \
    --output_dir data/{dataset}/captions/{experiment_name} --interval 10

# 2. LLM first-round scoring
bash scripts/query_llm_vad.sh

# 3. Sliding window suspicious segment extraction
python src/score_filter.py

# 4. Anomaly tag extraction
python src/summarize_window.py

# 5. Score refinement with tags
bash scripts/refine_score.sh

# 6. Evaluation
bash scripts/eval_{dataset_name}.sh
```

These original scripts remain unchanged. The new pipeline in `src/pipeline/` replaces steps 1–6 with the 5-stage architecture.

---

## Module Map

| New Module | Replaces (original) |
|------------|---------------------|
| `src/perception/vlm_engine.py` | `video_pre_caption.py`, `summarize_window.py` (VLM parts) |
| `src/reflection/context_memory.py` | `summarize_window.py` (context logic) |
| `src/reflection/conflict_detector.py` | `refine_with_tag.py` (new logic — contradiction detection) |
| `src/reflection/targeted_verifier.py` | New (no equivalent in original) |
| `src/pipeline/stage_*.py` | Orchestration (replaces shell scripts) |

| Retained (unchanged) | Purpose |
|-----------------------|---------|
| `src/eval.py` | AUC / PR evaluation |
| `src/score_filter.py` | Statistical interval extraction |
| `src/val_priors.py` | VAL task |
| `src/vau_priors.py` | VAU task |
| `src/data/` | Data classes (`VideoRecord`, `VideoBoxes`) |
| `src/utils/` | Utilities (paths, torch, visualization) |

---

## Video Anomaly Localisation (VAL)

Requires [UCFCrime BoundingBox Annotation](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation). We provide a preprocessed file `Test_annotation_naming_aligned.pkl` under `data/ucf_crime/`.

After tag extraction, run:
```bash
python src/val_priors.py
```

Localisation results are under `data/ucf_crime/localisations/`.

---

## Video Anomaly Understanding (VAU)

Requires [HIVAU-70K dataset](https://github.com/pipixin321/HolmesVAU). Video summaries have been preprocessed from `HolmesVAU/HIVAU-70k/raw_annotations/` and placed under `data/{dataset}/video_summaries.json`.

```bash
python src/vau_priors.py
```

Evaluate:
```bash
python src/compute_bleu.py <ground_truth.json> <predictions.json>
python src/gpt_score_eval.py
```

---

## BibTeX

```
@inproceedings{
    lin2025AUR,
    title={A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis},
    author={Dongheng Lin, Mengxue Qu, Kunyang Han, Jianbo Jiao, Xiaojie Jin, Yunchao Wei},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=Qla5PqFL0s}
}
```
