# CLAUDE.md — Project Context for Claude Code

## Project Identity

This is the codebase for the NeurIPS 2025 paper: **"A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis"** (URF-HVAA).

Repository: Zero-shot Video Anomaly Detection (VAD) / Localization (VAL) / Understanding (VAU) using VLM + LLM.

## Core Architecture

- **VLM (Vision)**: `DAMO-NLP-SG/VideoLLaMA3-7B` — loaded from HuggingFace at runtime
- **LLM (Text)**: `Meta-Llama-3.1-8B-Instruct` — stored locally at `libs/llama/llama3.1-8b/`
- **GPU requirement**: Single RTX 3090 (24GB). VLM and LLM MUST NOT be loaded simultaneously — each requires ~15-16GB.

## Directory Layout

```
.
├── ARCHITECTURE.md          # v3 refactored architecture design (READ THIS FIRST)
├── CLAUDE.md                # This file
├── README.md                # Original project README
├── environment.yml           # Conda environment (VAA)
├── requirements.txt          # Pip dependencies
├── assets/                   # Paper figures
├── data/                     # Datasets (MSAD, UBNormal, ucf_crime, xd_violence)
│   └── data.md               # Dataset download instructions
├── libs/
│   └── llama/
│       ├── llama/             # Llama model code (tokenizer, model, generation)
│       └── llama3.1-8b/       # Llama 3.1 8B checkpoint weights
├── scripts/                   # Shell scripts (eval, extract_frames, query_llm, refine)
└── src/
    ├── video_pre_caption.py   # [TO BE REPLACED] VLM offline captioning
    ├── llm_anomaly_scorer.py  # [KEEP] LLM scoring engine (Phase 1 & Phase 4)
    ├── score_filter.py        # [KEEP] Statistical interval extraction
    ├── summarize_window.py    # [TO BE REPLACED] VLM anomaly tag extraction
    ├── refine_with_tag.py     # [TO BE REPLACED] Tag-guided score refinement
    ├── eval.py                # [KEEP] AUC / PR evaluation
    ├── val_priors.py          # [KEEP] VAL task
    ├── vau_priors.py          # [KEEP] VAU task (future refactor)
    ├── extract_frames.py      # Frame extraction utility
    ├── draw_bboxes.py         # Bounding box drawing
    ├── compute_bleu.py        # BLEU metric
    ├── gpt_score_eval.py      # GPT-score evaluation
    ├── utils/                  # Utilities (vis, path, plot, image, frame_boxes, etc.)
    └── data/                   # Data classes (video_record, video_boxes, video_box_vis)
```

## Current State & Implementation Roadmap

We are implementing the **Asymmetric Dual-Pass Reflection** architecture (see `ARCHITECTURE.md` for full details).

### What exists (original pipeline):
1. `video_pre_caption.py` — VLM offline pre-captioning every 16 frames
2. `llm_anomaly_scorer.py` — LLM scores each caption (0–1 anomaly score)
3. `score_filter.py` — Finds highest/lowest score intervals
4. `summarize_window.py` — VLM extracts anomaly tags from suspicious intervals
5. `refine_with_tag.py` — LLM re-scores with tag priors (score gate mechanism)
6. `eval.py` — Computes ROC-AUC / PR-AUC

### What we're building (new pipeline):

```
Stage A [VLM]: Coarse blind captioning (interval=16)
Stage B [LLM]: Initial scoring
Stage C [LLM]: Context memory (Phase 2) + Conflict detection (Phase 3)
Stage D [VLM]: Fine targeted verification (interval=4, flagged frames only)
Stage E [LLM]: Final scoring + merge + eval
```

### Implementation order:
1. `src/perception/vlm_engine.py`
2. `src/reflection/context_memory.py`
3. `src/reflection/conflict_detector.py`
4. `src/reflection/targeted_verifier.py`
5. `src/pipeline/stage_*.py` (5 pipeline scripts)

### New dependency to add:
- `sentence-transformers` (for `all-MiniLM-L6-v2`, ~80MB, used in context drift detection)

## Key Design Decisions

- **No singleton VLM** — caller manages `load()`/`unload()` lifecycle explicitly
- **Serial GPU usage** — VLM and LLM never co-resident in GPU memory
- **File-based handoff between stages** — each stage reads JSON from previous, writes JSON for next
- **Dynamic percentile threshold** for normal frame selection (not hardcoded 0.3)
- **Time-based sliding windows** (seconds, not frame counts) for context memory
- **Anti-hallucination prompts** in Phase 4 — neutral, objective verification queries
- **Coarse-to-fine sampling** — interval=16 for patrol, interval=4 for focus
- **Global Gaussian smoothing** (sigma=2) on merged scores, not local smoothing
- **Compute baseline**: compare against `total_frames / 4` (full fine-grained scan), not `total_frames / 16`
- **Drift detection Plan A**: `all-MiniLM-L6-v2` embedding cosine similarity (~80MB, co-resident with LLM)
- **Drift detection Plan B (fallback)**: LLM prompt "Has the fundamental environment changed? YES/NO"
- **Score merge**: global `gaussian_filter1d(sigma=2)` on entire score array (not local smoothing around flagged frames)
- **Compute baseline in paper**: compare against `total_frames / 4` (naive full fine-grained scan), showing 60-80% VLM compute savings
- **Phase 3 output**: Flagged frames MUST include `conflict_reason`, `suspicious_element`, and `alternative_explanation` for Phase 4 anti-hallucination prompts

## Working Directory

The project is at `/z/URF-HVAA/` (RaiDrive mount from AutoDL server). All paths use Unix conventions.

## Dataset Mapping

| Dataset | Annotation File | FPS | Notes |
|---|---|---|---|
| UCF-Crime | `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | 30 | 1900+ videos, 13 anomaly types |
| XD-Violence | `temporal_anomaly_annotation_for_testing_videos.txt` | 24 | Multi-label temporal annotations |
| UB-Normal | `temporal.txt` | 30 | All-normal videos |
| MSAD | `msad_anomaly_index.txt` | 30 | Multi-scene anomaly detection |

## Conventions

- Python 3.10, PyTorch 2.5, CUDA 12.4
- Video frames stored as JPG sequences under `frames/{video_name}/000001.jpg`
- Video files as `.mp4` under `videos/`
- Captions as `{video_name}.json` keyed by frame index string
- Scores as `{video_name}.json` keyed by frame index string
- Annotation index files: `<path> <start_frame> <end_frame> <label>` per line
