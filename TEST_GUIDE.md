# Test Guide — Asymmetric Dual-Pass Reflection Architecture

## Quick context

**Goal**: validate that the new coarse (interval=16) + targeted (interval=4)
architecture saves 60-80% VLM compute while matching or improving AUC vs the
original pipeline.

**Key insight**: the original pipeline uses VLM captioning at a fixed coarse
interval (16 frames) with score refinement via LLM tag injection.  The new
pipeline adds a reflection loop: LLM detects "flagged" frames where captions
conflict with scene context → VLM re-examines only those frames at 4x density
→ scores are merged with global Gaussian smoothing.

**Hardware required**: single GPU with ≥24 GB VRAM (RTX 3090/4090).  VLM
(~15 GB) and LLM (~16 GB) run in separate processes — never co-resident.

---

## 1. Prerequisites checklist

Run these on the **AutoDL server** (not your local machine):

```bash
# 1. Environment
conda activate VAA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi  # confirm ≥24 GB available

# 2. LLM checkpoint
ls libs/llama/llama3.1-8b/consolidated.00.pth && echo "LLM: OK" || echo "LLM: MISSING"

# 3. VLM downloaded (for Stage A / D)
ls libs/VideoLLaMA3-7B/ && echo "VLM: OK" || echo "VLM: NOT DOWNLOADED"
# If missing, run:  python scripts/download_vlm.py

# 4. Extra dependency (Phase 2 drift detection)
pip install sentence-transformers

# 5. Precomputed data (reuse existing captions + scores for Test A)
ls data/ucf_crime/captions/video_llama3_json_results/*.json | wc -l  # should be 290
ls data/ucf_crime/scores/videollama3/*.json | wc -l                   # should be 290
ls data/ucf_crime/annotations/test.txt                                 # video list
ls data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt  # ground truth

# 6. Quick import check (30 seconds, no GPU needed for imports)
python -c "
from src.perception.vlm_engine import VLMEngine, FlaggedFrame, SceneContext; print('vlm_engine: OK')
from src.reflection.context_memory import SlidingContextMemory; print('context_memory: OK')
from src.reflection.conflict_detector import ConflictDetector; print('conflict_detector: OK')
from src.reflection.targeted_verifier import TargetedVerifier, ComputeTracker; print('targeted_verifier: OK')
from src.utils.torch_utils import ensure_single_gpu_distributed; print('torch_utils: OK')
print('ALL IMPORTS PASSED')
"
```

---

## 2. Two test tracks

|     | Test A                           | Test B                            |
|-----|----------------------------------|-----------------------------------|
| What | Stage C only (LLM, no video)    | Full 5-stage pipeline             |
| Time | ~10-15 min                       | hours (depends on video count)   |
| Needs video files | No             | Yes                              |
| Validates | Phase 2+3 logic correctness | End-to-end AUC + compute savings |
| Recommended first | **Yes**                 | After Test A passes              |

---

## 3. Test A — Stage C only (LLM, no video files needed)

This reuses existing precomputed captions and scores to test Phase 2 (context
memory) and Phase 3 (conflict detection).  No VLM needed.

### 3.1 Run

```bash
conda activate VAA
cd /root/autodl-tmp/URF-HVAA

python -m src.pipeline.stage_c_context_reflect \
    --root_path data/ucf_crime/frames \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/video_llama3_json_results \
    --scores_dir data/ucf_crime/scores/videollama3 \
    --context_output data/ucf_crime/context/phase2/ \
    --flagged_output data/ucf_crime/reflection/phase3_flagged/ \
    --ckpt_dir libs/llama/llama3.1-8b/ \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model \
    --default_fps 30.0
```

**Key parameters:**

| Arg | Default | Meaning |
|-----|---------|---------|
| `--default_fps` | `30.0` | FPS fallback when video files are missing (UCF-Crime = 30 fps) |
| `--window_seconds` | `60.0` | Time window for sliding context (sec) |
| `--stride_seconds` | `30.0` | Stride between windows (sec) |
| `--normality_percentile` | `30.0` | Bottom percentile of scores treated as "normal" |
| `--cap_max_flags` | `20` | Max flagged frames per video (controls Phase 4 VLM cost) |

### 3.2 Verify output

```bash
# Count output files
echo "Context windows:" && ls data/ucf_crime/context/phase2/*.json 2>/dev/null | wc -l
echo "Flagged frames:" && ls data/ucf_crime/reflection/phase3_flagged/*.json 2>/dev/null | wc -l

# Inspect one anomalous video (Abuse)
python -c "
import json
with open('data/ucf_crime/reflection/phase3_flagged/Abuse028_x264.json') as f:
    flags = json.load(f)
print(f'Flagged frames: {len(flags)}')
for ff in flags[:3]:
    print(f'  frame={ff[\"frame\"]}  reason={ff[\"conflict_reason\"][:100]}')
    print(f'  suspicious={ff[\"suspicious_element\"]}')
    print()
"

# Inspect one normal video
python -c "
import json
path = 'data/ucf_crime/reflection/phase3_flagged/Normal_Videos_001_x264.json'
import os
if os.path.exists(path):
    with open(path) as f:
        flags = json.load(f)
    print(f'Normal_Videos_001: {len(flags)} flagged (expect 0 or few)')
"
```

### 3.3 Success criteria

1. Most of 290 videos produce both context and flagged output files
2. Anomalous videos (Abuse, Arrest, Arson, etc.) have **flagged frames > 0**
3. Normal videos have **flagged frames = 0 or very few** (typical false positive: 0-2)
4. Sampled `conflict_reason` text is logically coherent (e.g. "dark empty street
   has sudden bright flash — this contradicts nighttime residential normality")

---

## 4. Test B — Full 5-stage pipeline (needs video files)

Run this once you have `.mp4` videos in `data/ucf_crime/videos/`.

### 4.1 Stage A — VLM coarse blind captioning

```bash
python -m src.pipeline.stage_a_coarse_caption \
    --video_folder data/ucf_crime/videos/ \
    --index_file data/ucf_crime/annotations/test.txt \
    --output_dir data/ucf_crime/captions/phase1_coarse/
```

**Output**: `data/ucf_crime/captions/phase1_coarse/{video}.json`  
**Time**: ~20 hours for full test set on single 3090  
**Shortcut**: reuse existing `data/ucf_crime/captions/video_llama3_json_results/`
for testing (already interval≈16).

### 4.2 Stage B — LLM initial scoring

```bash
python -m src.pipeline.stage_b_initial_scoring \
    --root_path data/ucf_crime/frames \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse/ \
    --output_dir data/ucf_crime/scores/phase1_initial/ \
    --ckpt_dir libs/llama/llama3.1-8b/ \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model
```

**Output**: `data/ucf_crime/scores/phase1_initial/{video}.json`

### 4.3 Stage C — Context memory + conflict detection

Same command as Test A, but point `--captions_dir` and `--scores_dir` to the
Stage A+B outputs:

```bash
python -m src.pipeline.stage_c_context_reflect \
    --root_path data/ucf_crime/frames \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --captions_dir data/ucf_crime/captions/phase1_coarse/ \
    --scores_dir data/ucf_crime/scores/phase1_initial/ \
    --video_folder data/ucf_crime/videos/ \
    --context_output data/ucf_crime/context/phase2/ \
    --flagged_output data/ucf_crime/reflection/phase3_flagged/ \
    --ckpt_dir libs/llama/llama3.1-8b/ \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model
```

### 4.4 Stage D — VLM targeted verification

```bash
python -m src.pipeline.stage_d_targeted_verify \
    --flagged_dir data/ucf_crime/reflection/phase3_flagged/ \
    --context_dir data/ucf_crime/context/phase2/ \
    --video_folder data/ucf_crime/videos/ \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --root_path data/ucf_crime/frames \
    --output_dir data/ucf_crime/captions/phase4_fine/
```

**Key**: only flagged frames get fine-grained VLM captioning (interval=4,
max_frames=16). This is where the 60-80% compute savings come from.

### 4.5 Stage E — Final scoring + merge + eval

```bash
python -m src.pipeline.stage_e_final_merge \
    --root_path data/ucf_crime/frames \
    --annotationfile_path data/ucf_crime/annotations/test.txt \
    --original_scores_dir data/ucf_crime/scores/phase1_initial/ \
    --refined_captions_dir data/ucf_crime/captions/phase4_fine/ \
    --output_dir data/ucf_crime/scores/final/ \
    --ckpt_dir libs/llama/llama3.1-8b/ \
    --tokenizer_path libs/llama/llama3.1-8b/tokenizer.model \
    --run_eval \
    --temporal_annotation_file data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

---

## 5. Comparing results — new vs old pipeline

### 5.1 AUC comparison

```bash
# New pipeline AUC
cat data/ucf_crime/scores/final/metrics/roc_auc.txt
cat data/ucf_crime/scores/final/metrics/pr_auc.txt

# Old pipeline AUC (from precomputed refined scores)
cat data/ucf_crime/refined_scores/videollama3/metrics/roc_auc.txt
cat data/ucf_crime/refined_scores/videollama3/metrics/pr_auc.txt
```

**Expected outcome**: ROC-AUC and PR-AUC should be **comparable or better** than
the old pipeline.  The new pipeline should NOT regress on AUC.

### 5.2 Compute savings

After Stage D, the compute tracker is saved:

```bash
cat data/ucf_crime/captions/phase4_fine/_compute_tracker.json
```

Expected format:
```json
{
  "total_frames": 12345,
  "naive_full_cost": 3086.25,
  "our_cost": 1080,
  "savings_percent": 65.0
}
```

**Expected outcome**: `savings_percent` should be **60-80%** vs naive full
fine-grained scan (interval=4 everywhere).  This is the primary paper claim.

### 5.3 Qualitative check — flagged frame relevance

Pick an annotated anomalous video and check that flagged frames overlap with
the ground-truth anomaly interval:

```bash
# Ground truth for Abuse028_x264
grep "Abuse028_x264" data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt

# Our flagged frames
python -c "
import json
with open('data/ucf_crime/reflection/phase3_flagged/Abuse028_x264.json') as f:
    flags = json.load(f)
frames = sorted([f['frame'] for f in flags])
print(f'Flagged frames ({len(frames)}): {frames}')
print(f'Range: {min(frames)} - {max(frames)}')
"
```

The flagged frames should cluster **near or within** the annotated anomaly
segment — not scattered randomly across the video.

---

## 6. Important design decisions (for paper context)

When interpreting results, keep these in mind:

| Decision | Why |
|----------|-----|
| Dynamic percentile threshold | `np.percentile(scores, 30)` per-video — adapts to each video's score distribution |
| Time-based windows (60s) | Frame counts differ across datasets; seconds are universal |
| Global Gaussian (sigma=2) | Smooths the entire merged score array, not just around flagged frames |
| Anti-hallucination prompts | Phase 4 VLM prompts include `conflict_reason` + `alternative_explanation` to prevent confirmation bias |
| Compute baseline | Compare against `total_frames / 4` (naive full fine scan), not `total_frames / 16` |

---

## 7. Troubleshooting

### CUDA out of memory

Each stage runs VLM or LLM, never both.  Confirm the previous process has fully
exited and GPU memory is freed:

```bash
nvidia-smi
# Look for orphaned Python processes; kill if needed
```

### Stage C skips all videos

Likely cause: `--default_fps` not being used because `--video_folder` is set to
a path that doesn't exist but isn't empty-string.

Fix: either provide a valid `--video_folder` OR omit it entirely to use
`--default_fps`.

### ModuleNotFoundError: sentence_transformers

Phase 2 Plan A (semantic drift detection) needs this. If missing, the code
automatically falls back to Plan B (LLM-based drift check), which is slower but
works. Still recommended to install:

```bash
pip install sentence-transformers
```

### chat_completion() got unexpected keyword argument

This means the code uses an older version of the Llama wrapper. Pull the latest
commits:

```bash
git pull origin main
```

### No flagged frames for anomalous videos

Possible causes:
- `--normality_percentile` too low (try 20 or 25)
- `--cap_max_flags` too small
- The captions/scores are from a different VLM/LLM version — make sure
  captions and scores match (both from videollama3, not mixed)

---

## 8. Quick reference — file layout after testing

```
data/ucf_crime/
├── captions/
│   ├── video_llama3_json_results/    # Existing precomputed (Stage A analogue)
│   ├── phase1_coarse/                # New Stage A output
│   └── phase4_fine/                  # Stage D output + _compute_tracker.json
├── scores/
│   ├── videollama3/                  # Existing precomputed (Stage B analogue)
│   ├── phase1_initial/               # New Stage B output
│   ├── final/                        # Stage E output + metrics/
│   └── refined_scores/               # Old pipeline (baseline for comparison)
├── context/
│   └── phase2/                       # Stage C output: scene contexts
├── reflection/
│   └── phase3_flagged/               # Stage C output: flagged frame lists
└── annotations/                      # test.txt + ground truth annotations
```

---

## 9. Current git state

```
0514bc7 fix: move ImageBind import inside function
0675740 fix: set single-GPU distributed env vars before Llama.build()
1ee0717 feat: asymmetric dual-pass reflection architecture for VAD
```

Key modules:
- `src/perception/vlm_engine.py` — unified VLM interface (blind + guided modes)
- `src/reflection/context_memory.py` — Phase 2 sliding-window scene context
- `src/reflection/conflict_detector.py` — Phase 3 full-traversal conflict detection
- `src/reflection/targeted_verifier.py` — Phase 4 targeted verification + score merge
- `src/pipeline/stage_a~e_*.py` — 5 orchestration scripts
- `src/utils/vlm_path.py` — VLM path resolver (local ModelScope > HuggingFace Hub)
- `scripts/download_vlm.py` — proactive VLM download from ModelScope
