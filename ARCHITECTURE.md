# Architecture Design v3 — Asymmetric Dual-Pass Reflection for VAD

## Overview

This document describes the refactored architecture for the paper *"A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis"* (NeurIPS 2025).

The core innovation is an **Asymmetric Dual-Pass Reflection Loop** — replacing the offline VLM pre-captioning pipeline with a closed-loop perception-reasoning system inspired by human cognitive patterns ("top-down perception guidance").

## Design Principles

1. **Serial model loading with file-based handoff** — GPU only hosts one model at a time (VLM ~15GB or LLM ~16GB, never both). Avoids OOM on single RTX 3090/4090.
2. **Asymmetric compute allocation** — LLM (cheap) does full-traversal reasoning; VLM (expensive) is invoked only on-demand for targeted re-examination.
3. **Anti-hallucination by design** — Phase 4 VLM prompts are neutral verification queries, never leading suggestions.
4. **Coarse-to-fine sampling** — Phase 1 uses interval=16 (patrol mode); Phase 4 uses interval=4 (focus mode).

---

## Data Flow (5 Stages)

```
+------------------------------------------------------------------+
| Stage A - Coarse Blind Captioning [VLM only, ~15GB]               |
|   video -> blind_caption(interval=16, max_frames=8)               |
|        -> captions/phase1_coarse/{video}.json                     |
|   Unload VLM                                                      |
+------------------------------------------------------------------+
| Stage B - Initial Scoring [LLM only, ~16GB]                       |
|   captions -> llm_anomaly_scorer (generic scoring)                |
|           -> scores/phase1_initial/{video}.json                   |
+------------------------------------------------------------------+
| Stage C - Context Memory + Conflict Detection [LLM only, ~16GB]   |
|   captions + scores                                               |
|     -> Phase 2: SlidingContextMemory (dynamic percentile +        |
|                 time-based windows + drift detection)             |
|        -> context/phase2/{video}_windows.json                     |
|     -> Phase 3: ConflictDetector (full traversal + reason output) |
|        -> reflection/phase3_flagged/{video}.json                  |
|   Unload LLM                                                      |
+------------------------------------------------------------------+
| Stage D - Fine-Grained Targeted Verification [VLM only, ~15GB]    |
|   flagged_list + scene_context                                    |
|     -> guided_caption(interval=4, max_frames=16,                 |
|                       anti-hallucination prompt)                  |
|        -> captions/phase4_fine/{video}.json (flagged frames only) |
|   Unload VLM                                                      |
+------------------------------------------------------------------+
| Stage E - Final Scoring + Merge [LLM only, ~16GB]                 |
|   refined captions -> LLM rescore                                 |
|   Score merge (replace flagged) + global Gaussian smooth          |
|     -> scores/final/{video}.json                                  |
|   -> eval.py (unchanged)                                          |
+------------------------------------------------------------------+
```

---

## Module Design

### 1. `src/perception/vlm_engine.py` — Unified VLM Interface

Replaces the duplicated `load_model()` / `infer()` across:
- `video_pre_caption.py`
- `summarize_window.py`
- `vau_priors.py`

Key design:
- **Explicit lifecycle**: caller controls `load()` / `unload()`, never singleton
- **Dual sampling density**:
  - `coarse`: interval=16, max_frames=8 (Phase 1 patrol)
  - `fine`: interval=4, max_frames=16 (Phase 4 focus)

```python
class VLMEngine:
    def load(self):
        """Load VideoLLaMA3-7B to GPU from DAMO-NLP-SG/VideoLLaMA3-7B"""

    def unload(self):
        """del model + torch.cuda.empty_cache()"""

    def blind_caption(self, video_path, frame_idx, mode="coarse") -> str:
        """
        Phase 1: No prior knowledge. Describe what you see.
        Prompt: "Summarize the main events or actions in a concise way."
        """

    def guided_caption(self, video_path, frame_idx, scene_context,
                        original_caption, conflict_reason, mode="fine") -> str:
        """
        Phase 4: Neutral verification with anti-hallucination prompt.
        """
```

**Phase 4 Anti-Hallucination Prompt Template:**

```
SYSTEM: You are an objective visual observer. Describe what you actually
see — not what you expect to see. Do not speculate.

SCENE CONTEXT: {scene_context}

IMPORTANT NOTE: A previous coarse scan reported "{original_caption}".
However, this may be INACCURATE because {conflict_reason}.

Please examine the frames carefully and answer:
1. What is actually visible? Describe only observable facts.
2. Is there genuinely any {suspicious_element}, or could the previous
   observation be explained by {alternative_explanation}?

Respond with a factual description. Do not assume any anomaly exists.
```

---

### 2. `src/reflection/context_memory.py` — Sliding Window Context Memory

Key decisions (v3):
- **Dynamic percentile threshold**: `np.percentile(scores, 30)` per-video, not hardcoded 0.3
- **Time-based windows**: `window_seconds=60`, `stride_seconds=30` (NOT absolute frame counts)
- **Drift detection Plan A (primary)**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB VRAM) cosine similarity. If `cosine_sim < 0.7`, trigger context refresh.
- **Drift detection Plan B (fallback)**: LLM prompt "Has the fundamental environment changed? YES/NO."

```python
class SlidingContextMemory:
    def __init__(self, window_seconds=60, stride_seconds=30,
                 normality_percentile=30, drift_threshold=0.7):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_normality_threshold(self, scores: dict) -> float:
        """Dynamic: bottom 30% percentile per-video."""
        return float(np.percentile(list(scores.values()), self.normalite_percentile))

    def get_normal_frames(self, captions, scores, fps):
        """Filter frames below dynamic threshold, slice by time windows."""

    def generate_context(self, llm, window_captions) -> SceneContext:
        """LLM summarizes normal scene context per window."""

    def check_drift(self, new_window_captions, current_context) -> bool:
        """
        Plan A (primary): all-MiniLM-L6-v2 embedding cosine similarity.
        Plan B (fallback): LLM prompt "Has the fundamental environment changed?"
        """
```

**Why time-based windows instead of frame counts?**
Different datasets have different FPS (10-30). A 60-second window ensures consistent semantic granularity across all videos.

**Why sliding windows for "normality"?**
"Normal" at t=0s (daytime parking lot) differs from "normal" at t=3600s (nighttime). Per-window context captures local normality.

---

### 3. `src/reflection/conflict_detector.py` — Full-Traversal Conflict Detection

Completely NEW logic — does NOT reuse `refine_with_tag.py`.

| | refine_with_tag.py | conflict_detector.py |
|---|---|---|
| Purpose | Adjust scores with tag hints | Detect local-global contradictions |
| Traversal | Conditional (score gate skips) | Full traversal, all frames examined |
| Prompt | "Consider these suspicious behaviors" | "Does action CONTRADICT context?" |
| Output | Modified score per frame | Flagged List with conflict reasons |

```python
class ConflictDetector:
    def __init__(self, cap_max_flags=20):
        self.cap_max_flags = cap_max_flags

    CONFLICT_PROMPT = """
    You are a logical consistency checker for surveillance footage.
    GLOBAL SCENE CONTEXT: {scene_context}

    For each frame description below, determine if the described action
    LOGICALLY CONTRADICTS the expected normality of this scene.

    For EACH flagged frame, output:
    - frame: int
    - caption_summary: original caption text
    - conflict_reason: why this contradicts the scene context
    - suspicious_element: the specific anomalous thing
    - alternative_explanation: what else it could be

    Respond as JSON array. If no contradiction, respond [].
    """

    def detect(self, llm, scene_context, captions, scores) -> list:
        """Returns flagged frames sorted by conflict strength, capped."""
```

**Output format (feeds directly into Phase 4):**
```json
[
  {
    "frame": 480,
    "caption_summary": "a bright explosion in the street",
    "conflict_reason": "Dark empty residential street at 3am. Sudden bright explosions do not occur in normal residential settings.",
    "suspicious_element": "bright explosion / flash",
    "alternative_explanation": "car headlight, camera glare, or lamp malfunction"
  }
]
```

---

### 4. `src/reflection/targeted_verifier.py` — Targeted Verification + Score Closed Loop

```python
class TargetedVerifier:
    def verify_frame(self, vlm, video_path, flagged_frame, scene_context):
        """
        Single-frame targeted re-examination:
        - interval=4 (4x denser than Phase 1)
        - max_frames=16 (2x more frames than Phase 1)
        - Prompt includes conflict_reason + alternative_explanation
        """

    def rescore(self, llm_scorer, refined_captions):
        """Reuse llm_anomaly_scorer logic on refined captions."""

    def merge_scores(self, original_scores, flagged_scores):
        """
        1. Replace flagged frame scores with refined scores
        2. Apply global 1D Gaussian filter (sigma=2) to ENTIRE score array
           - Smooths replacement boundaries
           - Removes LLM scoring jitter (usually improves AUC)
        """
```

**Why global Gaussian instead of local?**
- Local smoothing around flagged frames creates artifacts when multiple flags cluster
- Global smoothing also removes inherent LLM scoring noise
- `sigma=2` is mild — preserves anomaly peaks while smoothing jitter

---

### 5. Compute Tracking

```python
class ComputeTracker:
    """
    Baseline: naive full fine-grained scan at interval=4.
    Our cost: coarse scan (interval=16) + targeted fine scan (interval=4).

    savings = (1 - our_cost / naive_full_cost) * 100
    """

    @property
    def naive_full_cost(self):
        """Traditional: full fine-grained scan at interval=4"""
        return self.total_frames / 4

    @property
    def our_cost(self):
        """Our method: coarse patrol + targeted focus"""
        return self.phase1_vlm_calls + self.phase4_vlm_calls

    @property
    def savings_percent(self):
        return (1 - self.our_cost / self.naive_full_cost) * 100
```

Expected savings: 60-80%+ VLM compute vs. naive full fine-grained scan.

---

## Module <-> Existing Code Mapping

| New Module | Replaces |
|---|---|
| `src/perception/vlm_engine.py` | `video_pre_caption.py`, `summarize_window.py`, `vau_priors.py` (VLM parts) |
| `src/reflection/context_memory.py` | `summarize_window.py` (context logic) |
| `src/reflection/conflict_detector.py` | `refine_with_tag.py` (completely new logic) |
| `src/reflection/targeted_verifier.py` | New (no existing equivalent) |

| Retained (unchanged) | Purpose |
|---|---|
| `src/llm_anomaly_scorer.py` | Scoring engine reused in Stage B and Stage E |
| `src/score_filter.py` | Statistical interval extraction |
| `src/eval.py` | Final AUC/metric evaluation |
| `src/val_priors.py` | VAL task (future refactor) |
| `src/vau_priors.py` | VAU task (future refactor) |
| `src/data/` | Data classes (video_record, video_boxes) |
| `src/utils/` | Utility functions |

---

## Implementation Order

1. `src/perception/vlm_engine.py` — Foundation for all VLM calls
2. `src/reflection/context_memory.py` — Independent, testable alone
3. `src/reflection/conflict_detector.py` — Phase 3 core logic
4. `src/reflection/targeted_verifier.py` — Phase 4 closed loop
5. `src/pipeline/stage_*.py` (5 scripts) — Orchestration layer

## New Dependency

- `sentence-transformers` (for `all-MiniLM-L6-v2` in drift detection, ~80MB)

---

## Pipeline Shell Invocations

Each stage runs independently. GPU hosts only ONE model at a time.

```bash
# Stage A - Coarse blind captioning [VLM]
python -m src.pipeline.stage_a_coarse_caption \
    --video_folder ./data/{dataset}/videos/ \
    --index_file ./data/{dataset}/annotations/test.txt \
    --output_dir ./data/{dataset}/captions/phase1_coarse/

# Stage B - Initial LLM scoring [LLM]
python -m src.pipeline.stage_b_initial_scoring \
    --captions_dir ./data/{dataset}/captions/phase1_coarse/ \
    --output_dir ./data/{dataset}/scores/phase1_initial/ \
    --ckpt_dir ./libs/llama/llama3.1-8b/ \
    --tokenizer_path ./libs/llama/llama3.1-8b/tokenizer.model

# Stage C - Context memory + conflict detection [LLM]
python -m src.pipeline.stage_c_context_reflect \
    --captions_dir ./data/{dataset}/captions/phase1_coarse/ \
    --scores_dir ./data/{dataset}/scores/phase1_initial/ \
    --context_output ./data/{dataset}/context/phase2/ \
    --flagged_output ./data/{dataset}/reflection/phase3_flagged/ \
    --ckpt_dir ./libs/llama/llama3.1-8b/ \
    --tokenizer_path ./libs/llama/llama3.1-8b/tokenizer.model

# Stage D - Targeted VLM verification [VLM]
python -m src.pipeline.stage_d_targeted_verify \
    --flagged_list ./data/{dataset}/reflection/phase3_flagged/ \
    --scene_context ./data/{dataset}/context/phase2/ \
    --video_folder ./data/{dataset}/videos/ \
    --output_dir ./data/{dataset}/captions/phase4_fine/

# Stage E - Final scoring + merge + eval [LLM]
python -m src.pipeline.stage_e_final_merge \
    --original_scores ./data/{dataset}/scores/phase1_initial/ \
    --refined_captions ./data/{dataset}/captions/phase4_fine/ \
    --output_dir ./data/{dataset}/scores/final/ \
    --ckpt_dir ./libs/llama/llama3.1-8b/ \
    --tokenizer_path ./libs/llama/llama3.1-8b/tokenizer.model
```
