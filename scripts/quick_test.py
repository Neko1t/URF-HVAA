#!/usr/bin/env python
"""
Quick pre-experiment test (25-40 min) for the Asymmetric Dual-Pass Reflection pipeline.

Validates that the new architecture (context memory + conflict detection + targeted
verification + merge) outperforms the original pipeline on a 5-video subset of UCF-Crime.

Usage (run on AutoDL server):
    conda activate VAA
    cd /path/to/URF-HVAA
    python scripts/quick_test.py

If video .mp4 files are unavailable, the script automatically falls back to
"text-only" mode: Stage C (LLM) + Stage E (LLM) with conflict-aware rescoring,
skipping the VLM-dependent Stage D.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# ---------------------------------------------------------------------------
# Keep logging quiet — use tqdm for progress instead
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ============================================================================
# Configuration
# ============================================================================

# 5 test videos covering different anomaly types, each ~1400-2400 frames
TEST_VIDEOS = [
    ("Abuse",      "Abuse028_x264",     1412),
    ("Arrest",     "Arrest001_x264",    2374),
    ("Arson",      "Arson016_x264",     1795),
    ("Burglary",   "Burglary021_x264",  1537),
    ("Shooting",   "Shooting015_x264",  1713),
]

# Paths relative to project root
CAPTIONS_DIR    = "data/ucf_crime/captions/video_llama3_json_results"
SCORES_DIR      = "data/ucf_crime/scores/videollama3"
REFINED_DIR     = "data/ucf_crime/refined_scores/videollama3"
ANNO_INDEX      = "data/ucf_crime/annotations/test.txt"
ANNO_TEMPORAL   = "data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
VIDEO_DIR       = "data/ucf_crime/videos"
LLAMA_CKPT      = "libs/llama/llama3.1-8b"
LLAMA_TOKENIZER = "libs/llama/llama3.1-8b/tokenizer.model"

OUTPUT_DIR      = "data/ucf_crime/quick_test_results"


# ============================================================================
# Pre-flight checks
# ============================================================================

def check_environment() -> dict:
    """Validate everything needed for the test. Returns a status dict."""
    status = {"ok": True, "warnings": [], "errors": []}

    # Required packages
    for mod, desc in [("torch", "PyTorch"), ("numpy", "NumPy"), ("scipy", "SciPy"),
                       ("sklearn", "scikit-learn"), ("tqdm", "tqdm"),
                       ("cv2", "OpenCV"), ("fairscale", "fairscale")]:
        try:
            __import__(mod)
        except ImportError:
            status["errors"].append(f"Missing: {desc} ({mod})")
            status["ok"] = False

    try:
        __import__("sentence_transformers")
    except ImportError:
        status["warnings"].append("sentence-transformers missing → Plan B drift detection")

    try:
        __import__("transformers")
        status["has_transformers"] = True
    except ImportError:
        status["warnings"].append("transformers missing → Stage D disabled")
        status["has_transformers"] = False

    # Model checkpoint
    if not (Path(LLAMA_CKPT) / "consolidated.00.pth").exists():
        status["errors"].append(f"Llama checkpoint missing: {LLAMA_CKPT}")
        status["ok"] = False
    if not Path(LLAMA_TOKENIZER).exists():
        status["errors"].append(f"Llama tokenizer missing: {LLAMA_TOKENIZER}")
        status["ok"] = False

    # Data files
    for vtype, vname, _ in TEST_VIDEOS:
        for sub, label in [(CAPTIONS_DIR, "captions"), (SCORES_DIR, "scores")]:
            if not (Path(sub) / f"{vname}.json").exists():
                status["errors"].append(f"Missing {label}: {sub}/{vname}.json")
                status["ok"] = False
        if not (Path(REFINED_DIR) / f"{vname}.json").exists():
            status["warnings"].append(f"Missing refined: {REFINED_DIR}/{vname}.json")

    # Annotation files
    for anno_path in [ANNO_INDEX, ANNO_TEMPORAL]:
        if not Path(anno_path).exists():
            status["errors"].append(f"Missing annotation: {anno_path}")
            status["ok"] = False

    # Video files (optional)
    video_available = all(
        (Path(VIDEO_DIR) / f"{vname}.mp4").exists()
        for _, vname, _ in TEST_VIDEOS
    )
    status["video_available"] = video_available
    if not video_available:
        status["warnings"].append("Video .mp4 missing → Stage D disabled (TEXT-ONLY)")

    # GPU check
    try:
        import torch
        if not torch.cuda.is_available():
            status["errors"].append("No CUDA GPU available")
            status["ok"] = False
    except Exception:
        status["errors"].append("Cannot check GPU")
        status["ok"] = False

    # Print compact summary
    ok_count = sum(1 for _ in [1] if status["ok"]) + (6 - len(status["errors"]))
    print(f"  Pre-flight: {len(status['errors'])} errors, {len(status['warnings'])} warnings")
    if not status["ok"]:
        for e in status["errors"]:
            print(f"    ✗ {e}")
    for w in status["warnings"]:
        print(f"    ⚠ {w}")
    if status["ok"]:
        print(f"    ✓ All checks passed"
              + (f" (VLM: on)" if video_available else " (VLM: off)"))

    return status


# ============================================================================
# Data loading helpers
# ============================================================================

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_temporal_annotations(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parse ground-truth anomaly intervals.
    Returns {video_name: [(start_frame, end_frame), ...]}
    """
    annotations: Dict[str, List[Tuple[int, int]]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            video_name = parts[0].replace(".mp4", "")
            vals = [int(x) for x in parts[2:] if x != "-1"]
            intervals = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if intervals:
                annotations[video_name] = intervals
    return annotations


# ============================================================================
# Helpers
# ============================================================================

def _cleanup_llm(generator) -> None:
    """Release Llama generator and clean up distributed state."""
    import torch
    del generator
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    print("      LLM unloaded + GPU memory released.")


def _load_stage_c_summary(output_dir: str) -> dict:
    """Reconstruct Stage C summary dict from saved JSON files on disk."""
    import json as _json, os as _os
    summary = {}
    flagged_dir = _os.path.join(output_dir, "stage_c_flagged")
    context_dir = _os.path.join(output_dir, "stage_c_contexts")
    for vtype, vname, nframes in TEST_VIDEOS:
        flag_path = _os.path.join(flagged_dir, f"{vname}.json")
        ctx_path = _os.path.join(context_dir, f"{vname}_windows.json")
        n_flagged = 0
        flagged_frames = []
        window_descs = []
        n_windows = 0
        if _os.path.exists(flag_path):
            with open(flag_path, "r", encoding="utf-8") as f:
                flags = _json.load(f)
            n_flagged = len(flags)
            flagged_frames = [item["frame"] for item in flags]
        if _os.path.exists(ctx_path):
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctxs = _json.load(f)
            n_windows = len(ctxs)
            window_descs = [c.get("description", "") for c in ctxs]
        summary[vname] = {
            "type": vtype, "nframes": nframes,
            "n_windows": n_windows, "n_flagged": n_flagged,
            "flagged_frames": flagged_frames,
            "window_descriptions": window_descs,
        }
    return summary


# ============================================================================
# Stage C: Context Memory + Conflict Detection  [LLM only]
# ============================================================================

def run_stage_c(test_videos, output_dir) -> dict:
    """
    Phase 2: Build sliding-window scene context for each video.
    Phase 3: Full-traversal conflict detection → FlaggedFrame list.

    Returns summary dict keyed by video name.
    """
    from libs.llama.llama import Llama
    from src.reflection.context_memory import SlidingContextMemory
    from src.reflection.conflict_detector import ConflictDetector
    from src.utils.torch_utils import ensure_single_gpu_distributed
    from tqdm import tqdm

    print("\n" + "=" * 55)
    print("  STAGE C  Context Memory + Conflict Detection  [LLM]")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)
    context_out = os.path.join(output_dir, "stage_c_contexts")
    flagged_out = os.path.join(output_dir, "stage_c_flagged")
    os.makedirs(context_out, exist_ok=True)
    os.makedirs(flagged_out, exist_ok=True)

    print("Loading Llama 3.1 8B ...", end="", flush=True)
    t0 = time.time()
    ensure_single_gpu_distributed()
    generator = Llama.build(
        ckpt_dir=LLAMA_CKPT,
        tokenizer_path=LLAMA_TOKENIZER,
        max_seq_len=2048,
        max_batch_size=8,
        model_parallel_size=1,
        seed=1,
    )
    print(f" {time.time() - t0:.0f}s")

    memory = SlidingContextMemory(
        window_seconds=60.0, stride_seconds=30.0,
        normality_percentile=30.0, max_captions_per_context=10,
    )
    memory._init_embedder()
    detector = ConflictDetector(cap_max_flags=999, batch_size=5,
                                min_score_percentile=50.0)

    summary = {}
    total_flagged = 0
    total_windows = 0

    pbar = tqdm(test_videos, desc="  Stage C", unit="video", ncols=100)
    for vtype, vname, nframes in pbar:
        pbar.set_postfix_str(f"{vname}")
        pbar.write(f"  [{vtype}] {vname} ({nframes} frames)")
        captions = load_json(os.path.join(CAPTIONS_DIR, f"{vname}.json"))
        scores = load_json(os.path.join(SCORES_DIR, f"{vname}.json"))

        video_path = os.path.join(VIDEO_DIR, f"{vname}.mp4")
        if os.path.exists(video_path):
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
            cap.release()
            if fps <= 0:
                fps = 30.0
        else:
            fps = 30.0

        contexts = memory.process_video(captions, scores, fps, generator)
        total_windows += len(contexts)

        ctx_list = [{"window_start_sec": c.window_start_sec,
                      "window_end_sec": c.window_end_sec,
                      "description": c.description} for c in contexts]
        with open(os.path.join(context_out, f"{vname}_windows.json"), "w") as f:
            json.dump(ctx_list, f, indent=2, ensure_ascii=False)

        if not contexts:
            summary[vname] = {"type": vtype, "nframes": nframes,
                              "n_windows": 0, "n_flagged": 0,
                              "flagged_frames": [], "window_descriptions": []}
            continue

        flagged = detector.detect(captions, scores, fps, contexts, generator)
        total_flagged += len(flagged)

        flag_list = [{"frame": ff.frame, "caption_summary": ff.caption_summary,
                       "conflict_reason": ff.conflict_reason,
                       "suspicious_element": ff.suspicious_element,
                       "alternative_explanation": ff.alternative_explanation}
                     for ff in flagged]
        with open(os.path.join(flagged_out, f"{vname}.json"), "w") as f:
            json.dump(flag_list, f, indent=2, ensure_ascii=False)

        summary[vname] = {"type": vtype, "nframes": nframes,
                          "n_windows": len(contexts), "n_flagged": len(flagged),
                          "flagged_frames": [ff.frame for ff in flagged],
                          "window_descriptions": [c.description for c in contexts]}

    print(f"  → {total_windows} windows, {total_flagged} flagged frames")
    print(f"  LLM stays loaded for Stage E")
    return summary, generator


# ============================================================================
# Stage D: Targeted VLM Verification  [VLM only, OPTIONAL]
# ============================================================================

def run_stage_d(output_dir, stage_c_summary) -> Dict[str, Dict[int, str]]:
    """
    Re-examine flagged frames with fine-grained VLM sampling and
    anti-hallucination prompts.

    Returns {video_name: {frame_idx: refined_caption}}
    """
    from src.perception.vlm_engine import VLMEngine, FlaggedFrame, SceneContext
    from tqdm import tqdm

    print("\n" + "=" * 55)
    print("  STAGE D  Targeted VLM Verification  [VLM]")
    print("=" * 55)

    flagged_dir = os.path.join(output_dir, "stage_c_flagged")
    context_dir = os.path.join(output_dir, "stage_c_contexts")
    refined_out_dir = os.path.join(output_dir, "stage_d_refined_captions")
    os.makedirs(refined_out_dir, exist_ok=True)

    print("Loading VideoLLaMA3-7B ...", end="", flush=True)
    t0 = time.time()
    engine = VLMEngine()
    engine.load()
    print(f" {time.time() - t0:.0f}s")

    # Collect all flagged frames across videos
    all_tasks = []  # (vname, video_path, fps, contexts, flagged_frame)
    for vtype, vname, nframes in TEST_VIDEOS:
        flag_path = os.path.join(flagged_dir, f"{vname}.json")
        ctx_path = os.path.join(context_dir, f"{vname}_windows.json")
        video_path = os.path.join(VIDEO_DIR, f"{vname}.mp4")
        if not os.path.exists(flag_path) or not os.path.exists(ctx_path):
            continue
        if not os.path.exists(video_path):
            continue
        with open(flag_path) as f:
            raw_flags = json.load(f)
        with open(ctx_path) as f:
            raw_ctx = json.load(f)
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        cap.release()
        if fps <= 0:
            fps = 30.0
        contexts = [SceneContext(window_start_sec=c["window_start_sec"],
                                 window_end_sec=c["window_end_sec"],
                                 description=c["description"]) for c in raw_ctx]
        for item in raw_flags:
            ff = FlaggedFrame(
                frame=item["frame"],
                caption_summary=item.get("caption_summary", ""),
                conflict_reason=item.get("conflict_reason", ""),
                suspicious_element=item.get("suspicious_element", ""),
                alternative_explanation=item.get("alternative_explanation", ""),
            )
            all_tasks.append((vname, video_path, fps, contexts, ff))

    if not all_tasks:
        print("  No flagged frames to verify — skipping")
        engine.unload()
        return {}

    all_refined: Dict[str, Dict[int, str]] = {}
    ok_count = 0
    pbar = tqdm(all_tasks, desc="  VLM verify", unit="frame", ncols=100)
    for vname, video_path, fps, contexts, ff in pbar:
        pbar.set_postfix_str(f"{vname} f{ff.frame}")
        try:
            caption = engine.guided_caption(
                video_path=video_path, frame_idx=ff.frame,
                scene_context=contexts[0], flagged_frame=ff, mode="fine",
            )
            all_refined.setdefault(vname, {})[ff.frame] = caption
            ok_count += 1
        except Exception as e:
            pass  # silently skip failed frames

    # Save per video
    for vname, refined in all_refined.items():
        with open(os.path.join(refined_out_dir, f"{vname}.json"), "w") as f:
            json.dump(refined, f, indent=2, ensure_ascii=False)

    print(f"  → {ok_count}/{len(all_tasks)} verified")
    engine.unload()
    return all_refined


# ============================================================================
# Stage E: Final Scoring + Merge + Evaluation  [LLM only]
# ============================================================================

def run_stage_e(output_dir, stage_c_summary, refined_captions=None, generator=None):
    """
    1. Re-score refined captions (or original captions for flagged frames) via LLM.
    2. Merge refined scores into original Phase 1 score array.
    3. Apply global Gaussian smoothing (sigma=2).
    4. Evaluate ROC-AUC / PR-AUC on the 5-video subset.

    If *generator* is given (from Stage C), it is reused; otherwise a new
    Llama instance is built.
    """
    from scipy.ndimage import gaussian_filter1d
    from tqdm import tqdm

    print("\n" + "=" * 55)
    print("  STAGE E  Final Scoring + Merge + Evaluation  [LLM]")
    print("=" * 55)

    if refined_captions is None:
        refined_captions = {}
    has_vlm_refined = bool(refined_captions and any(refined_captions.values()))
    mode_str = "FULL" if has_vlm_refined else "TEXT-ONLY"
    print(f"Mode: {mode_str} ({'VLM refined' if has_vlm_refined else 'conflict-aware prompt'})")

    os.makedirs(output_dir, exist_ok=True)
    final_scores_dir = os.path.join(output_dir, "stage_e_final_scores")
    os.makedirs(final_scores_dir, exist_ok=True)

    if generator is None:
        from libs.llama.llama import Llama
        from src.utils.torch_utils import ensure_single_gpu_distributed
        print("Loading Llama 3.1 8B ...", end="", flush=True)
        t0 = time.time()
        ensure_single_gpu_distributed()
        generator = Llama.build(
            ckpt_dir=LLAMA_CKPT, tokenizer_path=LLAMA_TOKENIZER,
            max_seq_len=512, max_batch_size=1, model_parallel_size=1, seed=1,
        )
        print(f" {time.time() - t0:.0f}s")
    else:
        print("Reusing Llama from Stage C")

    context_prompt = (
        "How would you rate the scene described on a scale from 0 to 1, "
        "with 0 representing a standard scene and 1 denoting a scene with "
        "suspicious activities or potentially criminal activities?"
    )
    format_prompt = (
        "Please provide the response in the form of a Python list and respond "
        "with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, "
        "0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should "
        "begin with '[' and end with ']'."
    )
    system_prompt = context_prompt + " " + format_prompt

    gt_annotations = load_temporal_annotations(ANNO_TEMPORAL)

    # Collect all frames to rescore
    all_rescore_tasks = []  # (vname, fidx, caption, conflict_note)
    for vtype, vname, nframes in TEST_VIDEOS:
        video_refined = refined_captions.get(vname, {})
        orig_captions = load_json(os.path.join(CAPTIONS_DIR, f"{vname}.json"))
        flagged_path = os.path.join(output_dir, "stage_c_flagged", f"{vname}.json")
        flagged_info = {}
        if os.path.exists(flagged_path):
            with open(flagged_path) as f:
                for item in json.load(f):
                    flagged_info[item["frame"]] = item
        frames_to_rescore = list(video_refined.keys()) if video_refined else list(flagged_info.keys())
        for fidx in frames_to_rescore:
            caption = video_refined.get(fidx) or orig_captions.get(str(fidx), "")
            if not caption:
                continue
            finfo = flagged_info.get(fidx)
            if finfo and not video_refined:
                conflict_note = (
                    f"[Scene note: this segment may contain {finfo.get('suspicious_element', 'unusual activity')} "
                    f"but could also be {finfo.get('alternative_explanation', 'a benign variation')}].\n"
                )
            else:
                conflict_note = ""
            all_rescore_tasks.append((vname, fidx, caption, conflict_note))

    # ---- Re-score ----
    rescore_results: Dict[str, Dict[int, float]] = {}  # {vname: {fidx: score}}
    pbar = tqdm(all_rescore_tasks, desc="  Re-scoring", unit="frame", ncols=100)
    for vname, fidx, caption, conflict_note in pbar:
        pbar.set_postfix_str(f"{vname} f{fidx}")
        user_text = f"{conflict_note}{caption}."
        try:
            result = generator.chat_completion(
                [[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_text}]],
                max_gen_len=64, temperature=0.6, top_p=0.9,
            )
            response = result[0]["generation"]["content"]
            m = re.search(r"\[(\d+(?:\.\d+)?)\]", response)
            if m:
                rescore_results.setdefault(vname, {})[fidx] = float(m.group(1))
        except Exception:
            pass

    # ---- Merge + smooth per video ----
    all_new_scores: Dict[str, np.ndarray] = {}
    all_baseline_scores: Dict[str, np.ndarray] = {}
    all_phase1_scores: Dict[str, np.ndarray] = {}

    for vtype, vname, nframes in TEST_VIDEOS:
        orig_scores = load_json(os.path.join(SCORES_DIR, f"{vname}.json"))
        orig_dict = {int(k): float(v) for k, v in orig_scores.items()}

        # Build dense array, replace refined, smooth
        arr = np.full(nframes, np.nan)
        for fidx, score in orig_dict.items():
            if 0 <= fidx < nframes:
                arr[fidx] = score
        known = np.where(~np.isnan(arr))[0]
        arr = np.interp(np.arange(nframes), known, arr[known]) if len(known) >= 2 else np.full(nframes, arr[known[0]] if len(known) == 1 else 0.0)

        n_replaced = 0
        for fidx, score in rescore_results.get(vname, {}).items():
            if 0 <= fidx < nframes:
                arr[fidx] = score
                n_replaced += 1

        all_new_scores[vname] = gaussian_filter1d(arr, sigma=2.0)

        # Phase 1 only baseline
        p1_arr = np.full(nframes, np.nan)
        for fidx, score in orig_dict.items():
            if 0 <= fidx < nframes:
                p1_arr[fidx] = score
        p1_known = np.where(~np.isnan(p1_arr))[0]
        all_phase1_scores[vname] = np.interp(np.arange(nframes), p1_known, p1_arr[p1_known]) if len(p1_known) >= 2 else np.zeros(nframes)

        # Original refined baseline
        refined_path = os.path.join(REFINED_DIR, f"{vname}.json")
        if os.path.exists(refined_path):
            base_raw = load_json(refined_path)
            base_arr = np.full(nframes, np.nan)
            for k, v in base_raw.items():
                fidx = int(k)
                if 0 <= fidx < nframes:
                    base_arr[fidx] = float(v)
            b_known = np.where(~np.isnan(base_arr))[0]
            all_baseline_scores[vname] = np.interp(np.arange(nframes), b_known, base_arr[b_known]) if len(b_known) >= 2 else np.zeros(nframes)
        else:
            all_baseline_scores[vname] = all_new_scores[vname].copy()

    # ---- Save final scores ----
    for bucket, name in [(all_new_scores, ""), (all_phase1_scores, "_phase1_only")]:
        sub = final_scores_dir if name == "" else os.path.join(output_dir, "stage_e_phase1_only")
        os.makedirs(sub, exist_ok=True)
        for vname, arr in bucket.items():
            with open(os.path.join(sub, f"{vname}.json"), "w") as f:
                json.dump({str(i): round(float(arr[i]), 4) for i in range(len(arr))}, f, indent=2)

    print(f"  → {sum(len(v) for v in rescore_results.values())} scores replaced, sigma=2 smoothed")

    # ---- Evaluation ----
    print("Computing AUC ...", end="", flush=True)
    new_metrics = _compute_auc(all_new_scores, gt_annotations)
    base_metrics = _compute_auc(all_baseline_scores, gt_annotations)
    phase1_metrics = _compute_auc(all_phase1_scores, gt_annotations)
    print(" done")

    # ---- Report ----
    print(f"\n  {'=' * 55}")
    print(f"  RESULTS — Three-way comparison (5-video subset)")
    print(f"  {'=' * 55}")
    print(f"  {'Metric':<15} {'Phase1 Only':>12} {'New (ours)':>12} {'Orig+Refine':>14} {'Δ Ours-P1':>11} {'Δ Ours-Orig':>12}")
    print(f"  {'-' * 74}")
    for metric_name in ["roc_auc", "pr_auc"]:
        p1, nv, bv = phase1_metrics.get(metric_name, 0), new_metrics.get(metric_name, 0), base_metrics.get(metric_name, 0)
        nm = "ROC-AUC" if metric_name == "roc_auc" else "PR-AUC"
        print(f"  {nm:<15} {p1:>12.4f} {nv:>12.4f} {bv:>14.4f} {nv-p1:>+11.4f} {nv-bv:>+12.4f}")

    print(f"\n  Per-video ROC-AUC:")
    print(f"  {'Video':<24} {'Phase1':>8} {'New':>8} {'Orig':>8} {'Δ(N-P1)':>9} {'Δ(N-O)':>9}")
    print(f"  {'-' * 68}")
    for vtype, vname, nframes in TEST_VIDEOS:
        p1 = _compute_auc({vname: all_phase1_scores.get(vname, np.zeros(1))}, gt_annotations)
        nv = _compute_auc({vname: all_new_scores.get(vname, np.zeros(1))}, gt_annotations)
        bv = _compute_auc({vname: all_baseline_scores.get(vname, np.zeros(1))}, gt_annotations)
        p1v, nvv, bvv = p1.get("roc_auc", 0), nv.get("roc_auc", 0), bv.get("roc_auc", 0)
        print(f"  {vname:<24} {p1v:>8.4f} {nvv:>8.4f} {bvv:>8.4f} {nvv-p1v:>+9.4f} {nvv-bvv:>+9.4f}")

    # Flagged summary
    print(f"\n  Stage C Flagged Frames:")
    for vtype, vname, nframes in TEST_VIDEOS:
        info = stage_c_summary.get(vname, {})
        nf = info.get("n_flagged", 0)
        print(f"    {vname:<24} {nf:>3d} / {nframes:>5d}  ({100*nf/nframes:.1f}%)")

    # Save metrics
    metrics_out = os.path.join(output_dir, "stage_e_metrics")
    os.makedirs(metrics_out, exist_ok=True)
    for metric_name, val in new_metrics.items():
        with open(os.path.join(metrics_out, f"{metric_name}.txt"), "w") as f:
            f.write(f"{val:.6f}\n")
    with open(os.path.join(metrics_out, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write("Three-way comparison (5-video subset):\n")
        for metric_name in ["roc_auc", "pr_auc"]:
            nm = "ROC-AUC" if metric_name == "roc_auc" else "PR-AUC"
            p1 = phase1_metrics.get(metric_name, 0)
            nv = new_metrics.get(metric_name, 0)
            bv = base_metrics.get(metric_name, 0)
            f.write(f"\n{nm}:\n")
            f.write(f"  Phase 1 only:     {p1:.4f}\n")
            f.write(f"  New (ours):       {nv:.4f}  (Δ vs P1: {nv-p1:+.4f})\n")
            f.write(f"  Original refined: {bv:.4f}  (Δ vs P1: {bv-p1:+.4f})\n")

    print(f"\n  Metrics → {metrics_out}/")
    return new_metrics, base_metrics, phase1_metrics, generator


def _compute_auc(scores_dict: Dict[str, np.ndarray], gt: Dict[str, List[Tuple[int, int]]]):
    """Compute ROC-AUC and PR-AUC for a set of video scores against ground truth."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    flat_labels = []
    flat_scores = []

    for vname, scores_arr in scores_dict.items():
        intervals = gt.get(vname, [])
        if not intervals:
            continue

        n = len(scores_arr)
        labels = np.zeros(n, dtype=int)
        for start, end in intervals:
            si = max(0, start)
            ei = min(n, end)
            if si < ei:
                labels[si:ei] = 1

        min_len = min(len(scores_arr), len(labels))
        flat_scores.extend(scores_arr[:min_len].tolist())
        flat_labels.extend(labels[:min_len].tolist())

    if not flat_labels or sum(flat_labels) == 0:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall, precision)

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


# ============================================================================
# Qualitative Analysis
# ============================================================================

def run_qualitative_analysis(stage_c_summary, output_dir):
    """Check whether flagged frames overlap with ground truth anomaly intervals."""
    gt = load_temporal_annotations(ANNO_TEMPORAL)
    results = []

    for vtype, vname, nframes in TEST_VIDEOS:
        info = stage_c_summary.get(vname, {})
        flagged = info.get("flagged_frames", [])
        intervals = gt.get(vname, [])
        hits = sum(1 for ff in flagged for s, e in intervals if s <= ff <= e)
        precision = hits / len(flagged) if flagged else 0.0
        results.append((vname, vtype, hits, len(flagged), precision, intervals, flagged))

    print(f"\n  Flagged vs GT Overlap:")
    print(f"  {'Video':<24} {'Hits':>5} {'Flagged':>7} {'Prec':>7}  GT")
    for vname, vtype, hits, nflag, prec, intervals, flagged in results:
        gt_str = ", ".join(f"[{s}-{e}]" for s, e in intervals)
        print(f"  {vname:<24} {hits:>5d} {nflag:>7d} {prec:>6.1%}  {gt_str}")

    qual_out = os.path.join(output_dir, "qualitative_analysis.json")
    with open(qual_out, "w") as f:
        json.dump([{"video": vname, "type": vtype, "hits": hits,
                     "n_flagged": nflag, "precision": prec,
                     "gt_intervals": intervals, "flagged_frames": flagged}
                   for vname, vtype, hits, nflag, prec, intervals, flagged in results], f, indent=2)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Quick test for Asymmetric Dual-Pass Reflection")
    p.add_argument("--skip-stage-d", action="store_true",
                   help="Force skip VLM Stage D even if video files are available")
    p.add_argument("--resume-from", choices=["stage_c", "stage_e"], default=None,
                   help="Resume from a saved checkpoint: 'stage_c' re-uses saved "
                        "context+flagged JSONs (skips Stage C LLM); 'stage_e' skips "
                        "directly to Stage E")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    print("=" * 55)
    print("  URF-HVAA  Quick Pre-Experiment Test")
    print("  Asymmetric Dual-Pass Reflection Architecture")
    print("=" * 55)
    print(f"  5 videos  |  ~25-40 min  |  Output: {output_dir}/\n")

    # ---- Pre-flight checks (compact) ----
    status = check_environment()
    if not status["ok"]:
        print("\n[FATAL] Environment errors:")
        for e in status["errors"]:
            print(f"  ✗ {e}")
        sys.exit(1)
    if status["warnings"]:
        print(f"  ({len(status['warnings'])} warning(s), continuing)\n")

    resume_mode = args.resume_from

    # ---- Stage C / Resume ----
    if resume_mode in ("stage_c", "stage_e"):
        print(f"[RESUME] Loading saved Stage C output ...")
        stage_c_summary = _load_stage_c_summary(output_dir)
        if not stage_c_summary:
            print("[FATAL] No saved Stage C output; run without --resume-from first.")
            sys.exit(1)
        llm_generator = None
    else:
        stage_c_summary, llm_generator = run_stage_c(TEST_VIDEOS, output_dir)

    # ---- Qualitative analysis ----
    if resume_mode != "stage_e":
        run_qualitative_analysis(stage_c_summary, output_dir)

    # ---- Stage D: VLM targeted verification (conditional) ----
    run_vlm = status.get("video_available", False) and status.get("has_transformers", False)
    if args.skip_stage_d:
        run_vlm = False

    refined_captions = None
    if run_vlm:
        from src.perception.vlm_engine import VLMEngine
        from src.utils.vlm_path import get_vlm_path
        print("\n  Probing VLM ...", end="", flush=True)
        # Only unload LLM AFTER we confirm VLM can load
        vlm_ok = False
        try:
            vlm_path = get_vlm_path()
            print(f" path={vlm_path}")
            engine = VLMEngine(model_path=vlm_path)
            engine.load()
            vlm_ok = True
        except Exception as e:
            print(f"\n  [WARN] VLM unavailable: {e}")
            print("         Falling back to TEXT-ONLY mode (LLM preserved)")

        if vlm_ok:
            engine.unload()  # release probe, run_stage_d will load its own
            print("  Unloading LLM to free GPU for VLM ...")
            _cleanup_llm(llm_generator)
            llm_generator = None
            try:
                refined_captions = run_stage_d(output_dir, stage_c_summary)
            except Exception as e:
                print(f"  [WARN] Stage D failed: {e} → falling back to TEXT-ONLY")
                refined_captions = None
    else:
        print("\n" + "=" * 55)
        print("  STAGE D  SKIPPED (no videos) → TEXT-ONLY mode")
        print("=" * 55)

    # ---- Stage E: Final scoring + merge + eval ----
    new_metrics, base_metrics, phase1_metrics, _e_gen = run_stage_e(
        output_dir, stage_c_summary, refined_captions, llm_generator,
    )

    # Final cleanup
    final_gen = llm_generator if llm_generator is not None else _e_gen
    if final_gen is not None:
        _cleanup_llm(final_gen)

    # ---- Done ----
    print("\n" + "=" * 55)
    print("  TEST COMPLETE")
    print("=" * 55)
    full_mode = run_vlm and refined_captions
    print(f"  Mode: {'FULL (VLM)' if full_mode else 'TEXT-ONLY'}")
    print(f"  ROC-AUC  |  Phase1: {phase1_metrics.get('roc_auc', 0):.4f}  "
          f"Ours: {new_metrics.get('roc_auc', 0):.4f}  "
          f"Orig: {base_metrics.get('roc_auc', 0):.4f}")
    print()


if __name__ == "__main__":
    main()
