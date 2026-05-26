"""
Stage E: Final scoring + merge [LLM only, ~16 GB].

1. Re-scores refined captions from Stage D using the LLM.
2. Merges refined scores into the original Phase 1 score array.
3. Applies global 1D Gaussian smoothing (sigma=2).
4. Optionally runs eval.py to compute AUC metrics.

Output:  scores/final/{video}.json
         scores/final/metrics/  (if --run_eval)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from libs.llama.llama import Llama
from src.data.video_record import VideoRecord
from src.reflection.targeted_verifier import TargetedVerifier
from src.utils.torch_utils import ensure_single_gpu_distributed


# ---------------------------------------------------------------------------
# Context-aware scoring prompt (used when context_dir + flagged_dir provided)
# ---------------------------------------------------------------------------

CONTEXT_SCORING_SYSTEM = (
    "You are a surveillance anomaly scorer. Compare the OBSERVATION "
    "against the SCENE BASELINE and output a score.\n\n"
    "SCENE BASELINE (normal, expected activity in this area):\n"
    "{scene_context}\n\n"
    "SCORING GUIDE:\n"
    "  [0.0-0.3] Matches baseline — normal, routine, expected behavior\n"
    "  [0.3-0.6] Minor deviation — unusual but plausibly benign\n"
    "  [0.6-0.8] Concerning — clear deviation from expected activity\n"
    "  [0.8-1.0] Highly anomalous — strong indicators of criminal/suspicious behavior\n\n"
    "{conflict_note}\n"
    "Respond with ONLY a number in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] "
    "as a Python list like [0.7]. Begin with '[' and end with ']'. No explanation."
)


def _find_context_for_frame(fidx: int, fps: float, windows: list[dict]) -> str:
    """Return the scene context description covering *fidx*, or empty string."""
    timestamp = fidx / fps
    for w in windows:
        if w["window_start_sec"] <= timestamp < w["window_end_sec"]:
            return w.get("description", "")
    return ""


def _get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage E: Final scoring + merge")
    p.add_argument("--root_path", required=True,
                   help="Root path for video frames")
    p.add_argument("--annotationfile_path", required=True,
                   help="Annotation index file")
    p.add_argument("--original_scores_dir", required=True,
                   help="Directory of Phase 1 initial scores (Stage B)")
    p.add_argument("--refined_captions_dir", required=True,
                   help="Directory of Phase 4 refined captions (Stage D)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for final score JSONs")
    p.add_argument("--ckpt_dir", required=True,
                   help="Llama checkpoint directory")
    p.add_argument("--tokenizer_path", required=True,
                   help="Llama tokenizer.model path")
    p.add_argument("--context_prompt",
                   default="How would you rate the scene described on a "
                           "scale from 0 to 1, with 0 representing a standard "
                           "scene and 1 denoting a scene with suspicious or "
                           "potentially criminal activities?")
    p.add_argument("--format_prompt",
                   default="Respond with exactly one number in a Python list "
                           "[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "
                           "1.0]. Start with '[' and end with ']'. No extra text.")
    p.add_argument("--merge_sigma", type=float, default=2.0,
                   help="Gaussian sigma for score smoothing")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_gen_len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--context_dir", type=str, default=None,
                   help="Optional: Phase 2 scene context dir for context-aware scoring")
    p.add_argument("--flagged_dir", type=str, default=None,
                   help="Optional: Phase 3 flagged frames dir for conflict notes")
    p.add_argument("--run_eval", action="store_true",
                   help="Also compute AUC metrics after merging")
    p.add_argument("--temporal_annotation_file", type=str, default=None)
    p.add_argument("--frame_interval", type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    root_path: str,
    annotationfile_path: str,
    original_scores_dir: str,
    refined_captions_dir: str,
    output_dir: str,
    ckpt_dir: str,
    tokenizer_path: str,
    context_dir: Optional[str] = None,
    flagged_dir: Optional[str] = None,
    video_folder: Optional[str] = None,
    context_prompt: str = (
        "How would you rate the scene described on a scale from 0 to 1, "
        "with 0 representing a standard scene and 1 denoting a scene "
        "with suspicious or potentially criminal activities?"
    ),
    format_prompt: str = (
        "Respond with exactly one number in a Python list "
        "[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "
        "1.0]. Start with '[' and end with ']'. No extra text."
    ),
    merge_sigma: float = 2.0,
    max_seq_len: int = 512,
    max_gen_len: int = 64,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: int = 1,
    run_eval: bool = False,
    temporal_annotation_file: Optional[str] = None,
    refined_baseline_dir: Optional[str] = None,
    frame_interval: int = 16,
    video_filter: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """Stage E: final scoring + merge + eval (programmatic entry point).

    If both *context_dir* and *flagged_dir* are provided, uses context-aware
    scoring prompts with scene baseline + conflict notes (Phase 2+3 results).
    Otherwise falls back to the simple scoring prompt (backward compatible).

    If *refined_baseline_dir* is provided and *run_eval* is True, prints a
    three-way comparison: Phase 1 only vs Ours vs Original refined baseline.

    Args:
        video_filter: If given, only process videos whose names are in this list.

    Returns:
        {video_name: final_scores_array}
    """
    use_context = bool(context_dir and flagged_dir)
    mode_str = "context-aware" if use_context else "simple prompt"
    print(f"Stage E: {mode_str}")

    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    ensure_single_gpu_distributed()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        model_parallel_size=1,
        seed=seed,
    )

    simple_system_prompt = context_prompt + " " + format_prompt

    video_list = [
        VideoRecord(x.strip().split(), root_path)
        for x in open(annotationfile_path)
    ]

    if video_filter is not None:
        filter_set = set(video_filter)
        video_list = [
            v for v in video_list
            if Path(v.path).name.replace(".mp4", "") in filter_set
        ]

    if not video_list:
        print("Stage E: no videos to process.")
        del generator
        torch.cuda.empty_cache()
        return {}

    all_final_scores: dict[str, np.ndarray] = {}
    all_phase1_scores: dict[str, np.ndarray] = {}
    verifier = TargetedVerifier()

    for video in tqdm(video_list, desc="Stage E: final scoring"):
        video_name = Path(video.path).name.replace(".mp4", "")

        # Load original scores
        orig_path = os.path.join(original_scores_dir, f"{video_name}.json")
        if not os.path.exists(orig_path):
            continue
        with open(orig_path, "r") as f:
            original_scores_raw = json.load(f)
        original_scores = {int(k): float(v) for k, v in original_scores_raw.items()}

        # Load refined captions
        refined_path = os.path.join(refined_captions_dir, f"{video_name}.json")
        refined_captions: dict[int, str] = {}
        if os.path.exists(refined_path):
            with open(refined_path, "r") as f:
                refined_captions = {
                    int(k): v for k, v in json.load(f).items()
                }

        # ---- Load context + conflict info (if enabled) ----
        scene_windows: list[dict] = []
        flagged_info: dict[int, dict] = {}
        fps: float = 30.0

        if use_context:
            ctx_path = os.path.join(context_dir, f"{video_name}_windows.json")
            if os.path.exists(ctx_path):
                with open(ctx_path) as f:
                    scene_windows = json.load(f)

            flag_path = os.path.join(flagged_dir, f"{video_name}.json")
            if os.path.exists(flag_path):
                with open(flag_path) as f:
                    for item in json.load(f):
                        flagged_info[item["frame"]] = item

            if video_folder:
                vp = os.path.join(video_folder, f"{video_name}.mp4")
                if os.path.exists(vp):
                    fps = _get_fps(vp)

        # ---- Re-score refined captions ----
        refined_scores: dict[int, float] = {}
        frames_to_rescore = list(refined_captions.keys()) if refined_captions \
                            else list(flagged_info.keys())

        for fidx in frames_to_rescore:
            caption = refined_captions.get(fidx)
            if not caption:
                continue

            if use_context:
                scene_ctx = _find_context_for_frame(fidx, fps, scene_windows)
                scene_text = scene_ctx[:300] if scene_ctx else (
                    "A typical surveillance scene. "
                    "No unusual baseline information available."
                )
                finfo = flagged_info.get(fidx, {})
                conflict_note = _build_conflict_note(finfo)
                system_prompt = CONTEXT_SCORING_SYSTEM.format(
                    scene_context=scene_text,
                    conflict_note=conflict_note,
                )
            else:
                system_prompt = simple_system_prompt

            dialog = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{caption}."},
            ]
            try:
                result = generator.chat_completion(
                    [dialog],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                response = result[0]["generation"]["content"]
                m = re.search(r"\[(\d+(?:\.\d+)?)\]", response)
                if m:
                    refined_scores[fidx] = float(m.group(1))
            except Exception:
                pass

        # ---- Merge scores ----
        num_frames = video.num_frames
        arr = verifier._dict_to_array(original_scores, num_frames)
        # Store Phase1-only baseline BEFORE score replacement
        all_phase1_scores[video_name] = gaussian_filter1d(arr.copy(), sigma=merge_sigma)

        for fidx, score in refined_scores.items():
            if 0 <= fidx < num_frames:
                arr[fidx] = score

        arr = gaussian_filter1d(arr, sigma=merge_sigma)

        output_path = os.path.join(output_dir, f"{video_name}.json")
        final_dict = {str(i): round(float(arr[i]), 4) for i in range(num_frames)}
        with open(output_path, "w") as f:
            json.dump(final_dict, f, indent=2)

        all_final_scores[video_name] = arr

    del generator
    torch.cuda.empty_cache()

    if run_eval and temporal_annotation_file:
        annotations = _parse_annotations(temporal_annotation_file)
        labels_cache = _build_labels_cache(video_list, annotations)

        our_metrics = _compute_auc_for_scores(
            all_final_scores, video_list, labels_cache,
        )
        phase1_metrics = _compute_auc_for_scores(
            all_phase1_scores, video_list, labels_cache,
        )

        for name, val in our_metrics.items():
            with open(os.path.join(metrics_dir, f"{name}.txt"), "w") as f:
                f.write(f"{val:.6f}\n")

        baseline_metrics = None
        baseline_scores: dict[str, np.ndarray] = {}
        if refined_baseline_dir:
            baseline_scores = _load_refined_baseline(video_list, refined_baseline_dir)
            if baseline_scores:
                baseline_metrics = _compute_auc_for_scores(
                    baseline_scores, video_list, labels_cache,
                )

        _print_comparison(our_metrics, phase1_metrics, baseline_metrics)
        _print_per_video(video_list, all_final_scores, all_phase1_scores,
                         baseline_scores, labels_cache)
        _save_comparison(our_metrics, phase1_metrics, baseline_metrics, metrics_dir)

    print(f"Stage E done. Final scores → {output_dir}")
    return all_final_scores


def _parse_annotations(
    temporal_annotation_file: str,
) -> dict[str, list[str]]:
    """Parse temporal annotation file once."""
    annotations: dict[str, list[str]] = {}
    with open(temporal_annotation_file) as f:
        for line in f:
            parts = line.strip().split()
            annotations[parts[0].replace(".mp4", "")] = parts[2:]
    return annotations


def _build_labels_cache(
    video_list: list,
    annotations: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Pre-build binary label arrays for all videos."""
    cache: dict[str, np.ndarray] = {}
    for video in video_list:
        video_name = Path(video.path).name.replace(".mp4", "")
        anns = annotations.get(video_name, [])
        anns = [x for x in anns if x != "-1"]
        if not anns:
            continue
        labels = np.zeros(video.num_frames, dtype=int)
        starts, ends = anns[::2], anns[1::2]
        for s, e in zip(starts, ends):
            si, ei = int(s) - video.start_frame, int(e) - video.start_frame
            si = max(0, si); ei = min(video.num_frames, ei)
            if si < ei:
                labels[si:ei] = 1
        cache[video_name] = labels
    return cache


def _load_refined_baseline(
    video_list: list,
    baseline_dir: str,
) -> dict[str, np.ndarray]:
    """Load original refined baseline scores if available."""
    verifier = TargetedVerifier()
    scores: dict[str, np.ndarray] = {}
    for video in video_list:
        video_name = Path(video.path).name.replace(".mp4", "")
        path = os.path.join(baseline_dir, f"{video_name}.json")
        try:
            with open(path) as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        orig = {int(k): float(v) for k, v in raw.items()}
        scores[video_name] = verifier._dict_to_array(orig, video.num_frames)
    return scores


def _compute_auc_for_scores(
    scores_dict: dict[str, np.ndarray],
    video_list: list,
    labels_cache: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute ROC-AUC and PR-AUC against pre-built label arrays."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    flat_labels: list[int] = []
    flat_scores: list[float] = []

    for video in video_list:
        video_name = Path(video.path).name.replace(".mp4", "")
        arr = scores_dict.get(video_name)
        labels = labels_cache.get(video_name)
        if arr is None or labels is None:
            continue
        n = min(len(arr), len(labels))
        flat_scores.extend(arr[:n].tolist())
        flat_labels.extend(labels[:n].tolist())

    if not flat_labels or sum(flat_labels) == 0:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    precision, recall, _ = precision_recall_curve(flat_labels, flat_scores)
    return {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(recall, precision))}


def _print_comparison(
    ours: dict, phase1: dict, baseline: dict | None,
) -> None:
    """Print three-way AUC comparison table."""
    print(f"\n  {'=' * 55}")
    print(f"  RESULTS — Three-way comparison")
    print(f"  {'=' * 55}")
    hdr = f"  {'Metric':<15} {'Phase1 Only':>12} {'New (ours)':>12}"
    if baseline:
        hdr += f" {'Orig+Refine':>14} {'Δ Ours-P1':>11} {'Δ Ours-Orig':>12}"
    else:
        hdr += f" {'Δ Ours-P1':>11}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        p1 = phase1.get(key, 0)
        nv = ours.get(key, 0)
        if baseline:
            bv = baseline.get(key, 0)
            print(f"  {label:<15} {p1:>12.4f} {nv:>12.4f} {bv:>14.4f} "
                  f"{nv-p1:>+11.4f} {nv-bv:>+12.4f}")
        else:
            print(f"  {label:<15} {p1:>12.4f} {nv:>12.4f} {nv-p1:>+11.4f}")


def _print_per_video(
    video_list: list,
    ours: dict, phase1: dict, baseline: dict,
    labels_cache: dict[str, np.ndarray],
) -> None:
    """Print per-video ROC-AUC breakdown using pre-built labels."""
    print(f"\n  Per-video ROC-AUC:")
    has_base = bool(baseline)
    hdr = f"  {'Video':<24} {'Phase1':>8} {'New':>8}"
    hdr += f" {'Orig':>8} {'Δ(N-P1)':>9} {'Δ(N-O)':>9}" if has_base \
           else f" {'Δ':>9}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for video in video_list:
        vname = Path(video.path).name.replace(".mp4", "")
        p1v = _single_video_auc(vname, phase1, labels_cache)
        nvv = _single_video_auc(vname, ours, labels_cache)
        if has_base:
            bvv = _single_video_auc(vname, baseline, labels_cache)
            print(f"  {vname:<24} {p1v:>8.4f} {nvv:>8.4f} {bvv:>8.4f} "
                  f"{nvv-p1v:>+9.4f} {nvv-bvv:>+9.4f}")
        else:
            print(f"  {vname:<24} {p1v:>8.4f} {nvv:>8.4f} {nvv-p1v:>+9.4f}")


def _single_video_auc(
    vname: str,
    scores_dict: dict[str, np.ndarray],
    labels_cache: dict[str, np.ndarray],
) -> float:
    """Compute ROC-AUC for a single video (no annotation re-parsing)."""
    arr = scores_dict.get(vname)
    labels = labels_cache.get(vname)
    if arr is None or labels is None:
        return 0.0
    from sklearn.metrics import auc, roc_curve
    n = min(len(arr), len(labels))
    if n == 0 or labels[:n].sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(labels[:n], arr[:n])
    return float(auc(fpr, tpr))


def _save_comparison(
    ours: dict, phase1: dict, baseline: dict | None, metrics_dir: str,
) -> None:
    """Save comparison results to a text file."""
    path = os.path.join(metrics_dir, "comparison.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Three-way comparison:\n\n")
        for key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
            f.write(f"{label}:\n")
            f.write(f"  Phase 1 only:       {phase1.get(key, 0):.4f}\n")
            f.write(f"  New (ours):         {ours.get(key, 0):.4f}  "
                    f"(Δ vs P1: {ours.get(key, 0) - phase1.get(key, 0):+.4f})\n")
            if baseline:
                f.write(f"  Original refined:   {baseline.get(key, 0):.4f}  "
                        f"(Δ vs P1: {baseline.get(key, 0) - phase1.get(key, 0):+.4f})\n")
            f.write("\n")


def _build_conflict_note(finfo: dict) -> str:
    suspicious = finfo.get("suspicious_element", "")
    alternative = finfo.get("alternative_explanation", "")
    if suspicious and alternative:
        return (
            f"PRIOR NOTE: a previous scan flagged potential \"{suspicious}\", "
            f"but this could simply be \"{alternative}\". "
            f"Weigh this information but base your score on the actual observation."
        )
    elif suspicious:
        return (
            f"PRIOR NOTE: a previous scan flagged potential \"{suspicious}\". "
            f"Weigh this information but base your score on the actual observation."
        )
    return ""


def main() -> None:
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        original_scores_dir=args.original_scores_dir,
        refined_captions_dir=args.refined_captions_dir,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        context_dir=args.context_dir,
        flagged_dir=args.flagged_dir,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        merge_sigma=args.merge_sigma,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        run_eval=args.run_eval,
        temporal_annotation_file=args.temporal_annotation_file,
        frame_interval=args.frame_interval,
    )


# ---------------------------------------------------------------------------
# Eval (mirrors eval.py logic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
