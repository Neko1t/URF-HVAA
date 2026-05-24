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

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from libs.llama.llama import Llama
from src.data.video_record import VideoRecord


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
    p.add_argument("--run_eval", action="store_true",
                   help="Also compute AUC metrics after merging")
    p.add_argument("--temporal_annotation_file", type=str, default=None)
    p.add_argument("--frame_interval", type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Build LLM
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
        seed=args.seed,
    )

    system_prompt = args.context_prompt + " " + args.format_prompt

    # Read video list
    video_list = [
        VideoRecord(x.strip().split(), args.root_path)
        for x in open(args.annotationfile_path)
    ]

    all_final_scores: dict[str, np.ndarray] = {}

    for video in tqdm(video_list, desc="Stage E: final scoring"):
        video_name = Path(video.path).name.replace(".mp4", "")

        # Load original scores
        orig_path = os.path.join(args.original_scores_dir, f"{video_name}.json")
        if not os.path.exists(orig_path):
            continue
        with open(orig_path, "r") as f:
            original_scores_raw = json.load(f)
        original_scores = {int(k): float(v) for k, v in original_scores_raw.items()}

        # Load refined captions
        refined_path = os.path.join(args.refined_captions_dir, f"{video_name}.json")
        refined_captions: dict[int, str] = {}
        if os.path.exists(refined_path):
            with open(refined_path, "r") as f:
                refined_captions = {
                    int(k): v for k, v in json.load(f).items()
                }

        # ---- Re-score refined captions ----
        refined_scores: dict[int, float] = {}
        for fidx, caption in refined_captions.items():
            dialog = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{caption}."},
            ]
            try:
                result = generator.chat_completion(
                    [dialog],
                    max_gen_len=args.max_gen_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                response = result[0]["generation"]["content"]
                m = re.search(r"\[(\d+(?:\.\d+)?)\]", response)
                if m:
                    refined_scores[fidx] = float(m.group(1))
            except Exception:
                pass

        # ---- Merge scores ----
        # Build full array from original
        num_frames = video.num_frames
        arr = np.full(num_frames, np.nan)
        for fidx, score in original_scores.items():
            if 0 <= fidx < num_frames:
                arr[fidx] = score

        # Interpolate gaps
        known = np.where(~np.isnan(arr))[0]
        if len(known) >= 2:
            arr = np.interp(np.arange(num_frames), known, arr[known])
        elif len(known) == 1:
            arr = np.full(num_frames, arr[known[0]])
        else:
            arr = np.zeros(num_frames)

        # Replace flagged positions
        for fidx, score in refined_scores.items():
            if 0 <= fidx < num_frames:
                arr[fidx] = score

        # Global Gaussian smooth
        arr = gaussian_filter1d(arr, sigma=args.merge_sigma)

        # Save
        output_path = os.path.join(args.output_dir, f"{video_name}.json")
        final_dict = {str(i): round(float(arr[i]), 4) for i in range(num_frames)}
        with open(output_path, "w") as f:
            json.dump(final_dict, f, indent=2)

        all_final_scores[video_name] = arr

    del generator

    # ---- Optional eval ----
    if args.run_eval and args.temporal_annotation_file:
        _run_evaluation(args, all_final_scores, metrics_dir)

    print(f"Stage E done. Final scores → {args.output_dir}")


# ---------------------------------------------------------------------------
# Eval (mirrors eval.py logic)
# ---------------------------------------------------------------------------

def _run_evaluation(
    args: argparse.Namespace,
    all_scores: dict[str, np.ndarray],
    metrics_dir: str,
) -> None:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    annotations: dict[str, list[str]] = {}
    with open(args.temporal_annotation_file) as f:
        for line in f:
            parts = line.strip().split()
            video_name = str(parts[0]).replace(".mp4", "")
            annotations[video_name] = parts[2:]

    flat_labels: list[int] = []
    flat_scores: list[float] = []

    for video in [
        VideoRecord(x.strip().split(), args.root_path)
        for x in open(args.annotationfile_path)
    ]:
        video_name = Path(video.path).name.replace(".mp4", "")
        scores_arr = all_scores.get(video_name)
        if scores_arr is None:
            continue

        video_anns = annotations.get(video_name, [])
        video_anns = [x for x in video_anns if x != "-1"]
        if not video_anns:
            continue

        starts = video_anns[::2]
        ends = video_anns[1::2]

        # Build binary labels
        labels = np.zeros(video.num_frames, dtype=int)
        for s, e in zip(starts, ends):
            si, ei = int(s) - video.start_frame, int(e) - video.start_frame
            si = max(0, si)
            ei = min(video.num_frames, ei)
            if si < ei:
                labels[si:ei] = 1

        min_len = min(len(scores_arr), len(labels))
        flat_scores.extend(scores_arr[:min_len].tolist())
        flat_labels.extend(labels[:min_len].tolist())

    if not flat_labels or sum(flat_labels) == 0:
        print("[eval] No positive labels found; skipping metrics.")
        return

    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, th = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    # Optimal threshold (Youden's J)
    youden = tpr - fpr
    opt_thresh = float(th[np.argmax(youden)]) if len(th) > 0 else 0.5
    print(f"Optimal threshold (Youden): {opt_thresh:.4f}")

    with open(os.path.join(metrics_dir, "roc_auc.txt"), "w") as f:
        f.write(f"{roc_auc:.4f}\n")
    with open(os.path.join(metrics_dir, "pr_auc.txt"), "w") as f:
        f.write(f"{pr_auc:.4f}\n")
    with open(os.path.join(metrics_dir, "optimal_threshold.txt"), "w") as f:
        f.write(f"{opt_thresh:.4f}\n")


if __name__ == "__main__":
    main()
