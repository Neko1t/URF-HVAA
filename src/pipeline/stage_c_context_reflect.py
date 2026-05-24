"""
Stage C: Context memory + conflict detection [LLM only, ~16 GB].

Combines Phase 2 and Phase 3 in a single LLM loading cycle:

  Phase 2: SlidingContextMemory builds scene normality profiles.
  Phase 3: ConflictDetector identifies frames whose caption contradicts
           the scene context — producing a FlaggedFrame list.

Output:  context/phase2/{video}_windows.json
         reflection/phase3_flagged/{video}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from libs.llama.llama import Llama
from src.data.video_record import VideoRecord
from src.reflection.conflict_detector import ConflictDetector
from src.reflection.context_memory import SlidingContextMemory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage C: Context memory + conflict detection")
    p.add_argument("--root_path", required=True,
                   help="Root path for video frames")
    p.add_argument("--annotationfile_path", required=True,
                   help="Annotation index file")
    p.add_argument("--captions_dir", required=True,
                   help="Directory of caption JSONs from Stage A")
    p.add_argument("--scores_dir", required=True,
                   help="Directory of score JSONs from Stage B")
    p.add_argument("--video_folder", default=None,
                   help="Path to .mp4 video files (for FPS metadata). "
                        "If omitted, --default_fps must be set.")
    p.add_argument("--context_output", required=True,
                   help="Output directory for scene context JSONs")
    p.add_argument("--flagged_output", required=True,
                   help="Output directory for flagged frame JSONs")
    p.add_argument("--ckpt_dir", required=True,
                   help="Llama checkpoint directory")
    p.add_argument("--tokenizer_path", required=True,
                   help="Llama tokenizer.model path")
    p.add_argument("--window_seconds", type=float, default=60.0)
    p.add_argument("--stride_seconds", type=float, default=30.0)
    p.add_argument("--normality_percentile", type=float, default=30.0)
    p.add_argument("--cap_max_flags", type=int, default=20)
    p.add_argument("--default_fps", type=float, default=30.0,
                   help="Fallback FPS when video files are unavailable "
                        "(default: 30.0 for UCF-Crime)")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 0.0


def _scene_context_to_dict(ctx) -> dict:
    return {
        "window_start_sec": ctx.window_start_sec,
        "window_end_sec": ctx.window_end_sec,
        "description": ctx.description,
    }


def _flagged_frame_to_dict(ff) -> dict:
    return {
        "frame": ff.frame,
        "caption_summary": ff.caption_summary,
        "conflict_reason": ff.conflict_reason,
        "suspicious_element": ff.suspicious_element,
        "alternative_explanation": ff.alternative_explanation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.context_output, exist_ok=True)
    os.makedirs(args.flagged_output, exist_ok=True)

    # Build LLM generator (single instance for both Phase 2 and Phase 3)
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=8,
        seed=args.seed,
    )

    memory = SlidingContextMemory(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        normality_percentile=args.normality_percentile,
    )
    detector = ConflictDetector(cap_max_flags=args.cap_max_flags)

    # Read video list
    video_list = [
        VideoRecord(x.strip().split(), args.root_path)
        for x in open(args.annotationfile_path)
    ]

    for video in tqdm(video_list, desc="Stage C: context + conflict"):
        video_name = Path(video.path).name.replace(".mp4", "")
        video_path = (
            os.path.join(args.video_folder, f"{video_name}.mp4")
            if args.video_folder else ""
        )

        # Load captions
        caption_path = os.path.join(args.captions_dir, f"{video_name}.json")
        if not os.path.exists(caption_path):
            tqdm.write(f"[skip] no captions: {caption_path}")
            continue
        with open(caption_path, "r") as f:
            captions = json.load(f)

        # Load scores
        score_path = os.path.join(args.scores_dir, f"{video_name}.json")
        if not os.path.exists(score_path):
            tqdm.write(f"[skip] no scores: {score_path}")
            continue
        with open(score_path, "r") as f:
            scores = json.load(f)

        # Get FPS (from video file, or fallback to default)
        fps = _get_fps(video_path) if os.path.exists(video_path) else 0.0
        if fps <= 0:
            fps = args.default_fps
            if fps <= 0:
                tqdm.write(f"[skip] cannot determine FPS for: {video_name}")
                continue

        # ---- Phase 2: context memory ----
        contexts = memory.process_video(captions, scores, fps, generator)
        ctx_out_path = os.path.join(
            args.context_output, f"{video_name}_windows.json",
        )
        with open(ctx_out_path, "w") as f:
            json.dump([_scene_context_to_dict(c) for c in contexts], f, indent=2)

        if not contexts:
            tqdm.write(f"[warn] no contexts generated for {video_name}")

        # ---- Phase 3: conflict detection ----
        flagged = detector.detect(captions, scores, fps, contexts, generator)
        flag_out_path = os.path.join(
            args.flagged_output, f"{video_name}.json",
        )
        with open(flag_out_path, "w") as f:
            json.dump([_flagged_frame_to_dict(ff) for ff in flagged], f, indent=2)

        tqdm.write(
            f"{video_name}: {len(contexts)} windows, {len(flagged)} flagged"
        )

    del generator
    print(f"Stage C done.\n  contexts → {args.context_output}\n  flagged → {args.flagged_output}")


if __name__ == "__main__":
    main()
