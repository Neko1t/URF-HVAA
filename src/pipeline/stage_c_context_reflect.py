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
from typing import Optional

import cv2
import torch
from tqdm import tqdm

from libs.llama.llama import Llama
from src.data.video_record import VideoRecord
from src.reflection.conflict_detector import ConflictDetector
from src.reflection.context_memory import SlidingContextMemory
from src.utils.torch_utils import ensure_single_gpu_distributed


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
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_gen_len", type=int, default=2048)
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

def run(
    root_path: str,
    annotationfile_path: str,
    captions_dir: str,
    scores_dir: str,
    context_output: str,
    flagged_output: str,
    ckpt_dir: str,
    tokenizer_path: str,
    video_folder: str | None = None,
    window_seconds: float = 60.0,
    stride_seconds: float = 30.0,
    normality_percentile: float = 30.0,
    cap_max_flags: int = 20,
    max_captions_per_context: int = 30,
    default_fps: float = 30.0,
    max_seq_len: int = 4096,
    max_gen_len: int = 2048,
    seed: int = 1,
    video_filter: Optional[list[str]] = None,
) -> None:
    """Stage C: context memory + conflict detection (programmatic entry point).

    Args:
        video_filter: If given, only process videos whose names are in this list.
    """
    os.makedirs(context_output, exist_ok=True)
    os.makedirs(flagged_output, exist_ok=True)

    ensure_single_gpu_distributed()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=8,
        model_parallel_size=1,
        seed=seed,
    )

    memory = SlidingContextMemory(
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        normality_percentile=normality_percentile,
        max_captions_per_context=max_captions_per_context,
    )
    detector = ConflictDetector(
        cap_max_flags=cap_max_flags,
        max_gen_len=max_gen_len,
    )

    with open(annotationfile_path) as _f:
        video_list = [
            VideoRecord(x.strip().split(), root_path)
            for x in _f
        ]

    if video_filter is not None:
        filter_set = set(video_filter)
        video_list = [
            v for v in video_list
            if Path(v.path).name.replace(".mp4", "") in filter_set
        ]

    if not video_list:
        print("Stage C: no videos to process.")
        del generator
        torch.cuda.empty_cache()
        return

    for video in tqdm(video_list, desc="Stage C: context + conflict"):
        video_name = Path(video.path).name.replace(".mp4", "")
        video_path = (
            os.path.join(video_folder, f"{video_name}.mp4")
            if video_folder else ""
        )

        caption_path = os.path.join(captions_dir, f"{video_name}.json")
        if not os.path.exists(caption_path):
            tqdm.write(f"[skip] no captions: {caption_path}")
            continue
        with open(caption_path, "r") as f:
            captions = json.load(f)

        score_path = os.path.join(scores_dir, f"{video_name}.json")
        if not os.path.exists(score_path):
            tqdm.write(f"[skip] no scores: {score_path}")
            continue
        with open(score_path, "r") as f:
            scores = json.load(f)

        fps = _get_fps(video_path) if os.path.exists(video_path) else 0.0
        if fps <= 0:
            fps = default_fps
            if fps <= 0:
                tqdm.write(f"[skip] cannot determine FPS for: {video_name}")
                continue

        # Phase 2: context memory
        contexts = memory.process_video(captions, scores, fps, generator)
        ctx_out_path = os.path.join(
            context_output, f"{video_name}_windows.json",
        )
        with open(ctx_out_path, "w") as f:
            json.dump([_scene_context_to_dict(c) for c in contexts], f, indent=2)

        if not contexts:
            tqdm.write(f"[warn] no contexts generated for {video_name}")

        # Phase 3: conflict detection
        flagged = detector.detect(captions, scores, fps, contexts, generator)
        flag_out_path = os.path.join(
            flagged_output, f"{video_name}.json",
        )
        with open(flag_out_path, "w") as f:
            json.dump([_flagged_frame_to_dict(ff) for ff in flagged], f, indent=2)

        tqdm.write(
            f"{video_name}: {len(contexts)} windows, {len(flagged)} flagged"
        )

    del generator
    torch.cuda.empty_cache()
    print(f"Stage C done.\n  contexts → {context_output}\n  flagged → {flagged_output}")


def main() -> None:
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        captions_dir=args.captions_dir,
        scores_dir=args.scores_dir,
        context_output=args.context_output,
        flagged_output=args.flagged_output,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        video_folder=args.video_folder,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        normality_percentile=args.normality_percentile,
        cap_max_flags=args.cap_max_flags,
        default_fps=args.default_fps,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
