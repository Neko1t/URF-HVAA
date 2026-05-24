"""
Stage D: Targeted visual verification [VLM only, ~15 GB].

Re-examines ONLY the frames flagged by Phase 3, using fine-grained
sampling (interval≈2.5s, max_frames=16) and anti-hallucination prompts.

This is the "high-cost on-demand" stage of the asymmetric design.

Output:  captions/phase4_fine/{video}.json  (only flagged frames)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from src.perception.vlm_engine import FlaggedFrame, SceneContext, VLMEngine
from src.reflection.targeted_verifier import ComputeTracker, TargetedVerifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage D: Targeted VLM verification")
    p.add_argument("--flagged_dir", required=True,
                   help="Directory of flagged frame JSONs from Stage C")
    p.add_argument("--context_dir", required=True,
                   help="Directory of scene context JSONs from Stage C")
    p.add_argument("--video_folder", required=True,
                   help="Path to .mp4 video files")
    p.add_argument("--annotationfile_path", required=True,
                   help="Annotation index file (for video list)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for refined caption JSONs")
    p.add_argument("--root_path", required=True,
                   help="Root path for video frames (for VideoRecord)")
    p.add_argument("--mode", default="fine",
                   help="VLM sampling mode (default: fine)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_flagged(path: str) -> list[FlaggedFrame]:
    with open(path, "r") as f:
        raw = json.load(f)
    return [
        FlaggedFrame(
            frame=item["frame"],
            caption_summary=item.get("caption_summary", ""),
            conflict_reason=item.get("conflict_reason", ""),
            suspicious_element=item.get("suspicious_element", ""),
            alternative_explanation=item.get("alternative_explanation", ""),
        )
        for item in raw
    ]


def _load_contexts(path: str) -> list[SceneContext]:
    with open(path, "r") as f:
        raw = json.load(f)
    return [
        SceneContext(
            window_start_sec=item["window_start_sec"],
            window_end_sec=item["window_end_sec"],
            description=item["description"],
        )
        for item in raw
    ]


def _get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Read video list
    from src.data.video_record import VideoRecord

    video_list = [
        VideoRecord(x.strip().split(), args.root_path)
        for x in open(args.annotationfile_path)
    ]

    engine = VLMEngine()
    engine.load()

    verifier = TargetedVerifier()
    global_tracker = ComputeTracker()

    for video in tqdm(video_list, desc="Stage D: targeted verify"):
        video_name = Path(video.path).name.replace(".mp4", "")
        video_path = os.path.join(args.video_folder, f"{video_name}.mp4")

        # Load flagged frames
        flagged_path = os.path.join(args.flagged_dir, f"{video_name}.json")
        if not os.path.exists(flagged_path):
            continue
        flagged_frames = _load_flagged(flagged_path)
        if not flagged_frames:
            continue

        # Load scene contexts
        context_path = os.path.join(args.context_dir, f"{video_name}_windows.json")
        if not os.path.exists(context_path):
            tqdm.write(f"[skip] no contexts: {context_path}")
            continue
        contexts = _load_contexts(context_path)
        if not contexts:
            continue

        # Get FPS + frame count
        fps = _get_fps(video_path)
        total_frames = video.num_frames if video.num_frames > 0 else (
            int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        )

        if fps <= 0:
            tqdm.write(f"[skip] cannot read FPS for: {video_path}")
            continue

        # Phase 4a: VLM verification (only flagged frames)
        refined = verifier.verify_frames(
            engine, video_path, flagged_frames, contexts, fps, mode=args.mode,
        )

        # Track compute for this video
        global_tracker.total_frames += total_frames
        global_tracker.phase1_vlm_calls += total_frames // 16
        global_tracker.phase4_vlm_calls += verifier.tracker.phase4_vlm_calls
        global_tracker.flagged_count += len(flagged_frames)

        # Save refined captions
        output_path = os.path.join(args.output_dir, f"{video_name}.json")
        with open(output_path, "w") as f:
            json.dump(refined, f, indent=2)

        tqdm.write(
            f"{video_name}: {len(refined)}/{len(flagged_frames)} frames verified"
        )

    engine.unload()

    # Print compute savings
    global_tracker.print_report()
    tracker_path = os.path.join(args.output_dir, "_compute_tracker.json")
    with open(tracker_path, "w") as f:
        json.dump(global_tracker.report(), f, indent=2)
    print(f"Stage D done. Refined captions → {args.output_dir}")


if __name__ == "__main__":
    main()
