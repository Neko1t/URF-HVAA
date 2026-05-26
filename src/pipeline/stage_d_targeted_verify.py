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
from typing import Optional

import cv2
from tqdm import tqdm

from src.perception.vlm_engine import (
    AdversarialVerification,
    FlaggedFrame,
    SceneContext,
    VLMEngine,
)
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
    p.add_argument("--adversarial", action="store_true",
                   help="v2: Use dual-perspective adversarial VLM verification")
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


def _adversarial_to_dict(av: AdversarialVerification) -> dict:
    return {
        "frame": av.frame,
        "caption_refined": av.caption_refined,
        "positive_tag": av.positive_tag,
        "positive_confidence": av.positive_confidence,
        "negative_tag": av.negative_tag,
        "negative_confidence": av.negative_confidence,
    }


def _get_video_meta(video_path: str) -> tuple[float, int]:
    """Return (fps, frame_count) from a single VideoCapture open."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (float(fps) if fps > 0 else 0.0), nframes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    flagged_dir: str,
    context_dir: str,
    video_folder: str,
    annotationfile_path: str,
    output_dir: str,
    root_path: str,
    mode: str = "fine",
    adversarial: bool = False,
    video_filter: Optional[list[str]] = None,
) -> None:
    """Stage D: targeted VLM verification (programmatic entry point).

    Args:
        video_filter: If given, only process videos whose names are in this list.
    """
    from src.data.video_record import VideoRecord

    os.makedirs(output_dir, exist_ok=True)

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
        print("Stage D: no videos to process.")
        return

    engine = VLMEngine()
    engine.load()

    verifier = TargetedVerifier()
    global_tracker = ComputeTracker()

    for video in tqdm(video_list, desc="Stage D: targeted verify"):
        video_name = Path(video.path).name.replace(".mp4", "")
        video_path = os.path.join(video_folder, f"{video_name}.mp4")

        flagged_path = os.path.join(flagged_dir, f"{video_name}.json")
        if not os.path.exists(flagged_path):
            continue
        flagged_frames = _load_flagged(flagged_path)
        if not flagged_frames:
            continue

        context_path = os.path.join(context_dir, f"{video_name}_windows.json")
        if not os.path.exists(context_path):
            tqdm.write(f"[skip] no contexts: {context_path}")
            continue
        contexts = _load_contexts(context_path)
        if not contexts:
            continue

        fps, total_frames_from_video = _get_video_meta(video_path)
        if fps <= 0:
            tqdm.write(f"[skip] cannot read FPS for: {video_path}")
            continue

        total_frames = video.num_frames if video.num_frames > 0 else total_frames_from_video

        if adversarial:
            refined = verifier.verify_frames_adversarial(
                engine, video_path, flagged_frames, contexts, fps,
                mode=mode, progress=True,
            )
            global_tracker.phase4_vlm_calls += verifier.tracker.phase4_vlm_calls
            output_path = os.path.join(output_dir, f"{video_name}.json")
            with open(output_path, "w") as f:
                json.dump(
                    {str(k): _adversarial_to_dict(v) for k, v in refined.items()},
                    f, indent=2,
                )
            tqdm.write(
                f"{video_name}: {len(refined)}/{len(flagged_frames)} "
                f"adversarially verified"
            )
        else:
            refined = verifier.verify_frames(
                engine, video_path, flagged_frames, contexts, fps,
                mode=mode, progress=True,
            )
            global_tracker.phase4_vlm_calls += verifier.tracker.phase4_vlm_calls
            output_path = os.path.join(output_dir, f"{video_name}.json")
            with open(output_path, "w") as f:
                json.dump(refined, f, indent=2)
            tqdm.write(
                f"{video_name}: {len(refined)}/{len(flagged_frames)} "
                f"frames verified"
            )

        global_tracker.total_frames += total_frames
        global_tracker.phase1_vlm_calls += total_frames // 16
        global_tracker.flagged_count += len(flagged_frames)

    engine.unload()

    global_tracker.print_report()
    tracker_path = os.path.join(output_dir, "_compute_tracker.json")
    with open(tracker_path, "w") as f:
        json.dump(global_tracker.report(), f, indent=2)
    print(f"Stage D done. Refined captions → {output_dir}")


def main() -> None:
    args = parse_args()
    run(
        flagged_dir=args.flagged_dir,
        context_dir=args.context_dir,
        video_folder=args.video_folder,
        annotationfile_path=args.annotationfile_path,
        output_dir=args.output_dir,
        root_path=args.root_path,
        mode=args.mode,
        adversarial=args.adversarial,
    )


if __name__ == "__main__":
    main()
