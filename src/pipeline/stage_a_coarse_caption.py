"""
Stage A: Coarse blind captioning [VLM only, ~15 GB].

Walks every video at interval=16 (patrol mode), produces a
base caption for each segment WITHOUT any prior context.

Output:  captions/phase1_coarse/{video}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.perception.vlm_engine import VLMEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage A: Coarse blind captioning")
    p.add_argument("--video_folder", required=True,
                   help="Path to folder containing .mp4 files")
    p.add_argument("--index_file", required=True,
                   help="Annotation index file (test.txt)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for caption JSON files")
    p.add_argument("--frame_interval", type=int, default=16,
                   help="Frame step between captions (default: 16)")
    p.add_argument("--mode", default="coarse",
                   help="Sampling mode: 'coarse' or 'fine'")
    return p.parse_args()


def run(
    video_folder: str,
    index_file: str,
    output_dir: str,
    frame_interval: int = 16,
    mode: str = "coarse",
    video_filter: list[str] | None = None,
) -> None:
    """Stage A: coarse blind captioning (programmatic entry point).

    Args:
        video_filter: If given, only process videos whose names are in this list.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(index_file, "r") as f:
        video_names = [line.strip().split()[0] for line in f if line.strip()]

    if video_filter is not None:
        filter_set = set(video_filter)
        video_names = [v for v in video_names if v in filter_set]

    if not video_names:
        print("Stage A: no videos to process.")
        return

    engine = VLMEngine()
    engine.load()

    for video_name in tqdm(video_names, desc="Stage A: blind captioning"):
        video_path = os.path.join(video_folder, f"{video_name}.mp4")
        output_path = os.path.join(output_dir, f"{video_name}.json")

        if os.path.isfile(output_path):
            continue

        if not os.path.exists(video_path):
            tqdm.write(f"[skip] missing: {video_path}")
            continue

        duration, fps, total_frames = VLMEngine._get_video_info(video_path)
        if duration <= 0 or total_frames <= 0:
            tqdm.write(f"[skip] unreadable: {video_path}")
            continue

        results: dict[str, str] = {}
        for frame_idx in range(0, total_frames, frame_interval):
            caption = engine.blind_caption(
                video_path, frame_idx, mode=mode,
            )
            results[str(frame_idx)] = caption

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    engine.unload()
    print(f"Stage A done. Captions saved to {output_dir}")


def main() -> None:
    args = parse_args()
    run(
        video_folder=args.video_folder,
        index_file=args.index_file,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
