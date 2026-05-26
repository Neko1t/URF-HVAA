"""
Stage B: Initial LLM scoring [LLM only, ~16 GB].

Reads coarse captions from Stage A and scores each frame (0-1 anomaly)
using the Llama 3.1-8B text model.

Output:  scores/phase1_initial/{video}.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from libs.llama.llama import Dialog, Llama
from src.data.video_record import VideoRecord
from src.utils.torch_utils import ensure_single_gpu_distributed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B: Initial LLM scoring")
    p.add_argument("--root_path", required=True,
                   help="Root path for video frames")
    p.add_argument("--annotationfile_path", required=True,
                   help="Annotation index file")
    p.add_argument("--captions_dir", required=True,
                   help="Directory of caption JSONs from Stage A")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for score JSONs")
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
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--frame_interval", type=int, default=16)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_gen_len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scoring engine (lightweight wrapper around Llama)
# ---------------------------------------------------------------------------

class StageBScorer:
    def __init__(
        self,
        ckpt_dir: str,
        tokenizer_path: str,
        captions_dir: str,
        context_prompt: str,
        format_prompt: str,
        batch_size: int = 32,
        frame_interval: int = 16,
        max_seq_len: int = 512,
        max_gen_len: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.9,
        seed: int = 1,
    ):
        self.captions_dir = captions_dir
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

        ensure_single_gpu_distributed()
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            model_parallel_size=1,
            seed=seed,
        )

    def cleanup(self) -> None:
        if self.generator is not None:
            del self.generator
            self.generator = None
        torch.cuda.empty_cache()

    def score_video(self, video: VideoRecord) -> dict[str, float]:
        video_name = Path(video.path).name.replace(".mp4", "")
        caption_path = os.path.join(self.captions_dir, f"{video_name}.json")

        if not os.path.exists(caption_path):
            tqdm.write(f"[skip] no captions: {caption_path}")
            return {}

        with open(caption_path, "r") as f:
            captions = json.load(f)

        system_prompt = self.context_prompt + " " + self.format_prompt
        frame_step = self.frame_interval
        batch_size = self.batch_size

        video_scores: dict[str, float] = {}

        for batch_start in tqdm(
            range(0, video.num_frames, batch_size * frame_step),
            desc=f"Scoring {video_name}",
            unit="batch",
        ):
            batch_end = min(
                batch_start + batch_size * frame_step, video.num_frames,
            )
            batch_frames = list(range(batch_start, batch_end, frame_step))

            dialogs: list[Dialog] = []
            for fidx in batch_frames:
                caption = captions.get(str(fidx), "No activity.")
                dialogs.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{caption}."},
                ])

            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            for result, fidx in zip(results, batch_frames):
                response = result["generation"]["content"]
                score = self._parse_score(response)
                video_scores[str(fidx)] = score

        return self._interpolate(video_scores)

    @staticmethod
    def _parse_score(response: str) -> float:
        m = re.search(r"\[(\d+(?:\.\d+)?)\]", response)
        return float(m.group(1)) if m else -1.0

    @staticmethod
    def _interpolate(scores: dict[str, float]) -> dict[str, float]:
        valid = [(int(k), v) for k, v in scores.items() if v != -1.0]
        if not valid:
            return scores
        valid.sort()
        all_frames = sorted(int(k) for k in scores)
        interp = np.interp(all_frames, [x for x, _ in valid], [y for _, y in valid])
        return {str(k): round(float(v), 3) for k, v in zip(all_frames, interp)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    root_path: str,
    annotationfile_path: str,
    captions_dir: str,
    output_dir: str,
    ckpt_dir: str,
    tokenizer_path: str,
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
    batch_size: int = 32,
    frame_interval: int = 16,
    max_seq_len: int = 512,
    max_gen_len: int = 64,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: int = 1,
    video_filter: Optional[list[str]] = None,
) -> None:
    """Stage B: initial LLM scoring (programmatic entry point).

    Args:
        video_filter: If given, only process videos whose names are in this list.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save prompts for reproducibility
    with open(os.path.join(output_dir, "context_prompt.txt"), "w") as f:
        f.write(context_prompt)
    with open(os.path.join(output_dir, "format_prompt.txt"), "w") as f:
        f.write(format_prompt)

    scorer = StageBScorer(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        captions_dir=captions_dir,
        context_prompt=context_prompt,
        format_prompt=format_prompt,
        batch_size=batch_size,
        frame_interval=frame_interval,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
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
        print("Stage B: no videos to process.")
        scorer.cleanup()
        return

    for video in tqdm(video_list, desc="Stage B: initial scoring"):
        video_name = Path(video.path).name.replace(".mp4", "")
        output_path = os.path.join(output_dir, f"{video_name}.json")

        if os.path.isfile(output_path):
            continue

        scores = scorer.score_video(video)
        if not scores:
            continue

        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)

    scorer.cleanup()
    print(f"Stage B done. Scores saved to {output_dir}")


def main() -> None:
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        captions_dir=args.captions_dir,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
