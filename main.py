#!/usr/bin/env python
"""
URF-HVAA  —  Asymmetric Dual-Pass Reflection Pipeline
========================================================
Single entry point for both pre-experiment and full experiment.

Usage:
    python main.py                           # Full experiment (all videos, Stage A)
    python main.py --quick-test              # Pre-experiment (5 videos, Stage C)
    python main.py --quick-test --resume-from A  # Pre-experiment from Stage A
    python main.py --resume-from B           # Full experiment from Stage B
    python main.py --dataset xd_violence     # Switch dataset
    python main.py --skip-stage-d            # Skip VLM Stage D
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress harmless but noisy third-party diagnostics
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*set_default_tensor_type.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLAMA_CKPT = "libs/llama/llama3.1-8b"
LLAMA_TOKENIZER = "libs/llama/llama3.1-8b/tokenizer.model"

# 5 representative videos for quick pre-experiment testing
QUICK_TEST_VIDEOS = [
    "Abuse028_x264",
    "Arrest001_x264",
    "Arson016_x264",
    "Burglary021_x264",
    "Shooting015_x264",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="URF-HVAA: Asymmetric Dual-Pass Reflection Pipeline",
    )
    p.add_argument("--quick-test", action="store_true",
                   help="Run pre-experiment on 5 representative videos "
                        "(default: starts from Stage C, uses pre-computed "
                        "captions/scores)")
    p.add_argument("--resume-from", choices=["A", "B", "C", "D", "E"], default=None,
                   help="Stage to start from: A (captioning), B (scoring), "
                        "C (context+conflict), D (targeted verify), E (final merge). "
                        "Default: A for full experiment, C for --quick-test")
    p.add_argument("--dataset", default="ucf_crime",
                   help="Dataset name under data/ (default: ucf_crime)")
    p.add_argument("--output-base", default=None,
                   help="Override base output directory (default: data/<dataset>/)")
    p.add_argument("--skip-stage-d", action="store_true",
                   help="Force skip Stage D even if video files are available")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip final AUC evaluation")
    p.add_argument("--max-captions-per-context", type=int, default=30,
                   help="Max normal-frame captions per scene context window "
                        "(default: 30, 0 = no limit)")
    p.add_argument("--score-gate", action="store_true",
                   help="v2: Enable dual-threshold Score Gate between Stage B and C. "
                        "Confident videos skip reflection (C/D/E).")
    p.add_argument("--adversarial", action="store_true",
                   help="v2: Enable adversarial dual-perspective VLM verification "
                        "in Stage D (requires --score-gate implicitly for Stage D)")
    p.add_argument("--detection-mode", type=str, default="intervals",
                   choices=["intervals", "full"],
                   help="Stage C conflict detection mode: 'intervals' (v2, lighter) "
                        "or 'full' (v1 legacy)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(args: argparse.Namespace) -> dict:
    """Build the path dictionary used by all pipeline stages."""
    base = args.output_base or f"data/{args.dataset}"

    video_folder = f"{base}/videos"
    anno_index = f"{base}/annotations/test.txt"
    anno_temporal = f"{base}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

    paths = {
        "root_path": base,
        "video_folder": video_folder,
        "anno_index": anno_index,
        "anno_temporal": anno_temporal,
        "ckpt_dir": LLAMA_CKPT,
        "tokenizer_path": LLAMA_TOKENIZER,

        # Pipeline outputs (Stage A/B)
        "stage_a_out": f"{base}/captions/phase1_coarse",
        "stage_b_out": f"{base}/scores/phase1_initial",

        # Pre-computed legacy data (used when resuming from Stage C)
        "legacy_captions": f"{base}/captions/video_llama3_json_results",
        "legacy_scores": f"{base}/scores/videollama3",

        # Stage C/D/E outputs
        "stage_c_context": f"{base}/context/phase2",
        "stage_c_flagged": f"{base}/reflection/phase3_flagged",
        "stage_d_out": f"{base}/captions/phase4_fine",
        "stage_e_out": f"{base}/scores/final",
        # Original refined baseline for comparison
        "refined_baseline": f"{base}/refined_scores/videollama3",
    }
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_stage_header(stage: str, desc: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  STAGE {stage}  {desc}")
    print(f"{'=' * 60}")


def check_videos_available(video_folder: str, video_filter: Optional[list[str]],
                           anno_index: str) -> bool:
    """Return True if all needed .mp4 files exist."""
    if not os.path.isdir(video_folder):
        return False

    if video_filter is not None:
        needed = [f"{v}.mp4" for v in video_filter]
    else:
        with open(anno_index) as f:
            needed = [f"{line.strip().split()[0]}.mp4" for line in f if line.strip()]

    return all(os.path.isfile(os.path.join(video_folder, n)) for n in needed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    video_filter = QUICK_TEST_VIDEOS if args.quick_test else None

    # Resolve starting stage
    if args.resume_from:
        resume = args.resume_from
    elif args.quick_test:
        resume = "C"   # quick-test skips VLM-heavy A/B by default
    else:
        resume = "A"   # full experiment runs everything

    mode_label = "QUICK-TEST (5 videos)" if video_filter else "FULL EXPERIMENT"

    print("=" * 60)
    print("  URF-HVAA  Asymmetric Dual-Pass Reflection Pipeline")
    print("=" * 60)
    print(f"  Mode:       {mode_label}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Resume:     Stage {resume}")
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  GPU:        {gpu_name} (serial VLM/LLM)")
    print(f"{'=' * 60}")

    t_start = time.time()

    # ---- Resolve input directories for Stage C ----
    if resume <= "B":
        # Stage A/B were (or will be) run → use pipeline outputs
        captions_for_c = paths["stage_a_out"]
        scores_for_c = paths["stage_b_out"]
    else:
        # Resuming from C → use pre-computed legacy data
        captions_for_c = paths["legacy_captions"]
        scores_for_c = paths["legacy_scores"]
        print(f"\n  Using pre-computed captions: {captions_for_c}")
        print(f"  Using pre-computed scores:   {scores_for_c}")

    # =====================================================================
    # Stage A: VLM — Coarse blind captioning
    # =====================================================================
    if resume <= "A":
        print_stage_header("A", "Coarse Blind Captioning [VLM, ~15 GB]")
        t0 = time.time()

        from src.pipeline.stage_a_coarse_caption import run as run_a
        run_a(
            video_folder=paths["video_folder"],
            index_file=paths["anno_index"],
            output_dir=paths["stage_a_out"],
            frame_interval=16,
            mode="coarse",
            video_filter=video_filter,
        )
        print(f"  Stage A elapsed: {time.time() - t0:.0f}s")
    else:
        print_stage_header("A", "SKIPPED (resuming from later stage)")

    # =====================================================================
    # Stage B: LLM — Initial scoring
    # =====================================================================
    if resume <= "B":
        print_stage_header("B", "Initial Scoring [LLM, ~16 GB]")
        t0 = time.time()

        from src.pipeline.stage_b_initial_scoring import run as run_b
        run_b(
            root_path=paths["root_path"],
            annotationfile_path=paths["anno_index"],
            captions_dir=paths["stage_a_out"],
            output_dir=paths["stage_b_out"],
            ckpt_dir=paths["ckpt_dir"],
            tokenizer_path=paths["tokenizer_path"],
            video_filter=video_filter,
        )
        print(f"  Stage B elapsed: {time.time() - t0:.0f}s")
    else:
        print_stage_header("B", "SKIPPED (resuming from later stage)")

    # =====================================================================
    # Score Gate (v2, optional) — between Stage B and Stage C
    # =====================================================================
    gated_normal: list[str] = []
    gated_anomalous: list[str] = []

    if args.score_gate and resume <= "C":
        print_stage_header("GATE", "Dual-Threshold Score Gate")
        from src.reflection.score_gate import GATE_ANOMALOUS, GATE_NORMAL, ScoreGate

        gate = ScoreGate()
        gated_normal, gated_anomalous = [], []

        # Load all scores
        scores_dir = scores_for_c
        import glob as _glob
        for score_path in sorted(_glob.glob(f"{scores_dir}/*.json")):
            vname = Path(score_path).stem
            if video_filter and vname not in video_filter:
                continue
            with open(score_path) as f:
                scores = json.load(f)
            decision, reason = gate.decide(scores)

            if decision == GATE_NORMAL:
                gated_normal.append(vname)
                print(f"  [GATE:NORMAL]    {vname}: {reason}")
            elif decision == GATE_ANOMALOUS:
                gated_anomalous.append(vname)
                print(f"  [GATE:ANOMALOUS] {vname}: {reason}")
            else:
                pass  # will go through Stage C

        if gated_normal:
            # Write all-normal scores for gated videos
            out_dir = paths["stage_e_out"]
            os.makedirs(out_dir, exist_ok=True)
            for vname in gated_normal:
                gate_out = os.path.join(out_dir, f"{vname}.json")
                phase1_path = os.path.join(scores_dir, f"{vname}.json")
                if os.path.exists(phase1_path):
                    # Keep Phase 1 but set all to low scores
                    with open(phase1_path) as f:
                        p1 = json.load(f)
                    all_low = {k: min(float(v), 0.15) for k, v in p1.items()}
                    with open(gate_out, "w") as f:
                        json.dump(all_low, f, indent=2)

        if gated_anomalous:
            # Copy Phase 1 scores directly for gated anomalous videos
            out_dir = paths["stage_e_out"]
            os.makedirs(out_dir, exist_ok=True)
            from shutil import copy2
            for vname in gated_anomalous:
                src = os.path.join(scores_dir, f"{vname}.json")
                dst = os.path.join(out_dir, f"{vname}.json")
                if os.path.exists(src):
                    copy2(src, dst)

        print(f"  Gate result: {len(gated_normal)} normal, "
              f"{len(gated_anomalous)} anomalous "
              f"(skip C/D/E), remaining → Stage C")

    # =====================================================================
    # Stage C: LLM — Context memory + Conflict detection
    # =====================================================================
    if resume <= "C":
        print_stage_header("C", "Context Memory + Conflict Detection [LLM, ~16 GB]")
        t0 = time.time()

        from src.pipeline.stage_c_context_reflect import run as run_c

        # Exclude gated videos from Stage C
        skip_videos = set(gated_normal + gated_anomalous)
        c_video_filter = [v for v in video_filter if v not in skip_videos] \
                         if video_filter else None

        if c_video_filter or video_filter is None:
            run_c(
                root_path=paths["root_path"],
                annotationfile_path=paths["anno_index"],
                captions_dir=captions_for_c,
                scores_dir=scores_for_c,
                video_folder=paths["video_folder"],
                context_output=paths["stage_c_context"],
                flagged_output=paths["stage_c_flagged"],
                ckpt_dir=paths["ckpt_dir"],
                tokenizer_path=paths["tokenizer_path"],
                max_captions_per_context=args.max_captions_per_context,
                detection_mode=args.detection_mode,
                video_filter=c_video_filter,
            )
        print(f"  Stage C elapsed: {time.time() - t0:.0f}s")
    else:
        print_stage_header("C", "SKIPPED (resuming from later stage)")

    # =====================================================================
    # Stage D: VLM — Targeted verification (conditional)
    # =====================================================================
    videos_ok = check_videos_available(
        paths["video_folder"], video_filter, paths["anno_index"],
    )
    run_d = videos_ok and not args.skip_stage_d

    if resume > "D":
        run_d = False
        print_stage_header("D", "SKIPPED (resuming from Stage E)")

    if run_d:
        print_stage_header("D", "Targeted Visual Verification [VLM, ~15 GB]")
        if args.adversarial:
            print("  v2 ADVERSARIAL mode: dual-perspective VLM verification")
        t0 = time.time()

        skip_videos_d = set(gated_normal + gated_anomalous)
        d_video_filter = [v for v in video_filter if v not in skip_videos_d] \
                         if video_filter else None

        from src.pipeline.stage_d_targeted_verify import run as run_d_fn
        run_d_fn(
            flagged_dir=paths["stage_c_flagged"],
            context_dir=paths["stage_c_context"],
            video_folder=paths["video_folder"],
            annotationfile_path=paths["anno_index"],
            output_dir=paths["stage_d_out"],
            root_path=paths["root_path"],
            mode="fine",
            adversarial=args.adversarial,
            video_filter=d_video_filter,
        )
        print(f"  Stage D elapsed: {time.time() - t0:.0f}s")
    elif not run_d and resume <= "D":
        print_stage_header("D", "SKIPPED (videos unavailable or --skip-stage-d)")
        print("  TEXT-ONLY mode — Stage E will use conflict-aware prompts")

    # =====================================================================
    # Stage E: LLM — Final scoring + merge + eval
    # =====================================================================
    print_stage_header("E", "Final Scoring + Merge + Evaluation [LLM, ~16 GB]")
    t0 = time.time()

    from src.pipeline.stage_e_final_merge import run as run_e

    # Stage E reads original scores from wherever Stage C got them
    refined_dir = paths["stage_d_out"] if run_d else None

    # Exclude gated videos (already written by Score Gate)
    skip_videos_e = set(gated_normal + gated_anomalous)
    e_video_filter = [v for v in video_filter if v not in skip_videos_e] \
                     if video_filter else None

    run_e(
        root_path=paths["root_path"],
        annotationfile_path=paths["anno_index"],
        original_scores_dir=scores_for_c,
        refined_captions_dir=refined_dir or paths["stage_d_out"],
        output_dir=paths["stage_e_out"],
        ckpt_dir=paths["ckpt_dir"],
        tokenizer_path=paths["tokenizer_path"],
        context_dir=paths["stage_c_context"],
        flagged_dir=paths["stage_c_flagged"],
        video_folder=paths["video_folder"],
        run_eval=not args.no_eval,
        temporal_annotation_file=paths["anno_temporal"],
        refined_baseline_dir=paths["refined_baseline"],
        video_filter=e_video_filter,
    )
    print(f"  Stage E elapsed: {time.time() - t0:.0f}s")

    # =====================================================================
    # Done
    # =====================================================================
    total = time.time() - t_start
    vlm_mode = "FULL (VLM)" if run_d else "TEXT-ONLY"
    stages_run = ""
    if resume <= "A":
        stages_run = "A→B→C"
    elif resume == "B":
        stages_run = "(skip A)→B→C"
    elif resume == "C":
        stages_run = "(skip A,B)→C"
    elif resume == "D":
        stages_run = "(skip A,B,C)"
    else:
        stages_run = "(skip A,B,C,D)"
    if resume <= "D" and run_d:
        stages_run += "→D"
    stages_run += "→E"

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Mode:       {mode_label}  |  {vlm_mode}")
    print(f"  Stages:     {stages_run}")
    print(f"  Total time: {total / 60:.1f} min  ({total:.0f}s)")
    print(f"  Output:     {paths['stage_e_out']}/")
    if not args.no_eval:
        print(f"  Metrics:    {paths['stage_e_out']}/metrics/")
    print(f"{'=' * 60}")

    # Clean shutdown — destroy process group to suppress NCCL warning
    import torch
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
