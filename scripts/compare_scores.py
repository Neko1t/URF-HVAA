#!/usr/bin/env python
"""Compare original vs new pipeline scores for a given video in the GT anomaly region.

Usage:
    python scripts/compare_scores.py Shooting015_x264
    python scripts/compare_scores.py Shooting015_x264 --smooth-sigma 10
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Paths relative to project root
ORIG_REFINED  = "data/ucf_crime/refined_scores/videollama3"
NEW_FINAL     = "data/ucf_crime/quick_test_results/stage_e_final_scores"
GT_ANNO       = "data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

# Test videos with GT intervals from the quick_test configuration
TEST_VIDEOS = {
    "Abuse028_x264":    (1412, [(165, 240)]),
    "Arrest001_x264":   (2374, [(1185, 1485)]),
    "Arson016_x264":    (1795, [(1000, 1796)]),
    "Burglary021_x264": (1537, [(60, 200), (840, 1340)]),
    "Shooting015_x264": (1713, [(855, 1715)]),
}


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs new pipeline scores per video"
    )
    parser.add_argument(
        "video", nargs="?", default=None,
        help="Video name (e.g. Shooting015_x264). Omit to compare all 5."
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=None,
        help="Apply additional Gaussian smoothing before comparison "
             "(0 = no smoothing, default: no extra smoothing)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Print a simple ASCII chart of scores over time"
    )
    args = parser.parse_args()

    videos_to_check = [args.video] if args.video else list(TEST_VIDEOS.keys())

    for vname in videos_to_check:
        if vname not in TEST_VIDEOS:
            print(f"Unknown video: {vname}")
            print(f"  Known: {', '.join(TEST_VIDEOS.keys())}")
            continue

        nframes, gt_intervals = TEST_VIDEOS[vname]

        # ---- Load original refined scores ----
        orig_path = os.path.join(ORIG_REFINED, f"{vname}.json")
        if not os.path.exists(orig_path):
            print(f"[SKIP] Original scores not found: {orig_path}")
            continue
        orig_raw = load_json(orig_path)

        # Convert sparse {frame_str: score} to dense array via interpolation
        orig_arr = np.full(nframes, np.nan)
        for k, v in orig_raw.items():
            fidx = int(k)
            if 0 <= fidx < nframes:
                orig_arr[fidx] = float(v)
        known = np.where(~np.isnan(orig_arr))[0]
        if len(known) >= 2:
            orig_arr = np.interp(np.arange(nframes), known, orig_arr[known])
        elif len(known) == 1:
            orig_arr = np.full(nframes, orig_arr[known[0]])
        else:
            orig_arr = np.zeros(nframes)

        # ---- Load new pipeline final scores ----
        new_path = os.path.join(NEW_FINAL, f"{vname}.json")
        if not os.path.exists(new_path):
            print(f"[SKIP] New scores not found: {new_path}")
            continue
        new_raw = load_json(new_path)
        new_arr = np.array([float(new_raw.get(str(i), 0.0)) for i in range(nframes)])

        # ---- Extra smoothing (optional) ----
        if args.smooth_sigma is not None and args.smooth_sigma > 0:
            orig_arr = gaussian_filter1d(orig_arr, sigma=args.smooth_sigma)
            new_arr = gaussian_filter1d(new_arr, sigma=args.smooth_sigma)

        # ---- Report ----
        print(f"\n{'=' * 60}")
        print(f"  {vname}  ({nframes} frames)")
        print(f"  GT intervals: {gt_intervals}")
        print(f"{'=' * 60}")

        # Per-interval breakdown
        all_orig_gt = []
        all_new_gt = []
        all_orig_bg = []
        all_new_bg = []

        for start, end in gt_intervals:
            si = max(0, start)
            ei = min(nframes, end)
            o_gt = orig_arr[si:ei]
            n_gt = new_arr[si:ei]
            all_orig_gt.extend(o_gt.tolist())
            all_new_gt.extend(n_gt.tolist())
            print(f"  GT [{si}-{ei}] ({ei-si} frames):")
            print(f"    Original  avg={np.mean(o_gt):.4f}  max={np.max(o_gt):.4f}  "
                  f"min={np.min(o_gt):.4f}  std={np.std(o_gt):.4f}")
            print(f"    New       avg={np.mean(n_gt):.4f}  max={np.max(n_gt):.4f}  "
                  f"min={np.min(n_gt):.4f}  std={np.std(n_gt):.4f}")

        # Background (non-GT) region
        bg_mask = np.ones(nframes, dtype=bool)
        for start, end in gt_intervals:
            bg_mask[max(0, start):min(nframes, end)] = False
        all_orig_bg = orig_arr[bg_mask]
        all_new_bg = new_arr[bg_mask]

        # Summary
        print(f"\n  {'Region':<12} {'Original':>10} {'New':>10} {'Delta':>10}")
        print(f"  {'-' * 42}")
        print(f"  {'GT (anomaly)':<12} {np.mean(all_orig_gt):>10.4f} "
              f"{np.mean(all_new_gt):>10.4f} "
              f"{np.mean(all_new_gt)-np.mean(all_orig_gt):>+10.4f}")
        print(f"  {'Background':<12} {np.mean(all_orig_bg):>10.4f} "
              f"{np.mean(all_new_bg):>10.4f} "
              f"{np.mean(all_new_bg)-np.mean(all_orig_bg):>+10.4f}")

        separation_orig = np.mean(all_orig_gt) - np.mean(all_orig_bg)
        separation_new = np.mean(all_new_gt) - np.mean(all_new_bg)
        print(f"  {'Separation':<12} {separation_orig:>10.4f} "
              f"{separation_new:>10.4f} "
              f"{separation_new - separation_orig:>+10.4f}")
        print(f"  (Separation = GT_avg - BG_avg; higher is better)")

        # ---- ASCII plot ----
        if args.plot:
            print(f"\n  Score plot (GT region marked with █):")
            bin_width = max(1, nframes // 80)
            bins = np.arange(0, nframes, bin_width)
            orig_binned = [np.mean(orig_arr[i:i+bin_width]) for i in bins]
            new_binned = [np.mean(new_arr[i:i+bin_width]) for i in bins]
            for i, (o, n_val) in enumerate(zip(orig_binned, new_binned)):
                frame = i * bin_width
                in_gt = any(s <= frame < e for s, e in gt_intervals)
                bar_o = "█" * int(o * 20)
                bar_n = "█" * int(n_val * 20)
                marker = "▓" if in_gt else " "
                print(f"  {marker}F{frame:>5d} orig=[{bar_o:<20s}] {o:.2f}")
                print(f"  {marker}       new =[{bar_n:<20s}] {n_val:.2f}")

    print()


if __name__ == "__main__":
    main()
