#!/usr/bin/env python
"""Download specified UCF-Crime test videos from Kaggle.

The dataset stores videos inside ZIP archives (Anomaly-Videos-Part-1.zip ~ Part-4.zip).
This script downloads the needed ZIP(s), extracts only the target .mp4 files,
and discards the rest.

REQUIRES: Kaggle API token at ~/.kaggle/kaggle.json
    Get yours at: https://www.kaggle.com/settings/api

Usage:
    python scripts/download_videos.py                    # download 5 test videos
    python scripts/download_videos.py Abuse028_x264      # single video
"""

import os
import shutil
import subprocess
import sys
import zipfile

KAGGLE_DATASET = "minmints/ufc-crime-full-dataset"
TARGET_DIR = "data/ucf_crime/videos"

# Map video → category (for finding inside ZIP) + which ZIP it lives in
# Partition guesses based on UCF-Crime README naming convention
VIDEO_INFO = {
    "Abuse028_x264":    {"category": "Abuse",    "zip": "Anomaly-Videos-Part-1.zip"},
    "Arrest001_x264":   {"category": "Arrest",   "zip": "Anomaly-Videos-Part-1.zip"},
    "Arson016_x264":    {"category": "Arson",    "zip": "Anomaly-Videos-Part-1.zip"},
    "Burglary021_x264": {"category": "Burglary", "zip": "Anomaly-Videos-Part-2.zip"},
    "Shooting015_x264": {"category": "Shooting", "zip": "Anomaly-Videos-Part-3.zip"},
}


def check_kaggle_api():
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("[ERROR] ~/.kaggle/kaggle.json not found.")
        print("        Generate a token at: https://www.kaggle.com/settings/api")
        return False
    return True


def download_zip(zip_name, target_dir):
    """Download a single ZIP file from the dataset. Returns path to downloaded zip."""
    zip_path = os.path.join(target_dir, zip_name)
    if os.path.exists(zip_path):
        print(f"    [SKIP] {zip_name} already downloaded")
        return zip_path

    print(f"    Downloading {zip_name} ...", end="", flush=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-f", zip_name,
        "-p", target_dir,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f" FAILED")
        print(f"    {res.stdout.strip()}")
        return None

    # Kaggle wraps single-file downloads in a .zip too
    # → we get Anomaly-Videos-Part-1.zip.zip 😑
    double_zip = os.path.join(target_dir, f"{zip_name}.zip")
    if os.path.exists(double_zip):
        with zipfile.ZipFile(double_zip, "r") as zf:
            zf.extractall(target_dir)
        os.remove(double_zip)

    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f" OK ({size_mb:.0f} MB)")
    else:
        print(f" (not found after extract)")
    return zip_path if os.path.exists(zip_path) else None


def extract_video_from_zip(zip_path, video_name, category, target_dir):
    """Extract a single .mp4 from a ZIP archive, searching by category subdir."""
    final_path = os.path.join(target_dir, f"{video_name}.mp4")
    if os.path.exists(final_path):
        return True  # already done

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Try common path patterns inside the ZIP
        candidates = [
            f"{category}/{video_name}.mp4",
            f"Videos/{category}/{video_name}.mp4",
            f"{video_name}.mp4",
        ]
        for cand in candidates:
            try:
                zf.getinfo(cand)  # raises KeyError if not present
                print(f"    Extracting {cand} ...", end="", flush=True)
                with zf.open(cand) as src, open(final_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(" OK")
                return True
            except KeyError:
                continue

    # Not found — list what's actually in the ZIP (sample)
    all_names = zf.namelist()
    matches = [n for n in all_names if video_name in n or video_name.replace("_x264", "") in n]
    print(f"    NOT FOUND in {os.path.basename(zip_path)}")
    if matches:
        print(f"    Closest matches: {matches[:5]}")
    else:
        print(f"    ZIP contains {len(all_names)} entries. Samples: {all_names[:5]}")
    return False


def main():
    if not check_kaggle_api():
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target: {os.path.abspath(TARGET_DIR)}\n")

    # Parse args
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        targets = {k: v for k, v in VIDEO_INFO.items() if k in args}
        if not targets:
            print(f"No matching videos. Known: {', '.join(VIDEO_INFO)}")
            sys.exit(1)
    else:
        targets = VIDEO_INFO

    # Group by ZIP to avoid re-downloading the same archive
    zip_downloaded = {}  # zip_name → path

    ok = 0
    for vname, info in targets.items():
        zip_name = info["zip"]
        category = info["category"]
        final_path = os.path.join(TARGET_DIR, f"{vname}.mp4")

        if os.path.exists(final_path):
            print(f"  [SKIP] {vname}.mp4 already exists")
            ok += 1
            continue

        print(f"\n  [{vname}]")

        # Download ZIP (cached per zip_name)
        if zip_name not in zip_downloaded:
            zip_path = download_zip(zip_name, TARGET_DIR)
            zip_downloaded[zip_name] = zip_path
        else:
            zip_path = zip_downloaded[zip_name]

        if zip_path is None:
            print(f"    Skipping {vname} — ZIP download failed")
            continue

        # Extract just this video
        if extract_video_from_zip(zip_path, vname, category, TARGET_DIR):
            ok += 1

    # Cleanup: delete ZIP files (keep only .mp4)
    print(f"\n  Cleaning up ZIP archives ...")
    for zip_name, zip_path in zip_downloaded.items():
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"    Removed {zip_name}")

    print(f"\n[DONE] {ok}/{len(targets)} videos in {TARGET_DIR}/")
    if ok == len(targets):
        print("[INFO] All videos ready — you can now run:")
        print("       python scripts/quick_test.py")


if __name__ == "__main__":
    main()
