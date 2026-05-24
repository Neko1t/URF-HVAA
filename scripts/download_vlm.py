#!/usr/bin/env python
"""Download VideoLLaMA3-7B from ModelScope to libs/VideoLLaMA3-7B/.

Usage:
    python scripts/download_vlm.py              # download to default location
    python scripts/download_vlm.py --dir ./my_models/VideoLLaMA3  # custom path
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download VideoLLaMA3-7B from ModelScope")
    parser.add_argument(
        "--dir",
        default=None,
        help="Target directory (default: libs/VideoLLaMA3-7B relative to project root)",
    )
    args = parser.parse_args()

    # Resolve target directory
    if args.dir:
        target_dir = os.path.abspath(args.dir)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        target_dir = os.path.join(project_root, "libs", "VideoLLaMA3-7B")

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[INFO] Target directory already exists and is non-empty:")
        print(f"       {target_dir}")
        print(f"[INFO] Skipping download. Delete the directory first to re-download.")
        return

    os.makedirs(target_dir, exist_ok=True)

    print(f"[INFO] Downloading DAMO-NLP-SG/VideoLLaMA3-7B from ModelScope...")
    print(f"[INFO] Target: {target_dir}")
    print(f"[INFO] This is ~15 GB — may take a while depending on your network.")

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    snapshot_download(
        "DAMO-NLP-SG/VideoLLaMA3-7B",
        cache_dir=target_dir,
    )

    print(f"\n[DONE] Model downloaded to: {target_dir}")
    print(f"[INFO] VLM loading code will auto-detect this local copy.")


if __name__ == "__main__":
    main()
