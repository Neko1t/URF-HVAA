#!/usr/bin/env python
"""Download models from ModelScope to libs/.

Usage:
    python scripts/download_vlm.py                          # download VLM only
    python scripts/download_vlm.py --model vlm              # same as above
    python scripts/download_vlm.py --model embedder         # download all-MiniLM-L6-v2
    python scripts/download_vlm.py --model all              # download both
    python scripts/download_vlm.py --model vlm --dir ./custom/
"""

import argparse
import os
import sys

MODELS = {
    "vlm": {
        "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "target": "libs/VideoLLaMA3-7B",
        "size": "~15 GB",
        "desc": "VideoLLaMA3-7B (VLM vision model)",
    },
    "embedder": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "target": "libs/embedder",
        "size": "~80 MB",
        "desc": "all-MiniLM-L6-v2 (drift detection Plan A)",
    },
}


def download_model(model_id: str, target_dir: str, size_hint: str) -> None:
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[SKIP] {target_dir} already exists and is non-empty.")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"[INFO] Downloading {model_id} ({size_hint})...")
    print(f"       Target: {target_dir}")

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    snapshot_download(model_id, cache_dir=target_dir)
    print(f"       Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Download models from ModelScope")
    parser.add_argument(
        "--model", choices=["vlm", "embedder", "all"], default="vlm",
        help="Which model to download (default: vlm)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Override target directory (only valid with --model vlm)",
    )
    args = parser.parse_args()

    if args.model == "all":
        to_download = ["vlm", "embedder"]
    else:
        to_download = [args.model]

    for key in to_download:
        info = MODELS[key]
        if args.dir and key != "vlm":
            print(f"[WARN] --dir ignored for --model {key}")
        if args.dir and key == "vlm":
            target = os.path.abspath(args.dir)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            target = os.path.join(project_root, info["target"])

        print(f"\n--- {info['desc']} ---")
        download_model(info["model_id"], target, info["size"])

    print("[DONE] All downloads complete.")


if __name__ == "__main__":
    main()
