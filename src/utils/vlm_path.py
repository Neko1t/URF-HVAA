"""Resolve VLM model path — local ModelScope copy preferred, HF Hub as fallback."""

import os

# Project-relative preferred local path
_LOCAL_VLM_DIR = os.path.join("libs", "VideoLLaMA3-7B")
_HF_MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"


def get_vlm_path() -> str:
    """Return the path to use for loading the VLM.

    If ``libs/VideoLLaMA3-7B/`` exists and contains model files, use it.
    Otherwise fall back to the HuggingFace Hub model id (auto-download).
    """
    # Resolve relative to the project root (two levels up from this file)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(utils_dir)
    project_root = os.path.dirname(src_dir)
    local_path = os.path.join(project_root, _LOCAL_VLM_DIR)

    if os.path.isdir(local_path) and _has_model_files(local_path):
        return local_path
    return _HF_MODEL_ID


def _has_model_files(path: str) -> bool:
    """Return True if *path* looks like it contains model weights."""
    # Check for typical HuggingFace/model files
    indicators = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    # Also check for sharded checkpoints
    for entry in os.listdir(path):
        if entry.startswith("model-") and entry.endswith(".safetensors"):
            return True
        if entry.startswith("pytorch_model-") and entry.endswith(".bin"):
            return True
    return any(
        os.path.isfile(os.path.join(path, f)) for f in indicators
    )
