"""Resolve VLM model path — local ModelScope copy preferred, HF Hub as fallback."""

import os

_HF_MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"


def get_vlm_path() -> str:
    """Return the path to use for loading the VLM.

    Searches common local paths (including ModelScope's nested
    ``org/model_name/`` structure) before falling back to HF Hub.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(utils_dir))

    candidates = [
        os.path.join(project_root, "libs", "VideoLLaMA3-7B"),
        # ModelScope snapshot_download nests under cache_dir/{org}/{model}/
        os.path.join(project_root, "libs", "VideoLLaMA3-7B",
                     "DAMO-NLP-SG", "VideoLLaMA3-7B"),
    ]

    for local_path in candidates:
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
