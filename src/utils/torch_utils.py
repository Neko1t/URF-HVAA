import os

import torch


def initialize_vlm_model_and_device() -> torch.nn.Module:
    from libs.ImageBind.imagebind.models.imagebind_model import imagebind_huge

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_huge(pretrained=True).eval().to(device)
    return model, device


def ensure_single_gpu_distributed():
    """Set env vars for single-GPU torch.distributed (required by Llama.build).

    The Meta Llama model unconditionally calls ``init_process_group("nccl")``.
    On a single-GPU machine the required env vars are normally absent and the
    call fails.  Call this once before any ``Llama.build()``.
    """
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")
