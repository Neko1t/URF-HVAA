"""
Unified VLM interface for VideoLLaMA3-7B.

Replaces the duplicated load_model()/infer() across:
  - video_pre_caption.py
  - summarize_window.py
  - vau_priors.py

Design:
  - Explicit lifecycle: caller controls load()/unload(), never singleton
  - Dual sampling density: coarse (interval=16, max_frames=8) for Phase 1 patrol,
    fine (interval=4, max_frames=16) for Phase 4 focused verification
  - Anti-hallucination prompts in guided mode
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src.utils.vlm_path import get_vlm_path


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class FlaggedFrame:
    """A frame flagged by Phase 3 conflict detection, carrying all context
    needed by Phase 4 targeted verification."""
    frame: int
    caption_summary: str
    conflict_reason: str
    suspicious_element: str
    alternative_explanation: str


@dataclass
class SceneContext:
    """Global scene context produced by Phase 2 context memory."""
    window_start_sec: float
    window_end_sec: float
    description: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BLIND_SYSTEM_PROMPT = (
    "You are an AI assistant analyzing this video segment. "
    "Summarize the main events or actions in a concise way."
)

GUIDED_SYSTEM_PROMPT = (
    "You are an objective visual observer. Your task is to describe "
    "what you actually see — not what you expect to see. Do not speculate.\n\n"
    "SCENE CONTEXT: {scene_context}\n\n"
    "IMPORTANT NOTE: A previous coarse scan reported \"{original_caption}\". "
    "However, this may be INACCURATE because {conflict_reason}.\n\n"
    "Please examine the frames carefully and answer:\n"
    "1. What is actually visible in this segment? Describe only observable facts.\n"
    "2. Is there genuinely any {suspicious_element}, or could the previous "
    "observation be explained by {alternative_explanation}?\n\n"
    "Respond with a factual description without assuming any anomaly exists."
)

GUIDED_USER_PROMPT = (
    "Please describe this video segment objectively."
)


# ---------------------------------------------------------------------------
# VLM Engine
# ---------------------------------------------------------------------------

class VLMEngine:
    """Unified VideoLLaMA3-7B interface with explicit lifecycle.

    Usage::

        engine = VLMEngine()
        engine.load()
        captions = {}
        for frame_idx in range(0, total_frames, 16):
            captions[frame_idx] = engine.blind_caption(video_path, frame_idx)
        engine.unload()
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = get_vlm_path()
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._device = None
        self._float_dtype = None

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        """Load VideoLLaMA3-7B onto GPU."""
        if self.model is not None:
            return  # already loaded

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._float_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map=self._device,
            torch_dtype=self._float_dtype,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def unload(self) -> None:
        """Release GPU memory held by the VLM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # -- public API ---------------------------------------------------------

    def blind_caption(
        self,
        video_path: str,
        frame_idx: int,
        mode: str = "coarse",
        interval: Optional[int] = None,
    ) -> str:
        """Phase 1: caption a video segment WITHOUT any prior context.

        Args:
            video_path: Path to the .mp4 file.
            frame_idx: Center frame index for the segment.
            mode: 'coarse' (interval=16, max_frames=8) or 'fine' (interval=4,
                  max_frames=16). Overridden by *interval* if given.
            interval: Explicit time window half-width in seconds (defaults to
                      mode-dependent value).

        Returns:
            Caption string.
        """
        if not self.is_loaded:
            raise RuntimeError("VLMEngine.load() must be called before inference.")

        if not os.path.exists(video_path):
            return ""

        total_duration, fps, total_frames = self._get_video_info(video_path)
        if total_duration <= 0 or fps <= 0:
            return ""

        # Resolve sampling parameters
        if interval is None:
            interval = 5 if mode == "coarse" else 2.5
        max_frames = 8 if mode == "coarse" else 16

        # Clamp segment window to valid range
        center_sec = frame_idx / fps
        start_sec = max(0.0, center_sec - interval)
        end_sec = min(total_duration, center_sec + interval)

        if end_sec - start_sec <= 0:
            return ""

        conversation = self._build_blind_conversation(
            video_path, start_sec, end_sec, max_frames
        )
        response = self._infer(conversation)
        return response if response.strip() else "No detected activity."

    def guided_caption(
        self,
        video_path: str,
        frame_idx: int,
        scene_context: SceneContext,
        flagged_frame: FlaggedFrame,
        mode: str = "fine",
        interval: Optional[int] = None,
    ) -> str:
        """Phase 4: caption a video segment WITH neutral verification context.

        The prompt tells the VLM the scene context AND that a previous
        observation may have been inaccurate — but does NOT suggest that
        an anomaly exists.  This is the anti-hallucination design.

        Args:
            video_path: Path to the .mp4 file.
            frame_idx: Center frame index for the segment.
            scene_context: Global scene context from Phase 2.
            flagged_frame: Flagged frame data from Phase 3.
            mode: 'coarse' or 'fine' (default 'fine' for focused re-exam).
            interval: Explicit time window half-width in seconds.

        Returns:
            Refined caption string.
        """
        if not self.is_loaded:
            raise RuntimeError("VLMEngine.load() must be called before inference.")

        if not os.path.exists(video_path):
            return ""

        total_duration, fps, total_frames = self._get_video_info(video_path)
        if total_duration <= 0 or fps <= 0:
            return ""

        # Resolve sampling parameters — Phase 4 uses fine-grained by default
        if interval is None:
            interval = 2.5 if mode == "fine" else 5
        max_frames = 16 if mode == "fine" else 8

        center_sec = frame_idx / fps
        start_sec = max(0.0, center_sec - interval)
        end_sec = min(total_duration, center_sec + interval)

        if end_sec - start_sec <= 0:
            return ""

        conversation = self._build_guided_conversation(
            video_path, start_sec, end_sec, max_frames,
            scene_context, flagged_frame,
        )
        response = self._infer(conversation)
        return response if response.strip() else "No detected activity."

    # -- internal helpers ---------------------------------------------------

    @torch.inference_mode()
    def _infer(self, conversation: list[dict], temperature: float = 0.1) -> str:
        """Run a single inference pass."""
        if not conversation:
            return ""

        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    inputs[k] = v.to(self._device, dtype=self._float_dtype)
                else:
                    inputs[k] = v.to(self._device)

        output_ids = self.model.generate(
            **inputs, max_new_tokens=256, temperature=temperature
        )
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

    def _build_blind_conversation(
        self, video_path: str, start_sec: float, end_sec: float, max_frames: int,
    ) -> list[dict]:
        """Build a conversation for blind (no-context) captioning."""
        return [
            {"role": "system", "content": BLIND_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 2,
                            "start_time": start_sec,
                            "end_time": end_sec,
                            "max_frames": max_frames,
                        },
                    }
                ],
            },
        ]

    def _build_guided_conversation(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        max_frames: int,
        scene_context: SceneContext,
        flagged: FlaggedFrame,
    ) -> list[dict]:
        """Build a conversation for guided (anti-hallucination) verification."""
        system_content = GUIDED_SYSTEM_PROMPT.format(
            scene_context=scene_context.description,
            original_caption=flagged.caption_summary,
            conflict_reason=flagged.conflict_reason,
            suspicious_element=flagged.suspicious_element,
            alternative_explanation=flagged.alternative_explanation,
        )
        return [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 2,
                            "start_time": start_sec,
                            "end_time": end_sec,
                            "max_frames": max_frames,
                        },
                    },
                    {"type": "text", "text": GUIDED_USER_PROMPT},
                ],
            },
        ]

    @staticmethod
    def _get_video_info(video_path: str) -> tuple[float, float, int]:
        """Return (duration_seconds, fps, frame_count)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0, 0.0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0.0
        return duration, fps, frame_count
