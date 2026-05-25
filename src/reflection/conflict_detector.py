"""
Phase 3: Full-traversal conflict detection.

Identifies frames where the local caption LOGICALLY CONTRADICTS the
global scene context — the core of the asymmetric dual-pass design.

Key differences from refine_with_tag.py:

    | refine_with_tag.py         | conflict_detector.py            |
    |----------------------------|----------------------------------|
    | Adjust scores with tag     | Detect local-global             |
    | hints                      | contradictions                  |
    | Conditional (score gate    | Full traversal, all frames      |
    | skips many)                | examined                        |
    | "Consider these suspicious | "Does action CONTRADICT         |
    | behaviors when scoring"    | scene context?"                 |
    | Modified score per frame   | Flagged List + conflict reasons |

Each flagged frame carries the information needed by Phase 4 targeted
verification: the original caption, a specific conflict reason, the
suspicious element, and a plausible alternative explanation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

import numpy as np

from src.perception.vlm_engine import FlaggedFrame, SceneContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

CONFLICT_DETECTION_PROMPT = """\
You are a logical consistency checker for surveillance footage.

GLOBAL SCENE CONTEXT: {scene_context}

Below are frame-by-frame descriptions from this surveillance video.
For EACH description, determine whether the described action LOGICALLY
CONTRADICTS the expected normality of this scene.

A contradiction exists when: the action would be impossible or extremely
unexpected given the known scene conditions (time, environment, typical
activities). Do NOT flag actions that are merely uncommon — only flag
those representing a clear violation of what is physically or situationally
possible in this context.

For EACH flagged frame, you MUST output a JSON object with these fields:
  "frame": integer frame index,
  "caption_summary": what the original caption claimed (brief),
  "conflict_reason": why this contradicts the scene context,
  "suspicious_element": the specific thing that seems anomalous,
  "alternative_explanation": what else it could realistically be

FRAME DESCRIPTIONS:
{frame_entries}

Respond as a JSON array. If no frame contradicts the context, output [].
Do NOT include any text outside the JSON array."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_context_for_frame(
    frame_idx: int,
    fps: float,
    contexts: List[SceneContext],
) -> Optional[SceneContext]:
    """Return the SceneContext whose window covers *frame_idx*, or None."""
    timestamp = frame_idx / fps
    for ctx in contexts:
        if ctx.window_start_sec <= timestamp < ctx.window_end_sec:
            return ctx
    return None


def _extract_json_array(text: str) -> Optional[List[dict]]:
    """Robust JSON extraction from LLM output.

    Handles cases where the LLM wraps the JSON in markdown fences or
    adds trailing commentary.
    """
    if not text or not text.strip():
        return None

    # Try direct parse first
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding the outermost [...] in the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        try:
            result = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Full-traversal logical-consistency checker.

    Usage::

        detector = ConflictDetector(cap_max_flags=20, batch_size=15)
        flagged = detector.detect(
            captions={"0": "...", "16": "..."},
            scores={"0": 0.1, "16": 0.6},
            fps=30.0,
            contexts=[SceneContext(...), ...],
            llm_generator=llm_gen,
        )
        # flagged is List[FlaggedFrame], sorted by anomaly score desc
    """

    def __init__(
        self,
        cap_max_flags: int = 20,
        batch_size: int = 15,
        min_score_percentile: float = 50.0,
    ):
        """
        Args:
            cap_max_flags: Maximum number of flagged frames to return.
                Controls Phase 4 VLM cost.
            batch_size: Number of frame descriptions per LLM call.
            min_score_percentile: Only keep flagged frames whose anomaly
                score exceeds this percentile of **all** scores in the video
                (0–100).  Default 50 = median.  Set to 0 to disable.
        """
        self.cap_max_flags = cap_max_flags
        self.batch_size = batch_size
        self.min_score_percentile = min_score_percentile

    # -- public API ---------------------------------------------------------

    def detect(
        self,
        captions: Dict[str, str],
        scores: Dict[str, float],
        fps: float,
        contexts: List[SceneContext],
        llm_generator,
    ) -> List[FlaggedFrame]:
        """Run conflict detection across all frames.

        Args:
            captions: {frame_idx_str: caption_text}
            scores: {frame_idx_str: anomaly_score}
            fps: Video frames per second.
            contexts: SceneContext list from Phase 2.
            llm_generator: Object with ``chat_completion(dialogs, ...)``.

        Returns:
            FlaggedFrame list sorted by anomaly score descending, capped.
        """
        if not captions or not contexts:
            return []

        # Map frames to their covering SceneContext
        frame_ctx_map: Dict[int, SceneContext] = {}
        for frame_str in captions:
            try:
                fidx = int(frame_str)
            except ValueError:
                continue
            ctx = _find_context_for_frame(fidx, fps, contexts)
            if ctx is not None:
                frame_ctx_map[fidx] = ctx

        if not frame_ctx_map:
            logger.warning("No frames mapped to any SceneContext; cannot detect.")
            return []

        # Group frame indices by context window, then batch within each group
        all_flags: List[dict] = []

        # Group by context
        ctx_to_frames: Dict[int, List[int]] = {}  # id(ctx) → [frame_idx, ...]
        for fidx, ctx in frame_ctx_map.items():
            key = id(ctx)
            ctx_to_frames.setdefault(key, []).append(fidx)

        total_batches = sum(
            (len(frames) + self.batch_size - 1) // self.batch_size
            for frames in ctx_to_frames.values()
        )

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        pbar = _tqdm(total=total_batches, desc="    Phase 3: conflicts",
                      unit="batch", leave=False, ncols=90) if _tqdm else None
        batch_num = 0

        for ctx_frames in ctx_to_frames.values():
            ctx = frame_ctx_map[ctx_frames[0]]
            ctx_frames.sort()

            for batch_start in range(0, len(ctx_frames), self.batch_size):
                batch_num += 1
                batch_frames = ctx_frames[batch_start:batch_start + self.batch_size]
                flags = self._detect_batch(
                    llm_generator, ctx, batch_frames, captions,
                )
                all_flags.extend(flags)
                if pbar:
                    pbar.set_postfix_str(f"{len(all_flags)} conflicts")
                    pbar.update(1)

        if pbar:
            pbar.close()

        if not all_flags:
            return []

        # Deduplicate by frame index (keep first occurrence)
        seen: set[int] = set()
        unique_flags: List[dict] = []
        for f in all_flags:
            fidx = f.get("frame")
            if fidx is not None and fidx not in seen:
                seen.add(fidx)
                unique_flags.append(f)

        # Sort by anomaly score descending (proxy for conflict strength)
        unique_flags.sort(
            key=lambda f: float(scores.get(str(f["frame"]), 0.0)),
            reverse=True,
        )

        # Filter by minimum score percentile (dynamic per-video threshold)
        if self.min_score_percentile > 0:
            all_vals = [float(v) for v in scores.values() if v is not None]
            if all_vals:
                score_threshold = float(
                    np.percentile(all_vals, self.min_score_percentile)
                )
                n_before = len(unique_flags)
                unique_flags = [
                    f for f in unique_flags
                    if float(scores.get(str(f["frame"]), 0.0)) > score_threshold
                ]
                logger.info(
                    "Score gate (p%d=%.3f): %d → %d flagged",
                    int(self.min_score_percentile), score_threshold,
                    n_before, len(unique_flags),
                )

        # Safety cap (generous; score gate already controls quality)
        unique_flags = unique_flags[:self.cap_max_flags]

        # Convert to FlaggedFrame dataclasses
        return [
            FlaggedFrame(
                frame=f["frame"],
                caption_summary=f.get("caption_summary", ""),
                conflict_reason=f.get("conflict_reason", ""),
                suspicious_element=f.get("suspicious_element", ""),
                alternative_explanation=f.get("alternative_explanation", ""),
            )
            for f in unique_flags
        ]

    # -- internal -----------------------------------------------------------

    def _detect_batch(
        self,
        llm_generator,
        ctx: SceneContext,
        frame_indices: List[int],
        captions: Dict[str, str],
    ) -> List[dict]:
        """Send one batch of frames to the LLM for conflict detection."""
        entries = []
        for fidx in frame_indices:
            caption = captions.get(str(fidx), "")
            if caption:
                entries.append(f"Frame {fidx}: {caption}")

        if not entries:
            return []

        prompt = CONFLICT_DETECTION_PROMPT.format(
            scene_context=ctx.description,
            frame_entries="\n".join(entries),
        )

        dialogs = [[
            {"role": "system", "content": "You output only valid JSON arrays."},
            {"role": "user", "content": prompt},
        ]]

        try:
            results = llm_generator.chat_completion(
                dialogs,
                max_gen_len=1024,
                temperature=0.1,
                top_p=0.9,
            )
            response = results[0]["generation"]["content"]
        except Exception:
            logger.exception("LLM conflict detection failed for batch")
            return []

        parsed = _extract_json_array(response)
        if parsed is None:
            logger.warning(
                "Failed to parse LLM conflict output: %s",
                response[:200],
            )
            return []

        return parsed
