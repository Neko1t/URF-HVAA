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
from typing import Dict, List, Optional, Tuple

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
        max_gen_len: int = 2048,
        candidate_percentile: float = 70.0,
        min_interval_frames: int = 3,
    ):
        """
        Args:
            cap_max_flags: Maximum number of flagged frames to return.
                Controls Phase 4 VLM cost.
            batch_size: Number of frame descriptions per LLM call.
            min_score_percentile: Only keep flagged frames whose anomaly
                score exceeds this percentile of **all** scores in the video
                (0–100).  Default 50 = median.  Set to 0 to disable.
            max_gen_len: Max output tokens for conflict detection LLM calls.
            candidate_percentile: v2 — percentile threshold for identifying
                candidate anomaly intervals from score continuity.
            min_interval_frames: v2 — minimum number of consecutive frames
                above threshold to form a candidate interval.
        """
        self.cap_max_flags = cap_max_flags
        self.batch_size = batch_size
        self.min_score_percentile = min_score_percentile
        self.max_gen_len = max_gen_len
        self.candidate_percentile = candidate_percentile
        self.min_interval_frames = min_interval_frames

    # -- v2: interval-based candidate location --------------------------------

    def locate_candidate_intervals(
        self,
        scores: Dict[str, float],
    ) -> List[Tuple[int, int]]:
        """Find contiguous intervals where scores exceed the candidate percentile.

        These intervals are the "regions of interest" that Stage D will
        re-examine — replacing full-traversal LLM conflict detection.

        Args:
            scores: {frame_idx_str: anomaly_score}.

        Returns:
            List of (start_frame, end_frame) intervals, inclusive.
        """
        if not scores:
            return []

        values = np.array([float(v) for v in scores.values()], dtype=np.float32)
        keys = sorted(int(k) for k in scores.keys())
        if len(values) == 0:
            return []

        threshold = float(np.percentile(values, self.candidate_percentile))

        # Build a boolean mask of "above threshold" frames
        score_map = {int(k): float(v) for k, v in scores.items()}
        above = []
        for k in keys:
            above.append(score_map.get(k, 0.0) > threshold)

        # Group consecutive above-threshold frames into intervals
        intervals: List[Tuple[int, int]] = []
        i = 0
        while i < len(above):
            if above[i]:
                start = i
                while i < len(above) and above[i]:
                    i += 1
                end = i - 1
                if (end - start + 1) >= self.min_interval_frames:
                    intervals.append((keys[start], keys[end]))
            i += 1

        return intervals

    def compute_adaptive_window(
        self,
        frame_idx: int,
        scores: Dict[str, float],
        fps: float = 1.0,
        min_window_sec: float = 2.0,
        max_window_sec: float = 30.0,
    ) -> Tuple[int, int]:
        """Compute an adaptive temporal window around *frame_idx*.

        Expands backward/forward until the score drops below the candidate
        percentile — growing the window to capture the full anomaly event
        without any fixed hyperparameter.

        Args:
            frame_idx: Center frame index.
            scores: {frame_str: score} dict.
            fps: Video FPS (used only to clamp min/max window size).
            min_window_sec: Minimum window half-width in seconds.
            max_window_sec: Maximum window half-width in seconds.

        Returns:
            (start_frame, end_frame) inclusive.
        """
        if not scores:
            return (frame_idx, frame_idx)

        values = [float(v) for v in scores.values()]
        threshold = float(np.percentile(values, self.candidate_percentile))

        score_map = {int(k): float(v) for k, v in scores.items()}
        all_frames = sorted(score_map.keys())
        if not all_frames:
            return (frame_idx, frame_idx)

        min_half = max(1, int(min_window_sec * fps))
        max_half = int(max_window_sec * fps)

        # Expand backward
        left = frame_idx
        for _ in range(max_half):
            next_left = left - 1
            if next_left < all_frames[0]:
                break
            # Find closest keyed frame at or before next_left
            prev_frame = max((f for f in all_frames if f <= next_left), default=None)
            if prev_frame is not None and score_map.get(prev_frame, 0.0) > threshold:
                left = prev_frame
            else:
                break

        # Expand forward
        right = frame_idx
        for _ in range(max_half):
            next_right = right + 1
            if next_right > all_frames[-1]:
                break
            next_frame = min((f for f in all_frames if f >= next_right), default=None)
            if next_frame is not None and score_map.get(next_frame, 0.0) > threshold:
                right = next_frame
            else:
                break

        # Ensure minimum window
        left = min(left, frame_idx - min_half)
        right = max(right, frame_idx + min_half)
        left = max(all_frames[0], left)
        right = min(all_frames[-1], right)

        return (left, right)

    # -- public API ---------------------------------------------------------

    def detect(
        self,
        captions: Dict[str, str],
        scores: Dict[str, float],
        fps: float,
        contexts: List[SceneContext],
        llm_generator,
        mode: str = "intervals",
    ) -> List[FlaggedFrame]:
        """Run conflict detection.

        Args:
            captions: {frame_idx_str: caption_text}
            scores: {frame_idx_str: anomaly_score}
            fps: Video frames per second.
            contexts: SceneContext list from Phase 2.
            llm_generator: Object with ``chat_completion(dialogs, ...)``.
            mode: 'intervals' (v2, default) uses score-continuity-based
                candidate intervals + adaptive windows, sending only candidate
                frames to the LLM. 'full' (v1 legacy) checks every frame.

        Returns:
            FlaggedFrame list sorted by anomaly score descending, capped.
        """
        if mode == "intervals":
            return self._detect_intervals(captions, scores, fps, contexts, llm_generator)
        return self._detect_full(captions, scores, fps, contexts, llm_generator)

    def _detect_intervals(
        self,
        captions: Dict[str, str],
        scores: Dict[str, float],
        fps: float,
        contexts: List[SceneContext],
        llm_generator,
    ) -> List[FlaggedFrame]:
        """v2: Interval-based detection — only check candidate regions."""
        if not captions or not contexts:
            return []

        intervals = self.locate_candidate_intervals(scores)
        if not intervals:
            logger.info("No candidate intervals found; video likely normal.")
            return []

        # Collect frames within candidate intervals
        candidate_frames: set[int] = set()
        score_map = {int(k): float(v) for k, v in scores.items()}
        all_keys = sorted(score_map.keys())

        for start_f, end_f in intervals:
            for f in all_keys:
                if start_f <= f <= end_f:
                    candidate_frames.add(f)

        if not candidate_frames:
            return []

        # Map candidate frames to contexts and run LLM conflict detection
        frame_ctx_map: Dict[int, SceneContext] = {}
        for fidx in candidate_frames:
            ctx = _find_context_for_frame(fidx, fps, contexts)
            if ctx is not None:
                frame_ctx_map[fidx] = ctx

        if not frame_ctx_map:
            return []

        # Group by context and run detection
        ctx_to_frames: Dict[int, List[int]] = {}
        for fidx, ctx in frame_ctx_map.items():
            key = id(ctx)
            ctx_to_frames.setdefault(key, []).append(fidx)

        all_flags: List[dict] = []
        for ctx_frames in ctx_to_frames.values():
            ctx = frame_ctx_map[ctx_frames[0]]
            ctx_frames.sort()
            for batch_start in range(0, len(ctx_frames), self.batch_size):
                batch_frames = ctx_frames[batch_start:batch_start + self.batch_size]
                flags = self._detect_batch(llm_generator, ctx, batch_frames, captions)
                all_flags.extend(flags)

        return self._postprocess_flags(all_flags, scores)

    def _detect_full(
        self,
        captions: Dict[str, str],
        scores: Dict[str, float],
        fps: float,
        contexts: List[SceneContext],
        llm_generator,
    ) -> List[FlaggedFrame]:
        """v1 legacy: Full-traversal conflict detection."""
        if not captions or not contexts:
            return []

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

        all_flags: List[dict] = []
        ctx_to_frames: Dict[int, List[int]] = {}
        for fidx, ctx in frame_ctx_map.items():
            key = id(ctx)
            ctx_to_frames.setdefault(key, []).append(fidx)

        for ctx_frames in ctx_to_frames.values():
            ctx = frame_ctx_map[ctx_frames[0]]
            ctx_frames.sort()
            for batch_start in range(0, len(ctx_frames), self.batch_size):
                batch_frames = ctx_frames[batch_start:batch_start + self.batch_size]
                flags = self._detect_batch(llm_generator, ctx, batch_frames, captions)
                all_flags.extend(flags)

        return self._postprocess_flags(all_flags, scores)

    def _postprocess_flags(
        self, all_flags: List[dict], scores: Dict[str, float],
    ) -> List[FlaggedFrame]:
        """Deduplicate, score-filter, cap, and convert to FlaggedFrame list."""
        if not all_flags:
            return []

        seen: set[int] = set()
        unique_flags: List[dict] = []
        for f in all_flags:
            fidx = f.get("frame")
            if fidx is not None and fidx not in seen:
                seen.add(fidx)
                unique_flags.append(f)

        unique_flags.sort(
            key=lambda f: float(scores.get(str(f["frame"]), 0.0)),
            reverse=True,
        )

        if self.min_score_percentile > 0:
            all_vals = [float(v) for v in scores.values() if v is not None]
            if all_vals:
                score_threshold = float(np.percentile(all_vals, self.min_score_percentile))
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

        unique_flags = unique_flags[:self.cap_max_flags]

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
                max_gen_len=self.max_gen_len,
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
