"""
Phase 4: Targeted visual verification + score closed loop.

The "high-cost on-demand" half of the asymmetric dual-pass design.

1. For each flagged frame, calls VLM.guided_caption() with neutral
   anti-hallucination prompts (fine-grained: interval≈2.5s, max_frames=16).
2. Feeds refined captions back to the LLM scorer for final anomaly scores.
3. Merges refined scores into the original score array and applies a
   global 1D Gaussian filter (sigma=2) to smooth transitions and remove
   LLM scoring jitter.

Includes ComputeTracker for paper metrics — compares against a naive
full fine-grained scan (interval=4).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.perception.vlm_engine import FlaggedFrame, SceneContext, VLMEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComputeTracker
# ---------------------------------------------------------------------------

class ComputeTracker:
    """Tracks VLM compute savings of the asymmetric architecture.

    Baseline: naive full fine-grained scan at interval=4.
    Our cost: coarse patrol (interval=16) + targeted focus (interval=4).

    savings = (1 - our_cost / naive_full_cost) * 100
    """

    def __init__(self):
        self.total_frames: int = 0
        self.phase1_vlm_calls: int = 0   # coarse patrol (every 16 frames)
        self.phase4_vlm_calls: int = 0   # fine focus (flagged only)
        self.flagged_count: int = 0

    @property
    def naive_full_cost(self) -> float:
        """Traditional approach: full fine-grained scan at interval=4."""
        return self.total_frames / 4

    @property
    def our_cost(self) -> float:
        """Our approach: coarse patrol + on-demand fine focus."""
        return self.phase1_vlm_calls + self.phase4_vlm_calls

    @property
    def savings_percent(self) -> float:
        if self.naive_full_cost <= 0:
            return 0.0
        return round((1 - self.our_cost / self.naive_full_cost) * 100, 1)

    @property
    def flagged_percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return round(self.flagged_count / self.total_frames * 100, 1)

    def report(self) -> dict:
        """Return a dict suitable for logging or JSON export."""
        return {
            "total_frames": self.total_frames,
            "phase1_coarse_calls": self.phase1_vlm_calls,
            "phase4_fine_calls": self.phase4_vlm_calls,
            "flagged_count": self.flagged_count,
            "flagged_pct": self.flagged_percent,
            "naive_full_fine_cost": round(self.naive_full_cost, 1),
            "our_total_cost": round(self.our_cost, 1),
            "vlm_compute_saved_pct": self.savings_percent,
        }

    def print_report(self) -> None:
        r = self.report()
        logger.info(
            "ComputeTracker: %d/%d frames flagged (%.1f%%), "
            "VLM calls: %d coarse + %d fine = %d total, "
            "baseline (full fine): %.0f, savings: %.1f%%",
            r["flagged_count"], r["total_frames"], r["flagged_pct"],
            r["phase1_coarse_calls"], r["phase4_fine_calls"],
            r["our_total_cost"], r["naive_full_fine_cost"],
            r["vlm_compute_saved_pct"],
        )

# ---------------------------------------------------------------------------
# TargetedVerifier
# ---------------------------------------------------------------------------

# Type alias for the scoring function signature used in rescore()
ScoreFn = Callable[[Dict[str, str]], Dict[str, float]]


class TargetedVerifier:
    """Targeted visual verification with score closed loop.

    Usage::

        verifier = TargetedVerifier()
        refined = verifier.verify_frames(
            vlm_engine, video_path, flagged_frames, contexts, fps,
        )
        new_scores = verifier.rescore(score_fn, refined)
        final = verifier.merge_scores(original_scores, new_scores, num_frames)
    """

    def __init__(self, merge_sigma: float = 2.0):
        """
        Args:
            merge_sigma: Sigma for the global 1D Gaussian filter applied
                during score merging.
        """
        self.merge_sigma = merge_sigma
        self.tracker = ComputeTracker()

    # -- Phase 4a: VLM verification -----------------------------------------

    def verify_frames(
        self,
        vlm_engine: VLMEngine,
        video_path: str,
        flagged_frames: List[FlaggedFrame],
        contexts: List[SceneContext],
        fps: float,
        mode: str = "fine",
        progress: bool = False,
    ) -> Dict[int, str]:
        """Run targeted VLM verification on each flagged frame.

        Args:
            vlm_engine: A loaded VLMEngine instance.
            video_path: Path to the .mp4 file.
            flagged_frames: FlaggedFrame list from Phase 3.
            contexts: SceneContext list from Phase 2.
            fps: Video frames per second.
            mode: Sampling mode ('fine' = dense, high-quality).
            progress: If True, show per-frame tqdm progress bar.

        Returns:
            {frame_idx: refined_caption_text}
        """
        refined: Dict[int, str] = {}
        video_name = os.path.basename(video_path).replace(".mp4", "")

        ff_iter = tqdm(flagged_frames, desc=f"  VLM verify {video_name}",
                       unit="frame", leave=False, ncols=100) if progress \
                  else flagged_frames

        for ff in ff_iter:
            ctx = self._find_context(ff.frame, fps, contexts)
            if ctx is None:
                logger.debug("Frame %d: no SceneContext covers it, skipping.", ff.frame)
                continue

            if progress:
                ff_iter.set_postfix_str(f"f{ff.frame}")

            try:
                caption = vlm_engine.guided_caption(
                    video_path=video_path,
                    frame_idx=ff.frame,
                    scene_context=ctx,
                    flagged_frame=ff,
                    mode=mode,
                )
                refined[ff.frame] = caption
                self.tracker.phase4_vlm_calls += 1
            except Exception:
                logger.warning(
                    "VLM verify skipped frame %d (model internal error, frame OK)", ff.frame,
                )

        self.tracker.flagged_count = len(flagged_frames)
        return refined

    # -- Phase 4b: LLM rescore ----------------------------------------------

    def rescore(
        self,
        score_fn: ScoreFn,
        refined_captions: Dict[int, str],
    ) -> Dict[int, float]:
        """Re-score refined captions using the LLM.

        Args:
            score_fn: A callable that takes ``{frame_idx_str: caption}`` and
                returns ``{frame_idx_str: score_float}``. Typically wraps
                ``LLMAnomalyScorer._score_temporal_summaries``.
            refined_captions: {frame_idx: caption} from verify_frames.

        Returns:
            {frame_idx: score_float}
        """
        if not refined_captions:
            return {}

        # Convert int keys to strings for the scoring function
        str_keyed = {str(k): v for k, v in refined_captions.items()}

        try:
            raw = score_fn(str_keyed)
        except Exception:
            logger.exception("LLM rescore failed")
            return {}

        # Convert string keys back to int
        return {int(k): float(v) for k, v in raw.items()}

    # -- Phase 4c: merge + smooth -------------------------------------------

    def merge_scores(
        self,
        original_scores: Dict[int, float],
        refined_scores: Dict[int, float],
        num_frames: int,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Merge refined scores into the original array and globally smooth.

        Steps:
        1. Build a full-length array from *original_scores*.
        2. Replace entries at flagged frame positions with *refined_scores*.
        3. Apply global 1D Gaussian filter (sigma=``merge_sigma``) to the
           entire array.

        Args:
            original_scores: {frame_idx: score} from Phase 1.
            refined_scores: {frame_idx: score} from Phase 4 rescore.
            num_frames: Total number of frames in the video.
            sigma: Gaussian sigma override (defaults to self.merge_sigma).

        Returns:
            1D numpy array of final scores, length *num_frames*.
        """
        sigma = sigma if sigma is not None else self.merge_sigma

        # Build full array from original scores, filling gaps via linear interp
        arr = self._dict_to_array(original_scores, num_frames)

        # Replace flagged positions
        for fidx, score in refined_scores.items():
            if 0 <= fidx < num_frames:
                arr[fidx] = score

        # Global Gaussian smooth
        smoothed = gaussian_filter1d(arr, sigma=sigma)

        return smoothed

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _find_context(
        frame_idx: int,
        fps: float,
        contexts: List[SceneContext],
    ) -> Optional[SceneContext]:
        """Return the SceneContext whose window covers *frame_idx*."""
        timestamp = frame_idx / fps
        for ctx in contexts:
            if ctx.window_start_sec <= timestamp < ctx.window_end_sec:
                return ctx
        # Fallback: return nearest context if no exact match
        if contexts:
            best = min(
                contexts,
                key=lambda c: abs(
                    (c.window_start_sec + c.window_end_sec) / 2 - timestamp
                ),
            )
            return best
        return None

    @staticmethod
    def _dict_to_array(scores: Dict[int, float], num_frames: int) -> np.ndarray:
        """Convert a sparse {frame_idx: score} dict into a dense array.

        Missing values are linearly interpolated. Leading/trailing missing
        values are filled with the nearest known value.
        """
        arr = np.full(num_frames, np.nan, dtype=np.float64)

        for fidx, score in scores.items():
            if 0 <= fidx < num_frames:
                arr[fidx] = float(score)

        # Find known indices
        known = np.where(~np.isnan(arr))[0]
        if len(known) == 0:
            return np.zeros(num_frames)
        if len(known) == 1:
            return np.full(num_frames, arr[known[0]])

        # Interpolate
        arr = np.interp(
            np.arange(num_frames),
            known,
            arr[known],
        )
        return arr
