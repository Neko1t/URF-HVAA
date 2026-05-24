"""
Phase 2: Sliding-window context memory for video anomaly detection.

Builds a dynamic "global scene normality" model by:
1. Selecting high-confidence normal frames via dynamic percentile threshold
2. Partitioning normal frames into time-based sliding windows
3. Generating scene context summaries per window using LLM
4. Detecting semantic drift between windows (embedding cosine sim or LLM fallback)

Design decisions (v3):
  - Dynamic percentile threshold (np.percentile(scores, 30)) — not hardcoded 0.3
  - Time-based windows (60 s window, 30 s stride) — not absolute frame counts
  - Plan A drift detection: all-MiniLM-L6-v2 embedding (~80 MB) cosine similarity
  - Plan B drift detection (fallback): LLM prompt for environment change detection
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.perception.vlm_engine import SceneContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONTEXT_GENERATION_PROMPT = """\
You are a surveillance scene analyst. Below are descriptions of NORMAL, \
uneventful moments from a surveillance video within a specific time window.

Your task: synthesize these into a concise GLOBAL SCENE CONTEXT that describes:
1. The environment (indoor/outdoor, type of location)
2. The time of day / lighting conditions
3. Expected normal activities and behaviors
4. Objects or people commonly present

Be specific and factual. Do NOT mention any anomalies or suspicious activities.
Write 3-5 sentences.

NORMAL MOMENT DESCRIPTIONS:
{normal_descriptions}

GLOBAL SCENE CONTEXT:"""

DRIFT_CHECK_PROMPT = """\
You are a scene consistency checker.

PREVIOUS SCENE CONTEXT: {previous_context}

RECENT OBSERVATIONS FROM A NEW TIME WINDOW:
{new_observations}

Has the fundamental environment or state of the scene changed?
Consider: time of day, lighting, location, type of activity, people present.

Answer ONLY "YES" or "NO"."""


# ---------------------------------------------------------------------------
# SlidingContextMemory
# ---------------------------------------------------------------------------

class SlidingContextMemory:
    """Builds and maintains sliding-window scene context for a video.

    Usage::

        memory = SlidingContextMemory(window_seconds=60, stride_seconds=30)
        contexts = memory.process_video(
            captions={"0": "...", "16": "...", ...},
            scores={"0": 0.1, "16": 0.4, ...},
            fps=30.0,
            llm_generator=llm_gen,
        )
        # contexts is a list of SceneContext, one per distinct window
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        stride_seconds: float = 30.0,
        normality_percentile: float = 30.0,
        drift_threshold: float = 0.7,
        max_captions_per_context: int = 30,
    ):
        """
        Args:
            window_seconds: Duration of each time window in seconds.
            stride_seconds: Step size between consecutive windows.
            normality_percentile: Bottom percentile of scores treated as
                "normal" (e.g. 30 means bottom 30%).
            drift_threshold: Cosine similarity below which a context refresh
                is triggered (Plan A).
            max_captions_per_context: Max normal-frame captions fed to the
                LLM per window (truncated to avoid token overflow).
        """
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.normality_percentile = normality_percentile
        self.drift_threshold = drift_threshold
        self.max_captions_per_context = max_captions_per_context

        self._embedder = None          # lazy-init for Plan A
        self._embedder_failed = False  # set True if import fails

    # -- public API ---------------------------------------------------------

    def process_video(
        self,
        captions: dict[str, str],
        scores: dict[str, float],
        fps: float,
        llm_generator,
    ) -> List[SceneContext]:
        """Run the full context-memory pipeline on one video.

        Args:
            captions: {frame_idx_str: caption_text}
            scores: {frame_idx_str: anomaly_score}
            fps: Video frames per second.
            llm_generator: An object with a ``chat_completion(dialogs, ...)``
                method (e.g. the Llama instance from llm_anomaly_scorer).

        Returns:
            List of SceneContext ordered by window start time.
        """
        if not captions or not scores or fps <= 0:
            return []

        # Step 1 — dynamic normality threshold
        threshold = self._compute_threshold(scores)

        # Step 2 — partition normal frames into time windows
        windows = self._partition_windows(captions, scores, fps, threshold)
        if not windows:
            logger.warning("No normal-frame windows found; returning empty.")
            return []

        # Step 3 — generate context per window with drift gating
        contexts: List[SceneContext] = []

        for i, (start_sec, end_sec, window_captions) in enumerate(windows):
            # Drift check: skip if scene hasn't meaningfully changed
            if i > 0 and contexts:
                if not self._has_scene_changed(window_captions, contexts[-1],
                                               llm_generator):
                    continue

            description = self._generate_context(llm_generator, window_captions)
            if not description:
                continue

            contexts.append(SceneContext(
                window_start_sec=round(start_sec, 1),
                window_end_sec=round(end_sec, 1),
                description=description,
            ))

        return contexts

    # -- internal: threshold -------------------------------------------------

    def _compute_threshold(self, scores: dict[str, float]) -> float:
        """Return the score value at ``normality_percentile`` percentile."""
        values = [float(v) for v in scores.values() if v is not None]
        if not values:
            return 0.0
        return float(np.percentile(values, self.normality_percentile))

    # -- internal: windowing -------------------------------------------------

    def _partition_windows(
        self,
        captions: dict[str, str],
        scores: dict[str, float],
        fps: float,
        threshold: float,
    ) -> List[tuple[float, float, List[str]]]:
        """Group normal-frame captions into overlapping time windows.

        Returns:
            List of (window_start_sec, window_end_sec, [caption_str, ...]).
        """
        # Collect (timestamp, caption) for normal frames
        normal_frames: List[tuple[float, str]] = []
        for frame_str, caption in captions.items():
            score = scores.get(frame_str)
            if score is None:
                continue
            if float(score) > threshold:
                continue
            try:
                frame_idx = int(frame_str)
            except ValueError:
                continue
            timestamp = frame_idx / fps
            normal_frames.append((timestamp, caption))

        if not normal_frames:
            return []

        normal_frames.sort(key=lambda x: x[0])
        max_time = normal_frames[-1][0]

        windows: List[tuple[float, float, List[str]]] = []
        window_start = 0.0

        while window_start <= max_time:
            window_end = window_start + self.window_seconds
            window_captions = [
                cap for ts, cap in normal_frames
                if window_start <= ts < window_end
            ]
            if window_captions:
                windows.append((window_start, window_end, window_captions))
            window_start += self.stride_seconds

        return windows

    # -- internal: context generation ----------------------------------------

    def _generate_context(self, llm_generator, captions: List[str]) -> str:
        """Ask the LLM to synthesize a scene context from normal captions."""
        truncated = captions[:self.max_captions_per_context]
        joined = "\n".join(f"- {c}" for c in truncated)

        dialogs = [[
            {"role": "system", "content": "You are a surveillance scene analyst."},
            {"role": "user", "content": CONTEXT_GENERATION_PROMPT.format(
                normal_descriptions=joined,
            )},
        ]]

        try:
            results = llm_generator.chat_completion(
                dialogs,
                max_gen_len=256,
                temperature=0.3,
                top_p=0.9,
            )
            return results[0]["generation"]["content"].strip()
        except Exception:
            logger.exception("LLM context generation failed")
            return ""

    # -- internal: drift detection -------------------------------------------

    def _has_scene_changed(
        self,
        new_captions: List[str],
        prev_context: SceneContext,
        llm_generator,
    ) -> bool:
        """Return True if the scene has meaningfully drifted.

        Tries Plan A (embedding cosine similarity) first; falls back to
        Plan B (LLM prompt) if the embedding library is unavailable.
        """
        if not self._embedder_failed and self._embedder is None:
            self._init_embedder()

        if self._embedder is not None:
            return self._drift_check_embedding(new_captions, prev_context)
        else:
            return self._drift_check_llm(new_captions, prev_context, llm_generator)

    def _init_embedder(self) -> None:
        """Lazy-load the sentence-transformers model (Plan A)."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Plan A drift detection: all-MiniLM-L6-v2 loaded.")
        except ImportError:
            self._embedder_failed = True
            logger.warning(
                "sentence-transformers not installed; "
                "falling back to LLM-based drift detection (Plan B)."
            )

    def _drift_check_embedding(
        self,
        new_captions: List[str],
        prev_context: SceneContext,
    ) -> bool:
        """Plan A: cosine similarity between new captions and previous context."""
        new_text = " ".join(new_captions[:10])
        prev_text = prev_context.description

        new_emb = self._embedder.encode([new_text], show_progress_bar=False)[0]
        prev_emb = self._embedder.encode([prev_text], show_progress_bar=False)[0]

        cosine_sim = float(
            np.dot(new_emb, prev_emb)
            / (np.linalg.norm(new_emb) * np.linalg.norm(prev_emb) + 1e-8)
        )
        logger.debug("Drift cosine_sim=%.3f  threshold=%.2f", cosine_sim, self.drift_threshold)
        return cosine_sim < self.drift_threshold

    def _drift_check_llm(
        self,
        new_captions: List[str],
        prev_context: SceneContext,
        llm_generator,
    ) -> bool:
        """Plan B: ask the LLM whether the environment has fundamentally changed."""
        new_text = "\n".join(f"- {c}" for c in new_captions[:10])
        prompt = DRIFT_CHECK_PROMPT.format(
            previous_context=prev_context.description,
            new_observations=new_text,
        )
        dialogs = [[
            {"role": "system", "content": "Answer only YES or NO."},
            {"role": "user", "content": prompt},
        ]]

        try:
            results = llm_generator.chat_completion(
                dialogs,
                max_gen_len=8,
                temperature=0.0,
                top_p=0.9,
            )
            answer = results[0]["generation"]["content"].strip().upper()
            changed = answer.startswith("YES")
            logger.debug("Plan B drift check: changed=%s  (answer=%r)", changed, answer)
            return changed
        except Exception:
            logger.exception("Plan B drift check failed; assuming scene changed.")
            return True  # conservative: re-generate context on error
