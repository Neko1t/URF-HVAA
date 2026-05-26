"""
Dual-Threshold Score Gate — v2 optimization.

Placed between Stage B and Stage C.  Decides per-video whether the Phase 1
scores are already "confident enough" to skip the reflection loop (Stage C→D→E),
or whether the video is ambiguous and needs deeper analysis.

Conditions (per video):
  A (extremely normal):   Max_Score < 0.3  AND  Variance < 0.05
      → skip reflection, output all-normal scores.
  B (extremely anomalous): Max_Score > 0.85  AND  high_density > 10%
      → skip reflection, keep Phase 1 raw scores (no smoothing dilution).
  Otherwise → GATE_REFLECT: trigger Stage C/D/E.

The density constraint on condition B prevents a single hallucinated spike
(e.g. frame 64 scoring 1.0 on a VLM hallucination) from skipping reflection
for the entire video.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate decisions
# ---------------------------------------------------------------------------

GATE_NORMAL    = "normal"       # skip reflection, output all-normal
GATE_ANOMALOUS = "anomalous"    # skip reflection, keep Phase 1 raw scores
GATE_REFLECT   = "reflect"      # trigger Stage C → D → E


# ---------------------------------------------------------------------------
# ScoreGate
# ---------------------------------------------------------------------------

class ScoreGate:
    """Dual-threshold gating between Stage B and Stage C.

    Usage::

        gate = ScoreGate(
            max_score_normal=0.3,
            var_normal=0.05,
            max_score_anomalous=0.85,
            density_anomalous=0.10,
            high_score_threshold=0.7,
        )
        decision, reason = gate.decide(scores)

    Args:
        max_score_normal: Max_Score threshold for condition A.
        var_normal: Variance threshold for condition A.
        max_score_anomalous: Max_Score threshold for condition B.
        density_anomalous: Minimum fraction of frames with score >
            *high_score_threshold* to satisfy condition B.
        high_score_threshold: The score above which a frame is considered
            "high" for density computation.
    """

    def __init__(
        self,
        max_score_normal: float = 0.3,
        var_normal: float = 0.05,
        max_score_anomalous: float = 0.85,
        density_anomalous: float = 0.10,
        high_score_threshold: float = 0.7,
    ):
        self.max_score_normal = max_score_normal
        self.var_normal = var_normal
        self.max_score_anomalous = max_score_anomalous
        self.density_anomalous = density_anomalous
        self.high_score_threshold = high_score_threshold

    # -- public API -----------------------------------------------------------

    def decide(self, scores: Dict[str, float]) -> Tuple[str, str]:
        """Return (decision, reason) for a video.

        Args:
            scores: {frame_idx_str: anomaly_score} from Phase 1 / Stage B.

        Returns:
            (GATE_NORMAL | GATE_ANOMALOUS | GATE_REFLECT, human-readable reason).
        """
        if not scores:
            return GATE_REFLECT, "no scores available"

        values = np.array([float(v) for v in scores.values()], dtype=np.float32)
        n = len(values)
        if n == 0:
            return GATE_REFLECT, "empty score array"

        max_score = float(np.max(values))
        variance = float(np.var(values))

        # ---- Condition A: extremely normal ----
        if max_score < self.max_score_normal and variance < self.var_normal:
            reason = (
                f"Max={max_score:.3f} < {self.max_score_normal} "
                f"AND Var={variance:.4f} < {self.var_normal} "
                f"→ LLM consistently sees no anomaly"
            )
            return GATE_NORMAL, reason

        # ---- Condition B: extremely anomalous ----
        if max_score > self.max_score_anomalous:
            high_count = int(np.sum(values > self.high_score_threshold))
            high_density = high_count / n
            if high_density > self.density_anomalous:
                reason = (
                    f"Max={max_score:.3f} > {self.max_score_anomalous} "
                    f"AND high_density={high_density:.3f} "
                    f"({high_count}/{n}) > {self.density_anomalous} "
                    f"→ LLM confidently sees widespread anomaly"
                )
                return GATE_ANOMALOUS, reason
            else:
                reason = (
                    f"Max={max_score:.3f} > {self.max_score_anomalous} BUT "
                    f"high_density={high_density:.3f} "
                    f"({high_count}/{n}) <= {self.density_anomalous} "
                    f"→ possible hallucination spike, not dense enough"
                )
                return GATE_REFLECT, reason

        # ---- Default: trigger reflection ----
        reason = (
            f"Max={max_score:.3f}, Var={variance:.4f} "
            f"→ scores ambiguous, triggering reflection"
        )
        return GATE_REFLECT, reason

    # -- batch API ------------------------------------------------------------

    def classify(
        self, all_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Tuple[str, str]]:
        """Run the gate on multiple videos at once.

        Args:
            all_scores: {video_name: {frame_str: score}}.

        Returns:
            {video_name: (decision, reason)}.
        """
        results: Dict[str, Tuple[str, str]] = {}
        for vname, scores in all_scores.items():
            results[vname] = self.decide(scores)
        return results

    def print_summary(self, results: Dict[str, Tuple[str, str]]) -> None:
        """Print a summary of gate decisions."""
        counts = {GATE_NORMAL: 0, GATE_ANOMALOUS: 0, GATE_REFLECT: 0}
        for decision, _ in results.values():
            counts[decision] = counts.get(decision, 0) + 1

        total = sum(counts.values())
        print(f"\n  Score Gate Summary ({total} videos):")
        for label, desc in [
            (GATE_NORMAL,    "Normal (skip reflection)"),
            (GATE_ANOMALOUS, "Anomalous (keep raw)"),
            (GATE_REFLECT,   "Reflect (trigger C→D→E)"),
        ]:
            n = counts.get(label, 0)
            pct = n / total * 100 if total > 0 else 0
            print(f"    {desc:<30} {n:>4}  ({pct:>5.1f}%)")
