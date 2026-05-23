from __future__ import annotations

import numpy as np
import pandas as pd

TIER_LABELS = [
    "Top 5% premium",
    "Top 5-10% strong",
    "Top 10-20% growth",
    "Middle 20-50% broad",
    "Bottom 50% low-priority",
]

TIER_ACTIONS = {
    "Top 5% premium": "Prioritize for high-value personalized outreach and premium offers.",
    "Top 5-10% strong": "Use targeted offers with controlled acquisition cost and message testing.",
    "Top 10-20% growth": "Use lower-cost digital campaigns and monitor incremental response.",
    "Middle 20-50% broad": "Keep for broad awareness or small randomized tests, not expensive outreach.",
    "Bottom 50% low-priority": "Suppress from high-cost campaigns; use only low-frequency broad messaging.",
}


def add_score_tier(scores: np.ndarray) -> pd.Series:
    s = pd.Series(scores)
    pct_rank = s.rank(method="first", ascending=False) / len(s)
    return pd.cut(
        pct_rank,
        bins=[0, 0.05, 0.10, 0.20, 0.50, 1.0],
        labels=TIER_LABELS,
        include_lowest=True,
    ).astype(str)


def score_tier_summary(y_true, scores, sample_weight) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "score": np.asarray(scores),
            "weight": np.asarray(sample_weight, dtype=float),
            "tier": np.asarray(add_score_tier(scores)),
        }
    )
    base_rate = float((df["y"] * df["weight"]).sum() / df["weight"].sum())
    rows = []
    for tier in TIER_LABELS:
        g = df[df["tier"] == tier]
        if g.empty:
            continue
        positive_rate = float((g["y"] * g["weight"]).sum() / g["weight"].sum())
        rows.append(
            {
                "tier": tier,
                "n": int(len(g)),
                "weighted_share": float(g["weight"].sum() / df["weight"].sum()),
                "avg_score": float(np.average(g["score"], weights=g["weight"])),
                "positive_rate_w": positive_rate,
                "lift_vs_overall": positive_rate / base_rate if base_rate else 0.0,
                "recommended_action": TIER_ACTIONS.get(tier, "Review segment before campaign activation."),
            }
        )
    return pd.DataFrame(rows)
