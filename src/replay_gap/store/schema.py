"""The three tables the log becomes, and the conversion from arrays to rows.

The grains are deliberately separate. A session row is one decision, a candidate row is one item
the retrieval stage offered, and an impression row is one item a user could have looked at.
Flattening those into one wide table is what makes a position bias query impossible to write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from replay_gap.simulator.generate import InteractionLog

TABLES: tuple[str, ...] = ("sessions", "candidates", "impressions")


def log_to_frames(log: InteractionLog) -> dict[str, pd.DataFrame]:
    """Turn the array log into the three frames the SQL schema expects."""
    sessions = pd.DataFrame(
        {
            "session_id": log.session_id.astype(np.int64),
            "context_id": log.context_id.astype(np.int64),
            "log_propensity": log.log_propensity.astype(np.float64),
            "reward": log.reward.astype(np.int64),
        }
    )

    n_sessions, n_candidates = log.candidates.shape
    candidates = pd.DataFrame(
        {
            "session_id": np.repeat(log.session_id.astype(np.int64), n_candidates),
            "slot": np.tile(np.arange(n_candidates, dtype=np.int32), n_sessions),
            "item_id": log.candidates.reshape(-1).astype(np.int64),
        }
    )

    shown = log.slate_items()
    n_positions = shown.shape[1]
    impressions = pd.DataFrame(
        {
            "session_id": np.repeat(log.session_id.astype(np.int64), n_positions),
            "position": np.tile(np.arange(n_positions, dtype=np.int32), n_sessions),
            "item_id": shown.reshape(-1).astype(np.int64),
            "clicked": log.clicks.reshape(-1).astype(np.int16),
        }
    )
    return {"sessions": sessions, "candidates": candidates, "impressions": impressions}
