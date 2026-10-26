"""The seeded world that acts as referee.

A public learning to rank dataset has relevance labels but no logging policy and no propensities,
so the counterfactual this repository depends on does not exist in it. Everything under this
package exists to supply that counterfactual: a world with known true relevance, a deployed
policy that produced the log, and an exact value for any policy you care to ask about.
"""

from replay_gap.simulator.generate import InteractionLog, generate_log
from replay_gap.simulator.ground_truth import true_value, true_value_lift
from replay_gap.simulator.logging_policy import LoggingPolicy
from replay_gap.simulator.user_model import World, build_world

__all__ = [
    "InteractionLog",
    "LoggingPolicy",
    "World",
    "build_world",
    "generate_log",
    "true_value",
    "true_value_lift",
]
