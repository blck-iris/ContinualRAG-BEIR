"""ContinualRAG-BEIR: bounded-memory document eviction under sequential domain shift."""
from .config import Config, DOMAIN_SETS, EMBED_PRESETS
from .episode import Episode, retrieve
from .metrics import score_queries, holm, paired_test
from .runner import run_episode, run_suite
from .policies import REGISTRY, register_policy, make_policy

__version__ = "0.1.0"
__all__ = [
    "Config", "DOMAIN_SETS", "EMBED_PRESETS", "Episode", "retrieve",
    "score_queries", "holm", "paired_test", "run_episode", "run_suite",
    "REGISTRY", "register_policy", "make_policy",
]
