"""Policy registry. Importing this package registers all built-in policies."""
from .base import Policy, ScorePolicy, REGISTRY, register_policy, make_policy
from . import classic, structural, cage  # noqa: F401  (side effect: registration)

# Convenience sets used by the paper's confirmatory runs
CAGE_SET = {"Random", "Density+Quota", "Hubness+Quota", "HubDensity-Pure+Quota",
            "HitDensity+Quota", "CAGE+Quota (ours)"}
CORESET = {"K-Center+Quota (stratified coreset)"}
EXT8_SET = CAGE_SET | CORESET | {"DecayedHit"}

__all__ = ["Policy", "ScorePolicy", "REGISTRY", "register_policy", "make_policy",
           "CAGE_SET", "CORESET", "EXT8_SET"]
