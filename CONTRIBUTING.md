# Contributing to ContinualRAG-BEIR

Thank you for your interest in contributing! We welcome new bounded-memory eviction policies, benchmarks, and improvements to the core framework.

## Adding a New Eviction Policy
If you want to test your own eviction strategy against CAGE and the baselines:

1. Create a new file in `continual_rag/policies/` (e.g., `my_policy.py`).
2. Write your scoring function or stateful policy.
3. Register it using our decorator:
   ```python
   from continual_rag import register_policy
   from continual_rag.policies.base import ScorePolicy
   
   def my_score_fn(E, Q, w):
       # E: candidate embeddings, Q: served queries, w: decayed counts
       return ... # your scores
   
   register_policy("MyPolicy+Quota")(
       lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, my_score_fn, per_domain=True, quota=True)
   )
