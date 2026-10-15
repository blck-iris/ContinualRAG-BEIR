"""Benchmark configuration (mirrors the paper's protocol)."""
from dataclasses import dataclass, field
from typing import List, Optional

DOMAIN_SETS = {
    "base5": ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"],
    "extended8": ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs",
                  "trec-covid", "quora", "webis-touche2020"],
}

# Cap on positives kept per query for domains with very many judgments.
DOMAIN_POOL_CAP = {"trec-covid": 10}

# model name -> (query prefix, document prefix)
EMBED_PRESETS = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
}


@dataclass
class Config:
    domains: List[str] = field(default_factory=lambda: list(DOMAIN_SETS["base5"]))
    embed_model: str = "BAAI/bge-small-en-v1.5"       # retrieval / scoring embedder (E_ret)
    sel_model: Optional[str] = None                    # None = same embedder; "intfloat/e5-small-v2" = cross-embedder control (E_sel)
    max_seq_len: int = 256
    pool_neg: int = 1500
    docs_per_domain: int = 1000
    max_queries_per_domain: int = 200
    max_rel_per_query: int = 5
    rel_budget_frac: float = 0.6
    usage_frac: float = 0.5                            # share of queries in the usage pool; the rest is the held-out test pool
    n_usage_per_phase: int = 150
    zipf_s: float = 1.0
    top_k: int = 10
    cap: int = 2000                                    # memory budget M_max (40% of streamed docs in the paper)
    decay: float = 0.8                                 # per-phase decay of online hit counts
    cache_budget: Optional[int] = None                 # None = unconstrained exact-query result cache
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    cache_dir: str = "cache"
    data_dir: str = "beir_data"
