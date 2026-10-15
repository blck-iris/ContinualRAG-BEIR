"""Run policies through a continual stream and collect the paper's metrics."""
from dataclasses import replace
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

from .config import Config
from .data import Pool
from .episode import Episode, retrieve
from .metrics import score_queries, bwt, forgetting
from .policies import REGISTRY, make_policy


def run_episode(ep: Episode, name: str, cap: int, fb: str = "relevance", seed: int = 0) -> dict:
    """Stream one policy through one episode.

    Per phase t: policy.arrive(new docs) -> serve the Zipfian usage stream (retrieval uses E_ret) ->
    policy.feedback(...) -> evaluate novel (held-out test) nDCG@k on every seen domain.
    """
    cfg, k = ep.cfg, ep.cfg.top_k
    pol = make_policy(name, ep, cap, fb, seed)
    T = ep.T
    Rnov = np.full((T, T), np.nan)
    online, cache = [], {}                       # cache: first-served exact-query result cache {query id: nDCG}

    for t in range(T):
        pol.arrive(t, ep.tasks[t]["docs"])
        hit_docs, qidx, qcounts = [], [], np.zeros(T)
        for j in range(t + 1):
            recs = [ep.tasks[j]["usage"][i] for i in ep.stream[t][j]]
            ret = retrieve(pol.mem, recs, k, ep.QEMB, ep.DEMB, ep.N)
            nd, _, _ = score_queries(ret, recs, k)
            online.extend(nd.tolist())
            for r, row, n_i in zip(recs, ret, nd):
                hit_docs.append(row[np.isin(row, r["rel"])] if fb == "relevance" else row)
                if r["q"] not in cache and (cfg.cache_budget is None or len(cache) < cfg.cache_budget):
                    cache[r["q"]] = float(n_i)
            qcounts[j] = len(recs); qidx.extend(r["q"] for r in recs)
        hd = np.concatenate(hit_docs).astype(int) if hit_docs else np.zeros(0, int)
        pol.feedback(t, hd, qcounts, np.array(qidx, int))
        for j in range(t + 1):
            test = ep.tasks[j]["test"]
            if test:
                nd, _, _ = score_queries(retrieve(pol.mem, test, k, ep.QEMB, ep.DEMB, ep.N), test, k)
                Rnov[t, j] = nd.mean()

    # repeated-query performance under the final memory (popularity-weighted), live and behind the result cache
    rep, rep_c = [], []
    for j in range(T):
        recs = ep.tasks[j]["usage"]
        nd, _, _ = score_queries(retrieve(pol.mem, recs, k, ep.QEMB, ep.DEMB, ep.N), recs, k)
        nd_c = np.array([cache.get(r["q"], x) for r, x in zip(recs, nd)])
        rep.append(float((ep.pop[j] * nd).sum())); rep_c.append(float((ep.pop[j] * nd_c).sum()))

    R = np.nan_to_num(Rnov, nan=0.0)
    return dict(
        seed=ep.seed, strategy=name,
        final_ndcg=float(np.nanmean(Rnov[T - 1])), final_rep=float(np.mean(rep)), final_rep_cached=float(np.mean(rep_c)),
        bwt=bwt(R), forgetting=forgetting(R), online_ndcg=float(np.mean(online)),
        arrive_ms_per_doc=1000.0 * pol.evict_time / max(pol.n_seen, 1),
    )


def run_suite(cfg: Config, pools: List[Pool], seeds: Iterable[int], strategies: Iterable[str],
              fb: str = "relevance", verbose: bool = True) -> pd.DataFrame:
    """Run every strategy on every seed (policies within a seed share the same episode => paired comparison)."""
    unknown = [s for s in strategies if s not in REGISTRY]
    if unknown:
        raise KeyError(f"Unknown strategies: {unknown}")
    rows = []
    for seed in seeds:
        ep = Episode(pools, cfg, seed)
        for name in strategies:
            rows.append(run_episode(ep, name, cfg.cap, fb, seed))
        if verbose:
            print(f"seed {seed} done")
    return pd.DataFrame(rows)
