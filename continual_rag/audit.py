"""Pre-eviction predictability audit (paper Sec. 4 / Table 1).

For each domain: which observable signals predict relevance of a document to *held-out* test queries?
Reported as within-domain ROC-AUC averaged over seeds; "ALL" is the mean over domains.
"""
from typing import Iterable, List
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import Config
from .data import Pool
from .episode import Episode
from .policies.structural import knn_density


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos = y.astype(bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def audit_episode(ep: Episode) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(ep.names):
        docs = ep.tasks[j]["docs"]
        loc = {d: i for i, d in enumerate(docs)}
        y_use, y_test = np.zeros(len(docs), bool), np.zeros(len(docs), bool)
        for r in ep.tasks[j]["usage"]:
            for d in r["rel"]: y_use[loc[d]] = True
        for r in ep.tasks[j]["test"]:
            for d in r["rel"]: y_test[loc[d]] = True
        E = ep.DSEL[docs]
        dens = knn_density(E, 5)
        Qu = ep.QSEL[[r["q"] for r in ep.tasks[j]["usage"]]]
        S = E @ Qu.T
        top3 = np.sort(S, axis=1)[:, -3:].mean(1)
        nu = ~y_use
        rows.append({
            "domain": name,
            "base_rate": y_test.mean(),
            "repeat_rate": (y_test & y_use).sum() / max(y_test.sum(), 1),
            "relevant_to_usage_query": auc(y_use.astype(float), y_test),
            "max_sim_usage_q": auc(S.max(1), y_test),
            "top3_sim_usage_q": auc(top3, y_test),
            "outlierness(-dens)": auc(-dens, y_test),
            "knn_density": auc(dens, y_test),
            "knn_density|not_usage_rel": auc(dens[nu], y_test[nu]),
        })
    return pd.DataFrame(rows)


def run_audit(cfg: Config, pools: List[Pool], seeds: Iterable[int]) -> pd.DataFrame:
    """Domains x signals table averaged over seeds, plus an ALL column (mean over domains)."""
    df = pd.concat([audit_episode(Episode(pools, cfg, s)) for s in seeds])
    out = df.groupby("domain").mean().T
    out["ALL"] = out.mean(axis=1)
    return out
