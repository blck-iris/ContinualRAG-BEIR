"""nDCG scoring, continual-learning metrics, and the paper's statistical machinery."""
from typing import Tuple
import numpy as np
from scipy.stats import wilcoxon


def score_queries(ret: np.ndarray, recs, k: int):
    """Binary-relevance nDCG@k (ideal DCG over the full domain subsample), recall@k, success@k."""
    disc = 1.0 / np.log2(np.arange(k) + 2.0)
    nd, rc, su = [], [], []
    for i, r in enumerate(recs):
        rel = np.isin(ret[i], r["rel"]); L = len(rel)
        dcg = (rel * disc[:L]).sum(); idcg = disc[:min(k, len(r["rel"]))].sum()
        nd.append(dcg / idcg); rc.append(rel.sum() / len(r["rel"])); su.append(float(rel.any()))
    return np.array(nd), np.array(rc), np.array(su)


def bwt(R: np.ndarray) -> float:
    """Backward transfer: mean_j<T (R[T,j] - R[j,j]) with R[t, j] = novel nDCG on domain j after phase t."""
    T = R.shape[0]
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else 0.0


def forgetting(R: np.ndarray) -> float:
    """mean_j<T (max_{l in [j, T-2]} R[l, j] - R[T-1, j]) (0-indexed form of the paper's definition)."""
    T = R.shape[0]
    return float(np.mean([R[j:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)])) if T > 1 else 0.0


def mixed(final_rep: float, final_ndcg: float, r: float) -> float:
    return r * final_rep + (1 - r) * final_ndcg


def bootstrap_ci(x: np.ndarray, n: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    means = x[rng.randint(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a: np.ndarray, b: np.ndarray, alternative: str = "greater"):
    """Paired comparison over seeds: (mean diff, CI lo, CI hi, Wilcoxon signed-rank p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_ci(d)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, alternative=alternative).pvalue)
    return float(d.mean()), lo, hi, p


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj
