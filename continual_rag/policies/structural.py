"""Training-free structural centrality priors and the pure structural policies."""
import numpy as np
from scipy.stats import rankdata
from .base import ScorePolicy, register_policy


def _sim(E: np.ndarray) -> np.ndarray:
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    return S


def knn_density(E: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine similarity to the k nearest neighbours (Eq. 1)."""
    n = len(E)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    kd = min(k, n - 1)
    return np.partition(_sim(E), -kd, axis=1)[:, -kd:].mean(axis=1)


def knn_indegree(E: np.ndarray, k: int = 10) -> np.ndarray:
    """Hubness: how many other docs list i among their k nearest neighbours (Eq. 2)."""
    n = len(E)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    kh = min(k, n - 1)
    nn = np.argpartition(-_sim(E), kh - 1, axis=1)[:, :kh]
    return np.bincount(nn.ravel(), minlength=n).astype(np.float32)


def centroid_similarity(E: np.ndarray) -> np.ndarray:
    """Cosine similarity to the domain centroid / medoid proxy (Eq. 3)."""
    c = E.mean(axis=0)
    c = c / max(np.linalg.norm(c), 1e-9)
    return E @ c


def hub_density_centrality(E: np.ndarray) -> np.ndarray:
    """Composite structural centrality C_i in (0, 1] (Eq. 4): 0.45 R(Dens) + 0.45 R(Hub) + 0.10 R(Med)."""
    n = len(E)
    if n <= 2:
        return np.ones(n, dtype=np.float32)
    return (0.45 * rankdata(knn_density(E, 5)) / n
            + 0.45 * rankdata(knn_indegree(E, 10)) / n
            + 0.10 * rankdata(centroid_similarity(E)) / n).astype(np.float32)


# score functions: (E (n,d), Q (m,d), w (m,)) -> (n,)
def _dens(E, Q, w): return knn_density(E, 5)
def _hub(E, Q, w): return knn_indegree(E, 10)
def _hubdens(E, Q, w): return hub_density_centrality(E)


register_policy("Density")(lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, _dens, per_domain=False, quota=False, name="Density"))
register_policy("Density+Quota")(lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, _dens, per_domain=True, quota=True, name="Density+Quota"))
register_policy("Hubness+Quota")(lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, _hub, per_domain=True, quota=True, name="Hubness+Quota"))
register_policy("HubDensity-Pure+Quota")(lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, _hubdens, per_domain=True, quota=True, name="HubDensity-Pure+Quota"))
