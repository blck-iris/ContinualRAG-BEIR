"""Classic baselines: Random, FIFO, LRU/LFU/DecayedHit, WindowHit, K-Center coresets."""
import numpy as np
from .base import Policy, ScorePolicy, register_policy


class Random(Policy):
    name = "Random"
    def select(self, cand, t):
        return self.pick(cand, self.rng.rand(len(cand)))


class QuotaRandom(Policy):
    name = "Quota-Random"
    def select(self, cand, t):
        return self.pick(cand, self.rng.rand(len(cand)), quota=True)


class FIFO(Policy):
    name = "FIFO"
    def select(self, cand, t):
        return self.pick(cand, self.ep.doc_arr[cand].astype(float))


class HitPolicy(Policy):
    """Keep the docs with the highest usage signal (decayed hits / frequency / recency)."""
    def __init__(self, ep, cap, fb, seed, kind="decayed", quota=False):
        super().__init__(ep, cap, fb, seed)
        self.kind, self.quota = kind, quota

    def select(self, cand, t):
        if self.kind == "lru":
            s = self.last_hit[cand] + 1e-6 * self.ep.doc_arr[cand] / max(self.ep.N, 1)
        elif self.kind == "lfu":
            s = self.cnt_total[cand]
        else:
            s = self.hits[cand]
        return self.pick(cand, s, quota=self.quota)


class WindowHit(Policy):
    """Simplified W-TinyLFU: a recency window of the newest arrivals plus decayed-frequency main region."""
    name = "WindowHit (W-TinyLFU)"
    def __init__(self, ep, cap, fb, seed, window_frac=0.2):
        super().__init__(ep, cap, fb, seed)
        self.window_frac = window_frac

    def select(self, cand, t):
        w = int(self.window_frac * self.cap)
        newest = cand[np.argsort(-self.ep.doc_arr[cand])[:w]]
        rest = np.setdiff1d(cand, newest)
        main = self.pick(rest, self.hits[rest], n=self.cap - len(newest))
        return np.concatenate([newest, main])


def kcenter(E: np.ndarray, n: int, rng) -> np.ndarray:
    """Greedy farthest-point traversal on cosine similarity; returns n indices into E."""
    m = len(E)
    if n >= m:
        return np.arange(m)
    first = rng.randint(m)
    chosen = [first]
    max_sim = E @ E[first]
    for _ in range(n - 1):
        max_sim[chosen[-1]] = np.inf          # never re-pick
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, E @ E[nxt])
    return np.array(chosen)


class KCenter(Policy):
    """Coreset diversity baseline. quota=True is the stratified version (K-Center per domain)."""
    def __init__(self, ep, cap, fb, seed, quota=False):
        super().__init__(ep, cap, fb, seed)
        self.quota = quota
        self.name = "K-Center+Quota (stratified coreset)" if quota else "K-Center (global)"

    def select(self, cand, t):
        E = self.DSEL[cand]
        if not self.quota:
            return cand[kcenter(E, self.cap, self.rng)]
        doms = self.ep.doc_dom[cand]; ud = np.unique(doms); share = self.cap // len(ud); chosen = []
        for d in ud:
            ii = np.where(doms == d)[0]
            chosen.append(ii[kcenter(E[ii], share, self.rng)])
        chosen = np.concatenate(chosen)
        need = self.cap - len(chosen)
        if need > 0:
            rest = np.setdiff1d(np.arange(len(cand)), chosen)
            chosen = np.concatenate([chosen, self.rng.permutation(rest)[:need]])
        return cand[chosen]


register_policy("Random")(lambda ep, cap, fb, seed: Random(ep, cap, fb, seed))
register_policy("Quota-Random")(lambda ep, cap, fb, seed: QuotaRandom(ep, cap, fb, seed))
register_policy("FIFO")(lambda ep, cap, fb, seed: FIFO(ep, cap, fb, seed))
register_policy("DecayedHit")(lambda ep, cap, fb, seed: HitPolicy(ep, cap, fb, seed, "decayed"))
register_policy("LRU")(lambda ep, cap, fb, seed: HitPolicy(ep, cap, fb, seed, "lru"))
register_policy("LFU")(lambda ep, cap, fb, seed: HitPolicy(ep, cap, fb, seed, "lfu"))
register_policy("DecayedHit+Quota")(lambda ep, cap, fb, seed: HitPolicy(ep, cap, fb, seed, "decayed", quota=True))
register_policy("WindowHit (W-TinyLFU)")(lambda ep, cap, fb, seed: WindowHit(ep, cap, fb, seed))
register_policy("K-Center (global)")(lambda ep, cap, fb, seed: KCenter(ep, cap, fb, seed, quota=False))
register_policy("K-Center+Quota (stratified coreset)")(lambda ep, cap, fb, seed: KCenter(ep, cap, fb, seed, quota=True))
