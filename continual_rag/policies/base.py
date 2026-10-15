"""Policy base classes and the policy registry."""
import time
from typing import Callable, Dict
import numpy as np

# name -> factory(ep, cap, fb, seed) -> Policy
REGISTRY: Dict[str, Callable] = {}


def register_policy(name: str):
    """Decorator/utility: `register_policy("X")(factory)` where factory(ep, cap, fb, seed) returns a Policy."""
    def deco(factory):
        REGISTRY[name] = factory
        return factory
    return deco


def make_policy(name: str, ep, cap: int, fb: str, seed: int):
    if name not in REGISTRY:
        raise KeyError(f"Unknown policy '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](ep, cap, fb, seed)


class Policy:
    """Bounded-memory eviction policy.

    Lifecycle per phase t: arrive(t, new_docs) -> serve usage queries -> feedback(...).
    Subclasses implement select(cand, t) -> subset of `cand` with len <= cap.
    Selection features use ep.DSEL / ep.QSEL (the *selection* embedder); retrieval uses a different pair
    in the runner when running the cross-embedder control.
    """
    name = "Policy"

    def __init__(self, ep, cap: int, fb: str, seed: int):
        self.ep, self.cap, self.fb = ep, cap, fb
        self.DSEL, self.QSEL = ep.DSEL, ep.QSEL
        self.rng = np.random.RandomState(seed * 31 + 5)
        self.mem = np.zeros(0, int); self.n_seen = 0; self.phase = 0
        self.hits = np.zeros(ep.N)          # decayed online hit counts
        self.cnt_total = np.zeros(ep.N)     # undecayed hit counts (LFU)
        self.last_hit = -np.ones(ep.N)      # last phase with a hit (LRU)
        self.qc = np.zeros(ep.T)            # decayed #queries served per domain
        self.qw = np.zeros(self.QSEL.shape[0])   # decayed per-query service counts
        self.exposed = np.zeros(ep.N, bool)
        self.evict_time = 0.0

    # -- lifecycle ---------------------------------------------------------
    def arrive(self, t: int, new: np.ndarray):
        t0 = time.perf_counter()
        self.n_seen += len(new)
        cand = np.concatenate([self.mem, new])
        self.mem = cand if len(cand) <= self.cap else self.select(cand, t)
        assert len(self.mem) <= self.cap and len(np.unique(self.mem)) == len(self.mem)
        self.evict_time += time.perf_counter() - t0

    def feedback(self, t: int, hit_docs: np.ndarray, qcounts: np.ndarray, qidx: np.ndarray):
        d = self.ep.cfg.decay
        self.hits *= d; self.qc *= d; self.qw *= d
        np.add.at(self.qw, qidx, 1.0)
        if len(hit_docs):
            np.add.at(self.hits, hit_docs, 1.0); np.add.at(self.cnt_total, hit_docs, 1.0)
            self.last_hit[np.unique(hit_docs)] = self.phase
        self.qc += qcounts
        self.exposed[self.mem] = True
        self.phase += 1

    # -- helpers -----------------------------------------------------------
    def rate(self, idx: np.ndarray) -> np.ndarray:
        """Decayed hit rate of docs `idx` relative to queries served in their domain."""
        return self.hits[idx] / np.maximum(self.qc[self.ep.doc_dom[idx]], 1.0)

    def pick(self, cand: np.ndarray, scores: np.ndarray, quota: bool = False, n=None) -> np.ndarray:
        """Top-n by score. With quota=True each seen domain gets n // n_domains slots; the remainder is filled globally."""
        n = self.cap if n is None else int(n)
        if n <= 0:
            return cand[:0]
        if n >= len(cand):
            return cand.copy()
        scores = scores + self.rng.rand(len(scores)) * 1e-9   # random tie-breaking
        if not quota:
            return cand[np.argpartition(-scores, n - 1)[:n]]
        doms = self.ep.doc_dom[cand]; ud = np.unique(doms); share = n // len(ud); chosen = []
        for d in ud:
            ii = np.where(doms == d)[0]
            chosen.append(ii[np.argsort(-scores[ii])][:share])
        chosen = np.concatenate(chosen)
        rest = np.setdiff1d(np.arange(len(cand)), chosen); need = n - len(chosen)
        if need > 0:
            chosen = np.concatenate([chosen, rest[np.argsort(-scores[rest])][:need]])
        return cand[chosen]

    def select(self, cand: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError


class ScorePolicy(Policy):
    """Keep the top-`cap` candidates under score_fn(E, Q, w) -> (n,) scores.

    E: (n, d) selection embeddings of the scored candidates; Q: (m, d) selection embeddings of queries served so far
    (in the same domain when per_domain=True); w: (m,) decayed service counts. per_domain=True scores each domain
    separately; quota=True additionally splits the budget equally across domains.
    """
    def __init__(self, ep, cap, fb, seed, score_fn: Callable, per_domain: bool = False,
                 quota: bool = False, name: str = "ScorePolicy"):
        super().__init__(ep, cap, fb, seed)
        self.score_fn, self.per_domain, self.quota, self.name = score_fn, per_domain, quota, name

    def select(self, cand, t):
        used = np.where(self.qw > 0)[0]
        if not self.per_domain:
            s = np.asarray(self.score_fn(self.DSEL[cand], self.QSEL[used], self.qw[used]), float)
            return self.pick(cand, s, quota=self.quota)
        doms = self.ep.doc_dom[cand]; s = np.zeros(len(cand))
        for d in np.unique(doms):
            ii = np.where(doms == d)[0]
            ud = used[self.ep.q_dom[used] == d]
            s[ii] = self.score_fn(self.DSEL[cand[ii]], self.QSEL[ud], self.qw[ud])
        return self.pick(cand, s, quota=self.quota)
