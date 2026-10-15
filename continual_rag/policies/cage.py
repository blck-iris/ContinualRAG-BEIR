"""CAGE (Centrality-Anchored Gated Eviction) and the un-gated HitDensity hybrid."""
from typing import Callable
import numpy as np
from scipy.stats import rankdata
from .base import Policy, register_policy
from .structural import hub_density_centrality, knn_density


def _density_rank(E):
    return (rankdata(knn_density(E, 5)) / max(len(E), 1)).astype(np.float32)


class GatedQuotaPolicy(Policy):
    """Dual-memory, domain-stratified policy.

    Per domain quota B_d = cap // n_domains:
      * working memory: at most floor(rho * B_d) docs with decayed hits >= hit_thresh, ranked by
        rate * (mod_floor + (1 - mod_floor) * C)   (mod_floor=1 disables the centrality modulation)
      * core: the remaining slots go to the highest-prior (centrality) docs of the domain.
    rho = 0 gives the pure structural policy.
    """
    name = "GatedQuota"

    def __init__(self, ep, cap, fb, seed, prior: Callable = hub_density_centrality,
                 rho: float = 0.06, hit_thresh: float = 1.2, mod_floor: float = 0.5):
        super().__init__(ep, cap, fb, seed)
        self.prior, self.rho, self.hit_thresh, self.mod_floor = prior, rho, hit_thresh, mod_floor

    def select(self, cand, t):
        doms = self.ep.doc_dom[cand]; ud = np.unique(doms); share = self.cap // len(ud)
        C = np.zeros(len(cand), dtype=np.float32)
        for d in ud:
            ii = np.where(doms == d)[0]
            C[ii] = self.prior(self.DSEL[cand[ii]])
        if self.rho <= 0:
            return self.pick(cand, C, quota=True)

        rate, hits = self.rate(cand), self.hits[cand]
        hot_per_dom = max(1, int(self.rho * share))
        chosen = []
        for d in ud:
            ii = np.where(doms == d)[0]
            hot = np.where(hits[ii] >= self.hit_thresh)[0]                  # multi-hit admission gate (Eq. 5)
            if len(hot):
                score = rate[ii[hot]] * (self.mod_floor + (1 - self.mod_floor) * C[ii[hot]])
                top_h = hot[np.argsort(-score)[:min(hot_per_dom, len(hot), share)]]
            else:
                top_h = np.zeros(0, dtype=int)
            rem = np.setdiff1d(np.arange(len(ii)), top_h)
            need = min(share - len(top_h), len(rem))
            top_c = rem[np.argsort(-C[ii[rem]])[:need]]
            chosen.append(ii[np.concatenate([top_h, top_c])])
        chosen = np.concatenate(chosen)
        need = self.cap - len(chosen)
        if need > 0:                                                         # fill remainder globally by centrality
            rest = np.setdiff1d(np.arange(len(cand)), chosen)
            chosen = np.concatenate([chosen, rest[np.argsort(-C[rest])[:need]]])
        return cand[chosen]


class CAGEPolicy(GatedQuotaPolicy):
    name = "CAGE+Quota (ours)"
    def __init__(self, ep, cap, fb, seed, rho=0.06, hit_thresh=1.2):
        super().__init__(ep, cap, fb, seed, prior=hub_density_centrality, rho=rho, hit_thresh=hit_thresh, mod_floor=0.5)


register_policy("CAGE+Quota (ours)")(lambda ep, cap, fb, seed: CAGEPolicy(ep, cap, fb, seed, rho=0.06, hit_thresh=1.2))
# Naive hybrid from the paper: 25% of each quota for any doc with a (decayed) hit, density core, no centrality modulation.
register_policy("HitDensity+Quota")(lambda ep, cap, fb, seed: GatedQuotaPolicy(
    ep, cap, fb, seed, prior=_density_rank, rho=0.25, hit_thresh=1e-9, mod_floor=1.0))
