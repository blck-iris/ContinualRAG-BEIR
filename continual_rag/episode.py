"""Leak-free episode construction and exact top-k retrieval."""
from typing import List
import numpy as np

from .config import Config
from .data import Pool


class Episode:
    """One seeded continual stream.

    Domains arrive in a seed-dependent order. Within each domain the queries are split into a
    disjoint *usage* pool (drives Zipfian online feedback) and a held-out *test* pool (never
    observed by any policy). Global ids: doc = pool_doc_offset + local id (pool order, not arrival order).
    """

    def __init__(self, pools: List[Pool], cfg: Config, seed: int):
        self.cfg, self.seed = cfg, seed
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(pools))
        self.T = len(pools)
        self.N = sum(p.n_docs for p in pools)
        doc_off = np.cumsum([0] + [p.n_docs for p in pools])
        q_off = np.cumsum([0] + [p.Q.shape[0] for p in pools])

        # stacked embeddings (global ids)
        self.DEMB = np.vstack([p.D for p in pools])
        self.QEMB = np.vstack([p.Q for p in pools])
        cross = cfg.sel_model is not None and all(p.Ds is not None for p in pools)
        self.DSEL = np.vstack([p.Ds for p in pools]) if cross else self.DEMB
        self.QSEL = np.vstack([p.Qs for p in pools]) if cross else self.QEMB

        self.doc_dom = -np.ones(self.N, int)     # arrival position of each doc's domain
        self.doc_arr = -np.ones(self.N, int)     # global arrival sequence number
        self.q_dom = -np.ones(self.QEMB.shape[0], int)
        self.oracle_cnt = np.zeros(self.N)
        self.tasks, self.names = [], []
        seq = 0
        for pos, pi in enumerate(order):
            P = pools[pi]
            do, qo = doc_off[pi], q_off[pi]
            qids = list(rng.permutation(len(P.qrels)))
            budget = int(cfg.rel_budget_frac * cfg.docs_per_domain)
            chosen, rel_union = [], set()
            for q in qids:
                r = list(P.qrels[q]); rng.shuffle(r); r.sort(key=lambda x: -x[1])
                new = {d for d, _ in r[:cfg.max_rel_per_query]}
                if len(rel_union | new) > budget:
                    continue
                chosen.append((q, new)); rel_union |= new
                if len(chosen) >= cfg.max_queries_per_domain:
                    break
            rest = np.setdiff1d(np.arange(P.n_docs), np.fromiter(rel_union, int, len(rel_union)))
            n_neg = max(0, cfg.docs_per_domain - len(rel_union))
            negs = rng.choice(rest, size=min(n_neg, len(rest)), replace=False)
            local_docs = rng.permutation(np.concatenate([np.fromiter(rel_union, int, len(rel_union)), negs]))
            docs = (local_docs + do).astype(int)
            self.doc_dom[docs] = pos
            self.doc_arr[docs] = np.arange(seq, seq + len(docs)); seq += len(docs)
            recs = []
            for q, rel in chosen:
                relg = np.array(sorted(d + do for d in rel), int)
                sd = P.selfdoc.get(q, -1)
                recs.append(dict(q=int(q + qo), rel=relg, selfdoc=int(sd + do) if sd >= 0 else -1, dom=pos))
                self.oracle_cnt[relg] += 1
                self.q_dom[q + qo] = pos
            n_use = max(10, int(cfg.usage_frac * len(recs)))
            self.tasks.append(dict(docs=docs, usage=recs[:n_use], test=recs[n_use:]))
            self.names.append(P.name)

        # Zipfian usage stream (independent RNG so the stream does not depend on policy randomness)
        srng = np.random.RandomState(seed * 7919 + 13)
        self.pop = []
        for t in range(self.T):
            n = len(self.tasks[t]["usage"])
            w = 1.0 / (srng.permutation(n) + 1.0) ** cfg.zipf_s
            self.pop.append(w / w.sum())
        self.stream = [[srng.choice(len(self.tasks[j]["usage"]), size=cfg.n_usage_per_phase, p=self.pop[j])
                        for j in range(t + 1)] for t in range(self.T)]


def retrieve(mem: np.ndarray, recs: List[dict], k: int, QEMB: np.ndarray, DEMB: np.ndarray, n_docs: int) -> np.ndarray:
    """Exact cosine top-k over the documents in `mem` (embeddings are L2-normalised). Masks ArguAna self-docs."""
    if len(recs) == 0 or len(mem) == 0:
        return np.zeros((len(recs), 0), int)
    qi = np.array([r["q"] for r in recs])
    S = QEMB[qi] @ DEMB[mem].T
    lut = -np.ones(n_docs, int); lut[mem] = np.arange(len(mem))
    sd = np.array([r["selfdoc"] for r in recs]); ok = sd >= 0
    if ok.any():
        pos = lut[sd[ok]]; rows = np.where(ok)[0][pos >= 0]
        S[rows, pos[pos >= 0]] = -np.inf
    ke = min(k, len(mem))
    part = np.argpartition(-S, ke - 1, axis=1)[:, :ke]
    order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
    return mem[np.take_along_axis(part, order, 1)]
