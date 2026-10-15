"""Pools of documents/queries: BEIR loader (real) and a synthetic generator (tests, demos)."""
import os, io, json, zipfile, hashlib, urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import Config, EMBED_PRESETS, DOMAIN_POOL_CAP

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class Pool:
    name: str
    D: np.ndarray                               # (n_docs, dim) L2-normalised doc embeddings (retrieval embedder)
    Q: np.ndarray                               # (n_q, dim) query embeddings (retrieval embedder)
    qrels: Dict[int, List[Tuple[int, float]]]   # query -> [(doc, grade)]
    selfdoc: Dict[int, int]                     # query -> doc with identical id (ArguAna), masked at retrieval time
    n_docs: int = 0
    Ds: Optional[np.ndarray] = None             # doc embeddings for the *selection* embedder (cross-embedder control)
    Qs: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- BEIR
def download_beir(name: str, data_dir: str = "beir_data") -> str:
    path = os.path.join(data_dir, name)
    if os.path.exists(os.path.join(path, "corpus.jsonl")):
        return path
    os.makedirs(data_dir, exist_ok=True)
    zp = os.path.join(data_dir, name + ".zip")
    print(f"Downloading {name} from UKP BEIR ...")
    urllib.request.urlretrieve(BEIR_URL.format(name), zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(data_dir)
    os.remove(zp)
    return path


def load_beir(name: str, data_dir: str = "beir_data"):
    p = download_beir(name, data_dir)
    corpus, queries, qrels = {}, {}, {}
    with open(os.path.join(p, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["_id"]] = ((r.get("title") or "") + ". " + (r.get("text") or "")).strip()
    with open(os.path.join(p, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]
    with open(os.path.join(p, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")[:3]
            qrels.setdefault(parts[0], {})[parts[1]] = float(parts[2])
    return corpus, queries, qrels


def make_embedder(model_name: str, max_seq_len: int = 256):
    """Return f(texts, is_query) -> L2-normalised float embeddings. Needs `pip install -e .[embed]`."""
    from sentence_transformers import SentenceTransformer
    import torch
    qp, dp = EMBED_PRESETS[model_name]
    m = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = max_seq_len

    def f(texts, is_query):
        texts = [(qp if is_query else dp) + t for t in texts]
        return m.encode(texts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return f


def _cache_path(cfg: Config, name: str, model: str, cap_pp) -> str:
    qp, dp = EMBED_PRESETS[model]
    key = f"{name}|{model}|{cfg.pool_neg}|{cfg.max_seq_len}|{qp}|{dp}" + (f"|cap{cap_pp}" if cap_pp else "")
    return os.path.join(cfg.cache_dir, f"{name}_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz")


def build_pool_real(name: str, cfg: Config, emb_ret, emb_sel=None) -> Pool:
    corpus, queries, qrels = load_beir(name, cfg.data_dir)
    cap_pp = DOMAIN_POOL_CAP.get(name)
    prng = np.random.RandomState(1)
    pos = {}
    for q, rel in qrels.items():
        if q not in queries:
            continue
        r = [(d, s) for d, s in rel.items() if s > 0 and d in corpus]
        if not r:
            continue
        if cap_pp and len(r) > cap_pp:
            r = [r[i] for i in prng.permutation(len(r))]
        r = sorted(r, key=lambda x: -x[1])
        pos[q] = r[:cap_pp] if cap_pp else r
    pos_docs = sorted({d for r in pos.values() for d, _ in r})
    rng = np.random.RandomState(0)
    pos_set = set(pos_docs)
    others = sorted(d for d in corpus if d not in pos_set)
    neg = list(rng.choice(others, size=min(cfg.pool_neg, len(others)), replace=False)) if others else []
    doc_ids = pos_docs + neg
    id2loc = {d: i for i, d in enumerate(doc_ids)}
    qids = sorted(pos)
    dt, qt = [corpus[d] for d in doc_ids], [queries[q] for q in qids]

    cpath = _cache_path(cfg, name, cfg.embed_model, cap_pp)
    if os.path.exists(cpath):
        z = np.load(cpath); D, Q = z["D"], z["Q"]
    else:
        D, Q = emb_ret(dt, is_query=False), emb_ret(qt, is_query=True)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        np.savez_compressed(cpath, D=D, Q=Q)

    Ds = Qs = None
    if emb_sel is not None and cfg.sel_model and cfg.sel_model != cfg.embed_model:
        spath = _cache_path(cfg, name, cfg.sel_model, cap_pp)
        if os.path.exists(spath):
            sz = np.load(spath); Ds, Qs = sz["D"].astype(np.float32), sz["Q"].astype(np.float32)
        else:
            Ds = emb_sel(dt, is_query=False).astype(np.float32)
            Qs = emb_sel(qt, is_query=True).astype(np.float32)
            os.makedirs(cfg.cache_dir, exist_ok=True)
            np.savez_compressed(spath, D=Ds, Q=Qs)

    qrels_loc = {i: [(id2loc[d], s) for d, s in pos[q]] for i, q in enumerate(qids)}
    selfdoc = {i: id2loc[q] for i, q in enumerate(qids) if q in id2loc}
    return Pool(name, D.astype(np.float32), Q.astype(np.float32), qrels_loc, selfdoc, len(doc_ids), Ds, Qs)


def build_pools(cfg: Config) -> List[Pool]:
    """Load/embed every domain in cfg.domains (downloads BEIR on first use)."""
    emb_ret = make_embedder(cfg.embed_model, cfg.max_seq_len)
    emb_sel = make_embedder(cfg.sel_model, cfg.max_seq_len) if cfg.sel_model else None
    return [build_pool_real(d, cfg, emb_ret, emb_sel) for d in cfg.domains]


# --------------------------------------------------------------------------- synthetic
def _unit(x):
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def make_synthetic_pools(n_domains=3, n_docs=400, n_queries=80, dim=32, n_clusters=8,
                         rel_per_query=3, cross_embedder=False, seed=0) -> List[Pool]:
    """Clustered Gaussian pools with topical queries. Used for unit tests and the demo notebook.

    Dense clusters hold more documents than sparse ones, so structural-centrality
    policies have something to exploit. NOT a substitute for the real BEIR benchmark.
    """
    rng = np.random.RandomState(seed)
    pools = []
    for di in range(n_domains):
        centers = _unit(rng.randn(n_clusters, dim) + 3 * rng.randn(1, dim))
        sizes = rng.dirichlet(np.ones(n_clusters) * 0.7) * n_docs
        lab = np.repeat(np.arange(n_clusters), np.maximum(sizes.round().astype(int), 1))
        lab = np.concatenate([lab, rng.randint(0, n_clusters, max(0, n_docs - len(lab)))])[:n_docs]
        spread = 0.25 + 0.5 * rng.rand(n_clusters)
        D = _unit(centers[lab] + spread[lab, None] * rng.randn(n_docs, dim) / np.sqrt(dim) * 3)
        Q, qrels = [], {}
        for qi in range(n_queries):
            c = rng.randint(n_clusters)
            members = np.where(lab == c)[0]
            if len(members) < rel_per_query:
                members = np.arange(n_docs)
            rel = rng.choice(members, size=rel_per_query, replace=False)
            Q.append(D[rel].mean(0) + 0.1 * rng.randn(dim) / np.sqrt(dim))
            qrels[qi] = [(int(d), 1.0) for d in rel]
        Q = _unit(np.array(Q))
        Ds = Qs = None
        if cross_embedder:
            W = np.linalg.qr(rng.randn(dim, dim))[0]
            Ds, Qs = _unit(D @ W + 0.05 * rng.randn(n_docs, dim)), _unit(Q @ W)
        pools.append(Pool(f"synth{di}", D, Q, qrels, {}, n_docs, Ds, Qs))
    return pools
