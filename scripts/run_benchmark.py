#!/usr/bin/env python
"""Run one of the paper's three confirmatory regimes on real BEIR data (needs `pip install -e .[embed]`).

    python scripts/run_benchmark.py --regime base5-same
    python scripts/run_benchmark.py --regime base5-cross --seeds 600 614
    python scripts/run_benchmark.py --regime ext8 --out results/my_run

BEIR zips are downloaded from the UKP server on first use and embeddings are cached in ./cache.
"""
import argparse, os
from dataclasses import replace
from continual_rag import Config, DOMAIN_SETS, run_suite
from continual_rag.data import build_pools
from continual_rag.policies import CAGE_SET, CORESET, EXT8_SET
from continual_rag.report import print_verdicts, summary_table

REGIMES = {   # name: (domains, selection model, cap, default seeds (inclusive), strategies, output dir)
    "base5-same":  ("base5", None, 2000, (600, 614), CAGE_SET | CORESET, "base5_same_embedder"),
    "base5-cross": ("base5", "intfloat/e5-small-v2", 2000, (600, 614), CAGE_SET | CORESET, "base5_cross_embedder"),
    "ext8":        ("extended8", "intfloat/e5-small-v2", 3200, (700, 719), EXT8_SET, "extended8_stream"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--seeds", nargs=2, type=int, metavar=("FIRST", "LAST"), help="inclusive seed range")
    ap.add_argument("--feedback", choices=["relevance", "retrieval"], default="relevance")
    ap.add_argument("--out", default=None, help="output directory (default: results/<regime dir>_rerun)")
    a = ap.parse_args()

    dom, sel, cap, seeds, strategies, outname = REGIMES[a.regime]
    lo, hi = a.seeds or seeds
    cfg = Config(domains=list(DOMAIN_SETS[dom]), sel_model=sel, cap=cap)
    pools = build_pools(cfg)
    df = run_suite(cfg, pools, range(lo, hi + 1), sorted(strategies), fb=a.feedback)
    out = a.out or os.path.join("results", outname + "_rerun")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    summary_table(df).to_csv(os.path.join(out, "summary_table.csv"))
    print_verdicts(df, f"{a.regime}  seeds {lo}-{hi}")
