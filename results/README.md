# Results

Paper numbers as supplied by the author (`continualrag_beir_csv_results.zip`); they were **not** regenerated when this repository was assembled.

- `confirmatory_master.csv` - all three locked regimes in one table (paper Table 3)
- `base5_same_embedder/`, `base5_cross_embedder/`, `extended8_stream/` - the same rows split per regime (`summary_table.csv`)
- `audit.csv` - pre-eviction predictability audit (paper Table 1)

To regenerate: `python scripts/run_benchmark.py --regime {base5-same,base5-cross,ext8}`.
