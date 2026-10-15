"""Pre-declared hypothesis tests from the paper (H_a ... H_k) over a run_suite() DataFrame."""
import numpy as np
import pandas as pd
from .metrics import paired_test, holm

CAGE = "CAGE+Quota (ours)"
KC = "K-Center+Quota (stratified coreset)"


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("strategy")[["final_ndcg", "final_rep", "final_rep_cached", "bwt", "forgetting", "arrive_ms_per_doc"]].mean()
    s["mixed_0.25"] = 0.25 * s["final_rep"] + 0.75 * s["final_ndcg"]
    s["mixed_0.50"] = 0.50 * s["final_rep"] + 0.50 * s["final_ndcg"]
    return s.sort_values("mixed_0.25", ascending=False)


def verdicts(df: pd.DataFrame, margin: float = 0.01) -> dict:
    """H_a CAGE>Random (novel), H_b CAGE>Density+Quota (repeated), H_d HubDensity-Pure>Random (novel) with Holm
    correction; H_c novel non-inferiority of CAGE vs Density+Quota; H_k Random>K-Center (if present)."""
    pn = df.pivot(index="seed", columns="strategy", values="final_ndcg")
    pr = df.pivot(index="seed", columns="strategy", values="final_rep")
    ha = paired_test(pn[CAGE], pn["Random"])
    hb = paired_test(pr[CAGE], pr["Density+Quota"])
    hd = paired_test(pn["HubDensity-Pure+Quota"], pn["Random"])
    ph = holm([ha[3], hb[3], hd[3]])
    hc = paired_test(pn[CAGE], pn["Density+Quota"], alternative="two-sided")
    out = {
        "H_a": dict(diff=ha[0], ci=(ha[1], ha[2]), p_raw=ha[3], p_holm=ph[0], confirmed=bool(ph[0] < .05 and ha[1] > 0)),
        "H_b": dict(diff=hb[0], ci=(hb[1], hb[2]), p_raw=hb[3], p_holm=ph[1], confirmed=bool(ph[1] < .05 and hb[1] > 0)),
        "H_d": dict(diff=hd[0], ci=(hd[1], hd[2]), p_raw=hd[3], p_holm=ph[2], confirmed=bool(ph[2] < .05 and hd[1] > 0)),
        "H_c": dict(diff=hc[0], ci=(hc[1], hc[2]), non_inferior=bool(hc[1] > -margin)),
    }
    if KC in pn.columns:
        hk = paired_test(pn["Random"], pn[KC])
        out["H_k"] = dict(diff=hk[0], ci=(hk[1], hk[2]), p_raw=hk[3])
    return out


def print_verdicts(df: pd.DataFrame, title: str = ""):
    v = verdicts(df)
    print("=" * 80); print(title); print("=" * 80)
    for k in ("H_a", "H_b", "H_d"):
        h = v[k]
        print(f"{k}: diff={h['diff']:+.4f} [{h['ci'][0]:+.4f},{h['ci'][1]:+.4f}] raw p={h['p_raw']:.4f} "
              f"Holm p={h['p_holm']:.4f} -> {'CONFIRMED' if h['confirmed'] else 'not confirmed'}")
    h = v["H_c"]
    print(f"H_c: diff={h['diff']:+.4f} [{h['ci'][0]:+.4f},{h['ci'][1]:+.4f}] -> "
          f"{'NON-INFERIOR (margin 0.01)' if h['non_inferior'] else 'non-inferiority NOT shown'}")
    if "H_k" in v:
        h = v["H_k"]; print(f"H_k: Random - K-Center = {h['diff']:+.4f} [{h['ci'][0]:+.4f},{h['ci'][1]:+.4f}] p={h['p_raw']:.4f}")
    print(summary_table(df).round(4).to_string())
