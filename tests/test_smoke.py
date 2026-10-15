import numpy as np
from dataclasses import replace
from continual_rag import Config, Episode, run_suite, REGISTRY, register_policy
from continual_rag.data import make_synthetic_pools
from continual_rag.policies.base import ScorePolicy
from continual_rag.policies.structural import knn_density, knn_indegree, hub_density_centrality
from continual_rag.metrics import holm, forgetting, bwt
from continual_rag.audit import run_audit
from continual_rag.report import verdicts

CFG = Config(domains=["a", "b", "c"], docs_per_domain=300, max_queries_per_domain=60, n_usage_per_phase=40,
             cap=450, seeds=[0, 1, 2])


def pools(cross=False):
    return make_synthetic_pools(n_domains=3, n_docs=400, n_queries=80, cross_embedder=cross)


def test_episode_is_leak_free():
    ep = Episode(pools(), CFG, seed=0)
    for t in ep.tasks:
        use = {r["q"] for r in t["usage"]}; test = {r["q"] for r in t["test"]}
        assert use.isdisjoint(test) and len(test) > 0
    assert len(np.unique(ep.doc_arr[ep.doc_arr >= 0])) == (ep.doc_arr >= 0).sum()


def test_all_policies_respect_cap_and_run():
    names = sorted(REGISTRY)
    df = run_suite(CFG, pools(), [0], names, verbose=False)
    assert set(df.strategy) == set(names)
    assert df[["final_ndcg", "final_rep", "final_rep_cached"]].notna().all().all()
    assert (df.final_rep_cached >= 0).all()


def test_cross_embedder_uses_selection_embeddings():
    cfg = replace(CFG, sel_model="dummy-e5")
    ep = Episode(pools(cross=True), cfg, 0)
    assert not np.allclose(ep.DSEL, ep.DEMB)
    df = run_suite(cfg, pools(cross=True), [0], ["Random", "CAGE+Quota (ours)"], verbose=False)
    assert len(df) == 2


def test_structural_priors_shapes_and_ranges():
    E = pools()[0].D[:100]
    assert knn_density(E).shape == (100,) and knn_indegree(E).sum() == 100 * 10
    C = hub_density_centrality(E)
    assert C.min() > 0 and C.max() <= 1.0


def test_cage_rho0_equals_hubdensity_pure():
    ps = pools()
    df = run_suite(CFG, ps, [0], ["HubDensity-Pure+Quota"], verbose=False)
    register_policy("CAGE-rho0")(lambda ep, cap, fb, seed: REGISTRY["CAGE+Quota (ours)"](ep, cap, fb, seed))
    from continual_rag.policies.cage import CAGEPolicy
    register_policy("CAGE-rho0")(lambda ep, cap, fb, seed: CAGEPolicy(ep, cap, fb, seed, rho=0.0))
    df2 = run_suite(CFG, ps, [0], ["CAGE-rho0"], verbose=False)
    assert abs(df.final_ndcg[0] - df2.final_ndcg[0]) < 1e-9


def test_custom_policy_from_readme():
    def my_score(E, Q, w):
        return knn_density(E, k=5) + 0.5 * (knn_indegree(E, k=10) / len(E))
    register_policy("MyPolicy+Quota")(
        lambda ep, cap, fb, seed: ScorePolicy(ep, cap, fb, seed, my_score, per_domain=True, quota=True))
    df = run_suite(CFG, pools(), [0], ["MyPolicy+Quota"], verbose=False)
    assert df.final_ndcg[0] > 0


def test_metrics():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    R = np.array([[.5, 0, 0], [.4, .6, 0], [.3, .5, .7]])
    assert abs(bwt(R) - np.mean([.3 - .5, .5 - .6])) < 1e-9
    assert abs(forgetting(R) - np.mean([.5 - .3, .6 - .5])) < 1e-9


def test_audit_and_verdicts():
    ps = pools()
    a = run_audit(CFG, ps, [0, 1])
    assert "ALL" in a.columns and a.loc["base_rate", "ALL"] > 0
    names = ["Random", "Density+Quota", "HubDensity-Pure+Quota", "CAGE+Quota (ours)"]
    df = run_suite(CFG, ps, range(8), names, verbose=False)
    v = verdicts(df)
    assert {"H_a", "H_b", "H_c", "H_d"} <= set(v)
