"""Part 2 models, rules, bootstrap and verdict on a fake long frame (no raw data)."""

from datetime import date, timedelta

import mlflow
import numpy as np
import polars as pl
import pytest

from first_half_hour import config, part2

GROUPS = {
    "signal": ["abs_s", "intra"],
    "volatility": ["rv", "vix"],
    "volume": ["vol_rel"],
    "calendar": ["fomc"],
    "path": ["last30"],
}
FEATS = [f for g in GROUPS.values() for f in g]
TAUS = ["14:00", "15:30"]


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in {
        "PART2_GROUPS": GROUPS, "PART2_FEATURES": FEATS,
        "PART2_TIMES": tuple((t, 0) for t in TAUS),
        "PART2_COSTS_RT": (0.0, 0.0002, 0.0004), "PART2_TEST_START": date(2018, 1, 2),
        "PART2_BOOT_REPS": 200,
    }.items():  # fmt: skip
        monkeypatch.setattr(config, k, v, raising=False)
    monkeypatch.setattr(config, "PART2_HGB_GRID", ((2, 20), (3, 20)))


def fake_frame(seed=0, start=date(2015, 1, 1), end=date(2021, 12, 31), edge=0.0):
    """Long frame: g = d * r13 with r13 = noise + edge * (a feature), so edge=0 is no skill."""
    rng = np.random.default_rng(seed)
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    parts = []
    for tau in TAUS:
        n = len(days)
        x = rng.standard_normal((n, len(FEATS)))
        s = rng.standard_normal(n) * 0.005
        d = np.where(s >= 0, 1, -1)
        r13 = rng.standard_normal(n) * 0.001 + edge * x[:, 0] * 0.001
        g = d * r13
        df = pl.DataFrame(
            {"session": days, "tau": tau, "s": s, "d": d, "g": g, "r13": r13,
             **{f: x[:, i] for i, f in enumerate(FEATS)}}
        ).with_columns(pl.col("session").dt.year().alias("year"))  # fmt: skip
        for bp in (0, 2, 4):
            df = df.with_columns((pl.col("g") > bp * 1e-4).cast(pl.Int64).alias(f"y_{bp}"))
        parts.append(df)
    return pl.concat(parts).sort("tau", "session")


def test_canary_trips_auc_alarm(monkeypatch):
    f = fake_frame().with_columns(canary=pl.col("g") + 1e-5)  # leak: the payoff itself
    f = f.with_columns(y_2=(pl.col("g") > 0.0002).cast(pl.Int64))
    with pytest.raises(AssertionError, match="leak alarm"):
        part2.walk_forward(f, "15:30", 2, "logit", [*FEATS, "canary"], "all")
    # no leak, no alarm
    fc = part2.walk_forward(f, "15:30", 2, "logit", FEATS, "all")
    assert set(fc["kind"]) == {"pre", "test"} and fc["year"].min() == 2017


def test_forecasts_before_a_cut_do_not_move_and_pick_reads_validation_only():
    f = fake_frame()
    cut = date(2020, 1, 1)
    rng = np.random.default_rng(1)
    late = pl.col("session") >= cut
    g = f.with_columns(
        *[pl.when(late).then(pl.Series(rng.standard_normal(f.height))).otherwise(pl.col(c)).alias(c) for c in FEATS],
        y_2=pl.when(late).then(1 - pl.col("y_2")).otherwise(pl.col("y_2")),
    )  # fmt: skip
    for model in part2.MODELS:
        a = part2.walk_forward(f, "15:30", 2, model, FEATS, "all")
        b = part2.walk_forward(g, "15:30", 2, model, FEATS, "all")
        a, b = (x.filter(pl.col("session") < cut).sort("kind", "session") for x in (a, b))
        assert a.height > 0 and a.equals(b)  # bit-identical, incl. the picked option


def test_cutoff_is_past_only_and_coverage_is_q():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    seq = np.array([100.0, -50.0, 0.0])
    cut = part2.past_cuts(base, seq, (0.5,))
    assert cut[0, 0] == 2.5  # median of base only; the first test value is not in its own pool
    assert cut[1, 0] == 3.0  # base + [100]
    seq2 = seq.copy()
    seq2[2] = 1e9
    assert np.array_equal(
        part2.past_cuts(base, seq2, (0.5,))[:2], cut[:2]
    )  # later rows do not matter
    rng = np.random.default_rng(3)
    p = rng.random(3000)
    c = part2.past_cuts(p[:300], p[300:], (0.1, 0.2, 0.5))
    cov = (p[300:, None] >= c).mean(0)
    assert np.allclose(cov, [0.1, 0.2, 0.5], atol=0.03)


def test_run_all_rules_paired_se_and_counts():
    f = fake_frame(edge=0.0)
    res = part2.run_all(f, n_jobs=1, reps=300)
    r = res["rules"]
    s = res["summary"]
    assert s["verdict"] in {"amplified", "partly amplified", "not amplified"}
    # 2 taus x 3 c x 2 models cells, each with an 'all' and a 'vol_only' feature set
    assert s["n_cells"] == 2 * 3 * 2 * 2
    # always-follow mean net = mean(g) - c over the same test days (independent route: polars)
    te = f.filter((pl.col("tau") == "15:30") & (pl.col("year") >= 2018))
    af = r.filter((pl.col("rule") == "always_follow") & (pl.col("tau") == "15:30")
                  & (pl.col("c_eval_bp") == 2) & (pl.col("split") == "all"))  # fmt: skip
    assert np.isclose(af["mean_net"][0], te["g"].mean() - 0.0002)
    assert af["n_trades"][0] == te.height and af["coverage"][0] == 1.0
    m = r.filter((pl.col("rule") == "model") & (pl.col("q") == 0.2) & (pl.col("split") == "all"))
    assert (m["diff_vs_af_se"] > 0).all() and (m["diff_vs_bm_se"] > 0).all()
    af_all = r.filter((pl.col("rule") == "always_follow") & (pl.col("split") == "all")).select(
        "tau", "c_bp", "model", "c_eval_bp", af_mean="mean_net"
    )
    mm = m.join(af_all, on=["tau", "c_bp", "model", "c_eval_bp"])
    assert mm.height == m.height and np.allclose(mm["diff_vs_af"], mm["mean_net"] - mm["af_mean"])
    # no skill: coverage near q, quintile AUC near 0.5, no AUC alarm
    assert abs(m["coverage"].mean() - 0.2) < 0.06 and res["grid"]["auc"].max() < 0.6
    cum = res["cumulative"]
    assert set(cum["rule"]) == {"model", "always_follow", "big_move"}
    first = cum.filter(pl.col("rule") == "always_follow")["wealth"][0]
    assert np.isclose(first, 1 + te["g"][0] - 0.0002)
    assert res["quintiles"].filter(pl.col("tau") == "15:30")["n"].sum() == 3 * 2 * te.height
    assert res["groups"].height == 0 if not s["skill"] else res["groups"].height > 0


def test_verdict_three_outcomes():
    assert part2.verdict(3, 3, 3, 0.1, 0.01) == "amplified"
    assert part2.verdict(1, 5, 5, 0.001, 0.01) == "not amplified"  # groups alike, d_af small
    assert part2.verdict(3, 1, 3, 0.1, 0.01) == "partly amplified"
    assert part2.verdict(1, 1, 1, 0.1, 0.01) == "not amplified"  # groups differ but no gain vs af


def test_log_runs_counts_and_outputs(tmp_path):
    f = fake_frame().filter(pl.col("year") <= 2019)
    res = part2.run_all(f, n_jobs=1, reps=100, first_test_year=2018)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("part2-test")
    n_parent, n_child = part2.log_runs(res)
    assert (
        n_parent == res["summary"]["n_cells"] and n_child == n_parent * 2
    )  # test years 2018, 2019
    part2.write_outputs(res, tmp_path / "outputs")
    assert (tmp_path / "outputs" / "part2" / "cumulative.parquet").exists()
    assert (tmp_path / "outputs" / "part2_summary.json").exists()


def test_results_do_not_depend_on_worker_count():
    f = fake_frame().filter(pl.col("year") <= 2019)
    a = part2.run_all(f, n_jobs=1, reps=50, first_test_year=2018)
    b = part2.run_all(f, n_jobs=2, reps=50, first_test_year=2018)
    k = ["tau", "c_bp", "model", "feature_set", "kind", "session"]
    assert a["forecasts"].sort(k).equals(b["forecasts"].sort(k))
    assert a["rules"].equals(b["rules"])
