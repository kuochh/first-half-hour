"""Part 3 models: class pooling, findings rule, run_all on fake per-ETF frames (no raw data)."""

from datetime import date, timedelta

import mlflow
import numpy as np
import polars as pl
import pytest

from first_half_hour import config, part2, part3

GROUPS = {
    "signal": ["abs_s", "intra"],
    "volatility": ["rv", "vix"],
    "volume": ["vol_rel"],
    "calendar": ["fomc"],
    "path": ["last30"],
}
FEATS = [f for g in GROUPS.values() for f in g]
TAUS = ["14:00", "15:30"]
CLASSES = {"cls_a": ("AAA", "BBB"), "cls_b": ("CCC",)}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in {
        "PART2_GROUPS": GROUPS, "PART2_FEATURES": FEATS,
        "PART2_TIMES": tuple((t, 0) for t in TAUS), "PART2_TEST_START": date(2018, 1, 2),
        "PART3_CLASSES": CLASSES, "PART3_PERIODS": {}, "PART3_BOOT_REPS": 100,
    }.items():  # fmt: skip
        monkeypatch.setattr(config, k, v, raising=False)


def fake_frame(symbol, seed, end=date(2019, 12, 31)):
    """Part 2 style long frame for one symbol; no skill."""
    rng = np.random.default_rng(seed)
    days = [date(2015, 1, 1) + timedelta(i) for i in range((end - date(2015, 1, 1)).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    parts = []
    for tau in TAUS:
        n = len(days)
        s = rng.standard_normal(n) * 0.005
        d = np.where(s >= 0, 1, -1)
        r13 = rng.standard_normal(n) * 0.001
        df = pl.DataFrame(
            {"session": days, "tau": tau, "s": s, "d": d, "g": d * r13, "r13": r13,
             **{f: rng.standard_normal(n) for f in FEATS}}
        ).with_columns(pl.col("session").dt.year().alias("year"))  # fmt: skip
        for bp in (0, 2, 4):
            df = df.with_columns((pl.col("g") > bp * 1e-4).cast(pl.Int64).alias(f"y_{bp}"))
        parts.append(df)
    return pl.concat(parts).sort("tau", "session").with_columns(symbol=pl.lit(symbol))


def frames():
    return {s: fake_frame(s, i) for i, s in enumerate(["AAA", "BBB", "CCC"])}


def test_pooling_matches_a_day_by_day_hand_computation():
    # two ETFs; BBB has no forecast on the first 5 days (an ETF absent that day is left out of its mean)
    rng = np.random.default_rng(5)
    days = np.datetime64("2018-01-01") + np.arange(60).astype("timedelta64[D]")
    names = [("always_follow", 1.0), ("model", 0.2), ("big_move", 0.2), ("vol_only", 0.2)]

    def etf(sl):
        n = len(days[sl])
        return {"session": days[sl], "g": rng.standard_normal(n) * 0.001,
                "flags": {k: (rng.random(n) < (1 if k[0] == "always_follow" else 0.4)) for k in names}}  # fmt: skip

    a, b = etf(slice(0, 60)), etf(slice(5, 60))
    ce = 0.0002
    rows = part3._class_rows([a, b], "c", "15:30", part2._Boot(50), (ce,), ("all",))
    # independent route: plain python loop over days
    for i, (rule, q) in enumerate(names):
        sx = sc = sh = 0.0
        xs = []
        for t in range(60):
            mem = [m for m, off in ((a, 0), (b, 5)) if t - off >= 0]
            n_t = len(mem)
            x = cov = h = 0.0
            for m, off in ((a, 0), (b, 5)):
                if t - off < 0:
                    continue
                net = m["g"][t - off] - ce
                fl = m["flags"][(rule, q)][t - off]
                x += fl * net / n_t
                cov += fl / n_t
                h += fl * (net > 0) / n_t
            xs.append(x)
            sx, sc, sh = sx + x, sc + cov, sh + h
        r = rows.filter((pl.col("rule") == rule) & (pl.col("q") == q)).row(0, named=True)
        assert np.isclose(r["mean_net"], sx / sc) and np.isclose(r["hit"], sh / sc)
        assert np.isclose(r["sharpe"], np.mean(xs) / np.std(xs, ddof=1) * np.sqrt(252))
        assert np.isclose(r["coverage"], sc / 60) and r["n_days"] == 60
    af = rows.filter(pl.col("rule") == "always_follow").row(0, named=True)
    assert af["n_etf_trades"] == 115 and af["coverage"] == 1.0
    m = rows.filter(pl.col("rule") == "model").row(0, named=True)
    assert np.isclose(m["diff_vs_af"], m["mean_net"] - af["mean_net"]) and m["diff_vs_af_se"] > 0
    assert af["diff_vs_af"] is None


def test_finding_rule():
    assert part3.finding(2.5, [0.1, 0.2, 0.3])
    assert not part3.finding(2.0, [0.1, 0.2, 0.3])  # d_af strictly above 2
    assert not part3.finding(3.0, [0.1, -0.2, 0.3])  # any robustness version negative fails
    assert not part3.finding(3.0, [0.1, 0.0, 0.3])  # zero is not positive
    assert not part3.finding(3.0, [0.1, None, 0.3])  # an unreported split fails


def test_run_all_grid_counts_logging_and_worker_independence(tmp_path):
    f = frames()
    a = part3.run_all(f, n_jobs=1, reps=60, first_test_year=2018)
    b = part3.run_all(f, n_jobs=2, reps=60, first_test_year=2018)
    k = ["symbol", "tau", "feature_set", "kind", "session"]
    assert a["forecasts"].sort(k).equals(b["forecasts"].sort(k))
    assert a["rules"].equals(b["rules"]) and a["grid"].equals(b["grid"])
    s, g, r = a["summary"], a["grid"], a["rules"]
    assert g.height == s["n_cells"] == 2 * len(TAUS) and s["n_etfs"] == 3
    assert s["n_findings"] == g["finding"].sum() == len(s["findings"])
    # the grid's d_af is the pooled model-minus-always-follow row; finding rule re-applied by hand
    for row in g.iter_rows(named=True):
        want = row["d_af"] > 2 and min(row["diff_pre"], row["diff_post"], row["diff_nocovid"]) > 0
        assert row["finding"] == want
    # the one-ETF class equals that ETF's own rule numbers (always-follow = mean g - 2 bp, by polars)
    cc = f["CCC"].filter((pl.col("tau") == "15:30") & (pl.col("year") >= 2018))
    af = r.filter((pl.col("class") == "cls_b") & (pl.col("tau") == "15:30") & (pl.col("rule") == "always_follow")
                  & (pl.col("c_eval_bp") == 2) & (pl.col("split") == "all"))  # fmt: skip
    assert (
        np.isclose(af["mean_net"][0], cc["g"].mean() - 0.0002)
        and af["n_etf_trades"][0] == cc.height
    )
    er = a["etf_rules"].filter((pl.col("symbol") == "CCC") & (pl.col("tau") == "15:30") & (pl.col("rule") == "model")
                               & (pl.col("q") == 0.2) & (pl.col("c_eval_bp") == 2))  # fmt: skip
    m = r.filter((pl.col("class") == "cls_b") & (pl.col("tau") == "15:30") & (pl.col("rule") == "model")
                 & (pl.col("q") == 0.2) & (pl.col("c_eval_bp") == 2) & (pl.col("split") == "all"))  # fmt: skip
    assert np.isclose(er["mean_net"][0], m["mean_net"][0]) and er["auc"][0] < 0.6
    # counts and MLflow: parents per (ETF, tau, feature set), children per test year (2018, 2019)
    assert s["n_parent"] == 3 * len(TAUS) * 2 and s["n_child"] == s["n_parent"] * 2
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("part3-test")
    assert part3.log_runs(a) == (s["n_parent"], s["n_child"])
    part3.write_outputs(a, tmp_path / "outputs")
    assert (tmp_path / "outputs" / "part3" / "grid.parquet").exists()
    assert (tmp_path / "outputs" / "part3_summary.json").exists()


def test_class_day_without_any_forecast_is_counted(monkeypatch):
    f = frames()
    # CCC (a class of its own) loses a test-year day that the other ETFs have
    day = f["AAA"].filter((pl.col("tau") == "15:30") & (pl.col("year") == 2019))["session"][10]
    f["CCC"] = f["CCC"].filter(~((pl.col("tau") == "15:30") & (pl.col("session") == day)))
    s = part3.run_all(f, n_jobs=1, reps=20, first_test_year=2018)["summary"]
    missing = {k: v for k, v in s["class_days_missing"].items() if v}
    assert (
        len(missing) == 1
        and next(iter(missing.values())) == 1
        and next(iter(missing)).endswith("15:30")
    )
