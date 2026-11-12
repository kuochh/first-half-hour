"""Chapter 2 economics: trade P&L, the threshold rule's time order, the deflated Sharpe."""

from datetime import date, datetime, time

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from first_half_hour import calendar, config, economics, interpret, returns, synthetic


def test_trade_pnl_by_hand():
    # Hand arithmetic, P10 = 100, b = 0.004, cost 1 bp per side = 0.0002 round trip.
    p10 = np.full(4, 100.0)
    close = np.array([101.0, 99.5, 100.2, 99.0])
    y = np.array([1, 0, 0, 1])
    traded = np.array([True, True, True, False])
    r = economics.trade_returns(traded, p10, close, y, 0.004, 1.0)
    # hit: limit fills at +b, not the close; miss: close/P10 - 1; untraded: 0 even if y = 1.
    np.testing.assert_allclose(r, [0.004 - 0.0002, -0.005 - 0.0002, 0.002 - 0.0002, 0.0])


def test_half_day_uses_the_1259_close(tmp_path, monkeypatch):
    # Bars worked by hand: half-day 2019-11-29 has P10 = 117 (09:59 bar) and its close is the
    # 12:59 bar's close, 121; the 15:59 after-hours print (130) is not the close.
    d25, d26, d29 = date(2019, 11, 25), date(2019, 11, 26), date(2019, 11, 29)
    monkeypatch.setattr(config, "DATA_START", d25)
    monkeypatch.setattr(config, "DATA_END", d29)
    cal = calendar.sessions(d25, d29)
    rows = []
    for d, c10, tail, close in [(d25, 101.0, "15:59", 104.0), (d26, 106.0, "15:59", 110.0),
                                (d29, 117.0, "12:59", 121.0)]:  # fmt: skip
        key = ["14:59", "15:29"] if d != d29 else ["11:59", "12:29"]
        rows += [(d, "09:30", 100.0, 100.5), (d, "09:59", 100.5, c10)]
        rows += [(d, key[0], 102.0, 102.0), (d, key[1], 103.0, 103.0), (d, tail, close, close)]
    rows.append((d29, "15:59", 130.0, 130.0))
    rows.append((date(2019, 11, 27), "09:30", 111.0, 111.5))  # no key bars: dropped
    synthetic.write_bars(
        tmp_path,
        "SPY",
        pl.DataFrame(
            {
                "timestamp": [datetime.combine(d, time.fromisoformat(t)) for d, t, _, _ in rows],
                "open": [o for *_, o, _ in rows],
                "high": [max(o, c) for *_, o, c in rows],
                "low": [min(o, c) for *_, o, c in rows],
                "close": [c for *_, c in rows],
                "volume": [100] * len(rows),
            }
        ),
    )
    daily = returns.daily_table(tmp_path, "SPY", cal)
    target = pl.DataFrame(
        {"session": [d25, d26, d29], "y": [0, 1, 0], "p10": [101.0, 106.0, 117.0]}
    )
    px = economics.prices(daily, target)
    assert px["p_close"].to_list() == [104.0, 110.0, 121.0]
    # y = 0 on the half-day: the trade exits at the 12:59 close, 121 / 117 - 1, less 2 x 1 bp.
    r = economics.trade_returns(
        np.ones(3, dtype=bool), px["p10"].to_numpy(), px["p_close"].to_numpy(),
        px["y"].to_numpy(), 0.004, 1.0,
    )  # fmt: skip
    np.testing.assert_allclose(r[2], 121 / 117 - 1 - 0.0002)
    assert r[2] != 130 / 117 - 1 - 0.0002
    bad = target.with_columns(p10=pl.Series([101.0, 106.0, 118.0]))
    with pytest.raises(AssertionError, match="p10 != p_10"):
        economics.prices(daily, bad)


def _frame(year, p, y, close):
    n = len(p)
    return pl.DataFrame(
        {"session": [date(year, 1, 2 + i) for i in range(n)], "year": [year] * n, "p": p,
         "y": y, "p10": [100.0] * n, "p_close": close}
    ).with_columns(pl.col("year").cast(pl.Int32), pl.col("y").cast(pl.Int8))  # fmt: skip


def test_threshold_for_year_never_reads_year():
    # 2018 rows; gross per row: -0.01, +0.004 (hit), +0.004 (hit), +0.001. Round trip 0.0002.
    # Mean net per threshold: 0.3 -> -0.00045; 0.4 and 0.5 -> 0.0021 (tie, first wins); 0.6 ->
    # 0.00115; 0.7 -> 0.0002. So the 2019 threshold is 0.4.
    prev = _frame(2018, [0.35, 0.55, 0.65, 0.75], [0, 1, 1, 0], [99.0, 101.0, 101.0, 100.1])
    this = _frame(2019, [0.9, 0.9, 0.9, 0.9], [1, 1, 1, 1], [100.0] * 4)
    assert economics.choose_threshold(pl.concat([prev, this]), 2019) == 0.4
    wild = _frame(2019, [0.1, 0.2, 0.8, 0.8], [0, 0, 0, 0], [50.0, 60.0, 70.0, 80.0])
    assert economics.choose_threshold(pl.concat([prev, wild]), 2019) == 0.4
    # No threshold earns a positive mean net return: no trades.
    losing = _frame(2018, [0.35, 0.55, 0.65, 0.75], [0, 0, 0, 0], [99.0] * 4)
    assert economics.choose_threshold(pl.concat([losing, this]), 2019) is None


def test_strategy_positions_use_per_year_threshold():
    prev = _frame(2018, [0.35, 0.55, 0.65, 0.75], [0, 1, 1, 0], [99.0, 101.0, 101.0, 100.1])
    this = _frame(2019, [0.35, 0.45, 0.95, 0.2], [0, 0, 0, 0], [100.0] * 4)
    pos = economics.strategy_positions(
        pl.concat([prev, this]), (date(2019, 1, 1), date(2019, 12, 31))
    )
    assert pos["traded"].to_list() == [False, True, True, False]  # threshold 0.4


def test_deflated_sharpe_by_hand():
    r = np.array([0.010, -0.005, 0.020, 0.000, 0.015, -0.010, 0.005, 0.012])
    trials = np.array([0.01, 0.03, 0.05])  # daily Sharpes of N = 3 trials
    out = economics.deflated_sharpe(r, trials)
    # Written out independently: raw moments in numpy, normal quantiles from scipy.
    t = len(r)
    m = r.mean()
    sd = np.sqrt(((r - m) ** 2).sum() / (t - 1))
    sr = m / sd
    m2, m3, m4 = (((r - m) ** k).mean() for k in (2, 3, 4))
    skew, kurt = m3 / m2**1.5, m4 / m2**2
    v = ((trials - trials.mean()) ** 2).sum() / 2  # = 0.0004
    g = 0.5772156649015329
    sr0 = v**0.5 * ((1 - g) * stats.norm.ppf(1 - 1 / 3) + g * stats.norm.ppf(1 - 1 / (3 * np.e)))
    z = (sr - sr0) * (t - 1) ** 0.5 / (1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5
    assert np.isclose(sr0, 0.02 * (0.4227843 * 0.4307273 + 0.5772157 * 1.1634), atol=2e-4)
    np.testing.assert_allclose(out["sr0_daily"], sr0, rtol=1e-12)
    np.testing.assert_allclose(out["dsr"], stats.norm.cdf(z), rtol=1e-12)
    assert out["n_trials"] == 3


def test_deflated_sharpe_published_example():
    # Bailey and Lopez de Prado (2014), numerical example: annual SR 2.5, T = 1250 daily returns
    # (5 years x 250), skew -3, kurtosis 10, N = 100 trials with annual variance 0.5. Published
    # (and reproduced by hand from the paper's formulas): SR0 = 0.1132 daily, DSR = 0.9004.
    sr0, dsr = economics.deflated_sharpe_from_moments(
        sr=2.5 / np.sqrt(250), v=0.5 / 250, n=100, t=1250, skew=-3.0, kurt=10.0
    )
    assert round(sr0, 4) == 0.1132
    assert round(dsr, 4) == 0.9004


def test_deflated_sharpe_never_trading_rule():
    out = economics.deflated_sharpe(np.zeros(50), np.array([0.01, 0.02, 0.03]))
    assert out["dsr"] is None and "no trades" in out["note"]


def test_sharpe_interval_finite_when_trades_bunch():
    # 5 trades inside one year of a 1900-day window: some resamples hold no trade day (each has
    # probability about exp(-5)). The old interval was (nan, nan); now those draws have Sharpe 0
    # and are counted.
    rng = np.random.default_rng(1)
    r = np.zeros(1900)
    r[rng.choice(np.arange(600, 850), 5, replace=False)] = rng.normal(0.002, 0.004, 5)
    lo, hi, n_zero = economics.sharpe_ci0(r, 1000)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert n_zero > 0


def test_economics_table_and_interpretation(tmp_path):
    rng = np.random.default_rng(0)
    sessions = [d for d in pl.date_range(date(2018, 1, 1), date(2020, 12, 31), eager=True)
                if d.weekday() < 5]  # fmt: skip
    n = len(sessions)
    x = rng.normal(size=(n, 4))
    y = (rng.random(n) < 1 / (1 + np.exp(-x[:, 1] - 1.5))).astype(int)
    p10 = np.full(n, 100.0)
    close = 100 * (1 + rng.normal(0, 0.005, n))
    years = np.array([d.year for d in sessions], dtype=np.int32)
    prices = pl.DataFrame({"session": sessions, "p10": p10, "p_close": close, "y": y})
    names = ["t1", "t2", "t3", "formula_cal"]
    ps = {
        m: np.clip(1 / (1 + np.exp(-x[:, 1] - 1.5)) + 0.02 * i, 0, 1) for i, m in enumerate(names)
    }
    fc = pl.concat([
        pl.DataFrame({"session": sessions, "year": years, "target": "up_0.4",
                      "y": y.astype(np.int8), "model": m, "p": p})
        for m, p in ps.items()
    ])  # fmt: skip
    window = (date(2019, 1, 1), date(2020, 12, 31))
    tab = economics.economics_table(
        fc, prices, {"chosen_tier": "t1", "formula_cal": "formula_cal"}, window, reps=50
    )
    assert set(tab["rule"]) == {"chosen_tier", "formula_cal", "always_buy", "zero"}
    assert np.isfinite(tab["sharpe_ci_lo"].to_numpy()).all()
    # zero: no trades. always_buy: the same trade every session, costs every day.
    zero = tab.filter(pl.col("rule") == "zero")
    assert zero["sharpe"].item() == 0.0 and zero["n_traded"].item() == 0
    ab = tab.filter((pl.col("rule") == "always_buy") & (pl.col("cost_bp") == 1.0))
    assert ab["n_traded"].item() == ab["n_days"].item()
    win = prices.filter(pl.col("session").is_between(*window))
    by_hand = np.where(win["y"] == 1, 0.004, win["p_close"] / 100 - 1) - 0.0002
    np.testing.assert_allclose(ab["mean"].item(), by_hand.mean())
    # N is fixed at 20: 5 thresholds x 4 forecasts; the rule's returns share the 1 bp cost.
    ds = economics.trial_set(fc, prices, names, window)
    assert len(ds) == 20
    with pytest.raises(AssertionError):
        economics.trial_set(fc, prices, names[:3], window)
    r = economics.rule_returns(fc, prices, "t1", window)
    assert economics.deflated_sharpe(r, ds)["n_trials"] == 20
    # prices.y must agree with the forecasts' y; target and b must agree.
    with pytest.raises(AssertionError, match="prices.y differs"):
        economics.trial_set(fc, prices.with_columns(y=1 - pl.col("y")), names, window)
    with pytest.raises(AssertionError):
        economics.trial_set(fc, prices, names, window, b=0.002)

    cols = ["sigma", "f1", "f2", "f3"]
    data = pl.DataFrame({"session": sessions, "year": years, "y": y.astype(np.int8)}).hstack(
        pl.DataFrame(x, schema=cols)
    )
    fit = data.filter(pl.col("year") == 2018)
    est = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(
        fit.select(cols).to_numpy(), fit["y"].to_numpy()
    )
    imp = interpret.permutation_importance_by_year(
        {2019: est, 2020: est}, data, cols, ("sigma", "f2")
    )
    assert imp["feature"][0] == "f1"  # the one feature that drives y
    assert {"std", "n_years_positive"} <= set(imp.columns)
    assert imp.filter(pl.col("feature").str.starts_with("[group]")).height == 1
    assert (imp["std"] >= 0).all()
    others = interpret.top_other_features(imp)
    assert others[0] == "f1" and all(
        imp.filter(pl.col("feature") == f)["importance"].item() > 0 for f in others
    )
    with pytest.raises(AssertionError, match="differ in length"):
        interpret.permutation_importance_by_year({2019: est}, data, cols[:3])
    est.feature_names_in_ = np.array(cols[::-1])  # a model fitted in another column order
    with pytest.raises(AssertionError, match="feature order"):
        interpret.permutation_importance_by_year({2019: est}, data, cols)
    del est.feature_names_in_
    paths = interpret.write_figures(est, data, cols, imp, tmp_path)
    assert all(p.exists() for p in paths)
