"""Chapter 2 economics: trade P&L, the threshold rule's time order, the deflated Sharpe."""

from datetime import date

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from first_half_hour import economics, interpret


def test_trade_pnl_by_hand():
    # Hand arithmetic, P10 = 100, b = 0.004, cost 1 bp per side = 0.0002 round trip.
    p10 = np.full(5, 100.0)
    close = np.array([101.0, 99.5, 100.2, 99.0, 100.0])  # day 3 is a half-day close
    y = np.array([1, 0, 0, 1, 0])
    traded = np.array([True, True, True, False, False])
    r = economics.trade_returns(traded, p10, close, y, 0.004, 1.0)
    # hit: limit fills at +b, not the close; miss: close/P10 - 1; untraded: 0 even if y = 1.
    np.testing.assert_allclose(r, [0.004 - 0.0002, -0.005 - 0.0002, 0.002 - 0.0002, 0.0, 0.0])


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
    fc = pl.concat([
        pl.DataFrame({"session": sessions, "year": years, "target": "up_0.4",
                      "y": y.astype(np.int8), "model": m, "p": p})
        for m, p in (("tier:hgb", 1 / (1 + np.exp(-x[:, 1] - 1.5))), ("formula_cal", np.full(n, 0.6)))
    ])  # fmt: skip
    window = (date(2019, 1, 1), date(2020, 12, 31))
    tab = economics.economics_table(
        fc, prices, {"chosen_tier": "tier:hgb", "formula_cal": "formula_cal"}, window, reps=50
    )
    assert set(tab["rule"]) == {"chosen_tier", "formula_cal", "always_buy", "zero"}
    assert tab.filter(pl.col("rule") == "zero")["sharpe"].item() == 0.0
    ds = economics.trial_sharpes(fc, prices, ["tier:hgb", "formula_cal"], window)
    assert len(ds) == 10

    cols = ["sigma", "f1", "f2", "f3"]
    data = pl.DataFrame({"session": sessions, "year": years, "y": y.astype(np.int8)}).hstack(
        pl.DataFrame(x, schema=cols)
    )
    fit = data.filter(pl.col("year") == 2018)
    est = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(
        fit.select(cols).to_numpy(), fit["y"].to_numpy()
    )
    imp = interpret.permutation_importance_by_year({2019: est, 2020: est}, data, cols)
    assert imp["feature"][0] == "f1"  # the one feature that drives y
    assert interpret.top_other_features(imp)[0] == "f1"
    paths = interpret.write_figures(est, data, cols, imp, tmp_path)
    assert all(p.exists() for p in paths)
