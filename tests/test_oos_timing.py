"""Out-of-sample forecasts (time order, Clark-West) and the timing rules (costs, baselines)."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from first_half_hour import oos, timing


def _table(seed=0, start=date(2011, 1, 3), end=date(2015, 6, 30), beta=0.1):
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    rng = np.random.default_rng(seed)
    r1 = rng.normal(0, 0.005, len(days))
    r12 = rng.normal(0, 0.003, len(days))
    r13 = beta * r1 + rng.normal(0, 0.002, len(days))
    return pl.DataFrame({"session": days, "r1": r1, "r12": r12, "r13": r13})


def test_forecasts_use_only_earlier_months():
    t = _table()
    base = oos.recursive_forecasts(t)
    cut = date(2014, 7, 1)
    # Perturb every column from July 2014 on: forecasts through June 2014 must not move.
    noisy = t.with_columns(
        [pl.when(pl.col("session") >= cut).then(pl.col(c) * 7 + 0.05).otherwise(pl.col(c)).alias(c)
         for c in ("r1", "r12", "r13")]
    )  # fmt: skip
    early = pl.col("session") < cut
    assert base.filter(early).equals(oos.recursive_forecasts(noisy).filter(early))
    # Perturbing only r13 (the outcome) within July 2014 leaves July's forecasts unchanged.
    y_only = t.with_columns(
        r13=pl.when(pl.col("session") >= cut).then(pl.col("r13") + 1).otherwise(pl.col("r13"))
    )
    july = pl.col("month") == cut
    got = oos.recursive_forecasts(y_only).filter(july).drop("y")
    assert base.filter(july).drop("y").equals(got)


def test_each_month_is_refit_on_all_earlier_data_only():
    t = _table()
    all_fc = oos.recursive_forecasts(t)
    # Two months: the later one must use a longer window (monthly refit, not a fixed window).
    n_train = {}
    for month in (date(2014, 1, 1), date(2015, 3, 1)):
        fc = all_fc.filter((pl.col("spec") == "r13~r1") & (pl.col("month") == month))
        past = t.filter(pl.col("session") < month)
        assert fc["n_train"].unique().to_list() == [past.height]
        n_train[month] = past.height
        # Independent route: numpy least squares on the past rows only.
        slope, icpt = np.polyfit(past["r1"].to_numpy(), past["r13"].to_numpy(), 1)
        month_r1 = t.filter(pl.col("session").is_in(fc["session"].implode()))["r1"].to_numpy()
        np.testing.assert_allclose(fc["yhat"].to_numpy(), icpt + slope * month_r1, rtol=1e-9)
        assert np.allclose(fc["mean_fc"].to_numpy(), past["r13"].mean())
    assert n_train[date(2015, 3, 1)] > n_train[date(2014, 1, 1)] + 300


def test_null_rows_are_dropped_and_counted():
    t = _table().with_columns(
        r12=pl.when(pl.col("session") == date(2014, 5, 6)).then(None).otherwise(pl.col("r12"))
    )
    fc = oos.recursive_forecasts(t)
    assert fc["n_dropped_null"].unique().to_list() == [1]
    assert oos.oos_summary(fc)["n_dropped_null"].unique().to_list() == [1]


def test_clark_west_hand_worked_case():
    # y = [2, 0, 1], model = [1, 1, 1], benchmark = 0.
    # f_t = (y-b)^2 - (y-yhat)^2 + (b-yhat)^2 = [4-1+1, 0-1+1, 1-0+1] = [4, 0, 2]; mean 2.
    # Lag 0: var(mean) = sum((f-2)^2)/T^2 = (4+4+0)/9, se = sqrt(8)/3, t = 2/se = 3/sqrt(2) = 2.1213.
    # One-sided p = 1 - Phi(2.1213) = 0.01696 (normal table: Phi(2.12)=.98300, Phi(2.13)=.98341).
    t, p = oos.clark_west(np.array([2.0, 0, 1]), np.ones(3), np.zeros(3), lags=0)
    assert abs(t - 3 / np.sqrt(2)) < 1e-9
    assert abs(p - 0.01695) < 1e-4


def test_oos_summary_r2_against_mean_and_zero():
    s = oos.oos_summary(oos.recursive_forecasts(_table(beta=0.3)))
    row = s.filter(pl.col("spec") == "r13~r1").row(0, named=True)
    assert row["r2_os_vs_mean"] > 0 and row["cw_p_vs_mean"] < 0.05  # planted slope is found
    assert row["baseline_r2"] == 0.0


def test_r2_os_and_clark_west_benchmarks_hand_worked():
    # One spec, 4 forecasts. y = [2,0,1,3], model yhat = [1,1,1,2], expanding mean mu = [1,1,2,2].
    # SSE_model = 1+1+0+1 = 3. SSE_mean = (2-1)^2+(0-1)^2+(1-2)^2+(3-2)^2 = 4 -> R2 = 1-3/4 = 0.25.
    # (Against the test-period mean 1.5 the SSE would be 5, giving 0.4: the look-ahead trap.)
    # SSE_zero = 4+0+1+9 = 14 -> R2 = 1-3/14 = 11/14.
    # CW f_t = (y-b)^2 - (y-yhat)^2 + (b-yhat)^2.
    #   b = mu:   [1-1+0, 1-1+0, 1-0+1, 1-1+0] = [0, 0, 2, 0]  -> mean 0.5
    #   b = zero: [4-1+1, 0-1+1, 1-0+1, 9-1+4] = [4, 0, 2, 12] -> mean 4.5
    fc = pl.DataFrame(
        {
            "spec": "r13~r1",
            "session": [date(2014, 1, d) for d in (2, 3, 6, 7)],
            "y": [2.0, 0.0, 1.0, 3.0],
            "yhat": [1.0, 1.0, 1.0, 2.0],
            "mean_fc": [1.0, 1.0, 2.0, 2.0],
            "n_dropped_null": 0,
        }
    )
    row = oos.oos_summary(fc).row(0, named=True)
    assert abs(row["r2_os_vs_mean"] - 0.25) < 1e-12
    assert abs(row["r2_os_vs_zero"] - 11 / 14) < 1e-12
    # Each CW statistic is the hand-worked f series pushed through the same HAC regression;
    # t has the sign of mean(f) and the two benchmarks must give different statistics.
    y, yh, mu = fc["y"].to_numpy(), fc["yhat"].to_numpy(), fc["mean_fc"].to_numpy()
    lags = row["nw_lags"]
    f_mean, f_zero = np.array([0, 0, 2.0, 0]), np.array([4, 0, 2.0, 12])
    for key, f in (("cw_t_vs_mean", f_mean), ("cw_t_vs_zero", f_zero)):
        # Bartlett long-run variance of the mean of f with `lags` lags (statsmodels uses the
        # small-sample factor T/(T-1) for a constant-only HAC fit).
        e = f - f.mean()
        lrv = (e @ e) + 2 * sum((1 - k / (lags + 1)) * (e[k:] @ e[:-k]) for k in range(1, lags + 1))
        se = np.sqrt(lrv) / len(f)
        assert abs(row[key] - f.mean() / se) < 1e-9, key
    assert abs(row["cw_t_vs_mean"] - row["cw_t_vs_zero"]) > 0.1
    assert y.shape == yh.shape == mu.shape


def test_sharpe_fixed_value_and_annualisation():
    # r = [.0098, .0198, -.0302, .0048]: mean .00105; squared deviations from the mean sum to
    # .00141875, so sd (ddof 1) = sqrt(.00141875/3) = .021747; Sharpe = .00105/.021747*sqrt(252)
    # = 0.76647 (sqrt(365) would give 0.92245).
    r = np.array([0.0098, 0.0198, -0.0302, 0.0048])
    assert abs(timing.sharpe(r) - 0.766474) < 1e-5


def test_sharpe_ci_width_matches_theory():
    # iid normal daily returns, T = 4000, mean 0.0005, sd 0.01. The annualised Sharpe has
    # se = sqrt(252/T) * sqrt(1 + SR_daily^2/2) = sqrt(0.063) * 1.001 = 0.2513, so the 95% interval
    # is about 2 * 1.96 * 0.2513 = 0.985 wide. A widened (+-1) or a degenerate interval fails.
    r = np.random.default_rng(3).normal(0.0005, 0.01, 4000)
    lo, hi = timing.sharpe_ci(r, reps=1000)
    assert lo < timing.sharpe(r) < hi
    assert 0.8 < hi - lo < 1.2


def test_paired_difference_uses_the_same_resampled_days():
    # b = 0.5 * a has exactly the same Sharpe, so a paired bootstrap difference is 0 in every
    # replication (an independent bootstrap of the two series would not be).
    a = np.random.default_rng(4).normal(0.0004, 0.01, 500)
    delta, lo, hi = timing.paired_sharpe_diff_ci(a, 0.5 * a, reps=200)
    assert abs(delta) < 1e-12 and abs(lo) < 1e-9 and abs(hi) < 1e-9


# Table below at 1 bp, 100 days (25 repeats of the 4-day block; sd uses ddof 1 so sumsq/99):
# one-signal mean .00105, sumsq 25*.00141875 -> Sharpe .00105/sqrt(25*.00141875/99)*sqrt(252) = 0.88061
# always-long mean .00605, sumsq 25*.00126875 -> Sharpe 5.36556; difference -4.48495.
ONE1, LONG1 = 0.88061, 5.36556


def _timing_table(rep=25):
    # r13 = +1%, -2%, +3%, +0.5%; r1 sign + - - +; r12 sign + - + +. Repeated so the bootstrap runs.
    days = [date(2014, 1, 1) + timedelta(i) for i in range(1, 4 * rep + 1)]
    return pl.DataFrame(
        {
            "session": days,
            "r1": [0.01, -0.01, -0.01, 0.01] * rep,
            "r12": [0.01, -0.01, 0.01, 0.01] * rep,
            "r13": [0.01, -0.02, 0.03, 0.005] * rep,
        }
    )


def test_timing_pnl_with_costs_hand_worked():
    # Positions: one-signal [+1,-1,-1,+1]; two-signal [+1,-1,0,+1] (day 3: r1<=0 but r12>0, flat).
    # Cost 1 bp per side -> 2 bp = 2e-4 per active day.
    # One-signal net = [.0098, .0198, -.0302, .0048], mean = .0042/4 = .00105, hits 3 of 4.
    # Two-signal net = [.0098, .0198, 0, .0048], mean = .0344/4 = .0086, hits 3 of 3 active.
    # Break-even per side = mean gross per active day / 2: one = (.00125)/2 = 6.25 bp;
    # two = ((.01+.02+.005)/3)/2 = 58.33 bp. Always-long gross mean (.01-.02+.03+.005)/4 = .00625.
    res = timing.timing_results(_timing_table(), samples=("POST",), reps=50)
    one = res.filter((pl.col("rule") == "one_signal") & (pl.col("cost_bp") == 1.0)).row(
        0, named=True
    )
    two = res.filter((pl.col("rule") == "two_signal") & (pl.col("cost_bp") == 1.0)).row(
        0, named=True
    )
    assert abs(one["mean"] - 0.00105) < 1e-12 and abs(one["hit_rate"] - 0.75) < 1e-12
    assert abs(two["mean"] - 0.0086) < 1e-12 and two["hit_rate"] == 1.0
    assert two["n_active"] == 75 and one["n_active"] == 100
    assert abs(one["break_even_bp"] - 6.25) < 1e-9 and abs(two["break_even_bp"] - 175 / 3) < 1e-9
    long0 = res.filter((pl.col("rule") == "always_long") & (pl.col("cost_bp") == 0.0)).row(
        0, named=True
    )
    assert abs(long0["mean"] - 0.00625) < 1e-12
    assert one["sharpe_ci_lo"] <= one["sharpe_ci_hi"]
    assert abs(one["sharpe"] - ONE1) < 1e-4 and abs(one["sharpe_always_long"] - LONG1) < 1e-4
    # Always-long net at 1 bp = [.0098,-.0202,.0298,.0048]: 3 of 4 days win, as for one-signal;
    # on the two-signal active days 1, 2, 4 it wins 2 of 3.
    assert one["baseline_hit"] == 0.75 and abs(two["baseline_hit"] - 2 / 3) < 1e-12
    # Paired row: Sharpe difference = ONE1 - LONG1.
    diff = res.filter(
        (pl.col("rule") == "one_signal_minus_always_long") & (pl.col("cost_bp") == 1.0)
    ).row(0, named=True)
    assert abs(diff["sharpe"] - (ONE1 - LONG1)) < 1e-4
    assert diff["sharpe_ci_lo"] <= diff["sharpe"] <= diff["sharpe_ci_hi"]
    assert "costs" in diff["note"] and diff["n_dropped_null"] == 0


def test_no_break_even_when_the_gross_mean_is_not_positive():
    # r1 sign + + + -; r13 = -1%, -2%, -3%, +2%: one-signal gross P&L = [-.01,-.02,-.03,-.02] < 0.
    t = _timing_table().with_columns(
        r1=pl.Series([0.01, 0.01, 0.01, -0.01] * 25),
        r13=pl.Series([-0.01, -0.02, -0.03, 0.02] * 25),
    )
    res = timing.timing_results(t, samples=("POST",), reps=20)
    one = res.filter((pl.col("rule") == "one_signal") & (pl.col("cost_bp") == 0.0)).row(
        0, named=True
    )
    assert one["break_even_bp"] is None and one["note"] == "no break-even: gross mean <= 0"


def test_timing_counts_dropped_null_rows():
    t = _timing_table().with_columns(
        r1=pl.when(pl.col("session") == date(2014, 1, 2)).then(None).otherwise(pl.col("r1"))
    )
    res = timing.timing_results(t, samples=("POST",), reps=20)
    assert res["n_dropped_null"].unique().to_list() == [1]


def test_baselines_use_the_same_days():
    t = _table(start=date(2014, 1, 2), end=date(2014, 12, 31))
    t = t.with_columns(
        r1=pl.when(pl.col("session").dt.day() == 7).then(None).otherwise(pl.col("r1"))
    )
    res = timing.timing_results(t, samples=("POST",), reps=20)
    assert res["n_days"].unique().to_list() == [t.drop_nulls().height]
    long0 = res.filter((pl.col("rule") == "always_long") & (pl.col("cost_bp") == 0.0))["mean"][0]
    assert abs(long0 - t.drop_nulls()["r13"].mean()) < 1e-15
    assert res.filter(pl.col("rule") == "zero")["sharpe"][0] == 0.0
