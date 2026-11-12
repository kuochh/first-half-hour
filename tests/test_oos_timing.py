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


def test_first_forecast_month_fit_on_earlier_data_only():
    t = _table()
    fc = oos.recursive_forecasts(t).filter(
        (pl.col("spec") == "r13~r1") & (pl.col("month") == date(2014, 1, 1))
    )
    past = t.filter(pl.col("session") < date(2014, 1, 1))
    assert fc["n_train"].unique().to_list() == [past.height]
    # Independent route: numpy least squares on the past rows only.
    slope, icpt = np.polyfit(past["r1"].to_numpy(), past["r13"].to_numpy(), 1)
    month_r1 = t.filter(pl.col("session").is_in(fc["session"].implode()))["r1"].to_numpy()
    np.testing.assert_allclose(fc["yhat"].to_numpy(), icpt + slope * month_r1, rtol=1e-9)
    assert np.allclose(fc["mean_fc"].to_numpy(), past["r13"].mean())


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
