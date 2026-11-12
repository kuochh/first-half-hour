"""Chapter 1 regressions: lag rule, OLS against numpy, a planted break, robustness rows, the figure."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from first_half_hour import calendar, config, controls, regress

N_SEED = 11


def _table(
    slope_pre, slope_post=None, break_at=None, start=date(2011, 1, 3), end=date(2023, 12, 31)
):
    """r13 = slope * r1 + noise on every session; the slope changes at `break_at`."""
    cal = calendar.sessions(start, end)
    rng = np.random.default_rng(N_SEED)
    n = cal.height
    r1 = rng.normal(0, 0.006, n)
    post = np.zeros(n, bool) if break_at is None else (cal["session"] >= break_at).to_numpy()
    slope = np.where(post, slope_pre if slope_post is None else slope_post, slope_pre)
    r_on = r1 * 0.6
    return cal.select("session", "is_half_day").with_columns(
        r1=pl.Series(r1),
        r_on=pl.Series(r_on),
        r_fh=pl.Series(r1 - r_on),
        r12=pl.Series(rng.normal(0, 0.003, n)),
        rod=pl.Series(r1 + rng.normal(0, 0.004, n)),
        r13=pl.Series(slope * r1 + rng.normal(0, 0.003, n)),
    )


def test_newey_west_lag_rule():
    # floor(4 (T/100)^(2/9)) worked by hand: T=100 -> 4; T=1600 -> 4 * 2^(8/9) = 7.41 -> 7;
    # T=252 -> 4 * 2.52^(2/9) = 4.92 -> 4.
    assert (config.nw_lags(100), config.nw_lags(1600), config.nw_lags(252)) == (4, 7, 4)


def test_item1_matches_numpy_and_carries_the_baselines():
    table = _table(0.3)
    out = regress.item1(table).filter(
        (pl.col("sample") == "FULL") & (pl.col("variant") == "r13~r1")
    )
    full = table.filter(pl.col("session").is_between(*config.SAMPLES["FULL"]))
    x, y = full["r1"].to_numpy(), full["r13"].to_numpy()
    slope = np.polyfit(x, y, 1)[0]  # independent route: least squares without statsmodels
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    row = out.row(0, named=True)
    assert row["term"] == "r1" and row["T"] == full.height
    assert row["coef"] == pytest.approx(slope, rel=1e-9) and row["r2"] == pytest.approx(
        r2, rel=1e-9
    )
    assert row["nw_lags"] == config.nw_lags(full.height) and row["baseline_coef"] == 0.0
    assert row["baseline_r2"] == 0.0
    # Planted slope 0.3: within 4 standard errors of sd(noise) / (sd(r1) sqrt(T)).
    assert abs(row["coef"] - 0.3) < 4 * 0.003 / (0.006 * np.sqrt(full.height))
    assert regress.item1(table).height == 3 * (1 + 1 + 2)  # 3 samples x (r1, r12, r1 + r12)


def test_chow_test_detects_a_planted_break_and_not_a_flat_slope():
    brk = config.BREAK_DATES[0]
    broken = regress.item4(_table(0.5, 0.0, brk))
    joint = broken.filter(pl.col("term") == "D+D*r1")
    p_by = dict(zip(joint["variant"].to_list(), joint["p"].to_list(), strict=True))
    assert p_by[f"break_{brk.isoformat()}"] < 1e-6  # slope falls from 0.5 to 0 at 2014-01-02
    flat = regress.item4(_table(0.3))
    joint_flat = flat.filter(pl.col("term") == "D+D*r1")
    assert (joint_flat["p"] > 0.01).all()  # no break planted: fixed seed, p-values far from 0


def test_item2_variants_and_the_missing_dividend_file(tmp_path):
    table = _table(0.4)
    with pytest.raises(AssertionError, match="missing dividends"):
        regress.item2(table, tmp_path)
    ex = table["session"][[100, 500, 900]].to_list()
    pl.DataFrame({"ex_date": ex, "amount": [0.5] * 3}).write_csv(tmp_path / "spy_dividends.csv")
    out = regress.item2(table, tmp_path, reps=30).filter(pl.col("sample") == "FULL")
    full = table.filter(pl.col("session").is_between(*config.SAMPLES["FULL"]))
    by = {r["variant"]: r for r in out.iter_rows(named=True)}
    assert set(by) == {"huber", "median", "trim_top_1%", "trim_top_3%", "trim_top_5%",
                       "log_returns", "no_half_days", "no_ex_dividend"}  # fmt: skip
    assert by["no_ex_dividend"]["T"] == full.height - 3
    assert by["no_half_days"]["T"] == full.height - full["is_half_day"].sum()
    # trimming the top 3% of |r1|: at most 3% of rows go (ties aside), at least 2.9%
    cut = full.height - by["trim_top_3%"]["T"]
    assert 0.029 * full.height <= cut <= 0.031 * full.height
    for r in by.values():
        assert r["coef"] == pytest.approx(0.4, abs=0.05) and r["baseline_coef"] == 0.0


def test_rolling_slope_window_and_figure(tmp_path):
    table = _table(0.3)
    roll = regress.rolling_slope(table)
    usable = table.filter(pl.col("session").is_between(*config.SAMPLES["FULL"])).height
    assert roll.height == usable - config.ROLLING_WINDOW + 1
    w = config.ROLLING_WINDOW  # first window by numpy, independently
    full = table.filter(pl.col("session").is_between(*config.SAMPLES["FULL"]))
    first = np.polyfit(full["r1"].to_numpy()[:w], full["r13"].to_numpy()[:w], 1)[0]
    assert roll["slope"][0] == pytest.approx(first, rel=1e-9)
    assert roll["end_session"][0] == full["session"][w - 1]
    assert (roll["lo"] < roll["slope"]).all() and (roll["slope"] < roll["hi"]).all()
    png, pq = tmp_path / "figures" / "rolling_slope.png", tmp_path / "rolling_slope.parquet"
    regress.plot_rolling_slope(roll, png, pq)
    assert png.stat().st_size > 5_000 and pl.read_parquet(pq).height == roll.height


def test_item7_base_rows_match_numpy_and_interaction_drops_the_warmup(tmp_path):
    table = _table(0.3)
    n = table.height
    rng = np.random.default_rng(3)
    table = table.with_columns(
        rv_fh=pl.Series(rng.uniform(1e-6, 1e-4, n)),
        rv_day=pl.Series(rng.uniform(1e-5, 1e-3, n)),
        vol_fh=pl.Series(rng.integers(1_000, 90_000, n)),
    )
    ctrl = _fake_controls(table)
    out = regress.item7(table, ctrl).filter(pl.col("sample") == "FULL")
    # FULL starts 2011-01-03, the first session of this table: the sample is the whole table.
    base = out.filter((pl.col("variant") == "base_same_rows") & (pl.col("term") == "r1")).row(
        0, named=True
    )
    rows = table.join(ctrl, on="session").drop_nulls(["rv_day_lag", "relvol"])
    assert base["T"] == rows.height and base["n_dropped"] == n - rows.height
    assert base["coef"] == pytest.approx(np.polyfit(rows["r1"], rows["r13"], 1)[0], rel=1e-9)
    inter = out.filter(pl.col("variant") == "r1_x_top_rv_fh")
    assert inter["T"].unique().to_list() == [n - config.TERCILE_WARMUP]
    assert inter["n_dropped"].unique().to_list() == [config.TERCILE_WARMUP]
    assert set(inter["term"]) == {"r1", "rv_fh_top", "inter"}


def _fake_controls(table):
    """Same columns as controls.controls, filled from the table (the regression is under test)."""
    n = table.height
    rng = np.random.default_rng(5)
    flag = lambda p: pl.Series(rng.random(n) < p)
    out = table.select("session", "rv_fh").with_columns(
        rv_day_lag=pl.col("rv_fh").shift(1),
        relvol=pl.Series(rng.uniform(0.5, 2, n)),
        **{c: flag(0.2) for c in controls.WEEKDAY_COLUMNS},
        fomc=flag(0.05), cpi=flag(0.05), payroll=flag(0.05),
        vix_lag=pl.Series(rng.uniform(12, 30, n)), vix_ratio_lag=pl.Series(rng.uniform(0.7, 1.1, n)),
        opex_monthly=flag(0.05), opex_quarterly=flag(0.02), daily_expiry_era=pl.lit(False),
        month_end=flag(0.05), is_half_day=table["is_half_day"],
    )  # fmt: skip
    out = out.with_columns(news=pl.col("fomc") | pl.col("cpi") | pl.col("payroll"))
    w = config.TERCILE_WARMUP
    top = pl.Series(np.where(np.arange(n) < w, np.nan, rng.random(n) < 1 / 3), nan_to_null=True)
    return out.with_columns(
        rv_fh_top=top,
        relvol_top=top,
        rv_fh_cut=pl.lit(None, pl.Float64),
        relvol_cut=pl.lit(None, pl.Float64),
    )


def test_huber_and_median_rows_use_a_bootstrap_t_not_the_default_se(tmp_path):
    # AR(1) regressor and AR(1) errors: the default (iid) SE is far too small, so the default-SE
    # t of a Huber fit is much larger than a dependence-robust one.
    n = 1500
    rng = np.random.default_rng(2)
    x, e = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        x[i], e[i] = 0.8 * x[i - 1] + rng.normal(0, 0.004), 0.8 * e[i - 1] + rng.normal(0, 0.002)
    table = _table(0.0).head(n).with_columns(r1=pl.Series(x), r13=pl.Series(0.05 * x + e))
    pl.DataFrame({"ex_date": [table["session"][5]], "amount": [0.5]}).write_csv(
        tmp_path / "spy_dividends.csv"
    )
    out = regress.item2(table, tmp_path, reps=60).filter(pl.col("sample") == "FULL")
    by = {r["variant"]: r for r in out.iter_rows(named=True)}
    import statsmodels.api as sm

    d = table.filter(pl.col("session").is_between(*config.SAMPLES["FULL"]))
    X = sm.add_constant(d["r1"].to_numpy())
    fit = sm.RLM(d["r13"].to_numpy(), X, M=sm.robust.norms.HuberT()).fit()
    default_t = fit.params[1] / fit.bse[1]
    for v in ("huber", "median"):
        assert by[v]["se_kind"] == "bootstrap" and "60 reps" in by[v]["note"]
    assert by["huber"]["coef"] == pytest.approx(fit.params[1], rel=1e-9)
    assert abs(by["huber"]["t"]) < 0.6 * abs(default_t)
    assert {by[v]["se_kind"] for v in by if v not in ("huber", "median")} == {"nw"}
    assert "62 ex-dividend days" in by["no_ex_dividend"]["note"]


def test_item7_quarterly_expiry_note_and_no_exdividend_dummy():
    table = _table(0.3)
    n = table.height
    table = table.with_columns(
        rv_fh=pl.Series(np.random.default_rng(3).uniform(1e-6, 1e-4, n)),
        rv_day=pl.Series(np.random.default_rng(4).uniform(1e-5, 1e-3, n)),
        vol_fh=pl.Series(np.random.default_rng(5).integers(1_000, 90_000, n)),
    )
    out = regress.item7(table, _fake_controls(table))
    assert out["se_kind"].unique().to_list() == ["nw"]
    for variant in ("+opex_quarterly", "all"):
        notes = out.filter(pl.col("variant") == variant)["note"].drop_nulls().unique().to_list()
        assert any(
            "62 ex-dividend days are exactly the 62 quarterly expiry days" in x for x in notes
        )
    assert not any("div" in t for t in out["term"].unique().to_list())


def test_rolling_slope_marks_straddling_windows_and_labels_the_axis(tmp_path):
    roll = regress.rolling_slope(_table(0.3))
    brk = config.BREAK_DATES[0]
    assert (roll["start_session"] <= roll["end_session"]).all()
    assert (roll["end_session"].is_sorted()) and roll["start_session"].is_sorted()
    assert roll["start_session"][0] < brk < roll["end_session"].max()
    regress.plot_rolling_slope(roll, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 5_000
