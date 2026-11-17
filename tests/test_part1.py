"""Part 1: the official-close daily table on synthetic bars, past-only cut-offs, r_next_k, jumps,
the verdict arithmetic, windows, and the rows frame on a fake daily table."""

from datetime import date, time, timedelta

import numpy as np
import polars as pl
import pytest

from first_half_hour import bars, calendar, config, part1, returns, synthetic

# ------------------------------------------------------------------ synthetic bars, official close


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("p1")
    daily = root / "daily"
    daily.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DATA_START", date(2018, 1, 1))
        mp.setattr(config, "DATA_END", date(2018, 12, 31))
        cal = calendar.sessions(config.DATA_START, config.DATA_END)
        synthetic.write_synthetic(root, ["SPY"], date(2018, 1, 2), date(2018, 12, 31), 0.1, 7)
        pl.DataFrame(
            {"ex_date": [date(2018, 3, 16), date(2018, 6, 15), date(2018, 9, 21)],
             "amount": [0.5, 0.5, 0.5]}
        ).write_csv(daily / "spy_dividends.csv")  # fmt: skip
        base = returns.daily_table(root, "SPY", cal)
        synthetic.write_stub_official_close(root, daily, base, seed=7)
        table = part1.part1_table(root, daily, cal)
        rth = bars.rth_bars(root, "SPY", cal).collect()
    return table, rth, cal


def test_return_identities_and_prices(synth):
    t, rth, cal = synth
    # (1 + r_cc) = (1 + rod)(1 + r13) and prev_off is the previous kept session's off_close
    err = (t["r_cc"] + 1 - (1 + t["rod"]) * (1 + t["r13"])).abs().max()
    assert err < 1e-12
    shifted = t["off_close"].shift(1)
    assert (shifted[1:] == t["prev_off"][1:]).all()
    # the 15:59-bar returns are the chapter 1 definitions, so r13_bar * C-31 price = p_close
    assert ((1 + t["r13_bar"]) * t["p_c31"] - t["p_close"]).abs().max() < 1e-9
    # extra bars read straight from the raw frame (another route than the group_by in part1)
    day = date(2018, 5, 9)
    row = t.filter(pl.col("session") == day).row(0, named=True)
    one = rth.filter(pl.col("session") == day)

    def close_at(h, m):
        return one.filter(pl.col("minute") == time(h, m))["close"].item()

    assert row["p_c16"] == close_at(15, 44)  # bar C-16 closes at 15:45
    assert row["p_c6"] == close_at(15, 54)
    assert row["p_hh_0"] == close_at(9, 59) == row["p_10"]
    assert row["p_hh_5"] == close_at(12, 29)
    assert row["p_hh_11"] == close_at(15, 29) == row["p_c31"]
    # half-day (2018-11-23): only 6 boundaries, the last at C-31 = 12:29
    half = t.filter(pl.col("session") == date(2018, 11, 23)).row(0, named=True)
    assert half["p_hh_5"] == half["p_c31"] and half["p_hh_6"] is None and half["p_hh_11"] is None
    assert t["r_cc"].null_count() == 0 and t.height == cal.height


# ------------------------------------------------------------------ past-only cut-offs and jumps


def test_past_only_cutoffs_ignore_later_rows(monkeypatch):
    monkeypatch.setattr(config, "TERCILE_WARMUP", 20)
    rng = np.random.default_rng(1)
    v = rng.normal(size=120)
    base = part1.expanding_quantiles(v, (1 / 3, 2 / 3))
    assert np.isnan(base[:20]).all() and np.isfinite(base[20:]).all()
    changed = v.copy()
    changed[70:] = changed[70:] * 9 + 5  # later rows change ...
    new = part1.expanding_quantiles(changed, (1 / 3, 2 / 3))
    assert np.array_equal(
        base[:71], new[:71], equal_nan=True
    )  # ... cut-offs up to row 70 (which reads rows < 70) do not
    assert not np.array_equal(base[71:], new[71:])
    # hand check: the cut at row 30 is the quantile of rows 0..29 only
    assert base[30, 1] == pytest.approx(np.quantile(v[:30], 2 / 3))
    lab = part1.tercile_labels(v, base)
    assert (lab[:20] == -1).all() and set(lab[20:]) <= {0, 1, 2}
    assert np.array_equal(lab[:71], part1.tercile_labels(v, new)[:71])


def test_first_jump_picks_the_first_boundary():
    nan = np.nan
    x = np.array(
        [
            [0.001, -0.012, 0.020, 0.0],  # first beyond is -0.012, not the larger 0.020
            [0.000, 0.001, 0.002, 0.003],  # none
            [nan, nan, 0.015, 0.016],  # missing boundaries skipped; 0.015 first
            [0.011, 0.0, 0.0, 0.0],  # beyond at the very first boundary
        ]
    )
    lo, hi = np.full(4, -0.01), np.full(4, 0.01)
    got = part1.first_jump(x, lo, hi)
    assert got[0] == -0.012 and np.isnan(got[1]) and got[2] == 0.015 and got[3] == 0.011
    # a NaN cut-off (warm-up) means no jump
    assert np.isnan(part1.first_jump(x, np.full(4, np.nan), np.full(4, np.nan))).all()


def test_r_next_k_follows_calendar_and_is_null_across_a_gap():
    days = [date(2019, 3, d) for d in (11, 12, 13, 14, 15, 18)]
    cal = pl.DataFrame({"session": days})
    # session 03-13 is missing from the table (dropped); r_cc: 1%, 2%, -, 4%, 5%, 6%
    t = pl.DataFrame(
        {
            "session": [days[0], days[1], days[3], days[4], days[5]],
            "r_cc": [0.01, 0.02, 0.04, 0.05, 0.06],
        }
    )
    got = part1.next_k_returns(cal, t)
    k1 = got["r_next_1"].to_list()
    k2 = got["r_next_2"].to_list()
    k3 = got["r_next_3"].to_list()
    assert k1[0] == pytest.approx(0.02) and k1[1] is None  # 03-13 absent
    assert k2[0] is None and k3[0] is None  # reach across the gap
    assert k1[2] == pytest.approx(0.04) and k2[2] == pytest.approx(1.04 * 1.05 - 1)
    assert k3[2] == pytest.approx(1.04 * 1.05 * 1.06 - 1)
    assert k1[4] == pytest.approx(0.06) and k2[4] is None  # past the end
    assert k1[5] is None


# ------------------------------------------------------------------ verdict and windows


def test_verdict_three_outcomes():
    # paper 0.07, se 0.01, null 0
    rep = part1.verdict(0.065, 0.01, 0.07, 0.0, 0.01)
    assert rep["verdict"] == "replicated" and rep["d_null"] == pytest.approx(6.5)
    assert rep["d_paper"] == pytest.approx(0.5)
    assert rep["lo"] == pytest.approx(0.045) and rep["hi"] == pytest.approx(0.085)
    assert rep["mde"] == pytest.approx(2.84 * 0.01)
    assert part1.verdict(0.02, 0.01, 0.07, 0.0, 0.01)["verdict"] == "not replicated"  # d_paper 5
    assert part1.verdict(0.0, 0.01, 0.07, 0.0, 0.01)["verdict"] == "not replicated"
    assert part1.verdict(0.05, 0.03, 0.07, 0.0, 0.03)["verdict"] == "cannot tell"  # both <= 2
    none = part1.verdict(0.05, 0.01, None, 0.0, 0.01)
    assert none["verdict"] is None and none["d_paper"] is None
    # success row: the null is always-long's 0.50 and the SE of the difference (0.004) is used
    s = part1.verdict(0.5437, 0.01, 0.5437, 0.5042, 0.004)
    assert s["d_null"] == pytest.approx(0.0395 / 0.004) and s["verdict"] == "replicated"


def test_window_bounds_are_inclusive():
    w = config.PART1_WINDOWS
    assert w["gao_overlap"] == (date(2011, 1, 3), date(2013, 12, 31))
    assert w["balt_overlap"][1] == date(2020, 5, 1) and w["balt_after"][0] == date(2020, 5, 4)
    days = [date(2013, 12, 31), date(2014, 1, 2), date(2020, 5, 1), date(2020, 5, 4)]
    df = pl.DataFrame({"session": days, "v": [1, 2, 3, 4]})
    assert part1._window(df, "gao_overlap")["v"].to_list() == [1]
    assert part1._window(df, "gao_after")["v"].to_list() == [2, 3, 4]
    assert part1._window(df, "balt_overlap")["v"].to_list() == [1, 2, 3]
    assert part1._window(df, "balt_after")["v"].to_list() == [4]


# ------------------------------------------------------------------ rows on a fake daily table


@pytest.fixture(scope="module")
def fake():
    rng = np.random.default_rng(3)
    days = [
        date(2011, 1, 3) + timedelta(i)
        for i in range((date(2026, 8, 31) - date(2011, 1, 3)).days + 1)
    ]
    days = [d for d in days if d.weekday() < 5]
    n = len(days)
    prev = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    x = np.cumsum(rng.normal(0, 0.002, (n, part1.N_BOUNDARIES)), axis=1)
    r1 = x[:, 0] + rng.normal(0, 0.003, n)
    x[:, 0] = r1
    r12 = rng.normal(0, 0.002, n)
    rod = r1 + r12 + rng.normal(0, 0.001, n)
    r13 = 0.07 * r1 + 0.1 * r12 + rng.normal(0, 0.002, n)  # planted slopes
    cc = (1 + rod) * (1 + r13) - 1
    t = pl.DataFrame(
        {
            "session": days, "prev_off": prev, "r1": r1, "r12": r12, "rod": rod, "r13": r13,
            "r1_bar": r1, "rod_bar": rod, "r13_bar": r13 + rng.normal(0, 1e-4, n),
            "r_last15": 0.4 * r13 + rng.normal(0, 0.001, n), "r_last5": 0.2 * r13, "r_cc": cc,
            "rv_fh": rng.gamma(2.0, 1e-5, n),
        }
    ).with_columns(
        pl.Series(f"p_hh_{k}", prev * (1 + x[:, k])) for k in range(part1.N_BOUNDARIES)
    )  # fmt: skip
    cal = t.select("session")
    t = t.join(part1.next_k_returns(cal, t), on="session")
    ctrl = t.select(
        "session", fomc=pl.col("session").dt.day() == 15, cpi=pl.col("session").dt.day() == 12
    )
    return t, ctrl


@pytest.fixture(scope="module")
def rows(fake):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PART1_BOOT_REPS", 100)
        return part1.part1_rows(*fake)


def test_rows_columns_windows_and_planted_slope(rows):
    assert tuple(rows.columns) == part1.COLUMNS
    assert set(rows["window"]) == set(config.PART1_WINDOWS)
    assert rows.filter(pl.col("paper") == "gao")["window"].is_in(["gao_overlap", "gao_after"]).all()
    assert (
        rows.filter(pl.col("paper") == "baltussen")["window"]
        .is_in(["balt_overlap", "balt_after"])
        .all()
    )
    s = rows.filter((pl.col("item") == "gao_slope") & (pl.col("variant") == "main"))
    assert s.height == 2 and (s["se_kind"] == "nw").all()
    for r in s.iter_rows(named=True):
        assert abs(r["ours"] - 0.07) < 3 * r["se"] and r["paper_value"] == 0.069
        assert r["lo"] == pytest.approx(r["ours"] - 2 * r["se"])
    # no verdict on check and cost rows or rows without a paper value
    quiet = rows.filter(
        pl.col("variant").is_in(["check_1559", "cost_1bp"]) | pl.col("paper_value").is_null()
    )
    assert quiet["verdict"].null_count() == quiet.height
    loud = rows.filter(
        ~pl.col("variant").is_in(["check_1559", "cost_1bp"]) & pl.col("paper_value").is_not_null()
    )
    assert loud["verdict"].null_count() == 0
    # OOS forecasts start 2012-01-03: the overlap window holds 2012-2013 only (~500 forecasts; the fake table has no holidays)
    oos_row = rows.filter((pl.col("item") == "gao_oos_r2") & (pl.col("window") == "gao_overlap"))
    assert 500 <= oos_row["n"].item() <= 525 and oos_row["cw_p"].item() is not None
    # every paper value in config shows up on a row with its source and market
    keys = set(zip(rows["item"], rows["term"], rows["metric"], strict=True))
    assert set(config.PAPER) <= keys


def test_success_rows_use_the_paired_se_against_always_long(rows, fake):
    t, _ = fake
    s = rows.filter(
        (pl.col("item") == "gao_rule") & (pl.col("metric") == "success")
        & (pl.col("variant") == "main") & (pl.col("window") == "gao_after")
    ).row(0, named=True)  # fmt: skip
    d = part1._window(t, "gao_after")
    always_long = float((d["r13"] > 0).mean())
    assert s["null_label"] == "always-long success" and s["null_value"] == pytest.approx(
        always_long
    )
    assert s["se_null"] != s["se"]
    assert s["d_null"] == pytest.approx(abs(s["ours"] - s["null_value"]) / s["se_null"])
    assert s["ours"] == pytest.approx(float(((np.sign(d["r1"]) * d["r13"]) > 0).mean()))
    # paired SE: with the rule almost equal to always-long the difference is far less noisy
    r = d["r13"].to_numpy()
    flips = np.where(np.arange(r.size) % 50 == 0, -1.0, 1.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PART1_BOOT_REPS", 100)
        assert part1._success_paired_se(r * flips, r) < 0.01


def test_cumulative_and_rolling_series(fake):
    t, _ = fake
    c = part1.cumulative(t)
    al = c.filter((pl.col("rule") == "always_long") & (pl.col("cost_bp") == 0.0)).sort("session")
    assert al["wealth"][0] == pytest.approx(1 + t["r13"][0])
    assert al["wealth"][-1] == pytest.approx(float(np.prod(1 + t["r13"].to_numpy())))
    net = c.filter((pl.col("rule") == "eta_r1") & (pl.col("cost_bp") == 1.0)).sort("session")
    first = np.sign(t["r1"][0]) * t["r13"][0] - 2e-4
    assert net["wealth"][0] == pytest.approx(1 + first)
    roll = part1.rolling(t, window=300)
    assert set(roll["signal"]) == {"r1", "rod"}
    one = roll.filter(pl.col("signal") == "r1").sort("session")
    assert one.height == t.height - 299
    slope = np.polyfit(t["r1"].to_numpy()[:300], t["r13"].to_numpy()[:300], 1)[0]  # independent OLS
    assert one["slope"][0] == pytest.approx(slope) and one["se"][0] > 0
