"""Part 2 features on hand-built bars: values against numpy, the tau cut, past-only rolling
features, half-day and missing-price drops, zero signal = long, and frame() on synthetic bars."""

from datetime import date, time, timedelta

import numpy as np
import polars as pl
import pytest

from first_half_hour import (
    calendar,
    config,
    controls,
    part1,
    part2_features,
    returns,
    synthetic,
)

N_SESS = 14
MINUTES = [time(h, m) for h in range(9, 16) for m in range(60) if (h, m) >= (9, 30)]
HALF_DAY_INDEX = 6
FLAGS = (
    "fomc", "cpi", "payroll", "opex_monthly", "month_end", "dow_tue", "dow_wed", "dow_thu", "dow_fri",
)  # fmt: skip


def _days() -> list[date]:
    out, d = [], date(2018, 3, 5)
    while len(out) < N_SESS:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _hand_inputs(seed: int = 3):
    """(table, ctrl, cal, rth): random-walk minute bars and the table columns computed from them by
    plain numpy loops (another route than the polars code under test)."""
    rng = np.random.default_rng(seed)
    days, px, rows, tab = _days(), 100.0, [], []
    prev_off = 100.0
    for i, day in enumerate(days):
        closes, opens = [], []
        for _ in MINUTES:
            opens.append(px)
            px *= float(np.exp(rng.normal(0, 0.0005)))
            closes.append(px)
        closes, opens = np.array(closes), np.array(opens)
        highs, lows = np.maximum(opens, closes) * 1.0001, np.minimum(opens, closes) * 0.9999
        vols = rng.integers(100, 1000, len(MINUTES))
        rows.append(
            pl.DataFrame(
                {
                    "session": day,
                    "minute": MINUTES,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": vols,
                }
            )
        )
        off = closes[-1] * (1 + rng.normal(0, 0.0003))
        sq = np.log(closes / np.concatenate([[opens[0]], closes[:-1]])) ** 2
        tab.append(
            {"session": day, "is_half_day": i == HALF_DAY_INDEX, "p_open": opens[0],
             "prev_off": prev_off, "off_close": off, "p_c31": closes[MINUTES.index(time(15, 29))],
             "rv_fh": sq[:30].sum()}
        )  # fmt: skip
        prev_off = off
    table = pl.DataFrame(tab).with_columns(r13=pl.col("off_close") / pl.col("p_c31") - 1)
    ctrl = pl.DataFrame({"session": days, "vix_lag": np.linspace(15.0, 20.0, N_SESS)}).with_columns(
        pl.lit(False).alias(f) for f in FLAGS
    )
    ctrl = ctrl.with_columns(
        month_end=pl.col("session") == days[-1], fomc=pl.col("session") == days[4]
    )
    cal = table.select("session", close=pl.lit(time(16, 0)), is_half_day=pl.col("is_half_day"))
    return table, ctrl, cal, pl.concat(rows)


@pytest.fixture(autouse=True)
def _small_windows(monkeypatch):
    monkeypatch.setattr(config, "PART2_RANK_WINDOW", 3)
    monkeypatch.setattr(config, "PART2_ROLL_WINDOW", 2)


def _row(f: pl.DataFrame, day: date, tau: str) -> dict:
    return f.filter((pl.col("session") == day) & (pl.col("tau") == tau)).row(0, named=True)


def test_feature_values_against_numpy():
    table, ctrl, cal, rth = _hand_inputs()
    f = part2_features.build(table, ctrl, cal, rth)
    assert f.columns[:10] == ["session", "year", "tau", "s", "d", "g", "r13", "y_0", "y_2", "y_4"]
    assert f.columns[10:] == list(config.PART2_FEATURES)
    day, prev = _days()[9], _days()[8]
    tau, m = "14:00", 840
    one = rth.filter(pl.col("session") == day).sort("minute")
    px = one["close"].to_numpy()
    op = one["open"].to_numpy()
    k = m - 570  # bars 09:30 .. tau-1 are indices 0..k-1
    p_tau, p_30, p_60 = px[k - 1], px[k - 31], px[k - 61]
    t = table.filter(pl.col("session") == day).row(0, named=True)
    s = p_tau / t["prev_off"] - 1
    d = 1.0 if s >= 0 else -1.0
    r = _row(f, day, tau)
    sq = np.log(px[:k] / np.concatenate([[op[0]], px[: k - 1]])) ** 2
    pr = table.filter(pl.col("session") == prev).row(0, named=True)
    pbars = rth.filter(pl.col("session") == prev)
    exp = {
        "s": s, "d": d, "g": d * t["r13"], "abs_s": abs(s), "d_long": float(d > 0),
        "on_aligned": d * (t["p_open"] / t["prev_off"] - 1), "intra_aligned": d * (p_tau / t["p_open"] - 1),
        "rv_sofar": np.sqrt(sq.mean()), "rv_fh": np.sqrt(t["rv_fh"] / 30),
        "last30_aligned": d * (p_tau / p_30 - 1),
        "last60_share": float(np.clip(d * (p_tau - p_60) / (p_tau - t["prev_off"]), -2, 2)),
        "range_prev": (pbars["high"].max() - pbars["low"].min()) / pr["off_close"],
        "vix_lag": ctrl.filter(pl.col("session") == day)["vix_lag"].item(),
        "quarter_end": 0.0, "fomc": 0.0,
    }  # fmt: skip
    for name, value in exp.items():
        assert r[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
    for c, name in zip(config.PART2_COSTS_RT, ("y_0", "y_2", "y_4"), strict=True):
        assert r[name] == float(d * t["r13"] > c)
    # yesterday's signal and payoff at the same tau, aligned with today's d
    pp = rth.filter((pl.col("session") == prev) & (pl.col("minute") == time(13, 59)))[
        "close"
    ].item()
    assert r["yest_s_aligned"] == pytest.approx(d * (pp / pr["prev_off"] - 1), rel=1e-9)
    assert r["yest_r13_aligned"] == pytest.approx(d * pr["r13"], rel=1e-9)

    # volume ratio against the previous 2 sessions (window patched to 2), s_rank against previous 3
    def vol(dd):
        return rth.filter((pl.col("session") == dd) & (_minute_mod() < m))["volume"].sum()

    assert r["vol_rel"] == pytest.approx(vol(day) / np.mean([vol(_days()[7]), vol(prev)]))
    rows = f.filter(pl.col("tau") == tau).sort("session")
    before = rows.filter(pl.col("session") < day).tail(3)
    assert r["s_rank60"] == pytest.approx(np.mean(before["abs_s"].to_numpy() < abs(s)))
    assert r["hit20"] == pytest.approx(rows.filter(pl.col("session") < day).tail(2)["y_0"].mean())


def _minute_mod() -> pl.Expr:
    return pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(
        pl.Int32
    )


def test_tau_cut_and_past_only_rolling():
    table, ctrl, cal, rth = _hand_inputs()
    base = part2_features.build(table, ctrl, cal, rth)
    # plant wild bars at 12:00 and later in every session: rows at 10:00 and 12:00 cannot change
    wild = rth.with_columns(
        pl.when(_minute_mod() >= 720).then(pl.col(c) * 3).otherwise(pl.col(c)).alias(c)
        for c in ("open", "high", "low", "close")
    ).with_columns(
        volume=pl.when(_minute_mod() >= 720).then(pl.col("volume") * 50).otherwise(pl.col("volume"))
    )
    new = part2_features.build(table, ctrl, cal, wild)
    for tau in ("10:00", "12:00"):
        a = base.filter(pl.col("tau") == tau).drop(
            "range_prev"
        )  # range_prev uses whole earlier days
        b = new.filter(pl.col("tau") == tau).drop("range_prev")
        assert a.equals(b), tau
    assert not base.filter(pl.col("tau") == "14:00").equals(new.filter(pl.col("tau") == "14:00"))
    # later sessions change (bars, official close, r13): rolling features before the cut are unchanged
    cut = _days()[9]
    alt_table = table.with_columns(
        r13=pl.when(pl.col("session") > cut).then(-pl.col("r13") * 7).otherwise(pl.col("r13")),
        prev_off=pl.when(pl.col("session") > cut)
        .then(pl.col("prev_off") * 1.01)
        .otherwise(pl.col("prev_off")),
    )
    alt_rth = rth.with_columns(
        volume=pl.when(pl.col("session") > cut)
        .then(pl.col("volume") * 9)
        .otherwise(pl.col("volume"))
    )
    alt = part2_features.build(alt_table, ctrl, cal, alt_rth)
    cols = ["session", "tau", "s_rank60", "vol_rel", "hit20", "yest_s_aligned", "yest_r13_aligned"]
    keep = pl.col("session") <= cut
    assert base.filter(keep).select(cols).equals(alt.filter(keep).select(cols))
    assert not base.filter(~keep).select(cols).equals(alt.filter(~keep).select(cols))


def test_drops_zero_signal_and_half_day():
    table, ctrl, cal, rth = _hand_inputs()
    days = _days()
    # session 9: no bar closing at 12:00 (the 11:59 bar is absent) -> dropped at 12:00 only
    miss = days[9]
    rth2 = rth.filter(~((pl.col("session") == miss) & (pl.col("minute") == time(11, 59))))
    # session 10: zero signal at 15:30 (P_tau equals prev_off) -> counts long
    zero = days[10]
    p_1529 = rth2.filter((pl.col("session") == zero) & (pl.col("minute") == time(15, 29)))[
        "close"
    ].item()
    table2 = table.with_columns(
        prev_off=pl.when(pl.col("session") == zero).then(p_1529).otherwise(pl.col("prev_off"))
    )
    f = part2_features.build(table2, ctrl, cal, rth2)
    assert days[HALF_DAY_INDEX] not in f["session"].to_list()
    per_tau = f.group_by("tau").len().sort("tau")
    assert per_tau["len"].to_list() == [N_SESS - 1, N_SESS - 2, N_SESS - 1, N_SESS - 1, N_SESS - 1]
    assert f["tau"].to_list() == sorted(f["tau"].to_list())
    z = _row(f, zero, "15:30")
    assert z["s"] == 0.0 and z["d"] == 1.0 and z["d_long"] == 1.0 and z["last60_share"] == 0.0
    # the previous session's missing 12:00 price makes yesterday's signal null for the next session
    assert _row(f, days[10], "12:00")["yest_s_aligned"] is None
    assert _row(f, days[10], "14:00")["yest_s_aligned"] is not None


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2f")
    daily = root / "daily"
    daily.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DATA_START", date(2018, 1, 1))
        mp.setattr(config, "DATA_END", date(2018, 12, 31))
        cal = calendar.sessions(config.DATA_START, config.DATA_END)
        synthetic.write_synthetic(root, ["SPY"], date(2018, 1, 2), date(2018, 12, 31), 0.1, 7)
        pl.DataFrame(
            {"ex_date": [date(2018, 3, 16), date(2018, 6, 15)], "amount": [0.5, 0.5]}
        ).write_csv(daily / "spy_dividends.csv")  # fmt: skip
        synthetic.write_stub_official_close(root, daily, returns.daily_table(root, "SPY", cal), 7)
        table = part1.part1_table(root, daily, cal)
        return table, cal, root


def test_frame_on_synthetic_bars(synth, monkeypatch):
    table, cal, root = synth
    monkeypatch.setattr(config, "DATA_START", date(2018, 1, 1))
    monkeypatch.setattr(config, "DATA_END", date(2018, 12, 31))
    monkeypatch.setattr(config, "PART2_RANK_WINDOW", 60)
    monkeypatch.setattr(config, "PART2_ROLL_WINDOW", 20)
    ctrl = _fake_controls(table, cal)
    f = part2_features.frame(table, ctrl, root, cal)
    n_regular = int((~table["is_half_day"]).sum())
    assert f.group_by("tau").len()["len"].to_list() == [n_regular] * 5
    assert (f["year"] == 2018).all() and f["s"].null_count() == 0
    # the 15:30 signal is rod, the 15:30 target the Part 1 r13
    late = f.filter(pl.col("tau") == "15:30").join(table, on="session", suffix="_t")
    assert (late["s"] - late["rod"]).abs().max() < 1e-12
    assert (late["r13"] - late["r13_t"]).abs().max() == 0.0
    assert f.filter(pl.col("tau") == "15:30")["s_rank60"].null_count() == 60
    assert f["abs_s"].null_count() == 0 and f["rv_sofar"].null_count() == 0


def _fake_controls(table, cal):
    """Controls stand-in with the columns frame() reads (the real controls need the daily files)."""
    wd = table["session"].dt.weekday()
    out = table.select("session", vix_lag=pl.lit(18.0)).with_columns(
        pl.lit(False).alias(c) for c in ("fomc", "cpi", "payroll", "opex_monthly", "month_end")
    )
    return out.with_columns(
        (wd == i).alias(n) for n, i in zip(controls.WEEKDAY_COLUMNS, (2, 3, 4, 5), strict=True)
    )
