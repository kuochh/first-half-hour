"""Part 3 data: the symbol refactor leaves SPY's Part 1 table and Part 2 frame unchanged, the
per-ETF table on synthetic bars (non-SPY symbol, a dropped session never bridged), the coverage
table, class pooling and the verdict rule of the A rows on a hand-built two-ETF example."""

from datetime import date, time

import numpy as np
import polars as pl
import pytest

from first_half_hour import (
    calendar,
    config,
    controls,
    part1,
    part2_features,
    part3_data,
    returns,
    synthetic,
)

# ------------------------------------------------------------------ synthetic bars


def _fake_controls(table: pl.DataFrame) -> pl.DataFrame:
    """Controls stand-in with the columns part2_features.frame reads."""
    wd = table["session"].dt.weekday()
    out = table.select("session", vix_lag=pl.lit(18.0)).with_columns(
        pl.lit(False).alias(c) for c in ("fomc", "cpi", "payroll", "opex_monthly", "month_end")
    )
    return out.with_columns(
        (wd == i).alias(n) for n, i in zip(controls.WEEKDAY_COLUMNS, (2, 3, 4, 5), strict=True)
    )


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Synthetic SPY and QQQ bars for 2018, the stub official close, and SPY's Part 1 table."""
    root = tmp_path_factory.mktemp("p3d")
    daily = root / "daily"
    daily.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DATA_START", date(2018, 1, 1))
        mp.setattr(config, "DATA_END", date(2018, 12, 31))
        cal = calendar.sessions(config.DATA_START, config.DATA_END)
        synthetic.write_synthetic(
            root, ["SPY", "QQQ"], date(2018, 1, 2), date(2018, 12, 31), 0.1, 7
        )
        pl.DataFrame(
            {"ex_date": [date(2018, 3, 16), date(2018, 6, 15)], "amount": [0.5, 0.5]}
        ).write_csv(daily / "spy_dividends.csv")  # fmt: skip
        synthetic.write_stub_official_close(root, daily, returns.daily_table(root, "SPY", cal), 7)
        yield part1.part1_table(root, daily, cal), cal, root, daily


def test_spy_part1_table_and_part2_frame_unchanged_by_symbol_argument(synth):
    table, cal, root, _ = synth
    # Column sums pinned from the code BEFORE the symbol refactor (git HEAD 12998bb, same synthetic
    # bars, seed 7), computed by running that checkout; the refactor must not move them.
    assert table.height == 251
    for col, pinned in (
        ("p_c16", 25161.174499999994), ("p_hh_5", 25165.947700000004),
        ("r13", 0.023610992715205614), ("rod", -0.0025039049061676977),
        ("off_close", 25162.993791552733),
    ):  # fmt: skip
        assert table[col].sum() == pytest.approx(pinned, rel=1e-9, abs=1e-12), col
    f = part2_features.frame(table, _fake_controls(table), root, cal)
    assert f.height == 1240
    for col, pinned in (
        ("s", 0.0687008637350468), ("g", 0.16197794590425774), ("rv_sofar", 0.37549617327564266),
        ("last60_share", 10.09693570550333), ("range_prev", 11.807682501643269),
        ("vol_rel", 1140.2335010482784),
    ):  # fmt: skip
        assert f[col].sum() == pytest.approx(pinned, rel=1e-9, abs=1e-12), col
    # the explicit default is the same call
    same = part2_features.frame(table, _fake_controls(table), root, cal, "SPY")
    assert same.equals(f)
    a = part1._extra_bars(root, cal).sort("session")  # group_by order is not stable
    assert a.equals(part1._extra_bars(root, cal, "SPY").sort("session"))


def test_etf_table_non_spy_symbol(synth):
    _, cal, root, daily = synth
    t = part3_data.etf_table("QQQ", root, daily, cal)
    assert t.height == cal.height and t["session"].is_sorted()
    # bar basis: off_close is the 15:59 bar close; prev_off the previous session's
    assert (t["off_close"] == t["p_close"]).all()
    assert (t["prev_off"][1:] == t["off_close"].shift(1)[1:]).all() and t["prev_off"][0] is None
    assert t["r1"].null_count() == 1  # the first session has no previous close
    err = (t["r_cc"] + 1 - (1 + t["rod"]) * (1 + t["r13"])).abs().max()
    assert err < 1e-12
    # a boundary price read from the raw QQQ file (another route than the group_by)
    raw = pl.read_csv(root / "QQQ" / "QQQ_1min_2018-05.csv").with_columns(
        pl.col("timestamp").str.to_datetime()
    )
    day = raw.filter(pl.col("timestamp") == pl.datetime(2018, 5, 9, 12, 29))["close"].item()
    assert t.filter(pl.col("session") == date(2018, 5, 9))["p_hh_5"].item() == day
    # the QQQ table is QQQ's: its prices differ from SPY's synthetic path
    spy = part3_data.etf_table("SPY", root, daily, cal)
    assert (spy["p_close"] != t["p_close"]).any()
    # frame: Part 2 columns plus symbol, the first session (null prev_off) has no row
    f = part3_data.etf_frame("QQQ", t, _fake_controls(t), root, cal)
    assert f["symbol"].unique().to_list() == ["QQQ"] and f["s"].null_count() == 0
    assert t["session"][0] not in f["session"].to_list()
    assert set(config.PART2_FEATURES) <= set(f.columns)


def test_dropped_session_is_never_bridged(tmp_path, monkeypatch):
    """A session with no bars is dropped; the next session has null prev_off and no Part 2 row."""
    monkeypatch.setattr(config, "DATA_START", date(2018, 3, 1))
    monkeypatch.setattr(config, "DATA_END", date(2018, 3, 31))
    cal = calendar.sessions(config.DATA_START, config.DATA_END)
    gone = date(2018, 3, 14)
    rng = np.random.default_rng(1)
    frames, px = [], 100.0
    for day, close, _ in cal.iter_rows():
        n_bars = close.hour * 60 + close.minute - 570
        fr = synthetic._session_bars(rng, day, n_bars, px, 0.1)
        px = float(fr["close"][n_bars - 1])
        frames.append(fr)
    all_bars = pl.concat(frames)
    kept = all_bars.filter(pl.col("timestamp").dt.date() != gone)
    synthetic.write_bars(tmp_path, "TLT", kept)
    monkeypatch.setitem(config.KNOWN_MISSING_SESSIONS, "TLT", frozenset({gone}))
    t = part3_data.etf_table("TLT", tmp_path, tmp_path, cal)
    assert gone not in t["session"].to_list() and t.height == cal.height - 1
    nxt = t.filter(pl.col("session") > gone)["session"][0]
    row = t.filter(pl.col("session") == nxt).row(0, named=True)
    assert row["prev_off"] is None and row["r1"] is None and row["rod"] is None
    f = part3_data.etf_frame("TLT", t, _fake_controls(t), tmp_path, cal)
    assert nxt not in f["session"].to_list()
    cov = part3_data.coverage({"TLT": t}, cal)
    assert cov["sessions_dropped"].item() == 1 and cov["kept_sessions"].item() == t.height
    assert cov["first_session"].item() == t["session"].min()
    assert cov["share_r1_null"].item() == pytest.approx(2 / t.height)  # first row and `nxt`


def test_coverage_shares():
    t = pl.DataFrame(
        {
            "session": [date(2018, 1, d) for d in (2, 3, 4, 5)],
            "r13": [0.0, 0.0, 0.001, -0.001],
            "r1": [None, 0.01, 0.02, None],
        }
    )
    c = part3_data.coverage({"X": t}).row(0, named=True)
    assert (c["kept_sessions"], c["sessions_dropped"]) == (4, 0)
    assert c["share_r13_zero"] == 0.5 and c["share_r1_null"] == 0.5


# ------------------------------------------------------------------ A: pooling and verdicts


def _two_etf_tables(n: int = 300):
    """X has every day; Y misses r1 on every 5th day and r13 on day 7. Signals are random."""
    rng = np.random.default_rng(5)
    days = [d for d in pl.date_range(date(2018, 1, 1), date(2019, 12, 31), "1d", eager=True)
            if d.weekday() < 5][:n]  # fmt: skip
    out = {}
    for s in "XY":
        r1, rod = rng.normal(0, 0.005, n), rng.normal(0, 0.006, n)
        r13 = 0.05 * r1 + rng.normal(0, 0.002, n)
        out[s] = pl.DataFrame({"session": days, "r1": r1, "rod": rod, "r13": r13})
    out["Y"] = out["Y"].with_columns(
        r1=pl.when(pl.int_range(n) % 5 == 4).then(None).otherwise(pl.col("r1")),
        r13=pl.when(pl.int_range(n) == 7).then(None).otherwise(pl.col("r13")),
    )
    return out


@pytest.fixture
def small_part3(monkeypatch):
    monkeypatch.setattr(config, "PART3_CLASSES", {"c": ("X", "Y")})
    monkeypatch.setattr(config, "PART3_PERIODS", {"p": (date(2018, 1, 1), date(2019, 12, 31))})
    monkeypatch.setattr(config, "PART3_BOOT_REPS", 100)


def test_class_pooling_by_day(small_part3):
    tabs = _two_etf_tables()
    eff = part3_data.effect_rows(tabs, tabs["X"])
    x, y = tabs["X"], tabs["Y"]
    # slope_r1 by hand: per day the mean of the ETFs that have BOTH r1 and r13, OLS slope via numpy
    j = x.join(y, on="session", suffix="_y")
    ok_y = j["r1_y"].is_not_null() & j["r13_y"].is_not_null()
    m1 = np.where(ok_y, (j["r1"] + j["r1_y"].fill_null(0)) / 2, j["r1"])
    m13 = np.where(ok_y, (j["r13"] + j["r13_y"].fill_null(0)) / 2, j["r13"])
    slope = np.polyfit(m1, m13, 1)[0]
    row = eff.filter((pl.col("class") == "c") & (pl.col("item") == "slope_r1")).row(0, named=True)
    assert row["est"] == pytest.approx(slope, rel=1e-9) and row["n_days"] == 300
    assert row["n_etfs"] == 2 and row["level"] == "class" and row["cost_bp"] is None
    # rule eta(rod) at 2 bp: mean over the ETFs present of d_i * r13_i, minus the cost, x 252
    d = lambda r: np.where(r >= 0, 1.0, -1.0)
    gx = d(j["rod"].to_numpy()) * j["r13"].to_numpy()
    gy = d(j["rod_y"].to_numpy()) * j["r13_y"].to_numpy()
    both = ~np.isnan(gy)
    g = np.where(both, (gx + np.nan_to_num(gy)) / 2, gx) - 0.0002
    r = eff.filter(
        (pl.col("class") == "c") & (pl.col("item") == "rule_rod")
        & (pl.col("metric") == "ann_mean") & (pl.col("cost_bp") == 2.0)
    ).row(0, named=True)  # fmt: skip
    assert r["est"] == pytest.approx(252 * g.mean(), rel=1e-9) and r["n_days"] == 300
    # the appendix has one row set per ETF and the official-SPY row is computed from the table given
    assert set(eff.filter(pl.col("level") == "etf")["class"].unique()) == {"X", "Y"}
    spy = eff.filter(pl.col("level") == "spy_official")
    assert spy["n_etfs"].unique().to_list() == [1] and spy["class"].unique().to_list() == [
        "us_equity"
    ]
    xe = eff.filter((pl.col("class") == "X") & (pl.col("item") == "slope_rod"))["est"].item()
    so = eff.filter((pl.col("level") == "spy_official") & (pl.col("item") == "slope_rod"))
    assert so["est"].item() == pytest.approx(xe)


def test_verdict_two_se_rule():
    assert part3_data._verdict(2.5, 1.0) == (2.5, "clearly above")
    assert part3_data._verdict(-2.5, 1.0)[1] == "clearly below"
    assert part3_data._verdict(2.0, 1.0)[1] == "cannot tell"  # exactly 2 SE is not clear
    assert part3_data._verdict(-1.9, 1.0)[1] == "cannot tell"
    with pytest.raises(AssertionError):
        part3_data._verdict(1.0, 0.0)


def test_config_classes():
    flat = [s for syms in config.PART3_CLASSES.values() for s in syms]
    assert len(flat) == len(set(flat)) == 14 and {"IVV", "VOO"}.isdisjoint(flat)
    assert set(flat) <= set(config.SYMBOLS)
    assert list(config.PART3_CLASSES) == ["us_equity", "intl_equity", "bonds", "gold"]
    assert time(9, 30) == config.SESSION_OPEN  # keep the import honest
