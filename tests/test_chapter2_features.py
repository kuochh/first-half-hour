"""Chapter 2 target, formula and features: hand-made bars, a Monte Carlo, and a look-ahead check."""

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest

from first_half_hour import (
    bars,
    baselines,
    calendar,
    config,
    controls,
    features,
    returns,
    synthetic,
)


def _use_window(monkeypatch, start, end):
    monkeypatch.setattr(config, "DATA_START", start)
    monkeypatch.setattr(config, "DATA_END", end)
    return calendar.sessions(start, end)


def _frame(rows):
    """rows: (date, 'HH:MM', open, high, low, close)."""
    return pl.DataFrame(
        {
            "timestamp": [datetime.combine(d, time.fromisoformat(t)) for d, t, *_ in rows],
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[5] for r in rows],
            "volume": [100] * len(rows),
        }
    )


def test_target_from_hand_made_bars_incl_half_day_and_exact_touch(tmp_path, monkeypatch):
    # 2019-11-25 Mon, 26 Tue, 27 Wed, (28 Thanksgiving), 29 Fri = half-day closing 13:00.
    d25, d26, d27, d29 = (date(2019, 11, d) for d in (25, 26, 27, 29))
    cal = _use_window(monkeypatch, d25, d29)
    flat = (100.0, 100.0, 100.0, 100.0)  # open, high, low, close
    rows = []
    for d in (d25, d26, d27, d29):
        rows += [(d, "09:30", *flat), (d, "09:59", 100.0, 150.0, 100.0, 100.0)]  # P10 = 100;
        # the 09:59 high of 150 must never count (the window starts at 10:00)
    rows += [
        (d25, "10:00", 100.0, 100.1, 99.9, 100.0),
        (d25, "12:00", 100.0, 100.0 * (1 + 0.004), 99.7, 100.0),  # touch exactly at P10 (1 + b)
        (d26, "10:00", 100.0, 100.0 * (1 + 0.004) - 0.0001, 100.0 * (1 - 0.004), 100.0),
        (d27, "10:00", 100.0, 100.2, 99.9, 100.0),
        (d27, "15:59", 100.0, 101.0, 100.0, 100.0),  # C-1 counts
        (d29, "10:00", 100.0, 100.1, 99.9, 100.0),
        (d29, "12:59", 100.0, 100.0 * (1 + 0.002), 100.0, 100.0),  # half-day C-1
        (d29, "13:30", 100.0, 105.0, 100.0, 100.0),  # after-hours print: never counts
    ]
    synthetic.write_bars(tmp_path, "SPY", _frame(rows))
    up = features.target(tmp_path, "SPY", cal, 0.004, "up")
    assert up["session"].to_list() == [d25, d26, d27, d29]
    assert up["p10"].to_list() == [100.0] * 4
    # d25 touch -> 1; d26 a hair under -> 0; d27 spike at 15:59 -> 1; d29 high 100.2, 105 ignored -> 0
    assert up["y"].to_list() == [1, 0, 1, 0]
    assert up["y"].dtype == pl.Int8
    # b = 0.002: d29 now touches exactly at its last bar (12:59) (the 13:30 print is still
    # ignored: at 105 it would have hit at b = 0.004 too); d27's 100.2 at 10:00 also touches.
    assert features.target(tmp_path, "SPY", cal, 0.002, "up")["y"].to_list() == [1, 1, 1, 1]
    # mirror at 0.004: only d26 reaches the level below P10 (low exactly 99.6).
    assert features.target(tmp_path, "SPY", cal, 0.004, "down")["y"].to_list() == [0, 1, 0, 0]


@pytest.mark.parametrize(("tau", "sigma"), [(360, 0.0003), (180, 0.0004)])
def test_formula_matches_a_monte_carlo_of_a_driftless_walk(tau, sigma):
    b, n_paths = config.TARGET_B, 20_000
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(0.0, sigma, (n_paths, tau)), axis=1)
    hit = (walk.max(axis=1) >= math.log1p(b)).mean()
    p = float(baselines.formula_raw(np.array([sigma]), np.array([tau]), b)[0])
    # The formula is for continuous monitoring; the simulation checks only 'tau' minute marks, so
    # it undershoots by about 2 * phi(z) * 0.5826 / sqrt(tau) (Broadie-Glasserman-Kou), at most
    # ~0.02 here, plus Monte Carlo error 3 * sqrt(p (1-p) / n_paths) = 0.011. Tolerance 0.03.
    assert abs(p - hit) < 0.03, (p, hit)
    assert p > hit  # and the sign of that bias is the expected one


def test_fitted_baselines_recover_a_planted_logistic():
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 1.0, 20_000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-(0.5 - 1.5 * z)))).astype(int)
    sigma = config.TARGET_B / (z - z.min() + 1.0)  # tau = 1 so that formula z = ln(1+b)/sigma
    pred = baselines.fit_calibrated(sigma, np.ones_like(sigma), y, config.TARGET_B)
    assert abs(pred(sigma, np.ones_like(sigma)).mean() - y.mean()) < 1e-3  # logit MLE: mean match
    assert baselines.base_rate(y) == pytest.approx(y.mean())
    gao = baselines.fit_gao(z * 0.01, y)
    assert gao(np.array([0.0]))[0] == pytest.approx(1 / (1 + np.exp(-0.5)), abs=0.02)


# ---- look-ahead check -------------------------------------------------------------------------

START, END = date(2019, 6, 1), date(2019, 12, 31)  # holds three half-days


def _write_daily(path):
    days = [d for i in range(260) if (d := date(2019, 5, 1) + timedelta(i)).weekday() < 5]
    for name, base in (("VIXCLS", 15.0), ("VXVCLS", 18.0)):
        body = "".join(f"{d},{base + d.timetuple().tm_yday / 10:.2f}\n" for d in days)
        (path / f"{name}.csv").write_text(f"observation_date,{name}\n{body}")
    (path / "fomc_meetings.csv").write_text("date\n2019-07-30\n2019-09-17\n")
    (path / "fomc_emergency.csv").write_text("date\n2019-08-05\n")
    (path / "fomc_cancelled.csv").write_text("date,cancelled_on\n2019-10-30,2019-10-01\n")
    (path / "release_dates.csv").write_text("date,release\n2019-09-11,cpi\n2019-08-02,payroll\n")


def _build(root, daily, cal, leak=False):
    table = returns.daily_table(root, "SPY", cal)
    ctrl = controls.controls(table, cal, daily)
    out = features.features(table, ctrl, root, "SPY", cal)
    if leak:  # PLANTED LEAK: the session's own high after 10:00
        late = (
            bars.rth_bars(root, "SPY", cal)
            .filter(pl.col("minute") >= time(10, 0))
            .group_by("session")
            .agg(leak_high=pl.col("high").max())
            .collect()
        )
        out = out.join(late, on="session", how="left")
    return out


def _perturb(src, dst, mask):
    """Rewrite src's SPY bars into dst with every price scaled by 1.03 where `mask` holds."""
    raw = bars.scan_bars(src, "SPY").collect()
    scale = pl.when(mask).then(1.03).otherwise(1.0)
    out = raw.with_columns(pl.col("open", "high", "low", "close") * scale)
    synthetic.write_bars(dst, "SPY", out.drop("session", "minute"))


def _same(a, b, rows):
    return a.filter(pl.col("session").is_in(rows)).equals(b.filter(pl.col("session").is_in(rows)))


def _is_look_ahead_free(root, daily, cal, scratch, leak):
    """True if the features of every session <= t do not move when (A) t's own bars at/after 10:00
    change, or (B) every bar of every later session changes. Rows after t may move (lags, A)."""
    sessions = cal["session"].to_list()
    t = sessions[60]
    upto = [s for s in sessions if s <= t]
    base = _build(root, daily, cal, leak)
    masks = {
        "own_after_10": (pl.col("session") == t) & (pl.col("minute") >= time(10, 0)),
        "later_sessions": pl.col("session") > t,
    }
    ok = True
    for name, mask in masks.items():
        (scratch / name).mkdir(parents=True)
        _perturb(root, scratch / name, mask)
        moved = _build(scratch / name, daily, cal, leak)
        assert not _same(base, moved, sessions[61:])  # the perturbation does reach later rows
        ok &= _same(base, moved, upto)
    return ok


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root, daily = tmp_path_factory.mktemp("bars"), tmp_path_factory.mktemp("daily")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DATA_START", START)
    mp.setattr(config, "DATA_END", END)
    cal = calendar.sessions(START, END)
    synthetic.write_synthetic(root, ["SPY"], START, END, beta=0.3, seed=5)
    _write_daily(daily)
    yield root, daily, cal, tmp_path_factory.mktemp("alts")
    mp.undo()


def test_features_do_not_look_ahead_and_a_planted_leak_is_caught(world):
    root, daily, cal, scratch = world
    assert _is_look_ahead_free(root, daily, cal, scratch / "real", leak=False)
    assert not _is_look_ahead_free(root, daily, cal, scratch / "leaky", leak=True)


def test_ret_prev_never_bridges_a_dropped_session_and_tau_is_180_on_half_days(world):
    root, daily, cal, _ = world
    out = _build(root, daily, cal)
    assert set(out.filter(pl.col("is_half_day") == 1.0)["tau"]) == {180.0}
    assert set(out.filter(pl.col("is_half_day") == 0.0)["tau"]) == {360.0}
    # ret_prev of session s is (close of s-1) / (close of s-2) - 1: check one row against the bars
    # table by an independent route (the 15:59 close of each day read straight from the bars).
    closes = (
        bars.rth_bars(root, "SPY", cal)  # cut at the close, so the last bar is C-1
        .group_by("session")
        .agg(close=pl.col("close").sort_by("minute").last())
        .collect()
        .sort("session")
    )
    c = closes["close"].to_numpy()
    s = out["session"].to_list()
    i = 50
    assert s[i - 1] == closes["session"][i - 1] and s[i] == closes["session"][i]
    assert out["ret_prev"][i] == pytest.approx(c[i - 1] / c[i - 2] - 1)
    assert out["ret_prev"][0] is None and out["ret_prev"][1] is None


def test_ret_prev_and_r_on_never_bridge_a_dropped_session(world, monkeypatch):
    root, daily, cal, scratch = world
    sessions = cal["session"].to_list()
    gone = sessions[40]
    raw = bars.scan_bars(root, "SPY").collect()
    cut = raw.filter(~((pl.col("session") == gone) & (pl.col("minute") == time(9, 59))))
    (scratch / "dropped").mkdir()
    synthetic.write_bars(scratch / "dropped", "SPY", cut.drop("session", "minute"))
    out = _build(scratch / "dropped", daily, cal).sort("session")
    assert gone not in out["session"].to_list()
    nxt, after = (out.filter(pl.col("session") == s).row(0, named=True) for s in sessions[41:43])
    assert nxt["ret_prev"] is None and nxt["r_on"] is None  # previous session is gone
    assert after["ret_prev"] is None  # two sessions back is gone
    assert after["r_on"] is not None
    assert out.filter(pl.col("session") == sessions[43])["ret_prev"].item() is not None
