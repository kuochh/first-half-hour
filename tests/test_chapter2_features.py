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


@pytest.fixture(scope="module")
def unit_walks():
    """Per tau: the running max and min of 20,000 driftless unit-variance-per-minute Gaussian log
    price walks with 20 sub-steps per minute (built in time blocks to keep memory small). A walk
    with per-minute volatility sigma is sigma times the unit walk, so sigmas share the draws."""
    n_paths, sub, rng = 20_000, 20, np.random.default_rng(11)
    out = {}
    for tau in (360, 180):
        pos, hi, lo = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)
        for _ in range(tau * sub // 600):
            steps = rng.standard_normal((n_paths, 600), dtype=np.float32) / math.sqrt(sub)
            path = pos[:, None] + np.cumsum(steps, axis=1)
            hi, lo, pos = (
                np.maximum(hi, path.max(axis=1)),
                np.minimum(lo, path.min(axis=1)),
                path[:, -1],
            )
        out[tau] = (hi, lo)
    return out


@pytest.mark.parametrize("side", ["up", "down"])
@pytest.mark.parametrize("tau", [360, 180])
@pytest.mark.parametrize("sigma", [0.0002, 0.0003])
def test_formula_matches_a_monte_carlo_of_a_driftless_walk(unit_walks, sigma, tau, side):
    # Independent route: the walk is simulated directly; the formula is never used to build it.
    # 20 sub-steps per minute leave a discrete-monitoring bias of at most ~0.001 (BGK), far under
    # the 4 standard errors (~0.014) allowed.
    b = config.TARGET_B
    hi, lo = unit_walks[tau]
    if side == "up":
        hit = (sigma * hi >= math.log1p(b)).mean()
    else:
        hit = (sigma * -lo >= -math.log1p(-b)).mean()
    se = math.sqrt(hit * (1 - hit) / hi.size)
    p = float(baselines.formula_raw(np.array([sigma]), np.array([tau]), b, side)[0])
    assert abs(p - hit) < 4 * se, (p, hit, se)


def test_formula_barrier_takes_a_side():
    # worked by hand: ln(1.004) = 0.00399202, -ln(0.996) = 0.00400802
    s, t = np.array([0.0003]), np.array([360.0])
    assert baselines.formula_z(s, t, 0.004, "up")[0] == pytest.approx(
        0.00399202 / (0.0003 * math.sqrt(360)), rel=1e-5
    )
    assert baselines.formula_z(s, t, 0.004, "down")[0] == pytest.approx(
        0.00400802 / (0.0003 * math.sqrt(360)), rel=1e-5
    )
    assert (
        baselines.formula_raw(s, t, 0.004, "down")[0] < baselines.formula_raw(s, t, 0.004, "up")[0]
    )


def test_fitted_baselines_recover_a_planted_logistic():
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 1.0, 20_000)
    truth = lambda v: 1 / (1 + np.exp(-(0.5 - 1.5 * v)))
    y = (rng.uniform(size=z.size) < truth(z)).astype(int)
    shift = 1.0 - z.min()
    sigma = config.TARGET_B / (z + shift)  # tau = 1, so formula z = ln(1+b)/b * (z + shift)
    ones = np.ones_like(sigma)
    pred = baselines.fit_calibrated(sigma, ones, y, config.TARGET_B)
    assert abs(pred(sigma, ones).mean() - y.mean()) < 1e-3  # logit MLE: mean match
    assert baselines.base_rate(y) == pytest.approx(y.mean())
    # The fit must USE z: its forecasts at planted z = -1 and +1 follow the planted curve, which an
    # intercept-only fit (one constant, about 0.5) cannot do (curve: 0.88 and 0.12).
    for zz in (-1.0, 1.0):
        sig = np.array([config.TARGET_B / (zz + shift)])
        assert pred(sig, np.ones(1))[0] == pytest.approx(truth(zz), abs=0.03)
    gao = baselines.fit_gao(z * 0.01, y)
    assert gao(np.array([0.0]))[0] == pytest.approx(truth(0.0), abs=0.02)
    assert gao(np.array([0.01]))[0] == pytest.approx(truth(1.0), abs=0.03)
    assert gao(np.array([-0.01]))[0] == pytest.approx(truth(-1.0), abs=0.03)


def test_an_exact_touch_survives_float_rounding(tmp_path, monkeypatch):
    # 222.50 * 1.004 evaluates to 223.39000000000001 in floats, above the 223.39 bar; both are cent
    # grid levels, so a high of 223.39 is an exact touch. Same on the down side: 382.50 * 0.996.
    d1, d2 = date(2019, 11, 25), date(2019, 11, 26)
    cal = _use_window(monkeypatch, d1, d2)
    assert 222.5 * (1 + 0.004) > 223.39 and 382.5 * (1 - 0.004) < 380.97  # plain floats would miss
    rows = []
    for d in (d1, d2):
        rows += [(d, "09:30", 222.5, 222.5, 222.5, 222.5), (d, "09:59", 222.5, 222.5, 222.5, 222.5)]
    rows += [(d1, "10:00", 222.5, 223.39, 222.5, 222.5), (d2, "10:00", 222.5, 223.38, 222.5, 222.5)]
    synthetic.write_bars(tmp_path, "SPY", _frame(rows))
    assert features.target(tmp_path, "SPY", cal, 0.004, "up")["y"].to_list() == [1, 0]
    rows = []
    for d in (d1, d2):
        rows += [(d, "09:30", 382.5, 382.5, 382.5, 382.5), (d, "09:59", 382.5, 382.5, 382.5, 382.5)]
    rows += [(d1, "10:00", 382.5, 382.5, 380.97, 382.5), (d2, "10:00", 382.5, 382.5, 380.98, 382.5)]
    (tmp_path / "b").mkdir()
    synthetic.write_bars(tmp_path / "b", "SPY", _frame(rows))
    assert features.target(tmp_path / "b", "SPY", cal, 0.004, "down")["y"].to_list() == [1, 0]


# ---- look-ahead check -------------------------------------------------------------------------

START, END = date(2019, 6, 1), date(2019, 12, 31)  # holds three half-days


def _write_daily(path, vix_shift=0.0, after=None):
    """VIX/VXV files; values dated on/after `after` are moved by `vix_shift`."""
    days = [d for i in range(260) if (d := date(2019, 5, 1) + timedelta(i)).weekday() < 5]
    for name, base in (("VIXCLS", 15.0), ("VXVCLS", 18.0)):
        body = "".join(
            f"{d},{base + d.timetuple().tm_yday / 10 + (vix_shift if after and d >= after else 0):.2f}\n"
            for d in days
        )
        (path / f"{name}.csv").write_text(f"observation_date,{name}\n{body}")
    (path / "fomc_meetings.csv").write_text("date\n2019-07-30\n2019-09-17\n")
    (path / "fomc_emergency.csv").write_text("date\n2019-08-05\n")
    (path / "fomc_cancelled.csv").write_text("date,cancelled_on\n2019-10-30,2019-10-01\n")
    (path / "release_dates.csv").write_text("date,release\n2019-09-11,cpi\n2019-08-02,payroll\n")


LEAKS = (
    "fullday_volume",
    "same_day_vix",
    "last_half_hour_ret",
    "next_r_fh",
    "afternoon_rv",
    "late_high",
)


def _leak(out, root, daily, cal, kind):
    """Add one column that reads information from after 10:00 on t, or from later rows."""
    rth = bars.rth_bars(root, "SPY", cal)
    if kind == "fullday_volume":
        col = rth.group_by("session").agg(leak=pl.col("volume").sum()).collect()
    elif kind == "late_high":
        col = (
            rth.filter(pl.col("minute") >= time(10, 0))
            .group_by("session")
            .agg(leak=pl.col("high").max())
            .collect()
        )
    elif kind == "last_half_hour_ret":
        col = (
            rth.filter(pl.col("minute") >= time(15, 30))
            .group_by("session")
            .agg(
                leak=pl.col("close").sort_by("minute").last()
                / pl.col("close").sort_by("minute").first()
            )
            .collect()
        )
    elif kind == "afternoon_rv":
        col = (
            rth.filter(pl.col("minute") >= time(12, 0))
            .sort("session", "minute")
            .group_by("session")
            .agg(leak=(pl.col("close") / pl.col("close").shift(1)).log().pow(2).sum())
            .collect()
        )
    elif kind == "same_day_vix":
        col = pl.read_csv(daily / "VIXCLS.csv", try_parse_dates=True).select(
            session="observation_date", leak="VIXCLS"
        )
    elif kind == "next_r_fh":
        return out.sort("session").with_columns(leak=pl.col("r_fh").shift(-1))
    return out.join(col, on="session", how="left")


def _build(root, daily, cal, leak=None):
    table = returns.daily_table(root, "SPY", cal)
    ctrl = controls.controls(table, cal, daily)
    out = features.features(table, ctrl, root, "SPY", cal)
    return out if leak is None else _leak(out, root, daily, cal, leak)


def _perturb(src, dst, mask, seed):
    """Rewrite src's SPY bars into dst: where `mask` holds every bar gets its own random factor
    (lognormal, sd 1%) on open, high, low and close together (so the bar stays consistent), and
    a random volume."""
    raw = bars.scan_bars(src, "SPY").collect()
    rng = np.random.default_rng(seed)
    f = pl.Series(rng.lognormal(0.0, 0.01, raw.height))
    vol = pl.Series(rng.integers(1, 10**7, raw.height))
    scale = pl.when(mask).then(f).otherwise(1.0)
    out = raw.with_columns(
        pl.col("open", "high", "low", "close") * scale,
        volume=pl.when(mask).then(vol).otherwise(pl.col("volume")).cast(raw["volume"].dtype),
    )
    synthetic.write_bars(dst, "SPY", out.drop("session", "minute"))


def _same(a, b, rows):
    return a.filter(pl.col("session").is_in(rows)).equals(b.filter(pl.col("session").is_in(rows)))


def _is_look_ahead_free(root, daily, cal, scratch, leak=None):
    """True if the features of every session <= t do not move when (A) t's own bars at/after 10:00
    change (random factor per bar, random volume), or (B) every bar of every later session
    changes, together with every VIX/VXV row dated on/after t. Rows after t may move."""
    sessions = cal["session"].to_list()
    t = sessions[60]
    upto = [s for s in sessions if s <= t]
    base = _build(root, daily, cal, leak)
    masks = {
        "own_after_10": (pl.col("session") == t) & (pl.col("minute") >= time(10, 0)),
        "later_sessions": pl.col("session") > t,
    }
    other_daily = scratch / "daily"
    other_daily.mkdir(parents=True)
    _write_daily(other_daily, vix_shift=40.0, after=t)
    ok = True
    for i, (name, mask) in enumerate(masks.items()):
        (scratch / name).mkdir()
        _perturb(root, scratch / name, mask, seed=100 + i)
        moved = _build(scratch / name, other_daily, cal, leak)
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


def test_features_do_not_look_ahead(world):
    root, daily, cal, scratch = world
    assert _is_look_ahead_free(root, daily, cal, scratch / "real")


@pytest.mark.parametrize("leak", LEAKS)
def test_every_planted_leak_is_caught(world, leak):
    root, daily, cal, scratch = world
    assert not _is_look_ahead_free(root, daily, cal, scratch / leak, leak)


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


def test_a_short_first_half_hour_keeps_a_positive_sigma(world, caplog):
    root, daily, cal, scratch = world
    gone = cal["session"][30]
    raw = bars.scan_bars(root, "SPY").collect()
    cut = raw.filter(~((pl.col("session") == gone) & (pl.col("minute") == time(9, 45))))
    (scratch / "short").mkdir()
    synthetic.write_bars(scratch / "short", "SPY", cut.drop("session", "minute"))
    with caplog.at_level("INFO"):
        out = _build(scratch / "short", daily, cal)
    assert out.filter(pl.col("session") == gone)["sigma"].item() > 0
    assert any("1 with a short first half-hour" in r.message for r in caplog.records)
