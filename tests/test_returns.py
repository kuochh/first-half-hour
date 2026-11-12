"""The daily returns table: hand-made bars, the r1 identity, and a planted slope."""

import math
from datetime import date, datetime, time

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from first_half_hour import calendar, config, returns, synthetic


def _bars(rows):
    """rows: (date, 'HH:MM', open, close). High/low bracket the bar so the reader accepts it."""
    return pl.DataFrame(
        {
            "timestamp": [datetime.combine(d, time.fromisoformat(t)) for d, t, _, _ in rows],
            "open": [o for *_, o, _ in rows],
            "high": [max(o, c) for *_, o, c in rows],
            "low": [min(o, c) for *_, o, c in rows],
            "close": [c for *_, c in rows],
            "volume": [100] * len(rows),
        }
    )


def _use_window(monkeypatch, start, end):
    monkeypatch.setattr(config, "DATA_START", start)
    monkeypatch.setattr(config, "DATA_END", end)
    return calendar.sessions(start, end)


def test_returns_from_hand_made_bars_including_a_half_day(tmp_path, monkeypatch):
    # 2019-11-25 Mon, 26 Tue, 27 Wed, (28 Thanksgiving), 29 Fri = half-day, closes 13:00.
    cal = _use_window(monkeypatch, date(2019, 11, 25), date(2019, 11, 29))
    d25, d26, d27, d29 = (
        date(2019, 11, 25),
        date(2019, 11, 26),
        date(2019, 11, 27),
        date(2019, 11, 29),
    )
    rows = []
    for d, o, c10, c61, c31, cl in [
        (d25, 100.0, 101.0, 102.0, 103.0, 104.0),  # regular day
        (d26, 105.0, 106.0, 107.0, 108.0, 110.0),  # regular day
        (d27, 111.0, 112.0, 113.0, 114.0, 115.0),  # 09:59 bar missing: dropped
        (d29, 116.0, 117.0, 118.0, 119.0, 121.0),  # half-day
    ]:
        regular = d != d29
        tail = ["14:59", "15:29", "15:59"] if regular else ["11:59", "12:29", "12:59"]
        rows += [(d, "09:30", o, o + 0.5), (d, "09:31", o + 0.5, o + 0.25)]
        if d != d27:
            rows.append((d, "09:59", o + 0.25, c10))
        rows += [(d, t, 0.0 + p, p) for t, p in zip(tail, (c61, c31, cl), strict=True)]
    rows += [(d29, "13:30", 121.0, 125.0), (d29, "15:59", 125.0, 130.0)]  # after hours: cut
    rows = [(d, t, o if o else c, c) for d, t, o, c in rows]
    synthetic.write_bars(tmp_path, "SPY", _bars(rows))

    t = returns.daily_table(tmp_path, "SPY", cal)
    assert t["session"].to_list() == [d25, d26, d29]  # 27th dropped
    assert t["is_half_day"].to_list() == [False, False, True]
    row = {r["session"]: r for r in t.iter_rows(named=True)}
    # Expected values worked by hand from the bars above.
    assert row[d25]["r1"] is None and row[d25]["r_on"] is None  # no previous session in window
    assert row[d25]["r12"] == pytest.approx(103 / 102 - 1)
    assert row[d25]["r13"] == pytest.approx(104 / 103 - 1)
    assert row[d26]["prev_close"] == 104.0
    assert row[d26]["r1"] == pytest.approx(106 / 104 - 1)
    assert row[d26]["r_on"] == pytest.approx(105 / 104 - 1)
    assert row[d26]["r_fh"] == pytest.approx(106 / 105 - 1)
    assert row[d26]["rod"] == pytest.approx(108 / 104 - 1)
    assert row[d26]["r13"] == pytest.approx(110 / 108 - 1)
    # Half-day: C = 13:00, so r12 = 12:00-12:30 (bars 11:59, 12:29) and r13 = 12:30-13:00.
    assert row[d29]["r12"] == pytest.approx(119 / 118 - 1)
    assert row[d29]["r13"] == pytest.approx(121 / 119 - 1)
    assert row[d29]["p_close"] == 121.0  # the 15:59 after-hours print is not the close
    # Previous session (27th) was dropped: never bridge two nights.
    assert row[d29]["prev_close"] is None
    assert row[d29]["r1"] is None and row[d29]["r_on"] is None and row[d29]["rod"] is None
    # rv_fh on 26th: first bar open 105 -> close 105.5, 09:31 close 105.25, 09:59 close 106.
    rv = math.log(105.5 / 105) ** 2 + math.log(105.25 / 105.5) ** 2 + math.log(106 / 105.25) ** 2
    assert row[d26]["rv_fh"] == pytest.approx(rv, rel=1e-12)
    assert row[d26]["vol_fh"] == 300  # three bars of volume 100 up to 09:59


def test_r1_equals_overnight_times_first_half_hour(tmp_path, monkeypatch):
    start, end = date(2019, 1, 1), date(2019, 3, 31)
    cal = _use_window(monkeypatch, start, end)
    synthetic.write_synthetic(tmp_path, ["SPY"], start, end, beta=0.3, seed=1)
    t = returns.daily_table(tmp_path, "SPY", cal)
    both = t.drop_nulls(["r1", "r_on", "r_fh"])
    assert both.height == t.height - 1  # only the first session lacks a previous close
    lhs, rhs = (1 + both["r1"]).to_numpy(), ((1 + both["r_on"]) * (1 + both["r_fh"])).to_numpy()
    assert np.abs(lhs - rhs).max() < 1e-12


def test_a_planted_slope_is_recovered(tmp_path, monkeypatch):
    start, end = date(2017, 1, 1), date(2019, 12, 31)  # about 755 sessions
    cal = _use_window(monkeypatch, start, end)
    beta = 0.5
    synthetic.write_synthetic(tmp_path, ["SPY"], start, end, beta=beta, seed=7)
    t = returns.daily_table(tmp_path, "SPY", cal).drop_nulls("r1")
    fit = sm.OLS(t["r13"].to_numpy(), sm.add_constant(t["r1"].to_numpy())).fit()
    # Tolerance derived from the planted noise, not from the fit: the OLS slope has standard
    # error sd(noise) / (sd(r1) * sqrt(T)), with sd(r1) = hypot(overnight, first-half-hour sd).
    sd_r1 = math.hypot(synthetic.SD_OVERNIGHT, synthetic.SD_FIRST_HALF_HOUR)
    se = synthetic.SD_R13_NOISE / (sd_r1 * math.sqrt(t.height))
    assert abs(fit.params[1] - beta) < 4 * se, (fit.params[1], 4 * se)
    assert abs(fit.params[0]) < 4 * synthetic.SD_R13_NOISE / math.sqrt(t.height)


def _regular_day_rows(d, base, drop_open_bar=False):
    """A regular session of bars; every price is `base` plus a fixed offset."""
    rows = [(d, "09:30", base, base + 0.5), (d, "09:31", base + 0.5, base + 0.25)]
    rows += [(d, "09:59", base + 0.25, base + 1)]
    rows += [(d, t, base + 2 + i, base + 2 + i) for i, t in enumerate(("14:59", "15:29", "15:59"))]
    return [r for r in rows if not (drop_open_bar and r[1] == "09:30")]


def test_a_row_never_uses_later_sessions(tmp_path, monkeypatch):
    # Mon 18 - Fri 22 Nov 2019. Wrecking every price of the sessions after Wed 20th must leave
    # the rows for the 18th-20th bit-identical (r1 of the 20th uses only the 19th's close).
    days = [date(2019, 11, d) for d in (18, 19, 20, 21, 22)]
    cal = _use_window(monkeypatch, days[0], days[-1])
    tables = []
    for i, scale in enumerate((1.0, 3.7)):
        rows = []
        for j, d in enumerate(days):
            base = 100.0 + 10 * j
            rows += _regular_day_rows(d, base * (scale if j >= 3 else 1.0))
        root = tmp_path / str(i)
        synthetic.write_bars(root, "SPY", _bars(rows))
        tables.append(returns.daily_table(root, "SPY", cal))
    upto = pl.col("session") <= days[2]
    assert tables[0].height == 5 and tables[0].filter(upto).equals(tables[1].filter(upto))
    assert not tables[0].equals(tables[1])  # the perturbation did change the later rows


def test_a_missing_opening_bar_nulls_the_overnight_split_but_keeps_r1(tmp_path, monkeypatch):
    d18, d19 = date(2019, 11, 18), date(2019, 11, 19)
    cal = _use_window(monkeypatch, d18, d19)
    rows = _regular_day_rows(d18, 100.0) + _regular_day_rows(d19, 110.0, drop_open_bar=True)
    synthetic.write_bars(tmp_path, "SPY", _bars(rows))
    row = (
        returns.daily_table(tmp_path, "SPY", cal)
        .filter(pl.col("session") == d19)
        .row(0, named=True)
    )
    # No 09:30 bar: p_open, r_on and r_fh are null, but r1 = 09:59 close / previous close - 1
    # = 111 / 104 - 1 (18th closes at its 15:59 bar, 100 + 2 + 2 = 104; 19th's 09:59 close is 111).
    assert row["p_open"] is None and row["r_on"] is None and row["r_fh"] is None
    assert row["r1"] == pytest.approx(111 / 104 - 1)
