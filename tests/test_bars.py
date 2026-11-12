"""The bar reader on synthetic data: one month (2019-11, with a half-day on the 29th)."""

import hashlib
import json
from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from first_half_hour import bars, calendar, config

START, END = date(2019, 11, 1), date(2019, 11, 30)


def _write_month(root, rows, symbol="SPY", month="2019-11", sha=None):
    d = root / symbol
    d.mkdir(parents=True, exist_ok=True)
    lines = [config.CSV_HEADER] + [
        f"{ts:%Y-%m-%d %H:%M:%S},1.0,1.0,1.0,1.0,100" for ts in sorted(rows, reverse=True)
    ]
    body = ("\n".join(lines) + "\n").encode()
    (d / f"{symbol}_1min_{month}.csv").write_bytes(body)
    entry = {"status": "ok", "bytes": len(body), "sha256": sha or hashlib.sha256(body).hexdigest()}
    (d / config.MANIFEST_NAME).write_text(json.dumps({"symbol": symbol, "months": {month: entry}}))


def _full_month(cal, after_hours_on_half_days=True, skip=()):
    rows = []
    for session, close, _ in cal.iter_rows():
        end = time(15, 59) if after_hours_on_half_days else close
        t = datetime.combine(session, config.SESSION_OPEN)
        while t.time() <= end and (t.time() < close or after_hours_on_half_days):
            if (session, t.time()) not in skip:
                rows.append(t)
            t += timedelta(minutes=1)
    return rows


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(config, "DATA_START", START)
    monkeypatch.setattr(config, "DATA_END", END)
    return calendar.sessions(START, END)


def test_calendar_marks_the_half_day():
    cal = calendar.sessions(START, END)
    half = cal.filter(pl.col("is_half_day"))
    assert half["session"].to_list() == [date(2019, 11, 29)]
    assert half["close"].item() == config.EARLY_CLOSE
    assert date(2019, 11, 28) not in cal["session"].to_list()  # Thanksgiving


def test_half_day_after_hours_prints_are_cut_at_the_close(tmp_path, window):
    _write_month(tmp_path, _full_month(window))
    rth = bars.rth_bars(tmp_path, "SPY", window).collect()
    half = rth.filter(pl.col("session") == date(2019, 11, 29))
    assert half["minute"].max() == time(12, 59)
    assert rth.filter(pl.col("session") == date(2019, 11, 27)).height == 390


def test_a_tampered_file_fails_its_manifest_check(tmp_path, window):
    _write_month(tmp_path, _full_month(window), sha="0" * 64)
    with pytest.raises(AssertionError, match="sha256"):
        bars.verified_month_files(tmp_path, "SPY")


def test_a_repeated_bar_fails_loud(tmp_path, window):
    rows = _full_month(window)
    _write_month(tmp_path, rows + [rows[0]])
    with pytest.raises(AssertionError, match="repeated"):
        bars.rth_bars(tmp_path, "SPY", window)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (datetime(2019, 11, 12, 9, 29), "before the open"),
        (datetime(2019, 11, 9, 10, 0), "non-sessions"),  # a Saturday
        (datetime(2019, 11, 12, 10, 0, 30), "whole minute"),
    ],
)
def test_an_impossible_bar_fails_loud(tmp_path, window, extra, message):
    _write_month(tmp_path, _full_month(window) + [extra])
    with pytest.raises(AssertionError, match=message):
        bars.rth_bars(tmp_path, "SPY", window)


def test_a_half_day_with_only_after_hours_prints_is_dropped_not_lost(tmp_path, window):
    half = date(2019, 11, 29)
    rows = [t for t in _full_month(window) if t.date() != half or t.time() >= time(13, 0)]
    _write_month(tmp_path, rows)
    dropped = bars.sessions_missing_key_bar(bars.rth_bars(tmp_path, "SPY", window), window)
    assert dropped["session"].to_list() == [half]
    assert dropped["lacking"].item() == "09:59, 11:59, 12:29, 12:59"


def test_a_missing_session_fails_loud(tmp_path, window):
    gone = date(2019, 11, 12)
    _write_month(tmp_path, [t for t in _full_month(window) if t.date() != gone])
    with pytest.raises(AssertionError, match="missing sessions"):
        bars.rth_bars(tmp_path, "SPY", window)


def test_only_sessions_missing_a_key_bar_are_dropped(tmp_path, window):
    skip = {
        (date(2019, 11, 5), time(9, 59)),  # key bar: dropped
        (date(2019, 11, 29), time(12, 29)),  # half-day C-31: dropped
        (date(2019, 11, 6), time(11, 0)),  # a halt-like gap, every key bar present: kept
    }
    _write_month(tmp_path, _full_month(window, skip=skip))
    dropped = bars.sessions_missing_key_bar(bars.rth_bars(tmp_path, "SPY", window), window)
    assert dropped["session"].to_list() == [date(2019, 11, 5), date(2019, 11, 29)]
    assert dropped["lacking"].to_list() == ["09:59", "12:29"]
