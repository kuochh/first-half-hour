"""XNYS sessions and their wall-clock closes, as polars.

`exchange_calendars` speaks pandas and UTC; this module converts at the boundary, so nothing
pandas leaves it.
"""

from datetime import date, time

import exchange_calendars as xcals
import polars as pl

from first_half_hour.config import EARLY_CLOSE, REGULAR_CLOSE, SESSION_OPEN, TIMEZONE


def sessions(start: date, end: date) -> pl.DataFrame:
    """Every XNYS session in [start, end]: columns `session` (Date), `close` (Time), `is_half_day`.

    Any open other than 09:30 or close other than 16:00 / 13:00 fails loud.
    """
    assert start < end, f"start must be before end, got {start} .. {end}"
    cal = xcals.get_calendar("XNYS", start=start.isoformat(), end=end.isoformat())
    days: list[date] = []
    closes: list[time] = []
    for day, open_utc, close_utc in zip(cal.sessions, cal.opens, cal.closes, strict=True):
        open_wall = open_utc.tz_convert(TIMEZONE).time()
        close_wall = close_utc.tz_convert(TIMEZONE).time()
        assert open_wall == SESSION_OPEN, f"{day.date()} opens at {open_wall}"
        assert close_wall in (REGULAR_CLOSE, EARLY_CLOSE), f"{day.date()} closes at {close_wall}"
        days.append(day.date())
        closes.append(close_wall)
    assert days and len(set(days)) == len(days), "empty or repeated sessions"
    return pl.DataFrame(
        {
            "session": pl.Series(days, dtype=pl.Date),
            "close": pl.Series(closes, dtype=pl.Time),
        }
    ).with_columns(is_half_day=pl.col("close") == EARLY_CLOSE)
