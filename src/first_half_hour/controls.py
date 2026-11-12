"""Controls known by 10:00 of session t, one row per session of the daily table.

Every column uses only information available by 10:00: the first-half-hour variance and volume
of session t itself, anything dated before t (VIX, previous realised variance, the previous 20
sessions' volume, expanding terciles) and facts fixed in advance by the calendar (FOMC and data
releases, expiries, month-end, half-days). Row t never reads a row after t.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from first_half_hour import calendar, config

WEEKDAY_COLUMNS = ("dow_tue", "dow_wed", "dow_thu", "dow_fri")
CONTROL_COLUMNS = (
    "rv_fh", "rv_day_lag", "relvol", *WEEKDAY_COLUMNS, "fomc", "cpi", "payroll",
    "vix_lag", "vix_ratio_lag", "opex_monthly", "opex_quarterly", "daily_expiry_era",
    "month_end", "is_half_day",
)  # fmt: skip
"""Columns entered as controls in item 7 (booleans are cast to 0/1 there)."""


def _read_dates(path: Path) -> pl.DataFrame:
    assert path.exists(), f"missing input file {path}"
    return pl.read_csv(path, try_parse_dates=True)


def _read_fred(daily_dir: Path, series: str) -> pl.DataFrame:
    path = daily_dir / f"{series}.csv"
    assert path.exists(), f"missing FRED file {path}"
    frame = pl.read_csv(
        path, null_values=[config.FRED_MISSING_TOKEN, ""], schema_overrides={series: pl.Float64}
    )
    return (
        frame.with_columns(pl.col(config.FRED_DATE_COLUMN).str.to_date())
        .rename({config.FRED_DATE_COLUMN: "obs_date", series: "value"})
        .drop_nulls("value")
        .sort("obs_date")
    )


def _before(sessions: pl.DataFrame, obs: pl.DataFrame, name: str) -> pl.DataFrame:
    """For each session the latest observation dated STRICTLY before it (never the session's own)."""
    out = sessions.select("session").join_asof(
        obs.rename({"value": name}),
        left_on="session",
        right_on="obs_date",
        strategy="backward",
        allow_exact_matches=False,
    )
    assert out[name].null_count() == 0, f"{name}: sessions with no earlier observation"
    stale = (out["session"] - out["obs_date"]).dt.total_days().max()
    assert stale <= config.VIX_MAX_STALE_DAYS, f"{name}: observation {stale} days before a session"
    return out.select("session", name)


def _next_session_on_or_after(ext: pl.DataFrame, days: pl.Series, strict: bool) -> pl.DataFrame:
    """Map each date to the first session on/after it (strictly after if `strict`)."""
    left = pl.DataFrame({"day": days}).sort("day")
    right = ext.select(pl.col("session").alias("news_session")).sort("news_session")
    out = left.join_asof(
        right.with_columns(key=pl.col("news_session")),
        left_on="day",
        right_on="key",
        strategy="forward",
        allow_exact_matches=not strict,
    )
    assert out["news_session"].null_count() == 0, "calendar too short to place an event"
    assert (out["news_session"] - out["day"]).dt.total_days().max() <= 5, "event placed too far out"
    return out.select("day", "news_session")


def _event_flag(sessions: pl.Series, ext: pl.DataFrame, days: pl.Series, strict: bool) -> pl.Series:
    mapped = _next_session_on_or_after(ext, days, strict)["news_session"]
    return sessions.is_in(mapped.implode())


def _expiry_sessions(ext: pl.DataFrame) -> tuple[set[date], set[date]]:
    """Monthly expiry sessions (3rd Friday, or the Thursday before if closed) and the quarterly ones."""
    have = set(ext["session"].to_list())
    first, last = ext["session"].min(), ext["session"].max()
    monthly: set[date] = set()
    quarterly: set[date] = set()
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        day1 = date(year, month, 1)
        friday = day1 + timedelta(days=(4 - day1.weekday()) % 7 + 14)
        expiry = friday if friday in have else friday - timedelta(days=1)
        assert expiry in have or expiry > last, f"no session for the {year}-{month:02d} expiry"
        monthly.add(expiry)
        if month in (3, 6, 9, 12):
            quarterly.add(expiry)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return monthly, quarterly


def expanding_top_tercile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cut-off, top flag) per row. The cut-off at row i is the 2/3 quantile of rows BEFORE i
    (NaNs ignored); it is NaN for the first TERCILE_WARMUP rows. The flag is value > cut-off
    (NaN where either is NaN)."""
    cuts = np.full(values.size, np.nan)
    for i in range(config.TERCILE_WARMUP, values.size):
        past = values[:i]
        past = past[~np.isnan(past)]
        cuts[i] = np.quantile(past, 2 / 3)
    top = np.where(np.isnan(cuts) | np.isnan(values), np.nan, (values > cuts).astype(float))
    return cuts, top


def controls(table: pl.DataFrame, cal: pl.DataFrame, daily_dir: Path) -> pl.DataFrame:
    """Controls keyed by session, in the table's order. See the module docstring.

    `table` is the daily table (needs session, is_half_day, rv_fh, rv_day, vol_fh); `cal` is the
    session calendar. Previous-session quantities are lagged by CALENDAR session: if the previous
    session was dropped from the table the value is null (never bridged). The 20-session volume
    mean uses the previous 20 KEPT sessions (null until 20 exist). Output columns: CONTROL_COLUMNS
    plus rv_fh_cut, rv_fh_top, relvol_cut, relvol_top (expanding terciles from earlier rows,
    null in the warm-up) and news (fomc, cpi or payroll).
    """
    t = table.sort("session")
    assert t["session"].is_unique().all(), "repeated sessions"
    assert set(t["session"].to_list()) <= set(cal["session"].to_list()), "session not in calendar"
    ext = calendar.sessions(cal["session"].min(), cal["session"].max() + timedelta(days=40))
    sessions = t["session"]

    # previous calendar session's realised variance (null if that session is not in the table)
    prev_map = cal.sort("session").select("session", prev_session=pl.col("session").shift(1))
    rv_lag = (
        t.select("session")
        .join(prev_map, on="session", how="left")
        .join(
            t.select(prev_session=pl.col("session"), rv_day_lag=pl.col("rv_day")),
            on="prev_session",
            how="left",
        )
        .select("session", "rv_day_lag")
    )
    base = t.select("session", "is_half_day", "rv_fh", "vol_fh").join(rv_lag, on="session")
    base = base.with_columns(
        _mean20=pl.col("vol_fh")
        .rolling_mean(config.RELVOL_WINDOW, min_samples=config.RELVOL_WINDOW)
        .shift(1)
    ).with_columns(relvol=pl.col("vol_fh") / pl.col("_mean20"))

    wd = sessions.dt.weekday()  # Monday = 1
    weekdays = {name: wd == i for name, i in zip(WEEKDAY_COLUMNS, (2, 3, 4, 5), strict=True)}

    meetings = _read_dates(daily_dir / "fomc_meetings.csv")["date"]
    emergency = _read_dates(daily_dir / "fomc_emergency.csv")["date"]
    cancelled = _read_dates(daily_dir / "fomc_cancelled.csv")["date"]
    first, last = cal["session"].min(), cal["session"].max()
    meetings = meetings.filter(meetings.is_between(first, last))
    # Two-day meetings (the file lists the first day) announce on the next day; the one-day
    # meetings in config announce on the file date. The announcement must be a real session of the
    # XNYS calendar (`ext`), checked against the calendar, not against a value derived from it.
    one_day = meetings.is_in(list(config.FOMC_ONE_DAY_MEETINGS))
    assert set(meetings.filter(one_day).to_list()) == {
        d for d in config.FOMC_ONE_DAY_MEETINGS if first <= d <= last
    }, "a one-day FOMC meeting in config is missing from the meetings file"
    announce = (
        pl.DataFrame({"m": meetings, "one": one_day})
        .select(
            a=pl.when(pl.col("one")).then(pl.col("m")).otherwise(pl.col("m") + pl.duration(days=1))
        )
        .to_series()
    )
    assert announce.is_in(ext["session"].implode()).all(), "FOMC announcement is not a session"
    announce = announce.filter(announce <= last)
    # Emergency (unscheduled) and cancelled meetings are not known in advance: never flagged.
    unknown = pl.concat([emergency, cancelled]).unique()
    assert not unknown.is_in(announce.implode()).any(), "emergency/cancelled date flagged as FOMC"
    assert not unknown.is_in(meetings.implode()).any(), "emergency date in the scheduled file"
    fomc = sessions.is_in(announce.implode())

    rel = _read_dates(daily_dir / "release_dates.csv")
    assert set(rel["release"].unique().to_list()) <= {"cpi", "payroll"}, "unknown release kind"
    rel = rel.filter(pl.col("date").is_between(first, last))
    flags = {
        kind: _event_flag(sessions, ext, rel.filter(pl.col("release") == kind)["date"], False)
        for kind in ("cpi", "payroll")
    }

    vix = _read_fred(daily_dir, "VIXCLS")
    vxv = _read_fred(daily_dir, "VXVCLS")
    ratio = vix.join(vxv, on="obs_date", suffix="_3m").select(
        "obs_date", value=pl.col("value") / pl.col("value_3m")
    )
    vix_lag = _before(t.select("session"), vix, "vix_lag")
    ratio_lag = _before(t.select("session"), ratio, "vix_ratio_lag")

    monthly, quarterly = _expiry_sessions(ext)
    next_session = ext.sort("session").select("session", nxt=pl.col("session").shift(-1))
    month_end = (
        t.select("session")
        .join(next_session, on="session", how="left")
        .select(me=pl.col("nxt").dt.month() != pl.col("session").dt.month())
        .to_series()
    )
    assert month_end.null_count() == 0, "no next session to place month-end"

    out = (
        base.select("session", "rv_fh", "rv_day_lag", "relvol", "is_half_day")
        .with_columns(pl.Series(n, v) for n, v in weekdays.items())
        .with_columns(
            fomc=fomc,
            cpi=flags["cpi"],
            payroll=flags["payroll"],
            opex_monthly=sessions.is_in(list(monthly)),
            opex_quarterly=sessions.is_in(list(quarterly)),
            daily_expiry_era=sessions >= config.DAILY_EXPIRY_START,
            month_end=month_end,
        )
        .join(vix_lag, on="session")
        .join(ratio_lag, on="session")
        .with_columns(news=pl.col("fomc") | pl.col("cpi") | pl.col("payroll"))
    )
    for name in ("rv_fh", "relvol"):
        cut, top = expanding_top_tercile(out[name].cast(pl.Float64).to_numpy())
        out = out.with_columns(
            pl.Series(f"{name}_cut", cut, nan_to_null=True),
            pl.Series(f"{name}_top", top, nan_to_null=True),
        )
    out = out.select(
        "session", *CONTROL_COLUMNS, "news", "rv_fh_cut", "rv_fh_top", "relvol_cut", "relvol_top"
    )
    assert out.height == t.height
    return out
