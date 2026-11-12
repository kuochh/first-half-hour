"""Chapter 2: the further-move target and the features, all known by 10:00.

P10 is the close of the 09:59 bar. The target looks at the bars 10:00 .. C-1 only (C = 16:00, or
13:00 on half-days; `bars.rth_bars` already cuts the after-hours prints). Bars are read only
through `bars.rth_bars`.
"""

import logging
from pathlib import Path

import polars as pl

from first_half_hour import bars, config

log = logging.getLogger(__name__)

_FH_LAST = config.OPENING_BAR_LAST
_BOOL_FEATURES = tuple(
    c for c in config.FEATURE_COLUMNS if c in {
        "fomc", "cpi", "payroll", "opex_monthly", "opex_quarterly", "daily_expiry_era",
        "dow_tue", "dow_wed", "dow_thu", "dow_fri", "is_half_day", "month_end",
    }
)  # fmt: skip


def target(root: Path, symbol: str, cal: pl.DataFrame, b: float, side: str = "up") -> pl.DataFrame:
    """(session, y, p10) for sessions with a 09:59 bar.

    side "up": y = 1 if max(high of 10:00 .. C-1) >= P10 * (1 + b); side "down": y = 1 if
    min(low of 10:00 .. C-1) <= P10 * (1 - b). A touch exactly at the level counts.
    """
    assert side in ("up", "down"), f"side must be 'up' or 'down', got {side!r}"
    assert b > 0, f"b must be positive, got {b}"
    rth = bars.rth_bars(root, symbol, cal)
    after = pl.col("minute") > _FH_LAST
    per = (
        rth.group_by("session")
        .agg(
            p10=pl.col("close").filter(pl.col("minute") == _FH_LAST).first(),
            hi=pl.col("high").filter(after).max(),
            lo=pl.col("low").filter(after).min(),
            n_after=after.sum(),
        )
        .filter(pl.col("p10").is_not_null())
        .sort("session")
        .collect()
    )
    assert (per["n_after"] > 0).all(), f"{symbol}: a session with a 09:59 bar and none after it"
    hit = (
        pl.col("hi") >= pl.col("p10") * (1 + b)
        if side == "up"
        else (pl.col("lo") <= pl.col("p10") * (1 - b))
    )
    return per.select("session", y=hit.cast(pl.Int8), p10="p10")


def features(
    table: pl.DataFrame, ctrl: pl.DataFrame, root: Path, symbol: str, cal: pl.DataFrame
) -> pl.DataFrame:
    """One row per session of `table`: session, tau, r1, then `config.FEATURE_COLUMNS`.

    `table` is `returns.daily_table`, `ctrl` is `controls.controls` on it (calendar and VIX columns
    are taken from it unchanged). Nulls are kept: a session with any null feature is dropped by the
    caller for every model alike. sigma = sqrt(rv_fh / 30) and is null when the first half-hour
    lacks one of its 30 bars. ret_prev is the previous calendar session's close-to-close return,
    null when that session or the one before it is missing or dropped. r1 (Gao's regressor) and
    tau (minutes from 10:00 to the close) are not model features.
    """
    t = table.sort("session")
    assert t["session"].is_unique().all(), "repeated sessions"
    assert ctrl["session"].to_list() == t["session"].to_list(), "ctrl and table sessions differ"
    opening = (
        bars.rth_bars(root, symbol, cal)
        .filter(pl.col("minute") <= _FH_LAST)
        .group_by("session")
        .agg(
            hi=pl.col("high").max(),
            lo=pl.col("low").min(),
            n_bars=pl.len(),
        )
        .collect()
    )
    # Previous calendar session's close-to-close return, from the table's own columns.
    day_ret = t.select("session", ret=pl.col("p_close") / pl.col("prev_close") - 1)
    prev_map = cal.sort("session").select("session", prev_session=pl.col("session").shift(1))
    ret_prev = prev_map.join(
        day_ret.rename({"session": "prev_session", "ret": "ret_prev"}),
        on="prev_session",
        how="left",
    ).select("session", "ret_prev")

    out = (
        t.join(opening, on="session", how="left", validate="1:1")
        .join(ret_prev, on="session", how="left", validate="1:1")
        .join(ctrl.drop("rv_fh", "is_half_day"), on="session", how="left", validate="1:1")
        .with_columns(
            abs_r_on=pl.col("r_on").abs(),
            abs_r_fh=pl.col("r_fh").abs(),
            sigma=pl.when(pl.col("n_bars") == config.FIRST_HALF_HOUR_BARS)
            .then((pl.col("rv_fh") / config.FIRST_HALF_HOUR_BARS).sqrt())
            .otherwise(None),
            or_high=pl.col("hi") / pl.col("p_10") - 1,
            or_low=pl.col("lo") / pl.col("p_10") - 1,
            tau=pl.when(pl.col("is_half_day"))
            .then(config.TAU_MIN_HALF_DAY)
            .otherwise(config.TAU_MIN_REGULAR)
            .cast(pl.Float64),
        )
        .with_columns(pl.col(c).cast(pl.Float64) for c in _BOOL_FEATURES)
        .select("session", "tau", "r1", *config.FEATURE_COLUMNS)
    )
    assert out.height == t.height
    short = t.height - int(opening.filter(pl.col("n_bars") == config.FIRST_HALF_HOUR_BARS).height)
    log.info("features: %d sessions, %d with a short first half-hour", out.height, short)
    return out


def complete_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows where every feature, tau and r1 are present; the count dropped is logged."""
    keep = frame.drop_nulls()
    log.info("features: %d of %d sessions complete", keep.height, frame.height)
    return keep
