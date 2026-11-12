"""The daily returns table: one row per kept session, built from the one-minute bars.

Bars are labelled by their START minute, so the close of bar m is the price at m+1. C is the
session close (16:00, or 13:00 on half-days). Nothing from session t+1 enters row t.
"""

import logging
from pathlib import Path

import polars as pl

from first_half_hour import bars, config

log = logging.getLogger(__name__)

IDENTITY_TOLERANCE = 1e-12
"""(1 + r1) == (1 + r_on)(1 + r_fh) must hold to this, wherever all three exist."""

_FH_LAST_MOD = config.OPENING_BAR_LAST.hour * 60 + config.OPENING_BAR_LAST.minute


def _close_at(mod: pl.Expr, target: pl.Expr, col: str = "close") -> pl.Expr:
    """The `col` value of the bar whose minute-of-day equals `target` (null if that bar is absent)."""
    return pl.col(col).filter(mod == target).first()


def daily_table(root: Path, symbol: str, cal: pl.DataFrame) -> pl.DataFrame:
    """One row per kept session, sorted. Columns are documented in private/specs/m1-chapter1.md.

    A session lacking any key bar (09:59, C-61, C-31, C-1) is dropped and logged. prev_close is
    the previous XNYS session's close, and only if that session is kept; otherwise r1, r_on and
    rod are null (two nights are never bridged).
    """
    rth = bars.rth_bars(root, symbol, cal)
    dropped = bars.sessions_missing_key_bar(rth, cal)
    log.info("%s: %d sessions dropped for a missing key bar", symbol, dropped.height)

    mod = pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(
        pl.Int32
    )
    close_mod = pl.col("session_close").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "session_close"
    ).dt.minute().cast(pl.Int32)
    open_mod = config.SESSION_OPEN.hour * 60 + config.SESSION_OPEN.minute
    first_half_hour = mod <= _FH_LAST_MOD
    # Squared 1-minute log return of each bar's close on the previous present bar's close; the
    # session's first present bar uses its own open.
    prev_px = pl.coalesce(pl.col("close").shift(1).over("session"), pl.col("open"))
    sq = (pl.col("close") / prev_px).log() ** 2

    per_session = (
        rth.filter(~pl.col("session").is_in(dropped["session"].implode()))
        .sort("session", "minute")
        .group_by("session")
        .agg(
            p_close=_close_at(mod, close_mod - 1),
            p_open=_close_at(mod, pl.lit(open_mod), "open"),
            p_10=_close_at(mod, pl.lit(_FH_LAST_MOD)),
            p_c61=_close_at(mod, close_mod - 61),
            p_c31=_close_at(mod, close_mod - 31),
            rv_fh=sq.filter(first_half_hour).sum(),
            rv_day=sq.sum(),
            vol_fh=pl.col("volume").filter(first_half_hour).sum(),
        )
        .collect()
    )
    kept = per_session.select("session", "p_close")
    # Previous XNYS session over the FULL calendar, so a dropped neighbour yields null.
    prev = (
        cal.sort("session")
        .select("session", "is_half_day", prev_session=pl.col("session").shift(1))
        .join(
            kept.rename({"session": "prev_session", "p_close": "prev_close"}),
            on="prev_session",
            how="left",
        )
        .select("session", "is_half_day", "prev_close")
    )
    out = (
        per_session.join(prev, on="session", how="left", validate="1:1")
        .with_columns(
            r1=pl.col("p_10") / pl.col("prev_close") - 1,
            r_on=pl.col("p_open") / pl.col("prev_close") - 1,
            r_fh=pl.col("p_10") / pl.col("p_open") - 1,
            r12=pl.col("p_c31") / pl.col("p_c61") - 1,
            r13=pl.col("p_close") / pl.col("p_c31") - 1,
            rod=pl.col("p_c31") / pl.col("prev_close") - 1,
        )
        .select(
            "session",
            "is_half_day",
            "p_close",
            "p_open",
            "p_10",
            "p_c61",
            "p_c31",
            "prev_close",
            "r1",
            "r_on",
            "r_fh",
            "r12",
            "r13",
            "rod",
            "rv_fh",
            "rv_day",
            "vol_fh",
        )
        .sort("session")
    )
    assert out["session"].is_unique().all(), f"{symbol}: repeated sessions in the table"
    assert out.height == cal.height - dropped.height, f"{symbol}: row count != kept sessions"
    for col in ("p_close", "p_10", "p_c61", "p_c31", "r13", "r12", "rv_fh", "rv_day", "vol_fh"):
        assert out[col].null_count() == 0, f"{symbol}: nulls in {col}"
    err = identity_error(out)
    assert err < IDENTITY_TOLERANCE, f"{symbol}: (1+r1) != (1+r_on)(1+r_fh), max error {err:.3g}"
    log.info(
        "%s: %d rows, %d with null r1 (previous session absent or dropped)",
        symbol, out.height, out["r1"].null_count(),
    )  # fmt: skip
    return out


def identity_error(table: pl.DataFrame) -> float:
    """Max abs of (1+r1) - (1+r_on)(1+r_fh) over rows where all three exist (0.0 if none)."""
    err = (
        table.drop_nulls(["r1", "r_on", "r_fh"])
        .select(((1 + pl.col("r1")) - (1 + pl.col("r_on")) * (1 + pl.col("r_fh"))).abs().max())
        .item()
    )
    return 0.0 if err is None else float(err)
