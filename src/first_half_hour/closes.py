"""The official (auction) close of SPY, put on the minute bars' price basis.

The vendor's bars are split- and dividend-adjusted as of the pull date; Yahoo's `Close` is the
as-traded official close. Step 1 applies the CRSP adjustment factor from the dividends. Step 2
absorbs the small drift between the vendor's adjustment and CRSP's with one constant per dividend
period (sessions sharing a factor): the median of bar-close / (Close * f). Cost of step 2: the
median gap between the official close and the 15:59 bar is zero in the first period by construction
and only near zero in later ones; the day-to-day gap (what the closing auction adds) is kept.

Known by the session: period p uses c_{p-1}, the previous dividend period's median, which reads
earlier sessions only. The one exception is the first period of the table (2011-01-03 up to the
first ex-date), which uses its own median. The CRSP factor itself depends on later dividends (as
the vendor's adjusted bars do).
"""

import hashlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from first_half_hour import calendar, config


def official_close(table: pl.DataFrame, csv_path: Path, dividends_path: Path) -> pl.DataFrame:
    """`table` (needs `session`, `p_close` = the 15:59 bar close) plus `off_close`, `prev_off`, `c_p`.

    off_close is the adjusted official close of the session; prev_off the previous XNYS session's
    (read from the CSV, so a session dropped from `table` does not matter); c_p the period constant.
    Fails loud (AssertionError) on any check listed in private/specs/p1-part1.md.
    """
    if config.OFFICIAL_CLOSE_VERIFY:  # the real file only; reproduce --synthetic writes a stub
        body = csv_path.read_bytes()
        assert len(body) == config.OFFICIAL_CLOSE_BYTES, (
            f"{csv_path}: {len(body)} bytes, recorded {config.OFFICIAL_CLOSE_BYTES}"
        )
        assert hashlib.sha256(body).hexdigest() == config.OFFICIAL_CLOSE_SHA256, (
            f"{csv_path}: sha256 differs from the recorded one"
        )
    csv = (
        pl.read_csv(csv_path, schema_overrides={"date": pl.Date})
        .select("date", "Close", "Dividends", "Stock Splits")
        .sort("date")
    )
    assert csv["date"].is_unique().all(), "repeated dates in the official close CSV"
    t = table.sort("session")
    first, last = t["session"].min(), t["session"].max()
    missing = set(t["session"].to_list()) - set(csv["date"].to_list())
    assert not missing, f"sessions without an official close: {sorted(missing)[:5]}"
    in_window = csv.filter(pl.col("date").is_between(first, last))
    assert (in_window["Stock Splits"] == 0).all(), "a stock split inside the window"

    # Dividends: Yahoo against the owner's dividend file, same dates and amounts.
    div = csv.filter((pl.col("Dividends") > 0) & (pl.col("date") <= last))
    ours = pl.read_csv(dividends_path, schema_overrides={"ex_date": pl.Date}).filter(
        pl.col("ex_date").is_between(first, last)
    )
    theirs = div.filter(pl.col("date") >= first).select(ex_date="date", amount=pl.col("Dividends"))
    both = ours.sort("ex_date").join(theirs.sort("ex_date"), on="ex_date", how="full", suffix="_y")
    assert both["amount"].null_count() == 0 and both["amount_y"].null_count() == 0, (
        f"dividend dates differ between Yahoo and the dividend file: "
        f"{both.filter(pl.col('amount').is_null() | pl.col('amount_y').is_null())['ex_date'].to_list()}"
    )
    err = (both["amount"] - both["amount_y"]).abs().max()
    assert err < config.OFFICIAL_DIVIDEND_TOL, f"dividend amounts differ by {err}"

    # CRSP factor: f_t = product over ex-dates e > t of (1 - D_e / Close_{e-1}).
    dates = csv["date"].to_numpy()
    close = csv["Close"].to_numpy()
    f = np.ones(csv.height)
    ex_dates = div["date"].to_numpy()
    for e, d in zip(ex_dates, div["Dividends"].to_numpy(), strict=True):
        i = int(np.searchsorted(dates, e))
        assert i > 0, "a dividend on the first CSV row has no previous close"
        f[dates < e] *= 1 - d / close[i - 1]
    period = np.searchsorted(ex_dates, dates, side="right")
    raw = pl.DataFrame({"session": dates, "raw": close * f, "period": period})

    # One constant per dividend period: median of bar close / (Close * f) over kept sessions.
    ratio = t.select("session", "p_close").join(raw, on="session", validate="1:1")
    ratio = ratio.with_columns(ratio=pl.col("p_close") / pl.col("raw"))
    c_p = ratio.group_by("period").agg(c_p=pl.col("ratio").median()).sort("period")
    steps = (c_p["c_p"] / c_p["c_p"].shift(1) - 1).drop_nulls().abs()
    assert steps.max() < config.OFFICIAL_CLOSE_MAX_PERIOD_STEP, (
        f"adjacent period constants differ by {steps.max():.2e}"
    )
    # Period p is adjusted with the previous period's median (past only); the table's first
    # period, which has no predecessor, uses its own.
    assert (c_p["period"].diff().drop_nulls() == 1).all(), "a dividend period with no kept session"
    c_p = c_p.with_columns(c_p=pl.col("c_p").shift(1).fill_null(pl.col("c_p")))
    adj = raw.join(c_p, on="period", how="left").with_columns(off_adj=pl.col("raw") * pl.col("c_p"))

    # The previous XNYS session of every table session, read from the CSV.
    cal = calendar.sessions(first - timedelta(days=10), last)
    prev = cal.select("session", prev_session=pl.col("session").shift(1))
    out = (
        t.join(prev, on="session", how="left", validate="1:1")
        .join(adj.select("session", "off_adj", "c_p"), on="session", how="left", validate="1:1")
        .join(
            adj.select(prev_session="session", prev_off="off_adj"),
            on="prev_session",
            how="left",
            validate="m:1",
        )
        .rename({"off_adj": "off_close"})
        .drop("prev_session")
    )
    assert out["off_close"].null_count() == 0, "a session without an adjusted official close"
    assert out["prev_off"].null_count() == 0, "a previous session without an official close"
    gap = (out["off_close"] / out["p_close"] - 1).abs().max()
    assert gap < config.OFFICIAL_CLOSE_MAX_GAP, f"official close is {gap:.4f} from the 15:59 bar"
    return out
