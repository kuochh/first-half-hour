"""Raw one-minute bars: verify each monthly file, read them, keep the regular session.

Every surprise is an AssertionError naming the offending file or dates; nothing is silently
repaired.
"""

import hashlib
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from first_half_hour import config

RAW_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}
_CSV_SCHEMA = {"timestamp": pl.String, **{k: v for k, v in RAW_SCHEMA.items() if k != "timestamp"}}


def month_labels(first: date, last: date) -> list[str]:
    """Every `YYYY-MM` from `first`'s month to `last`'s, inclusive."""
    a, b = first.year * 12 + first.month - 1, last.year * 12 + last.month - 1
    assert a <= b, f"empty span {first}..{last}"
    return [f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(a, b + 1)]


def verified_month_files(root: Path, symbol: str) -> list[Path]:
    """The symbol's monthly CSVs inside the study window, each checked against its manifest.

    Each month must be in the manifest with status "ok", and its file must match the recorded
    byte count, sha256 and header. A file outside the window is never opened.
    """
    directory = root / symbol
    manifest = json.loads((directory / config.MANIFEST_NAME).read_text())
    assert manifest["symbol"] == symbol, f"{directory} holds {manifest['symbol']}'s manifest"
    entries = manifest["months"]
    files = []
    for month in month_labels(config.DATA_START, config.DATA_END):
        assert month in entries, f"{symbol} {month}: not in the manifest (never pulled)"
        entry = entries[month]
        assert entry["status"] == config.MANIFEST_OK, f"{symbol} {month}: {entry['status']!r}"
        path = directory / f"{symbol}_1min_{month}.csv"
        body = path.read_bytes()
        assert len(body) == entry["bytes"], f"{path}: {len(body)} bytes, manifest {entry['bytes']}"
        assert hashlib.sha256(body).hexdigest() == entry["sha256"], f"{path}: sha256 mismatch"
        header = body.split(b"\n", 1)[0].rstrip(b"\r").decode()
        assert header == config.CSV_HEADER, f"{path}: header {header!r}"
        files.append(path)
    return files


def scan_bars(root: Path, symbol: str) -> pl.LazyFrame:
    """All raw bars in the window, ascending, with `session` (Date) and `minute` (Time) added.

    A blank timestamp stays null (it is not filtered out), so `rth_bars` fails loud on it.
    """
    stamp = pl.col("timestamp").str.strptime(pl.Datetime("us"), config.TIMESTAMP_FORMAT)
    day = pl.col("timestamp").dt.date()
    lf = (
        pl.scan_csv(verified_month_files(root, symbol), schema=_CSV_SCHEMA)
        .with_columns(timestamp=stamp)
        .filter(day.is_null() | day.is_between(config.DATA_START, config.DATA_END))
        .sort("timestamp")
    )
    schema = dict(lf.collect_schema())
    assert schema == RAW_SCHEMA, f"{symbol}: schema {schema} != {RAW_SCHEMA}"
    return lf.with_columns(
        session=pl.col("timestamp").dt.date(), minute=pl.col("timestamp").dt.time()
    )


def rth_bars(root: Path, symbol: str, cal: pl.DataFrame) -> pl.LazyFrame:
    """The regular-session bars: each raw bar from 09:30 up to (not including) its session's close.

    Fails loud on a repeated (session, minute), a bar before 09:30 or at/after 16:00, a date that
    is not an XNYS session, or a session coverage different from `config.KNOWN_MISSING_SESSIONS`.
    The close cut removes exactly the after-hours prints that half-days carry to 15:59.
    """
    lf = scan_bars(root, symbol)
    price = ("open", "high", "low", "close")
    t = (
        lf.select(
            nulls=pl.sum_horizontal(pl.all().null_count()),
            off_minute=(pl.col("timestamp").dt.second() != 0).sum(),
            bad_price=(
                (pl.min_horizontal(price) <= 0)
                | (pl.col("low") > pl.min_horizontal("open", "close"))
                | (pl.col("high") < pl.max_horizontal("open", "close"))
            ).sum(),
            n=pl.len(),
            pairs=pl.struct("session", "minute").n_unique(),
            lo=pl.col("minute").min(),
            hi=pl.col("minute").max(),
            days=pl.col("session").unique().sort().implode(),
        )
        .collect()
        .row(0, named=True)
    )
    assert t["nulls"] == 0, f"{symbol}: {t['nulls']} empty cells (a blank timestamp or price)"
    assert t["off_minute"] == 0, f"{symbol}: {t['off_minute']} timestamps not on a whole minute"
    assert t["bad_price"] == 0, f"{symbol}: {t['bad_price']} bars with a price <= 0 or low/high"
    assert t["n"] == t["pairs"], f"{symbol}: {t['n'] - t['pairs']} repeated (session, minute)"
    assert t["lo"] >= config.SESSION_OPEN, f"{symbol}: a bar at {t['lo']}, before the open"
    assert t["hi"] < config.REGULAR_CLOSE, f"{symbol}: a bar at {t['hi']}, at or after 16:00"
    raw_days, cal_days = set(t["days"]), set(cal["session"].to_list())
    orphans = sorted(raw_days - cal_days)
    assert not orphans, f"{symbol}: bars on non-sessions {orphans}"
    absent = cal_days - raw_days
    known = config.KNOWN_MISSING_SESSIONS[symbol]
    assert absent == known, (
        f"{symbol}: missing sessions {sorted(absent - known)}; recorded missing but present "
        f"{sorted(known - absent)}"
    )
    return (
        lf.join(cal.lazy().rename({"close": "session_close"}), on="session", how="inner")
        .filter(pl.col("minute") >= config.SESSION_OPEN)
        .filter(pl.col("minute") < pl.col("session_close"))
    )


def key_bars(close: time) -> tuple[time, ...]:
    """The bars a session needs: 09:59 and close minus 61, 31 and 1 minutes."""

    def back(m: int) -> time:
        return (datetime.combine(date.min, close) - timedelta(minutes=m)).time()

    return (config.OPENING_BAR_LAST, *(back(m) for m in config.CLOSE_OFFSETS_MIN))


def sessions_missing_key_bar(rth: pl.LazyFrame, cal: pl.DataFrame) -> pl.DataFrame:
    """Calendar sessions lacking any key bar (09:59, C-61, C-31, C-1); dropped downstream.

    Every session in `cal` is judged, so one left with no regular-session bar at all (e.g. a
    half-day whose only prints came after 13:00) is listed as lacking all four rather than
    vanishing. Halt days with every key bar present are kept.
    """
    minutes = rth.group_by("session").agg(pl.col("minute"))
    per = cal.lazy().join(minutes, on="session", how="left").select("session", "close", "minute")
    rows = []
    for session, close, minutes in per.collect().iter_rows():
        have = set(minutes or [])
        lacking = [k for k in key_bars(close) if k not in have]
        if lacking:
            rows.append(
                {"session": session, "lacking": ", ".join(k.strftime("%H:%M") for k in lacking)}
            )
    return pl.DataFrame(rows, schema={"session": pl.Date, "lacking": pl.String}).sort("session")
