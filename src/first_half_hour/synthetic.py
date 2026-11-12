"""Synthetic one-minute bars in exactly the vendor's format, with a planted r13-on-r1 slope.

Used by the tests and by `fhh reproduce --synthetic`. The files pass `bars.rth_bars` unchanged:
monthly CSVs (newest row first, vendor header) plus a manifest with bytes and sha256. Callers
set `config.DATA_START` / `config.DATA_END` to the same window (whole months) before reading.
"""

import hashlib
import json
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from first_half_hour import bars, calendar, config

# Half-hour-level volatilities (simple returns) of the planted process.
SD_OVERNIGHT = 0.004
SD_FIRST_HALF_HOUR = 0.003
SD_R13_NOISE = 0.002
SD_R12 = 0.002
SD_MIDDAY = 0.005
SD_MINUTE_PATH = 0.0003
START_PRICE = 100.0
_OPEN_MIN = config.SESSION_OPEN.hour * 60 + config.SESSION_OPEN.minute


def write_bars(root: Path, symbol: str, frame: pl.DataFrame) -> None:
    """Write `frame` (timestamp, open, high, low, close, volume) as the vendor's monthly files.

    Every month of the current `config` window gets a file (a month with no rows holds only the
    header) and a manifest entry. Rows are newest first, prices have four decimals.
    """
    directory = root / symbol
    directory.mkdir(parents=True, exist_ok=True)
    months = {}
    for month in bars.month_labels(config.DATA_START, config.DATA_END):
        chunk = frame.filter(pl.col("timestamp").dt.strftime("%Y-%m") == month).sort(
            "timestamp", descending=True
        )
        body = (
            chunk.with_columns(timestamp=pl.col("timestamp").dt.strftime(config.TIMESTAMP_FORMAT))
            .write_csv(float_precision=4)
            .encode()
        )
        assert body.split(b"\n", 1)[0].decode() == config.CSV_HEADER
        (directory / f"{symbol}_1min_{month}.csv").write_bytes(body)
        months[month] = {
            "status": config.MANIFEST_OK,
            "rows": chunk.height,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    manifest = {"symbol": symbol, "months": months}
    (directory / config.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def _bridge(rng: np.random.Generator, log_start: float, log_end: float, steps: int) -> np.ndarray:
    """Log prices at the end of each of `steps` minutes: a Brownian bridge ending at `log_end`."""
    walk = np.cumsum(rng.normal(0.0, SD_MINUTE_PATH, steps))
    k = np.arange(1, steps + 1) / steps
    return log_start + (log_end - log_start) * k + walk - k * walk[-1]


def _session_bars(
    rng: np.random.Generator, day: date, n_bars: int, prev_close: float, beta: float
) -> pl.DataFrame:
    """One session's bars (09:30 up to and including the 15:59 print), plus its closing price."""
    r_on = rng.normal(0.0, SD_OVERNIGHT)
    r_fh = rng.normal(0.0, SD_FIRST_HALF_HOUR)
    r1 = (1 + r_on) * (1 + r_fh) - 1
    r13 = beta * r1 + rng.normal(0.0, SD_R13_NOISE)
    r12 = rng.normal(0.0, SD_R12)
    p_open = prev_close * (1 + r_on)
    p_10 = p_open * (1 + r_fh)
    p_c61 = p_10 * (1 + rng.normal(0.0, SD_MIDDAY))
    p_c31 = p_c61 * (1 + r12)
    p_close = p_c31 * (1 + r13)
    # Close-price anchors at bar indices 29 (09:59), n-61, n-31, n-1; index -1 is the open.
    anchors = [(-1, p_open), (29, p_10), (n_bars - 61, p_c61), (n_bars - 31, p_c31)]
    anchors.append((n_bars - 1, p_close))
    closes = np.empty(n_bars)
    for (a, pa), (b, pb) in pairwise(anchors):
        closes[a + 1 : b + 1] = np.exp(_bridge(rng, np.log(pa), np.log(pb), b - a))
    n_total = 390  # the vendor carries after-hours prints to 15:59 on half-days
    if n_bars < n_total:
        tail = np.exp(_bridge(rng, np.log(p_close), np.log(p_close), n_total - n_bars))
        closes = np.concatenate([closes, tail])
    closes = closes.round(4)
    opens = np.concatenate([[round(p_open, 4)], closes[:-1]])
    n = len(closes)
    highs = (np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 1e-4, n)))).round(4)
    lows = (np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 1e-4, n)))).round(4)
    stamp = np.datetime64(day, "m") + np.arange(_OPEN_MIN, _OPEN_MIN + n).astype("timedelta64[m]")
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series(stamp.astype("datetime64[us]")),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(1_000, 60_000, n),
        }
    )
    return frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def write_synthetic(
    root: Path, symbols: tuple[str, ...] | list[str], start: date, end: date, beta: float, seed: int
) -> None:
    """Write synthetic bars for every XNYS session in [start, end] for each symbol.

    r13 = beta * r1 + noise is planted at the half-hour level, minute paths are Brownian bridges
    between the anchors, and half-days (13:00 close) carry after-hours prints like the vendor's.
    """
    assert config.DATA_START <= start < end <= config.DATA_END, (
        f"window {start}..{end} must sit inside config {config.DATA_START}..{config.DATA_END}"
    )
    cal = calendar.sessions(start, end)
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng([seed, i])
        price = START_PRICE
        frames = []
        for day, close, _ in cal.iter_rows():
            n_bars = (close.hour * 60 + close.minute) - _OPEN_MIN
            frame = _session_bars(rng, day, n_bars, price, beta)
            price = float(frame["close"][n_bars - 1])
            frames.append(frame)
        write_bars(root, symbol, pl.concat(frames))
