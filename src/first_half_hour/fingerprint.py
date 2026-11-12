"""A per-symbol fingerprint of the raw bars, so anyone can check their own pull against ours."""

import hashlib
from pathlib import Path

import polars as pl

from first_half_hour import bars, calendar, config


def symbol_row(root: Path, symbol: str, cal: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    """One symbol's fingerprint row and its dropped sessions (missing a key bar)."""
    raw_rows = bars.scan_bars(root, symbol).select(pl.len()).collect().item()
    rth = bars.rth_bars(root, symbol, cal)
    dropped = bars.sessions_missing_key_bar(rth, cal)
    kept = rth.filter(~pl.col("session").is_in(dropped["session"].implode()))
    per = kept.group_by("session").agg(n=pl.len(), half=pl.col("is_half_day").first()).collect()
    span = kept.select(first=pl.col("timestamp").min(), last=pl.col("timestamp").max()).collect()
    assert per.height + dropped.height == cal.height, (
        f"{symbol}: {per.height} kept + {dropped.height} dropped != {cal.height} sessions"
    )
    row = {
        "symbol": symbol,
        "sessions": per.height,
        "dropped": dropped.height,
        "half_days": int(per["half"].sum()),
        "first_bar": span["first"].item().strftime("%Y-%m-%d %H:%M"),
        "last_bar": span["last"].item().strftime("%Y-%m-%d %H:%M"),
        "raw_rows": raw_rows,
        "bars_min": int(per["n"].min()),
        "bars_median": float(per["n"].median()),
    }
    return row, dropped.with_columns(symbol=pl.lit(symbol))


def fingerprint(root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The fingerprint table for every symbol, and every dropped session with its missing bars."""
    cal = calendar.sessions(config.DATA_START, config.DATA_END)
    results = [symbol_row(root, s, cal) for s in config.SYMBOLS]
    return pl.DataFrame([r for r, _ in results]), pl.concat([d for _, d in results])


def file_hashes(root: Path) -> list[str]:
    """`sha256  path` for every monthly file in the window (after its manifest check)."""
    lines = []
    for s in config.SYMBOLS:
        for path in bars.verified_month_files(root, s):
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {s}/{path.name}")
    return lines


def to_markdown(table: pl.DataFrame) -> str:
    head = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.iter_rows()]
    return "\n".join([head, rule, *body])
