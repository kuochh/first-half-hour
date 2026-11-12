"""Download SPY's past ex-dividend dates from Alpha Vantage into data/daily/spy_dividends.csv.

An acquisition script, not a package module. One request (function=DIVIDENDS, free key works).
The file is vendor-derived, so `data/` is gitignored and it is never committed. Every run
overwrites the file; there is no local state. Ex-dates are kept within DATA_START..DATA_END.

Key hygiene: the key is read from ALPHAVANTAGE_KEY and sent only in the request URL; every
failure is printed through redact(), never raw.

Fail loud: each ex-date is an XNYS session, ascending and unique, amounts are positive, each
full calendar year holds 4 ex-dates in Mar/Jun/Sep/Dec, and a partial final year holds only
those months.

Run from the repo root: `uv run --env-file .env python scripts/fetch_dividends.py [--out PATH]`.
"""

import argparse
import json
import os
import sys
import traceback
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from first_half_hour import config
from first_half_hour.calendar import sessions

URL = "https://www.alphavantage.co/query?function=DIVIDENDS&symbol=SPY&datatype=json&apikey={key}"
KEY_ENV_VAR = "ALPHAVANTAGE_KEY"
REQUEST_TIMEOUT_SECONDS = 60
QUARTER_MONTHS = (3, 6, 9, 12)
PER_YEAR = 4
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "daily" / "spy_dividends.csv"


def redact(text: str, key: str) -> str:
    return text.replace(key, "***")


def parse(body: str) -> pl.DataFrame:
    """The API's JSON -> (ex_date, amount), in the study window, ascending. Fails on any surprise."""
    payload = json.loads(body)
    assert payload.get("symbol") == "SPY" and payload.get("data"), (
        f"unexpected payload keys {list(payload)}"
    )
    rows = [
        (date.fromisoformat(r["ex_dividend_date"]), float(r["amount"])) for r in payload["data"]
    ]
    df = pl.DataFrame(rows, schema={"ex_date": pl.Date, "amount": pl.Float64}, orient="row")
    df = df.filter(pl.col("ex_date").is_between(config.DATA_START, config.DATA_END)).sort("ex_date")
    assert df.height and df["ex_date"].is_unique().all(), "empty or repeated ex-dates"
    assert (df["amount"] > 0).all(), "non-positive dividend amount"
    assert set(df["ex_date"].dt.month()) <= set(QUARTER_MONTHS), "ex-date outside Mar/Jun/Sep/Dec"
    trading = sessions(df["ex_date"].min(), df["ex_date"].max())["session"]
    off = df.filter(~pl.col("ex_date").is_in(trading.implode()))
    assert off.is_empty(), f"ex-dates that are not XNYS sessions: {off['ex_date'].to_list()}"
    last_year = config.DATA_END.year
    for year, count in df.group_by(df["ex_date"].dt.year().alias("y")).len().iter_rows():
        expected = (
            sum(m <= config.DATA_END.month for m in QUARTER_MONTHS)
            if year == last_year
            else PER_YEAR
        )
        assert count == expected, f"{year}: {count} ex-dates, expected {expected}"
    assert df["ex_date"].min().year == config.DATA_START.year, "first ex-date not in the first year"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out: Path = parser.parse_args().out
    key = os.environ[KEY_ENV_VAR]
    assert key, f"{KEY_ENV_VAR} is empty"
    try:
        with urllib.request.urlopen(URL.format(key=key), timeout=REQUEST_TIMEOUT_SECONDS) as r:
            assert r.status == 200, f"HTTP {r.status}"
            df = parse(r.read().decode("utf-8"))
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 -- the redaction boundary
        print(redact(traceback.format_exc(), key), file=sys.stderr, end="")
        raise SystemExit(1) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    print(f"{df.height} ex-dates, {df['ex_date'].min()} .. {df['ex_date'].max()} -> {out}")


if __name__ == "__main__":
    main()
