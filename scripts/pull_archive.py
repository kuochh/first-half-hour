"""Download the study's one-minute bars from Alpha Vantage, one month at a time.

An acquisition script, not a package module. For each symbol in `config.SYMBOLS` and each month
of the study window (2011-01 .. 2026-08) it requests the regular-session minute bars, validates
the response, and keeps the vendor's bytes verbatim as `<out>/<SYM>/<SYM>_1min_YYYY-MM.csv`
beside a `manifest.json` of sizes and sha256s. `first_half_hour.bars` reads the files only
after checking each against that manifest.

**Licensing.** Alpha Vantage data is never redistributed or committed; `data/` is gitignored.
You need your own key (premium for the 75 requests/min pace used here; a free key is far
slower and daily-capped, and the rate-limit handling will stop the run).

**Resumable.** The manifest is rewritten atomically after every month. A month already in the
manifest whose file matches the recorded size and sha256 is skipped without a request, so a
crash or Ctrl-C costs one month and a finished archive makes zero requests.

**Fail loud** on a malformed or truncated body, a row outside the requested month, a bar
outside 09:30-15:59 (the `extended_hours=false` contract), a duplicate timestamp, an empty
month, a vendor error, or a rate limit that outlasts its retries. A month with FEWER sessions
than the XNYS calendar is recorded (`short: true`) and listed at the end, not a crash.

**Key hygiene.** The key lives only in the environment (ALPHAVANTAGE_KEY) and the request URL.
Nothing prints a URL, the manifest excludes the key (asserted), and every failure is printed
through `redact()`.

Run from the repo root, with the key in the gitignored .env:
    uv run --env-file .env python scripts/pull_archive.py [--symbols SPY,QQQ] [--out data/raw]
`--dry-run` reads no key and makes no request.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import polars as pl

from first_half_hour import config
from first_half_hour.calendar import sessions

ENDPOINT = "https://www.alphavantage.co/query?"
REQUEST_PARAMS = {
    "function": "TIME_SERIES_INTRADAY",
    "interval": "1min",
    "outputsize": "full",
    "extended_hours": "false",
    "datatype": "csv",
}
"""Sent as is; `adjusted` is not passed, so the endpoint default (split- and dividend-adjusted)
applies. `symbol`, `month` and the key are added per request."""

KEY_ENV_VAR = "ALPHAVANTAGE_KEY"
KEY_PARAM = "apikey"
REDACTION = "***"
USER_AGENT = "first-half-hour/0.1 (research)"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

REQUESTS_PER_MINUTE = 75  # the vendor's premium cap
REQUEST_SPACING_SECONDS = 60 / REQUESTS_PER_MINUTE * 1.1  # 10% under the cap
REQUEST_TIMEOUT_SECONDS = 60
NETWORK_BACKOFF_SECONDS = (5, 15, 45)
"""Sleep before network retry 1, 2, 3: a socket failure, a timeout, HTTP 5xx or 429. Any other
HTTP error raises at once."""
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_SLEEP_SECONDS = 60
"""A rate-limit notice arrives as an HTTP 200 JSON body; sleep this long, then ask again."""
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500

LAST_RTH_BAR = time(15, 59)
FIELDS = config.CSV_HEADER.split(",")
PRICE_FIELDS = FIELDS[1:5]
MONTH_PATTERN = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
FLOAT_PATTERN = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
RATE_LIMIT_PATTERN = re.compile(r"\b(call frequency|rate|limit|premium)", re.IGNORECASE)
"""A "Note"/"Information" body mentioning any of these is a rate limit; any other is an error."""

Fetch = Callable[[str, str], bytes]
"""(symbol, month) -> the response body. Injected so the pull can be driven offline."""


def say(line: str) -> None:
    print(line, flush=True)


# --- Months and the calendar --------------------------------------------------------------


def month_index(month: str) -> int:
    assert MONTH_PATTERN.fullmatch(month), f"not a YYYY-MM month: {month!r}"
    year, number = month.split("-")
    return int(year) * 12 + int(number) - 1


def months_between(first: str, last: str) -> list[str]:
    """Every month from `first` to `last`, both inclusive, oldest first."""
    assert month_index(first) <= month_index(last), f"{first} is after {last}"
    return [
        f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(month_index(first), month_index(last) + 1)
    ]


def session_calendar(months: Sequence[str]) -> dict[str, frozenset[date]]:
    """Each month's XNYS sessions (the sets, so a missing day and a bar on a closed day cannot
    cancel out in a count)."""
    start = date.fromisoformat(f"{months[0]}-01")
    following = month_index(months[-1]) + 1
    end = date(following // 12, following % 12 + 1, 1) - timedelta(days=1)
    by_month = (
        sessions(start, end)
        .select("session", month=pl.col("session").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(pl.col("session"))
    )
    grouped = {month: frozenset(days) for month, days in by_month.iter_rows()}
    missing = [month for month in months if month not in grouped]
    assert not missing, f"the XNYS calendar holds no sessions in {missing}"
    return {month: grouped[month] for month in months}


# --- Response validation ------------------------------------------------------------------


def parse_row(line: str, place: str, month: str) -> datetime:
    """Validate one data row strictly and return its timestamp."""
    cells = line.split(",")
    assert len(cells) == len(FIELDS), f"{place}: expected {len(FIELDS)} fields: {line[:80]!r}"
    stamp_text, *prices, volume = cells
    assert TIMESTAMP_PATTERN.fullmatch(stamp_text), f"{place}: bad timestamp {stamp_text!r}"
    stamp = datetime.fromisoformat(stamp_text)
    for name, cell in zip(PRICE_FIELDS, prices, strict=True):
        assert FLOAT_PATTERN.fullmatch(cell), f"{place}: {name} {cell!r} is not a finite number"
    assert volume.isascii() and volume.isdigit(), f"{place}: volume {volume!r} is not an integer"
    assert stamp_text[:7] == month, f"{place}: bar {stamp_text} is outside the month {month}"
    assert config.SESSION_OPEN <= stamp.time() <= LAST_RTH_BAR, (
        f"{place}: bar at {stamp.time()} is outside 09:30-15:59 (extended_hours=false)"
    )
    return stamp


def check_body(body: bytes, symbol: str, month: str) -> tuple[str, set[datetime] | str]:
    """Sort one HTTP-200 body: ("data", timestamps), ("rate_limit", message) or ("error", message).

    Alpha Vantage answers everything with HTTP 200: data as CSV, errors and rate-limit notices
    as a JSON object. A malformed CSV raises.
    """
    where = f"{symbol} {month}"
    if body.lstrip().startswith(b"{"):
        reply = json.loads(body.decode("utf-8"))
        assert isinstance(reply, dict), f"{where}: a JSON reply that is not an object"
        for notice in ("Note", "Information"):
            if notice in reply and RATE_LIMIT_PATTERN.search(str(reply[notice])):
                return "rate_limit", str(reply[notice])
        return "error", str(reply.get("Error Message", reply.get("Note", reply)))[:300]

    lines = body.decode("utf-8").splitlines()
    header = lines[0] if lines else ""
    assert header == config.CSV_HEADER, f"{where}: header {header[:80]!r}"
    stamps: set[datetime] = set()
    for number, line in enumerate(lines[1:], start=2):
        place = f"{where} line {number}" + (
            " (final line: truncated?)" if number == len(lines) else ""
        )
        stamp = parse_row(line, place, month)
        assert stamp not in stamps, f"{place}: duplicate timestamp {stamp}"
        stamps.add(stamp)
    return "data", stamps


# --- The network boundary -----------------------------------------------------------------


def request_url(symbol: str, month: str, key: str) -> str:
    """The request URL. It carries the key: never print, log or store it."""
    return ENDPOINT + urllib.parse.urlencode(
        {**REQUEST_PARAMS, "symbol": symbol, "month": month, KEY_PARAM: key}
    )


def redact(text: str, key: str) -> str:
    """`text` with every raw or URL-encoded occurrence of `key` replaced."""
    if not key:
        return text
    forms = {key, urllib.parse.quote_plus(key), urllib.parse.quote(key, safe="")}
    for form in sorted(forms, key=len, reverse=True):
        text = text.replace(form, REDACTION)
    return text


def read_key() -> str:
    key = os.environ.get(KEY_ENV_VAR, "")
    if not key:
        raise SystemExit(
            f"{KEY_ENV_VAR} is not set: put it in the gitignored .env and use --env-file"
        )
    if key != key.strip():
        raise SystemExit(f"{KEY_ENV_VAR} has surrounding whitespace; fix the .env line")
    return key


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        assert response.status == HTTP_OK, f"HTTP {response.status}"
        payload: bytes = response.read()
    assert payload, "empty response body"
    return payload


class AlphaVantageFetcher:
    """The real `Fetch`: spaced to the rate cap, retrying transient network failures."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.requests = 0
        self.last_start: float | None = None

    def __call__(self, symbol: str, month: str) -> bytes:
        url = request_url(symbol, month, self.key)
        for attempt in range(len(NETWORK_BACKOFF_SECONDS) + 1):
            if self.last_start is not None:
                sleep(max(0.0, self.last_start + REQUEST_SPACING_SECONDS - monotonic()))
            self.last_start = monotonic()
            self.requests += 1
            try:
                return http_get(url)
            except urllib.error.HTTPError as error:
                transient = error.code >= HTTP_SERVER_ERROR or error.code == HTTP_TOO_MANY_REQUESTS
                if not transient or attempt == len(NETWORK_BACKOFF_SECONDS):
                    raise
                reason = f"HTTP {error.code}"
            except (OSError, http.client.HTTPException) as error:  # timeout, reset, short read
                if attempt == len(NETWORK_BACKOFF_SECONDS):
                    raise
                reason = type(error).__name__  # the type only: its text could carry the URL
            delay = NETWORK_BACKOFF_SECONDS[attempt]
            say(f"{symbol} {month}  retry {attempt + 1} after {reason}; sleeping {delay}s")
            sleep(delay)
        raise AssertionError("unreachable: the final attempt returns or raises")


# --- Files and the manifest ---------------------------------------------------------------


def write_atomic(path: Path, payload: bytes) -> None:
    """Write via a sibling temp file and `os.replace`, so a crash never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_entries(manifest_path: Path, symbol: str) -> dict[str, dict[str, Any]]:
    """A previous run's month entries (none if no manifest); refuses a manifest made with other
    parameters, so two different requests never mix in one archive."""
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    assert manifest["symbol"] == symbol, f"{manifest_path} belongs to {manifest['symbol']}"
    assert manifest["params"] == REQUEST_PARAMS, f"{manifest_path} was pulled with other params"
    return dict(manifest["months"])


def write_manifest(path: Path, symbol: str, entries: dict[str, dict[str, Any]]) -> None:
    assert KEY_PARAM not in REQUEST_PARAMS, "the key must never reach a manifest"
    manifest = {
        "symbol": symbol,
        "params": REQUEST_PARAMS,
        "months": {month: entries[month] for month in sorted(entries)},
    }
    write_atomic(path, (json.dumps(manifest, indent=2) + "\n").encode())


def is_on_disk(entry: dict[str, Any], csv_path: Path) -> bool:
    """True when the month's file exists with exactly the recorded size and sha256."""
    if entry["status"] != config.MANIFEST_OK or not csv_path.exists():
        return False
    body = csv_path.read_bytes()
    return len(body) == entry["bytes"] and hashlib.sha256(body).hexdigest() == entry["sha256"]


def fetch_month(symbol: str, month: str, fetch: Fetch) -> tuple[bytes, set[datetime]]:
    """Fetch and validate one month, sleeping through rate-limit notices. Fails on any error
    reply and on an empty month: every month of the window must hold data."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        body = fetch(symbol, month)
        kind, result = check_body(body, symbol, month)
        if kind == "data":
            assert isinstance(result, set)
            assert result, f"{symbol} {month}: an empty month (no bars) inside the study window"
            return body, result
        assert kind == "rate_limit", f"{symbol} {month}: the vendor answered: {result}"
        assert attempt < RATE_LIMIT_RETRIES, (
            f"{symbol} {month}: still rate-limited after {RATE_LIMIT_RETRIES} retries; {result}"
        )
        say(f"{symbol} {month}  rate-limited; retry {attempt + 1}/{RATE_LIMIT_RETRIES}")
        sleep(RATE_LIMIT_SLEEP_SECONDS)
    raise AssertionError("unreachable")


def month_entry(body: bytes, stamps: set[datetime], expected: frozenset[date]) -> dict[str, Any]:
    days = {stamp.date() for stamp in stamps}
    present = days & expected
    return {
        "status": config.MANIFEST_OK,
        "rows": len(stamps),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "sessions_present": len(present),
        "sessions_expected": len(expected),
        "short": len(present) < len(expected),
        "off_calendar": sorted(day.isoformat() for day in days - expected),
        "first_ts": min(stamps).isoformat(sep=" "),
        "last_ts": max(stamps).isoformat(sep=" "),
        "pulled_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def pull_symbol(
    symbol: str,
    months: Sequence[str],
    fetch: Fetch,
    out: Path,
    calendar: dict[str, frozenset[date]],
) -> list[str]:
    """Pull every month of one symbol, oldest first. Returns the notes for the final summary
    (short months and bars on non-session days)."""
    directory = out / symbol
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / config.MANIFEST_NAME
    entries = load_entries(manifest_path, symbol)
    notes: list[str] = []
    for month in months:
        csv_path = directory / f"{symbol}_1min_{month}.csv"
        if month in entries and is_on_disk(entries[month], csv_path):
            say(f"{symbol} {month}  skip   on disk")
        else:
            body, stamps = fetch_month(symbol, month, fetch)
            write_atomic(csv_path, body)
            entries[month] = month_entry(body, stamps, calendar[month])
            write_manifest(manifest_path, symbol, entries)  # before the line: printed = saved
            entry = entries[month]
            say(
                f"{symbol} {month}  ok     rows={entry['rows']:,} "
                f"sessions={entry['sessions_present']}/{entry['sessions_expected']}"
            )
        entry = entries[month]
        if entry["short"]:
            notes.append(
                f"{symbol} {month}: {entry['sessions_present']}/{entry['sessions_expected']} sessions"
            )
        if entry["off_calendar"]:
            notes.append(f"{symbol} {month}: bars on non-session days {entry['off_calendar']}")
    write_manifest(manifest_path, symbol, entries)
    return notes


# --- CLI ----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--symbols", default=",".join(config.SYMBOLS), help="comma-separated subset"
    )
    parser.add_argument("--start-month", default=config.DATA_START.strftime("%Y-%m"))
    parser.add_argument("--end-month", default=config.DATA_END.strftime("%Y-%m"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no key, no network")
    args = parser.parse_args(argv)
    args.symbols = tuple(s.strip() for s in args.symbols.split(","))
    unknown = [s for s in args.symbols if s not in config.SYMBOLS]
    if unknown or len(set(args.symbols)) != len(args.symbols):
        parser.error(f"unknown or duplicate symbols: {args.symbols}")
    last_complete = (datetime.now(UTC).date().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    if (
        not month_index(args.start_month)
        <= month_index(args.end_month)
        <= month_index(last_complete)
    ):
        parser.error(f"need start <= end <= {last_complete} (the last complete month)")
    return args


def run(args: argparse.Namespace, key: str) -> None:
    started = monotonic()
    months = months_between(args.start_month, args.end_month)
    say(
        f"{len(args.symbols)} symbols, {months[0]} .. {months[-1]} ({len(months)} months) -> {args.out}"
    )
    if args.dry_run:
        say(f"dry run: at most {len(args.symbols) * len(months):,} requests, no key read")
        return
    calendar = session_calendar(months)
    fetcher = AlphaVantageFetcher(key)
    notes: list[str] = []
    for symbol in args.symbols:
        notes += pull_symbol(symbol, months, fetcher, args.out, calendar)
    say("\nSHORT MONTHS / OFF-CALENDAR BARS: " + ("none" if not notes else ""))
    for note in notes:
        say(f"  {note}")
    say(f"elapsed {(monotonic() - started) / 60:.1f} min; {fetcher.requests:,} requests")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    key = "" if args.dry_run else read_key()
    try:
        run(args, key)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- the redaction boundary
        # Deliberately blind: every failure is printed only through redact(), never raw.
        print(redact("".join(traceback.format_exception(error)), key), file=sys.stderr, end="")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
