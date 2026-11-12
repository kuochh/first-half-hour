"""Download the daily FRED series the study reads (VIXCLS, VXVCLS) into data/daily/.

An acquisition script, not a package module. The files are Cboe data under Cboe's terms, so
`data/` is gitignored and they are never committed. Every run overwrites each file with the
full history FRED publishes; there is no local state.

Fail loud: each file is checked before it is kept: the expected two-column header, ascending
unique ISO dates, every value a float or FRED's "." marker (counted, never dropped silently),
and coverage from the study start through the study end.

Run from the repo root: `uv run python scripts/fetch_daily.py [--out data/daily]`.
"""

import argparse
import urllib.request
from datetime import date
from pathlib import Path

from first_half_hour import config

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
"""The no-API-key CSV endpoint."""
REQUEST_TIMEOUT_SECONDS = 60
HTTP_OK = 200
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "daily"


def fetch_series(series_id: str) -> bytes:
    """One FRED series as CSV bytes. Fails on any non-200 status or an empty body."""
    # urllib's default User-Agent: FRED stalls requests carrying a custom one (seen 2026-11-11).
    request = urllib.request.Request(FRED_CSV_URL.format(series_id=series_id))
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        assert response.status == HTTP_OK, f"{series_id}: HTTP {response.status}"
        payload: bytes = response.read()
    assert payload, f"{series_id}: empty response body"
    return payload


def validate_series(series_id: str, body: bytes) -> tuple[date, date, int, int]:
    """Check one series; return (first date, last date, rows, rows marked missing)."""
    lines = body.decode("utf-8").splitlines()
    assert len(lines) >= 2, f"{series_id}: file holds no observations"
    header = lines[0].split(",")
    assert header == [config.FRED_DATE_COLUMN, series_id], f"{series_id}: header {header}"

    dates: list[date] = []
    missing = 0
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split(",")
        assert len(cells) == 2, f"{series_id}:{number}: expected 2 columns, got {len(cells)}"
        dates.append(date.fromisoformat(cells[0]))
        if cells[1] in (config.FRED_MISSING_TOKEN, ""):
            missing += 1
        else:
            float(cells[1])  # any other non-numeric cell raises here

    assert dates == sorted(set(dates)), f"{series_id}: dates not ascending or not unique"
    assert dates[0] <= config.DATA_START, f"{series_id}: starts {dates[0]}"
    assert dates[-1] >= config.DATA_END, f"{series_id}: ends {dates[-1]}, before the study end"
    assert missing < len(dates), f"{series_id}: every observation is missing"
    return dates[0], dates[-1], len(dates), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out: Path = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for series_id in config.FRED_SERIES:
        body = fetch_series(series_id)
        first, last, rows, missing = validate_series(series_id, body)
        (out / f"{series_id}.csv").write_bytes(body)
        print(f"{series_id}: {rows} rows, {first} .. {last}, {missing} marked missing -> {out}")


if __name__ == "__main__":
    main()
