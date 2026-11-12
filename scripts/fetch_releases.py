"""Download past CPI and payroll (Employment Situation) release dates from FRED into data/daily/release_dates.csv.

An acquisition script, not a package module. Three requests: `fred/releases` (to confirm the
release ids by name), then `fred/release/dates` for each release. The file is derived from a
vendor API, so `data/` is gitignored and it is never committed. Every run overwrites the file.

Both releases come out at 08:30 ET, before the 09:30 open, so a release date is known by 10:00
of its own session (same-day use is allowed under the known-by-10:00 rule).

Key hygiene: the key is read from FRED_API_KEY and sent only in the request URL; every failure
is printed through redact(), never raw.

Fail loud: release ids carry the expected names, dates lie in DATA_START..DATA_END, are unique,
ascending and XNYS sessions except Good Friday (the BLS publishes while NYSE is shut; such a
date is kept, and its news reaches the next session), each full year holds 10-14 dates per
release, and a few known dates are present.

Run from the repo root: `uv run --env-file .env python scripts/fetch_releases.py [--out PATH]`.
"""

import argparse
import json
import os
import sys
import traceback
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from first_half_hour import config
from first_half_hour.calendar import sessions

BASE = "https://api.stlouisfed.org/fred/"
KEY_ENV_VAR = "FRED_API_KEY"
REQUEST_TIMEOUT_SECONDS = 60
RELEASES = {"cpi": (10, "Consumer Price Index"), "payroll": (50, "Employment Situation")}
PER_YEAR_RANGE = (10, 14)
KNOWN = {
    ("cpi", date(2024, 1, 11)),
    ("payroll", date(2024, 1, 5)),
    ("cpi", date(2013, 1, 16)),
    ("payroll", date(2013, 1, 4)),
}
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "daily" / "release_dates.csv"


def redact(text: str, key: str) -> str:
    return text.replace(key, "***")


def get(endpoint: str, key: str, **params: object) -> dict:
    query = urllib.parse.urlencode({**params, "api_key": key, "file_type": "json"})
    with urllib.request.urlopen(f"{BASE}{endpoint}?{query}", timeout=REQUEST_TIMEOUT_SECONDS) as r:
        assert r.status == 200, f"HTTP {r.status}"
        return json.loads(r.read().decode("utf-8"))


def check_ids(payload: dict) -> None:
    names = {r["id"]: r["name"] for r in payload["releases"]}
    assert payload["count"] == len(names), "releases list is paginated or truncated"
    for release, (rid, name) in RELEASES.items():
        assert names.get(rid) == name, f"{release}: id {rid} is {names.get(rid)!r}, not {name!r}"


def one_per_month(df: pl.DataFrame) -> pl.DataFrame:
    """Keep one date per release per month: FRED's release calendar also lists side releases.

    CPI: each February lists the seasonal-factor revision a few days BEFORE the January CPI, so
    the later date is kept. Payroll: a few months list a correction a few days AFTER the
    Employment Situation, so the earlier date is kept. Any other pattern fails loud.
    """
    month = pl.col("date").dt.truncate("1mo")
    cpi_feb = (pl.col("release") == "cpi") & (pl.col("date").dt.month() == 2)
    keep_last = (
        df.filter(pl.col("release") == "cpi").group_by(month.alias("m")).agg(pl.col("date").max())
    )
    keep_first = (
        df.filter(pl.col("release") == "payroll")
        .group_by(month.alias("m"))
        .agg(pl.col("date").min())
    )
    kept = pl.concat(
        [
            keep_last.with_columns(release=pl.lit("cpi")),
            keep_first.with_columns(release=pl.lit("payroll")),
        ]
    ).select("date", "release")
    dropped = df.join(kept, on=["date", "release"], how="anti")
    gap = dropped.join(kept.rename({"date": "kept"}), on="release").filter(
        (pl.col("kept") - pl.col("date")).dt.total_days().abs() <= 7
    )
    assert gap.select("date", "release").unique().height == dropped.height, (
        f"a dropped side release is not within 7 days of a kept one: {dropped.rows()}"
    )
    assert dropped.filter(pl.col("release") == "cpi").select(cpi_feb.all()).item() is not False, (
        "a CPI side release outside February"
    )
    print(f"dropped {dropped.height} side releases (one date per release per month kept)")
    return kept.sort("date", "release")


def parse(rows: list[tuple[str, date]]) -> pl.DataFrame:
    """(release, date) pairs -> frame sorted by date, validated. Fails on any surprise."""
    df = pl.DataFrame(rows, schema={"release": pl.String, "date": pl.Date}, orient="row")
    df = df.select("date", "release").sort("date", "release")
    assert df.height and not df.is_duplicated().any(), "empty or repeated (date, release) rows"
    assert df["date"].is_between(config.DATA_START, config.DATA_END).all(), "date outside window"
    trading = sessions(df["date"].min(), df["date"].max())["session"]
    off = df.filter(~pl.col("date").is_in(trading.implode()))
    # Good Friday: the BLS publishes on it while NYSE is shut. The date is kept (the analysis maps
    # it to the next session); any other non-session date fails.
    good_friday = (pl.col("date").dt.weekday() == 5) & pl.col("date").dt.month().is_in([3, 4])
    other = off.filter(~good_friday)
    assert other.is_empty(), f"release dates that are not XNYS sessions: {other.rows()}"
    print(f"{off.height} Good Friday releases kept (market closed that day): {off.rows()}")
    df = one_per_month(df)
    last_year = config.DATA_END.year
    counts = df.group_by(pl.col("date").dt.year().alias("year"), "release").len()
    for year, release, n in counts.iter_rows():
        if year != last_year:  # the final year is partial
            assert PER_YEAR_RANGE[0] <= n <= PER_YEAR_RANGE[1], f"{release} {year}: {n} dates"
    for release in RELEASES:
        assert df.filter(pl.col("release") == release)["date"].min().year == config.DATA_START.year
    missing = KNOWN - {(r, d) for d, r in df.rows()}
    assert not missing, f"known release dates absent: {sorted(missing)}"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out: Path = parser.parse_args().out
    key = os.environ[KEY_ENV_VAR]
    assert key, f"{KEY_ENV_VAR} is empty"
    try:
        check_ids(get("releases", key))
        rows = []
        for release, (rid, _) in RELEASES.items():
            payload = get(
                "release/dates",
                key,
                release_id=rid,
                realtime_start=config.DATA_START,
                realtime_end=config.DATA_END,
                sort_order="asc",
                include_release_dates_with_no_data="false",
                limit=10000,
            )
            assert payload["count"] == len(payload["release_dates"]), f"{release}: truncated"
            rows += [(release, date.fromisoformat(d["date"])) for d in payload["release_dates"]]
        df = parse(rows)
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 -- the redaction boundary
        print(redact(traceback.format_exc(), key), file=sys.stderr, end="")
        raise SystemExit(1) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    print(f"{df.height} release dates, {df['date'].min()} .. {df['date'].max()} -> {out}")


if __name__ == "__main__":
    main()
