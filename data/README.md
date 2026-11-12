# Data availability

The vendor's data is **not** in this repository. Its licence is for personal, non-commercial use and does not allow
redistribution. Everything published here is derived from it.

## What the analysis reads

| input | source | how to get it |
|---|---|---|
| One-minute bars, 2011-01 to 2026-08, for SPY and the robustness ETFs (QQQ, IWM, IVV, VOO, VTI, RSP, XLE, XLF, EFA, EEM; TLT, AGG, BND, HYG, GLD) | Alpha Vantage `TIME_SERIES_INTRADAY`, `interval=1min`, `outputsize=full`, `datatype=csv`, `month=YYYY-MM`, `extended_hours=false`; `adjusted` is not sent, so the endpoint's default applies (split- and dividend-adjusted) | Put your own key (`ALPHAVANTAGE_KEY=...`) in a gitignored `.env`, then `uv run --env-file .env python scripts/pull_archive.py` (writes `data/raw/<SYM>/` and a `manifest.json` per symbol; resumable; add `--symbols SPY` or `--dry-run` as needed). It paces itself to the premium cap of 75 requests per minute; a free key will not work. |
| VIX and VIX3M daily closes | FRED `VIXCLS`, `VXVCLS` (Cboe data, under Cboe's terms; not committed) | `uv run python scripts/fetch_daily.py` (writes `data/daily/VIXCLS.csv` and `data/daily/VXVCLS.csv`) |
| Exchange calendar, half-days | `exchange-calendars` (XNYS) | installed with the package |
| FOMC meeting dates | Federal Reserve FOMC calendar | a hand-made CSV kept under `data/daily/`; no script fetches it |
| CPI and payroll release dates | not yet obtained | to be added |
| SPY ex-dividend dates | not yet obtained | to be added |

## Checking that your data matches

A fresh pull may differ slightly (vendors re-adjust history). After downloading, run `uv run fhh fingerprint --data
data/raw` and compare with the table below.

| symbol | sessions | dropped | half_days | first_bar | last_bar | raw_rows | bars_min | bars_median |
|---|---|---|---|---|---|---|---|---|
| SPY | 3937 | 1 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1534237 | 210 | 390.0 |
| QQQ | 3936 | 2 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1532658 | 210 | 390.0 |
| IWM | 3937 | 1 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1532145 | 210 | 390.0 |
| IVV | 3923 | 15 | 32 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1525708 | 208 | 390.0 |
| VOO | 3484 | 454 | 31 | 2011-01-11 09:30 | 2026-08-31 15:59 | 1428785 | 102 | 390.0 |
| VTI | 3895 | 43 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1520477 | 203 | 390.0 |
| RSP | 3282 | 656 | 25 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1395879 | 126 | 381.0 |
| XLE | 3937 | 1 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1530740 | 209 | 390.0 |
| XLF | 3937 | 1 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1530703 | 209 | 390.0 |
| EFA | 3937 | 1 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1529568 | 210 | 390.0 |
| EEM | 3936 | 2 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1530021 | 209 | 390.0 |
| TLT | 3935 | 3 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1527501 | 201 | 390.0 |
| AGG | 3636 | 302 | 30 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1460679 | 159 | 388.0 |
| BND | 3617 | 321 | 28 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1453796 | 173 | 386.0 |
| HYG | 3885 | 53 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1516229 | 191 | 390.0 |
| GLD | 3935 | 3 | 33 | 2011-01-03 09:30 | 2026-08-31 15:59 | 1530578 | 208 | 390.0 |

The window holds 3938 NYSE sessions, 33 of them half-days (13:00 close), so `sessions` plus
`dropped` is 3938 for every symbol. A session is dropped when it lacks any bar the analysis
needs (09:59, or 61, 31 or 1 minutes before the close). `raw_rows` counts every row in the
window's files; the other columns count regular-session bars of kept sessions. A `bars_min` of
210 is a half-day (09:30-13:00).

## Known limits

- Thinly traded funds miss minutes, so they lose more sessions to the key-bar rule: RSP 656, VOO 454,
  BND 321 and AGG 302 of 3938, mostly before 2016. SPY loses one (2021-04-12, no 09:59 bar).
- The close used is the last one-minute bar's close (15:59, or 12:59 on half-days), not the official closing-auction
  price.
- Prices are dividend-adjusted, so a return spanning an ex-dividend date is a total return. A robustness check will
  drop ex-dividend days once their dates are added (see the table above).
