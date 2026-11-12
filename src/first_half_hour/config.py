"""Every constant the package uses, in one place.

Times are tz-naive America/New_York wall clock, the convention of the vendor's minute bars.
A bar is labelled by its START minute: the 09:30 bar covers 09:30:00-09:30:59.
"""

from datetime import date, time

SYMBOLS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "IVV", "VOO", "VTI", "RSP", "XLE", "XLF",
    "EFA", "EEM", "TLT", "AGG", "BND", "HYG", "GLD",
)  # fmt: skip
"""SPY and the robustness ETFs. VUG is left out: it split 6-for-1 between pulls."""

DATA_START: date = date(2011, 1, 1)
DATA_END: date = date(2026, 8, 31)
"""The study window (2011-01..2026-08). Nothing outside it is read."""

TIMEZONE: str = "America/New_York"
SESSION_OPEN: time = time(9, 30)
REGULAR_CLOSE: time = time(16, 0)
EARLY_CLOSE: time = time(13, 0)

OPENING_BAR_LAST: time = time(9, 59)
"""The last bar of the first half-hour; its close is the 10:00 price."""

CLOSE_OFFSETS_MIN: tuple[int, ...] = (61, 31, 1)
"""Key bars counted back from the session close: C-61, C-31 and C-1 (14:59, 15:29 and 15:59 on
a regular day; 11:59, 12:29 and 12:59 on a half-day). Their closes are the prices at the start
of the 12th and 13th half-hours and at the close."""

MANIFEST_NAME: str = "manifest.json"
MANIFEST_OK: str = "ok"
CSV_HEADER: str = "timestamp,open,high,low,close,volume"
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"

KNOWN_MISSING_SESSIONS: dict[str, frozenset[date]] = {s: frozenset() for s in SYMBOLS}
"""XNYS sessions inside the window that a symbol's pull lacks. Empty means full coverage is
demanded; a new gap fails loud rather than being widened here."""

FRED_SERIES: tuple[str, ...] = ("VIXCLS", "VXVCLS")
"""Daily FRED series the study reads: the VIX and VIX3M closes (Cboe data, under Cboe's terms)."""
FRED_DATE_COLUMN: str = "observation_date"
FRED_MISSING_TOKEN: str = "."

COSTS_BP: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
"""Timing-rule trading costs, basis points per side (two sides per active day)."""
TRADING_DAYS: int = 252
"""Annualisation: Sharpe = daily mean / sd * sqrt(TRADING_DAYS)."""
BOOTSTRAP_REPS: int = 10_000
BOOTSTRAP_SEED: int = 20260511
"""Stationary-bootstrap replications and the fixed seed for the Sharpe intervals."""

SAMPLES: dict[str, tuple[date, date]] = {
    "OVERLAP": (date(2011, 1, 3), date(2013, 12, 31)),
    "POST": (date(2014, 1, 2), date(2026, 8, 31)),
    "FULL": (date(2011, 1, 3), date(2026, 8, 31)),
}
"""Chapter 1 samples, inclusive. OVERLAP is the replication window (the paper's sample ran to 2013);
POST is data the paper never saw."""


def nw_lags(n_rows: int) -> int:
    """Newey-West lag count, floor(4 * (T / 100) ** (2 / 9)), T = rows in that regression."""
    assert n_rows > 0, f"need at least one row, got {n_rows}"
    return int(4 * (n_rows / 100) ** (2 / 9))


ROLLING_WINDOW: int = 252
"""Sessions in the rolling-slope window (about one year)."""

BREAK_DATES: tuple[date, ...] = (date(2014, 1, 2), date(2020, 3, 2), date(2022, 4, 18))
"""Structural-break candidates: publication era, Covid, Tuesday expiries listed."""

TRIM_SHARES: tuple[float, ...] = (0.01, 0.03, 0.05)
"""Shares of the largest |r1| trimmed in the descriptive robustness rows."""

DAILY_EXPIRY_START: date = date(2022, 5, 11)
"""First session of the daily-expiry era (Thursday SPX expiries listed on this date)."""

RELVOL_WINDOW: int = 20
"""Previous kept sessions in the relative-volume denominator."""

TERCILE_WARMUP: int = 252
"""Sessions before the expanding tercile cut-offs exist."""

VIX_MAX_STALE_DAYS: int = 7
"""A VIX observation older than this (calendar days) before the session fails loud."""

# ETF panel (S4)
EQUITY_ETFS: tuple[str, ...] = (
    "QQQ",
    "IWM",
    "IVV",
    "VOO",
    "VTI",
    "RSP",
    "XLE",
    "XLF",
    "EFA",
    "EEM",
)
NON_EQUITY_ETFS: tuple[str, ...] = ("TLT", "AGG", "BND", "HYG", "GLD")
"""Panel groups, reported separately: Holm is applied within each group."""

FOMC_ONE_DAY_MEETINGS: frozenset[date] = frozenset(
    {date(2011, 3, 15), date(2011, 8, 9), date(2011, 12, 13), date(2012, 3, 13)}
)
"""Scheduled FOMC meetings that lasted ONE day (statement the same afternoon), so the announcement
session is the file date itself. Every other scheduled meeting in data/daily/fomc_meetings.csv is
two-day and announces on the next session. Source: the Fed's historical meeting calendars,
federalreserve.gov/monetarypolicy/fomchistorical2011.htm and ...2012.htm."""

# Chapter 2, economics and interpretability (T3)
TARGET_B: float = 0.004
"""The further move before the close the trade targets: limit exit at P10 * (1 + b)."""
TRADE_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
"""Candidate probability thresholds; year Y's is picked on the same model's year Y-1 forecasts."""
THRESHOLD_COST_BP: float = 1.0
"""Cost per side at which the threshold is chosen."""
PERMUTATION_REPEATS: int = 10
PERMUTATION_SEED: int = 20260512
"""Permutation-importance shuffles per feature and the fixed seed."""
