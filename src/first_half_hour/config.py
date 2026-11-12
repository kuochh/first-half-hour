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
