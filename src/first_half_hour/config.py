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


# Chapter 2 (walk-forward). Read at call time so synthetic mode can override them.
CH2_FIRST_TEST_YEAR: int = 2015
"""First walk-forward test year; the last is the last year in the frame (2026, to 2026-08-31)."""
CH2_VALID: tuple[date, date] = (date(2016, 1, 1), date(2018, 12, 31))
"""Validation window: the only data the tier choice reads."""
CH2_JUDGE: tuple[date, date] = (date(2019, 1, 2), date(2026, 8, 31))
"""Judging window (seen data, exploration)."""
CH2_ENET_C: tuple[float, ...] = (0.01, 0.1, 1.0)
CH2_ENET_L1: tuple[float, ...] = (0.2, 0.5, 0.8)
CH2_HGB_LR: tuple[float, ...] = (0.03, 0.1)
CH2_HGB_DEPTH: tuple[int, ...] = (2, 3)
CH2_HGB_ITER: tuple[int, ...] = (100, 300)
CH2_HGB_MIN_LEAF: int = 50
CH2_SEED: int = 20261112
"""random_state of the boosted trees, the elastic net (saga) and the permutation shuffles."""
CH2_TIERS: tuple[str, ...] = ("logit", "enet", "hgb")
"""Candidate tiers, simplest first (ties in the choice rule go to the simpler tier)."""
CH2_MAX_ITER: int = 5000
"""max_iter of every logistic / elastic-net fit; n_iter_ < CH2_MAX_ITER is asserted on each fit."""
CH2_SPA_REPS: int = 1_000
CH2_SPA_SEED: int = 20261113
CH2_PRIMARY_ALPHA: float = 0.05
"""One-sided Diebold-Mariano level of the primary criterion (D19)."""
CH2_REGISTERED_MODEL: str = "fhh-chapter2-hit"

# Chapter 2 (target, features, baselines)
TARGET_B: float = 0.004
"""The further move: the day's later high reaches the 10:00 price times (1 + B) before the close."""
DESCRIPTIVE_BS: tuple[float, ...] = (0.002, 0.006)
"""Descriptive-only thresholds, run beside TARGET_B and counted as runs."""
MIRROR_B: float = 0.004
"""Threshold of the descriptive mirror target (a later low at or below P10 * (1 - MIRROR_B))."""
TAU_MIN_REGULAR: int = 360
TAU_MIN_HALF_DAY: int = 180
"""Minutes from 10:00 to the close, regular and half-day sessions."""
FIRST_HALF_HOUR_BARS: int = 30
"""One-minute bars (hence returns) in 09:30-10:00; sigma = sqrt(rv_fh / FIRST_HALF_HOUR_BARS)."""
PROB_CLIP_EPS: float = 1e-6
"""Probabilities are clipped to [eps, 1 - eps] for log loss only."""

CH2_Z_FEATURE: bool = True
"""Add z = ln(1+TARGET_B) / (sigma sqrt(tau)) to the feature list so the logistic tiers nest the
calibrated formula (added on the plan auditor's advice before any real-data result was seen)."""
TOUCH_REL_TOL: float = 1e-9
"""Relative tolerance for a touch exactly at P10 * (1 +/- b), so float rounding cannot turn it into a miss."""

_BASE_FEATURES: tuple[str, ...] = (
    "r_on", "r_fh", "abs_r_on", "abs_r_fh", "sigma", "or_high", "or_low", "relvol",
    "rv_day_lag", "ret_prev", "vix_lag", "vix_ratio_lag", "fomc", "cpi", "payroll",
    "opex_monthly", "opex_quarterly", "daily_expiry_era", "dow_tue", "dow_wed", "dow_thu",
    "dow_fri", "is_half_day",
)  # fmt: skip
FEATURE_COLUMNS: tuple[str, ...] = _BASE_FEATURES + (("z",) if CH2_Z_FEATURE else ())
"""The chapter 2 feature list, all known by 10:00 (booleans enter as 0/1)."""

# Chapter 2, economics and interpretability (T3)
TARGET_B_BY_NAME: dict[str, float] = {"up_0.4": 0.004}
"""Target name to its b; the economics code asserts the b it is given matches the target."""
N_TRADED_FORECASTS: int = 4
"""The three tuned tiers and the calibrated formula."""
TRADE_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
"""Candidate probability thresholds; year Y's is picked on the same model's year Y-1 forecasts."""
N_TRIALS: int = 20
"""Deflated-Sharpe trial count: 5 thresholds x 4 traded forecasts (fixed; asserted in trial_set)."""
THRESHOLD_COST_BP: float = 1.0
"""Cost per side at which the threshold is chosen."""
PERMUTATION_REPEATS: int = 10
PERMUTATION_SEED: int = 20260512
"""Permutation-importance shuffles per feature and the fixed seed."""


# Part 1 (replication of Gao et al. and Baltussen et al. on SPY)
PART1_WINDOWS: dict[str, tuple[date, date]] = {
    "gao_overlap": (date(2011, 1, 3), date(2013, 12, 31)),
    "gao_after": (date(2014, 1, 2), date(2026, 8, 31)),
    "balt_overlap": (date(2011, 1, 3), date(2020, 5, 1)),
    "balt_after": (date(2020, 5, 4), date(2026, 8, 31)),
}
"""Part 1 windows, inclusive. Gao's data ends 2013-12; Baltussen's ends 2020-05-01 (Table A1, ES)."""
PART1_BOOT_REPS: int = 2_000
"""Stationary-bootstrap replications of every Part 1 bootstrap SE (reproduce --synthetic lowers it)."""
PART1_OOS_START: date = date(2012, 1, 3)
"""First Part 1 OOS forecast session (2011 is the one-year warm-up, as in Gao's first window)."""
PART1_SMALL_N: int = 250
"""A Part 1 row with fewer observations is flagged small_sample."""
PART1_VERDICT_Z: float = 2.0
PART1_MDE_Z: float = 2.0 + 0.84
"""Verdict threshold (2 SE) and minimum detectable effect multiplier (2 + 0.84: 80% power)."""
PART1_COST_BP: float = 1.0
"""Cost per side of the `cost_1bp` variants."""
PART1_JUMP_PCT: tuple[float, float] = (0.10, 0.90)
"""Percentiles of daily close-to-close returns that define a Baltussen 'jump'."""
OFFICIAL_CLOSE_FILE: str = "daily_close/SPY_yahoo_daily.csv"
"""The official close CSV, relative to the bars' data root."""
OFFICIAL_CLOSE_VERIFY: bool = True
"""Check the official close CSV against the recorded size and sha256 (reproduce --synthetic sets
False: its CSV is a stub)."""
OFFICIAL_CLOSE_BYTES: int = 466_684
OFFICIAL_CLOSE_SHA256: str = "0013f5c0fee4641ef6293a89a0583dbe825ff5062a30206fa65189aee72fa6b4"
"""Size and sha256 of the real daily_close/SPY_yahoo_daily.csv (Yahoo pull of 2026-11-16; Yahoo revises history)."""
OFFICIAL_CLOSE_MAX_PERIOD_STEP: float = 5e-4
"""Adjacent dividend periods' calibration constants c_p may differ by less than 5 bp (largest real step:
+3.0 bp into the period starting 2020-03-20, Covid; the rest are 2 bp or less)."""
OFFICIAL_CLOSE_MAX_GAP: float = 0.015
"""|official close on the bars' basis / 15:59 bar - 1| must be below 150 bp on every session."""
OFFICIAL_DIVIDEND_TOL: float = 0.005
"""Yahoo and data/daily/spy_dividends.csv amounts must agree within this (dollars per share)."""

# Paper values keyed (item, term, metric): (value, source, market). Coefficients in Baltussen's tables are x100 in the paper;
# divided by 100 here. Versions held: "Gao WP Oct 2014", "Baltussen SSRN Aug 2020".
_GAO = "Gao WP Oct 2014"
_BALT = "Baltussen SSRN Aug 2020"
_SPY = "SPY"
_FUT = "pooled equity-index futures"
PAPER: dict[tuple[str, str, str], tuple[float, str, str]] = {
    ("gao_slope", "r1", "slope"): (0.069, f"{_GAO}, Table 1 col 1", _SPY),
    ("gao_r12", "r12", "slope"): (0.118, f"{_GAO}, Table 1 col 2", _SPY),
    ("gao_joint", "r1", "slope"): (0.068, f"{_GAO}, Table 1 col 3", _SPY),
    ("gao_joint", "r12", "slope"): (0.114, f"{_GAO}, Table 1 col 3", _SPY),
    ("gao_vol_tercile", "low", "slope"): (0.023, f"{_GAO}, Table 2", _SPY),
    ("gao_vol_tercile", "mid", "slope"): (0.054, f"{_GAO}, Table 2", _SPY),
    ("gao_vol_tercile", "high", "slope"): (0.072, f"{_GAO}, Table 2", _SPY),
    ("gao_oos_r2", "r1", "r2_os"): (0.0169, f"{_GAO}, Table 3", _SPY),
    ("gao_rule", "rule", "ann_mean"): (0.0667, f"{_GAO}, Table 5", _SPY),
    ("gao_rule", "rule", "sharpe"): (1.08, f"{_GAO}, Table 5", _SPY),
    ("gao_rule", "rule", "success"): (0.5437, f"{_GAO}, Table 5", _SPY),
    ("gao_rule", "rule", "success_excess"): (0.0395, f"{_GAO}, Table 5 (54.37 - 50.42)", _SPY),
    ("gao_news", "FOMC", "r2"): (0.110, f"{_GAO}, Table 11", _SPY),
    ("gao_news", "non-FOMC", "r2"): (0.025, f"{_GAO}, Table 11", _SPY),
    ("gao_news", "CPI", "r2"): (0.050, f"{_GAO}, Table 11", _SPY),
    ("gao_news", "non-CPI", "r2"): (0.025, f"{_GAO}, Table 11", _SPY),
    ("balt_slope", "rod", "slope"): (0.0618, f"{_BALT}, Table B1 (ES)", "S&P 500 E-mini futures"),
    ("balt_horse", "r1", "slope"): (0.0114, f"{_BALT}, Table 3 Panel A", _FUT),
    ("balt_horse", "rod", "slope"): (0.0365, f"{_BALT}, Table 3 Panel A", _FUT),
    ("balt_large_move", "x_jump", "slope"): (0.0516, f"{_BALT}, Table 10 (rLH)", _FUT),
    ("balt_reversal", "k1", "slope"): (-0.1451, f"{_BALT}, Table 11 Panel A", _FUT),
    ("balt_reversal", "k2", "slope"): (-0.2905, f"{_BALT}, Table 11 Panel A", _FUT),
    ("balt_reversal", "k3", "slope"): (-0.2798, f"{_BALT}, Table 11 Panel A", _FUT),
    ("balt_last_k", "r13", "slope"): (0.0598, f"{_BALT}, Table 12 Panel A", "S&P 500 cash index"),
    ("balt_last_k", "r_last15", "slope"): (0.0283, f"{_BALT}, Table 12 Panel A", "S&P 500 cash index"),
    ("balt_last_k", "r_last5", "slope"): (0.0194, f"{_BALT}, Table 12 Panel A", "S&P 500 cash index"),
    ("balt_rule", "rule", "ann_mean"): (0.0686, f"{_BALT}, Table 6 Panel A", "equity-index futures, 1/N portfolio"),
    ("balt_rule", "rule", "sharpe"): (1.73, f"{_BALT}, Table 6 Panel A", "equity-index futures, 1/N portfolio"),
    ("balt_rule", "rule", "success"): (0.55, f"{_BALT}, Table 6 Panel A", "equity-index futures, 1/N portfolio"),
    ("balt_rule", "rule", "success_excess"): (0.02, f"{_BALT}, Table 6 Panel A (55 - 53)", "equity-index futures, 1/N portfolio"),
}  # fmt: skip
_ONE_N = "equity-index futures, 1/N portfolio"
PAPER_ALWAYS_LONG: dict[tuple[str, str], tuple[float, str, str]] = {
    ("gao_rule", "ann_mean"): (-0.0111, f"{_GAO}, Table 5 Panel B (always long)", _SPY),
    ("gao_rule", "sharpe"): (-0.18, f"{_GAO}, Table 5 Panel B (always long)", _SPY),
    ("gao_rule", "success"): (0.5042, f"{_GAO}, Table 5 Panel B (always long)", _SPY),
    ("balt_rule", "ann_mean"): (0.0044, f"{_BALT}, Table 6 Panel A (always long)", _ONE_N),
    ("balt_rule", "sharpe"): (0.11, f"{_BALT}, Table 6 Panel A (always long)", _ONE_N),
    ("balt_rule", "success"): (0.53, f"{_BALT}, Table 6 Panel A (always long)", _ONE_N),
}
"""The papers' own always-long figures, shown as paper values on the always_long rows (no verdict)."""

# Part 1 report (figures and report; P1-3)
PART1_REPORT_ROLLING_WINDOW: int = 756
"""Sessions in the Part 1 rolling-slope figure (three years)."""
PART1_PAPER_SLOPES: dict[str, float] = {"r1": 0.069, "rod": 0.0618}
"""Paper slopes drawn in the rolling figure: Gao WP Oct 2014 Table 1; Baltussen SSRN Aug 2020 Table B1."""
PART1_FIGURES: tuple[str, str] = ("part1_cumulative", "part1_rolling_slope")
"""File stems under `<out>/figures/`."""

# Part 2 models (P2-B): walk-forward, rules, baselines, bootstrap
PART2_LOGIT_C: tuple[float, ...] = (0.01, 0.1, 1.0)
"""Inverse regularisation options of the Part 2 logit (picked per test year by last-year log loss)."""
PART2_HGB_GRID: tuple[tuple[int, int], ...] = ((2, 100), (2, 300), (3, 100), (3, 300))
"""(max_depth, max_iter) options of the Part 2 boosted trees, in tie-break order."""
PART2_HGB_LR: float = 0.05
PART2_HGB_MIN_LEAF: int = 50
PART2_MAX_ITER: int = 2000
"""max_iter of the Part 2 logit; n_iter_ < PART2_MAX_ITER is asserted on each fit."""
PART2_QS: tuple[float, ...] = (0.10, 0.20, 0.50)
PART2_MAIN_Q: float = 0.20
PART2_BOOT_REPS: int = 2_000
PART2_AUC_ALARM: float = 0.75
"""Every cell's test AUC must stay below this (a leak alarm); asserted."""
PART2_SEED: int = 20261118
PART2_SPLITS: dict[str, tuple[date, date]] = {
    "pre_2020_05": (date(2014, 1, 2), date(2020, 5, 1)),
    "post_2020_05": (date(2020, 5, 4), DATA_END),
}
PART2_COVID_OUT: tuple[date, date] = (date(2020, 2, 1), date(2020, 4, 30))
PART2_EVAL_COSTS: tuple[float, ...] = (0.0, 0.0002, 0.0004)
"""Evaluation costs, round trip (return units)."""
PART2_MAIN_C_BP: int = 2
PART2_HEADLINE_TAU: str = "15:30"
PART2_SPLIT_MIN_DAYS: int = 30
"""A split with fewer test days is not reported (only matters on synthetic data)."""
