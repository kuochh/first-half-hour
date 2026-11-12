"""Chapter 2 economics: a limit-exit trade on the hit forecast, and the deflated Sharpe.

Trade: buy at P10 on day t when p_t > threshold. If y = 1 the limit order at P10 * (1 + b) fills
exactly (return b); else sell at the close (p_close / P10 - 1). Costs are per side, two sides per
traded day; untraded days return 0.

Inputs are plain frames. `forecasts`: session, year, target, y, model, p. `prices`: session, p10,
p_close, y (built by `prices()`; y is the target at the same b as the forecasts, and is checked
against the forecasts' y).
"""

import logging
from datetime import date

import numpy as np
import polars as pl
from arch.bootstrap import StationaryBootstrap
from scipy import stats

from first_half_hour import config, timing

log = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329


def trade_returns(
    traded: np.ndarray,
    p10: np.ndarray,
    p_close: np.ndarray,
    y: np.ndarray,
    b: float,
    cost_bp: float,
) -> np.ndarray:
    """Daily net return: gross (b on a hit, else close / P10 - 1) minus 2 * cost, on traded days."""
    gross = np.where(y == 1, b, p_close / p10 - 1)
    return traded * (gross - 2 * cost_bp / 1e4)


def prices(daily: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """The per-session price frame: session, p10, p_close, y.

    `daily` is `returns.daily_table` (p_close is the 16:00 close, or the 12:59 bar's close on
    half-days; p_10 is the 09:59 bar's close). `target` is chapter 2's target frame (session, y,
    p10). The two 10:00 prices must agree.
    """
    j = target.select("session", "y", "p10").join(
        daily.select("session", "p_10", "p_close"), on="session", how="inner"
    )
    assert j.height == target.height, "target sessions missing from the daily table"
    assert j["p_close"].null_count() == 0 and j["p10"].null_count() == 0
    assert np.allclose(j["p10"].to_numpy(), j["p_10"].to_numpy(), rtol=0, atol=1e-12), "p10 != p_10"
    return j.select("session", "p10", "p_close", "y").sort("session")


def _check_b(target: str, b: float) -> None:
    assert target in config.TARGET_B_BY_NAME, f"unknown target {target!r}"
    assert config.TARGET_B_BY_NAME[target] == b, f"target {target!r} is not b = {b}"


def _model_frame(
    forecasts: pl.DataFrame, prices: pl.DataFrame, model: str, target: str
) -> pl.DataFrame:
    """One model's forecasts for one target joined to the price frame, sorted by session."""
    f = forecasts.filter((pl.col("model") == model) & (pl.col("target") == target))
    assert f.height > 0, f"no forecasts for model {model!r}, target {target!r}"
    joined = f.select("session", "year", "p", y_fc="y").join(prices, on="session", how="inner")
    assert joined.height == f.height, f"{model}: sessions missing from the price frame"
    assert (joined["y_fc"] == joined["y"]).all(), f"{model}: prices.y differs from forecasts.y"
    return joined.drop("y_fc").sort("session")


def _mean_net(frame: pl.DataFrame, threshold: float, b: float, cost_bp: float) -> float:
    r = trade_returns(
        (frame["p"] > threshold).to_numpy(), frame["p10"].to_numpy(), frame["p_close"].to_numpy(),
        frame["y"].to_numpy(), b, cost_bp,
    )  # fmt: skip
    return float(r.mean())


def choose_threshold(frame: pl.DataFrame, year: int, b: float = config.TARGET_B) -> float | None:
    """Threshold for `year`: the highest mean net return at 1 bp on the SAME model's year-1 rows.

    `frame` has year, p, p10, p_close, y. Only rows with year == `year` - 1 are read. The mean is
    over every day of that year (untraded days count 0). None when no threshold is > 0 (no trades).
    Ties go to the first (lowest) threshold.
    """
    prev = frame.filter(pl.col("year") == year - 1)
    assert prev.height > 0, f"no forecasts in {year - 1} to choose the {year} threshold"
    means = [_mean_net(prev, t, b, config.THRESHOLD_COST_BP) for t in config.TRADE_THRESHOLDS]
    best = int(np.argmax(means))
    return config.TRADE_THRESHOLDS[best] if means[best] > 0 else None


def strategy_positions(
    frame: pl.DataFrame, window: tuple[date, date], b: float = config.TARGET_B
) -> pl.DataFrame:
    """Per-session `traded` flag over the window, each year using its own chosen threshold."""
    lo, hi = window
    win = frame.filter(pl.col("session").is_between(lo, hi))
    assert win.height > 0, "empty window"
    parts = []
    for year in sorted(win["year"].unique().to_list()):
        th = choose_threshold(frame, year, b)
        log.info("threshold %d: %s", year, th)
        part = win.filter(pl.col("year") == year)
        traded = pl.lit(False) if th is None else pl.col("p") > th
        parts.append(part.with_columns(traded=traded, threshold=pl.lit(th, dtype=pl.Float64)))
    return pl.concat(parts)


def _sharpe0(r: np.ndarray) -> float:
    """Annualised Sharpe; 0 when the series has no spread (nothing traded, or a resample with none)."""
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else timing.sharpe(r)


def sharpe_ci0(r: np.ndarray, reps: int) -> tuple[float, float, int]:
    """95% stationary-bootstrap percentile interval of the Sharpe, never NaN.

    Same bootstrap as `timing.sharpe_ci` (arch, block from `timing._block`, config seed) but a
    resample with zero spread has Sharpe 0 (the all-zero rule above). Returns (lo, hi, n_zero)
    where n_zero counts such resamples; the bounds are asserted finite.
    """
    bs = StationaryBootstrap(timing._block(r), r, seed=config.BOOTSTRAP_SEED)
    draws = [pos[0] for pos, _ in bs.bootstrap(reps)]
    n_zero = sum(int(d.std(ddof=1) == 0) for d in draws)
    lo, hi = np.percentile([_sharpe0(d) for d in draws], [2.5, 97.5])
    assert np.isfinite(lo) and np.isfinite(hi), "Sharpe interval is not finite"
    return float(lo), float(hi), n_zero


def _sharpe_ci_row(r: np.ndarray, reps: int) -> tuple[float, float, float, int]:
    """Sharpe, CI and zero-spread resample count; an all-zero series (no trades) is Sharpe 0."""
    if np.all(r == r[0]) and r[0] == 0:
        return 0.0, 0.0, 0.0, 0
    lo, hi, n_zero = sharpe_ci0(r, reps)
    return _sharpe0(r), lo, hi, n_zero


def _row(name: str, traded: np.ndarray, d: pl.DataFrame, b: float, cost: float, reps: int) -> dict:
    p10, pc, y = d["p10"].to_numpy(), d["p_close"].to_numpy(), d["y"].to_numpy()
    r = trade_returns(traded, p10, pc, y, b, cost)
    sh, lo, hi, n_zero = _sharpe_ci_row(r, reps)
    gross = trade_returns(traded, p10, pc, y, b, 0.0)[traded]
    gross_mean = float(gross.mean()) if gross.size else 0.0
    return {
        "rule": name, "cost_bp": cost, "n_days": d.height, "n_traded": int(traded.sum()),
        "mean": float(r.mean()), "sharpe": sh, "sharpe_ci_lo": lo, "sharpe_ci_hi": hi, "ci_zero_spread_draws": n_zero,
        "break_even_bp": gross_mean / 2 / 1e-4 if gross_mean > 0 else None,
    }  # fmt: skip


def economics_table(
    forecasts: pl.DataFrame, prices: pl.DataFrame, models: dict[str, str], window: tuple[date, date],
    target: str = "up_0.4", b: float = config.TARGET_B, reps: int = config.BOOTSTRAP_REPS,
) -> pl.DataFrame:  # fmt: skip
    """Rows (rule, cost_bp) on the window: each named model, always_buy and zero.

    `models` maps a row label to a model name in `forecasts` (e.g. {"chosen_tier": "tier:hgb",
    "formula_cal": "formula_cal"}). Each model trades with its per-year threshold. All rules run
    on the same sessions (the intersection of the models' sessions in the window). Break-even cost
    per side in bp = mean gross return per traded day / 2 / 1e-4; null when that is <= 0.
    """
    _check_b(target, b)
    strat = {k: strategy_positions(_model_frame(forecasts, prices, m, target), window, b)
             for k, m in models.items()}  # fmt: skip
    sessions = set.intersection(*(set(s["session"].to_list()) for s in strat.values()))
    base = prices.filter(pl.col("session").is_in(sorted(sessions))).sort("session")
    assert base.height > 0, "no common sessions"
    rows = []
    for c in config.COSTS_BP:
        for k, s in strat.items():
            s = s.filter(pl.col("session").is_in(sorted(sessions))).sort("session")
            assert s["session"].to_list() == base["session"].to_list()
            rows.append(_row(k, s["traded"].to_numpy(), base, b, c, reps))
        rows.append(_row("always_buy", np.ones(base.height, dtype=bool), base, b, c, reps))
    rows.append(_row("zero", np.zeros(base.height, dtype=bool), base, b, 0.0, reps))
    return pl.DataFrame(rows, infer_schema_length=None)


def trial_set(
    forecasts: pl.DataFrame, prices: pl.DataFrame, models: list[str], window: tuple[date, date],
    target: str = "up_0.4", b: float = config.TARGET_B,
) -> np.ndarray:  # fmt: skip
    """The deflated Sharpe's N = 20 trials: 5 thresholds x the 4 forecasts traded.

    `models` is the three tuned tiers and the calibrated formula. Each variant is a fixed
    threshold over the whole window at 1 bp, on the sessions all four share. Returns the 20 DAILY
    (non-annualised) Sharpes; a variant with no trades has Sharpe 0.
    """
    _check_b(target, b)
    assert len(set(models)) == config.N_TRADED_FORECASTS, "the trial set is 4 distinct forecasts"
    frames = [
        _model_frame(forecasts, prices, m, target).filter(pl.col("session").is_between(*window))
        for m in models
    ]
    common = set.intersection(*(set(f["session"].to_list()) for f in frames))
    assert common, "no common sessions"
    out = []
    for d in frames:
        d = d.filter(pl.col("session").is_in(sorted(common)))
        for t in config.TRADE_THRESHOLDS:
            r = trade_returns(
                (d["p"] > t).to_numpy(), d["p10"].to_numpy(), d["p_close"].to_numpy(),
                d["y"].to_numpy(), b, config.THRESHOLD_COST_BP,
            )  # fmt: skip
            out.append(0.0 if r.std(ddof=1) == 0 else float(r.mean() / r.std(ddof=1)))
    assert len(out) == config.N_TRIALS, f"expected {config.N_TRIALS} trials, got {len(out)}"
    return np.asarray(out)


def rule_returns(
    forecasts: pl.DataFrame, prices: pl.DataFrame, model: str, window: tuple[date, date],
    target: str = "up_0.4", b: float = config.TARGET_B,
) -> np.ndarray:  # fmt: skip
    """Daily net returns at 1 bp of one model's per-year-threshold rule over the window."""
    _check_b(target, b)
    pos = strategy_positions(_model_frame(forecasts, prices, model, target), window, b)
    return trade_returns(
        pos["traded"].to_numpy(), pos["p10"].to_numpy(), pos["p_close"].to_numpy(),
        pos["y"].to_numpy(), b, config.THRESHOLD_COST_BP,
    )  # fmt: skip


def deflated_sharpe_from_moments(
    sr: float, v: float, n: int, t: int, skew: float, kurt: float
) -> tuple[float, float]:
    """(SR0, DSR) from per-period Sharpe `sr`, trial-Sharpe variance `v`, N trials, T periods.

    Bailey and Lopez de Prado (2014). `kurt` is NON-excess (3 for a normal).
        SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))),  g = Euler's gamma
        DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) ).
    """
    assert n >= 2, "need at least two trials"
    sr0 = np.sqrt(v) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    )
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(sr0), float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))


def deflated_sharpe(r: np.ndarray, trial_sr: np.ndarray) -> dict:
    """Deflated Sharpe ratio of the daily return series `r` against the trial Sharpes.

    Own code because neither arch nor statsmodels has it. All Sharpes are per period (daily), not
    annualised; V = variance (ddof 1) of `trial_sr`, N = len(trial_sr), T = len(r); skewness and
    non-excess kurtosis are the sample's. A series with no spread (the rule never traded) has no
    Sharpe: returns dsr None and a note, never NaN.
    """
    n = len(trial_sr)
    assert n >= 2, "need at least two trials to estimate their variance"
    if r.std(ddof=1) == 0:
        return {"sharpe_daily": 0.0, "sr0_daily": None, "n_trials": n, "dsr": None,
                "note": "no trades: the rule never traded, so the deflated Sharpe is undefined"}  # fmt: skip
    sr = float(r.mean() / r.std(ddof=1))
    sr0, dsr = deflated_sharpe_from_moments(
        sr, float(np.var(trial_sr, ddof=1)), n, len(r),
        float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)),
    )  # fmt: skip
    return {"sharpe_daily": sr, "sr0_daily": sr0, "n_trials": n, "dsr": dsr, "note": ""}
