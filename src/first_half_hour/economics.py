"""Chapter 2 economics: a limit-exit trade on the hit forecast, and the deflated Sharpe.

Trade: buy at P10 on day t when p_t > threshold. If y = 1 the limit order at P10 * (1 + b) fills
exactly (return b); else sell at the close (p_close / P10 - 1). Costs are per side, two sides per
traded day; untraded days return 0.

Inputs are plain frames. `forecasts`: session, year, target, y, model, p. `prices`: session, p10,
p_close, y (the per-session price frame; y is the target at the same b as the forecasts).
"""

import logging
from datetime import date

import numpy as np
import polars as pl
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


def _model_frame(
    forecasts: pl.DataFrame, prices: pl.DataFrame, model: str, target: str
) -> pl.DataFrame:
    """One model's forecasts for one target joined to the price frame, sorted by session."""
    f = forecasts.filter((pl.col("model") == model) & (pl.col("target") == target))
    assert f.height > 0, f"no forecasts for model {model!r}, target {target!r}"
    joined = f.select("session", "year", "p").join(prices, on="session", how="inner")
    assert joined.height == f.height, f"{model}: sessions missing from the price frame"
    return joined.sort("session")


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


def _sharpe_ci_row(r: np.ndarray, reps: int) -> tuple[float, float, float]:
    """Sharpe and CI; an all-zero series (no trades) is Sharpe 0 by definition, as in timing."""
    if np.all(r == r[0]) and r[0] == 0:
        return 0.0, 0.0, 0.0
    lo, hi = timing.sharpe_ci(r, reps)
    return timing.sharpe(r), lo, hi


def _row(name: str, traded: np.ndarray, d: pl.DataFrame, b: float, cost: float, reps: int) -> dict:
    p10, pc, y = d["p10"].to_numpy(), d["p_close"].to_numpy(), d["y"].to_numpy()
    r = trade_returns(traded, p10, pc, y, b, cost)
    sh, lo, hi = _sharpe_ci_row(r, reps)
    gross = trade_returns(traded, p10, pc, y, b, 0.0)[traded]
    gross_mean = float(gross.mean()) if gross.size else 0.0
    return {
        "rule": name, "cost_bp": cost, "n_days": d.height, "n_traded": int(traded.sum()),
        "mean": float(r.mean()), "sharpe": sh, "sharpe_ci_lo": lo, "sharpe_ci_hi": hi,
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


def trial_sharpes(
    forecasts: pl.DataFrame, prices: pl.DataFrame, models: list[str], window: tuple[date, date],
    target: str = "up_0.4", b: float = config.TARGET_B,
) -> np.ndarray:  # fmt: skip
    """Daily (non-annualised) Sharpes of every trading-rule variant: model x fixed threshold.

    N = len(models) * 5 variants, each a fixed threshold over the whole window at 1 bp. A variant
    with no trades has Sharpe 0. Feeds `deflated_sharpe`.
    """
    out = []
    for m in models:
        d = _model_frame(forecasts, prices, m, target)
        d = d.filter(pl.col("session").is_between(*window))
        for t in config.TRADE_THRESHOLDS:
            r = trade_returns(
                (d["p"] > t).to_numpy(), d["p10"].to_numpy(), d["p_close"].to_numpy(),
                d["y"].to_numpy(), b, config.THRESHOLD_COST_BP,
            )  # fmt: skip
            out.append(0.0 if r.std(ddof=1) == 0 else float(r.mean() / r.std(ddof=1)))
    return np.asarray(out)


def deflated_sharpe(r: np.ndarray, trial_sr: np.ndarray) -> dict[str, float]:
    """Deflated Sharpe ratio of Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio".

    Own code because neither arch nor statsmodels has it. All Sharpes are per period (daily), not
    annualised. With N = len(trial_sr) trials and V their variance (ddof 1):
        SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))),  g = Euler's gamma
    the expected maximum Sharpe of N trials with true Sharpe 0. Then the probabilistic Sharpe at
    benchmark SR0 with sample skew g3 and (non-excess) kurtosis g4 of `r`, T = len(r):
        DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2) ).
    """
    n = len(trial_sr)
    assert n >= 2, "need at least two trials to estimate their variance"
    t = len(r)
    sr = float(r.mean() / r.std(ddof=1))
    v = float(np.var(trial_sr, ddof=1))
    sr0 = np.sqrt(v) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    )
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
    return {"sharpe_daily": sr, "sr0_daily": float(sr0), "n_trials": n, "dsr": dsr}
