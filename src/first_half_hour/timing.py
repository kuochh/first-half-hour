"""Timing rules (item 9): trade the last half-hour (C-31 to C) on r1 and r12, with costs.

Both signals are known at the entry (r1 at 10:00, r12 at C-31). A trade enters at the C-31 close
and exits at the C-1 close, so it earns r13 and pays the cost on two sides.
"""

import numpy as np
import polars as pl
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from first_half_hour import config

RULES = ("one_signal", "two_signal", "always_long")


def positions(table: pl.DataFrame) -> pl.DataFrame:
    """Add one position column per rule: +1 long, -1 short, 0 flat.

    one_signal: long if r1 > 0 else short. two_signal: long if r1 > 0 and r12 > 0, short if both
    <= 0, else flat. always_long: +1.
    """
    r1_up, r12_up = pl.col("r1") > 0, pl.col("r12") > 0
    return table.with_columns(
        one_signal=pl.when(r1_up).then(1).otherwise(-1),
        two_signal=pl.when(r1_up & r12_up).then(1).when(~r1_up & ~r12_up).then(-1).otherwise(0),
        always_long=pl.lit(1),
    )


def net_returns(pos: np.ndarray, r13: np.ndarray, cost_bp: float) -> np.ndarray:
    """Daily P&L: pos * r13 minus 2 * cost per side on every active (pos != 0) day."""
    return pos * r13 - (pos != 0) * 2 * cost_bp / 1e4


def sharpe(r: np.ndarray) -> float:
    """Annualised Sharpe of a daily series: mean / sd (ddof 1) * sqrt(252)."""
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def sharpe_ci(r: np.ndarray, reps: int = config.BOOTSTRAP_REPS) -> tuple[float, float]:
    """95% percentile interval of the Sharpe from a stationary bootstrap (arch).

    Block length: arch's optimal_block_length (stationary). Seed fixed in config.
    """
    # arch needs block >= 1; its estimate is below 1 for serially uncorrelated series (then iid).
    block = max(1.0, float(optimal_block_length(r).to_numpy()[0, 0]))
    bs = StationaryBootstrap(block, r, seed=config.BOOTSTRAP_SEED)
    ci = bs.conf_int(sharpe, reps=reps, method="percentile", size=0.95)
    return float(ci[0, 0]), float(ci[1, 0])


def timing_results(
    table: pl.DataFrame, samples: tuple[str, ...] = ("POST", "OVERLAP", "FULL"),
    reps: int = config.BOOTSTRAP_REPS,
) -> pl.DataFrame:  # fmt: skip
    """Tidy rows (sample, rule, cost_bp) with mean, sd, Sharpe, CI, hit rate and break-even cost.

    Every rule runs on the same sessions of a sample (non-null r1, r12, r13). always_long is the
    baseline (also paid the costs, on every day) and `sharpe_always_long` sits beside every row;
    a `zero` row (no trades, Sharpe 0 by definition) is added once per sample. Break-even cost per
    side, in bp = mean gross return per active day / 2 / 1e-4.
    """
    rows = []
    for name in samples:
        lo, hi = config.SAMPLES[name]
        d = positions(
            table.filter(pl.col("session").is_between(lo, hi))
            .select("session", "r1", "r12", "r13")
            .drop_nulls()
            .sort("session")
        )
        assert d.height > 0, f"no sessions in {name}"
        r13 = d["r13"].to_numpy()
        long_sharpe = {c: sharpe(net_returns(np.ones(d.height), r13, c)) for c in config.COSTS_BP}
        for rule in RULES:
            pos = d[rule].to_numpy()
            active = pos != 0
            gross_active = (pos * r13)[active]
            break_even = float(gross_active.mean() / 2 / 1e-4)
            for c in config.COSTS_BP:
                r = net_returns(pos, r13, c)
                lo_ci, hi_ci = sharpe_ci(r, reps)
                rows.append(
                    {
                        "sample": name,
                        "rule": rule,
                        "cost_bp": c,
                        "n_days": d.height,
                        "n_active": int(active.sum()),
                        "mean": float(r.mean()),
                        "sd": float(r.std(ddof=1)),
                        "sharpe": sharpe(r),
                        "sharpe_ci_lo": lo_ci,
                        "sharpe_ci_hi": hi_ci,
                        "hit_rate": float((r[active] > 0).mean()),
                        "break_even_bp": break_even,
                        "sharpe_always_long": long_sharpe[c],
                    }
                )
        rows.append(
            {
                "sample": name, "rule": "zero", "cost_bp": 0.0, "n_days": d.height,
                "n_active": 0, "mean": 0.0, "sd": 0.0, "sharpe": 0.0, "sharpe_ci_lo": 0.0,
                "sharpe_ci_hi": 0.0, "hit_rate": None, "break_even_bp": None,
                "sharpe_always_long": long_sharpe[0.0],
            }
        )  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)
