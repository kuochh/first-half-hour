"""Part 2 features and targets: one row per (session, decision time tau), all known by tau.

Every feature uses only minute bars that start strictly before tau of the session itself, earlier
sessions, daily series shifted to the previous session (VIX) and calendar facts. Half-days are out
at every decision time; a session with no bar closing at tau is out at that time only. Rolling
features (s_rank60, vol_rel, hit20) read the previous rows of the same tau only. Direction-aligned
features are multiplied by d = +1 if s >= 0 else -1.
"""

from pathlib import Path

import numpy as np
import polars as pl

from first_half_hour import bars, config

SYMBOL = "SPY"
_OPEN_MOD = config.SESSION_OPEN.hour * 60 + config.SESSION_OPEN.minute
_CALENDAR_FLAGS = (
    "fomc", "cpi", "payroll", "opex_monthly", "month_end",
    "dow_tue", "dow_wed", "dow_thu", "dow_fri",
)  # fmt: skip
_MOD = pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(pl.Int32)


def bp_suffix(c: float) -> str:
    """Column suffix of cost threshold c: 0.0002 -> '2'."""
    return str(round(c * 1e4))


def _price_at(minute_of_day: int, m_tau: int) -> pl.Expr:
    """Price at a time T <= tau: the close of the bar starting at T - 1; at or before 09:30 the open
    of the 09:30 bar. Null if that bar is absent."""
    assert minute_of_day <= m_tau
    if minute_of_day <= _OPEN_MOD:
        return pl.col("open").filter(_MOD == _OPEN_MOD).first()
    return pl.col("close").filter(_MOD == minute_of_day - 1).first()


def _bar_stats(rth: pl.DataFrame, m_tau: int) -> pl.DataFrame:
    """Per session from the bars STARTING strictly before tau: p_tau, p_30, p_60, rv_sofar, vol_sofar."""
    sofar = rth.filter(_MOD < m_tau).sort("session", "minute")
    prev_px = pl.coalesce(pl.col("close").shift(1).over("session"), pl.col("open"))
    sq = ((pl.col("close") / prev_px).log() ** 2).alias("_sq")
    return (
        sofar.with_columns(sq)
        .group_by("session")
        .agg(
            p_tau=_price_at(m_tau, m_tau),
            p_30=_price_at(max(m_tau - 30, _OPEN_MOD), m_tau),
            p_60=_price_at(max(m_tau - 60, _OPEN_MOD), m_tau),
            rv_sofar=pl.col("_sq").mean().sqrt(),
            vol_sofar=pl.col("volume").sum().cast(pl.Float64),
        )
    )


def _past_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Share of the previous `window` values below values[i]; NaN until `window` exist."""
    out = np.full(values.size, np.nan)
    for i in range(window, values.size):
        out[i] = np.mean(values[i - window : i] < values[i])
    return out


def _one_tau(
    label: str, m_tau: int, base: pl.DataFrame, rth: pl.DataFrame, prev_map: pl.DataFrame,
    range_prev: pl.DataFrame,
) -> pl.DataFrame:  # fmt: skip
    """The rows of one decision time. `base` is the non-half-day sessions with ctrl columns joined."""
    rank_w, roll_w = config.PART2_RANK_WINDOW, config.PART2_ROLL_WINDOW
    stats = _bar_stats(rth.filter(pl.col("session").is_in(base["session"].implode())), m_tau)
    df = (
        base.join(stats, on="session", how="inner")
        .drop_nulls("p_tau")
        .sort("session")
        .with_columns(tau=pl.lit(label))
    )
    df = df.with_columns(s=pl.col("p_tau") / pl.col("prev_off") - 1).with_columns(
        d=pl.when(pl.col("s") >= 0).then(1.0).otherwise(-1.0)
    )
    df = df.with_columns(g=pl.col("d") * pl.col("r13"))
    df = df.with_columns(
        (pl.col("g") > c).cast(pl.Float64).alias(f"y_{bp_suffix(c)}") for c in config.PART2_COSTS_RT
    )
    d = pl.col("d")
    df = df.with_columns(
        abs_s=pl.col("s").abs(),
        s_rank60=pl.Series(_past_rank(df["s"].abs().to_numpy(), rank_w), nan_to_null=True),
        on_aligned=d * (pl.col("p_open") / pl.col("prev_off") - 1),
        intra_aligned=d * (pl.col("p_tau") / pl.col("p_open") - 1),
        d_long=(d > 0).cast(pl.Float64),
        rv_fh=(pl.col("rv_fh") / config.FIRST_HALF_HOUR_BARS).sqrt(),
        vol_rel=pl.col("vol_sofar")
        / pl.col("vol_sofar").rolling_mean(roll_w, min_samples=roll_w).shift(1),
        quarter_end=(pl.col("month_end") & pl.col("session").dt.month().is_in([3, 6, 9, 12])),
        last30_aligned=d * (pl.col("p_tau") / pl.col("p_30") - 1),
        last60_share=pl.when(pl.col("p_tau") == pl.col("prev_off"))
        .then(0.0)
        .otherwise(
            (d * (pl.col("p_tau") - pl.col("p_60")) / (pl.col("p_tau") - pl.col("prev_off"))).clip(
                -2, 2
            )
        ),
        hit20=pl.col("y_0").rolling_mean(roll_w, min_samples=roll_w).shift(1),
    )
    # previous calendar session's signal and payoff at this tau (null if it has no row here)
    yest = df.select(prev_session=pl.col("session"), _ys=pl.col("s"), _yr13=pl.col("r13"))
    df = (
        df.join(prev_map, on="session", how="left", validate="1:1")
        .join(yest, on="prev_session", how="left", validate="m:1")
        .join(range_prev, on="prev_session", how="left", validate="m:1")
        .with_columns(
            yest_s_aligned=d * pl.col("_ys"),
            yest_r13_aligned=d * pl.col("_yr13"),
        )
    )
    return df


def build(
    table: pl.DataFrame, ctrl: pl.DataFrame, cal: pl.DataFrame, rth: pl.DataFrame
) -> pl.DataFrame:
    """The long frame from the Part 1 table, controls, calendar and the collected regular-session bars."""
    assert table["session"].is_unique().all() and ctrl["session"].is_unique().all()
    sessions = set(table["session"].to_list())
    assert sessions <= set(ctrl["session"].to_list()), "controls miss a table session"
    prev_map = cal.sort("session").select("session", prev_session=pl.col("session").shift(1))
    rth = rth.filter(pl.col("session").is_in(table["session"].implode()))
    # previous session's range, over its regular-session bars (null if it is not in the table)
    day_range = (
        rth.group_by("session")
        .agg(hi=pl.col("high").max(), lo=pl.col("low").min())
        .join(table.select("session", "off_close"), on="session")
        .select(
            prev_session="session", range_prev=(pl.col("hi") - pl.col("lo")) / pl.col("off_close")
        )
    )
    flags = ctrl.select("session", "vix_lag", *_CALENDAR_FLAGS)
    base = (
        table.filter(~pl.col("is_half_day") & pl.col("prev_off").is_not_null())
        .select("session", "p_open", "prev_off", "off_close", "r13", "rv_fh")
        .join(flags, on="session", how="left", validate="1:1")
        .with_columns(year=pl.col("session").dt.year())
    )
    parts = [_one_tau(lab, m, base, rth, prev_map, day_range) for lab, m in config.PART2_TIMES]
    out = pl.concat(parts).sort("tau", "session")
    y_cols = [f"y_{bp_suffix(c)}" for c in config.PART2_COSTS_RT]
    return out.select(
        "session", "year", "tau", "s", "d", "g", "r13", *y_cols, *config.PART2_FEATURES
    ).with_columns(pl.col(c).cast(pl.Float64) for c in config.PART2_FEATURES)


def frame(
    table: pl.DataFrame,
    ctrl: pl.DataFrame,
    data_root: Path,
    cal: pl.DataFrame,
    symbol: str = SYMBOL,
) -> pl.DataFrame:
    """Part 2's long frame (columns in the module docstring and the spec). `table` is
    part1.part1_table (or part3_data.etf_table), `ctrl` is controls.controls(table, cal, daily_dir).
    A session whose previous calendar session is not in the table (null prev_off) has no signal
    and is left out (never happens for SPY's official-close table)."""
    rth = bars.rth_bars(data_root, symbol, cal).collect()
    return build(table, ctrl, cal, rth)
