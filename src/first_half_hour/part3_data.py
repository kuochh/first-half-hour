"""Part 3 data: per-ETF daily tables on the 15:59-bar basis, the Part 2 frame per ETF, the coverage
table and the "A" rows (is the last half-hour effect there, per class and period).

Every ETF is read on the bar basis: off_close is the close of bar C-1, prev_off the previous
kept session's (null if the previous calendar session is not kept: never bridged).
"""

from pathlib import Path

import numpy as np
import polars as pl
from arch.bootstrap import StationaryBootstrap

from first_half_hour import config, part1, part2_features, returns, timing

COLUMNS = (
    "class", "period", "item", "metric", "cost_bp", "n_days", "n_etfs", "est", "se", "t",
    "verdict", "level",
)  # fmt: skip
"""`class` holds the class key (level "class" and "spy_official") or the symbol (level "etf")."""
CLEARLY_ABOVE, CLEARLY_BELOW, CANNOT_TELL = "clearly above", "clearly below", "cannot tell"
_SIGNALS = {"r1": "r1", "rod": "rod"}


def etf_table(symbol: str, data_root: Path, daily_dir: Path, cal: pl.DataFrame) -> pl.DataFrame:
    """The Part 1 table columns for `symbol` on the 15:59-bar basis (one row per kept session).

    `returns.daily_table` (r1, r12, r13, rod, ...) plus p_c16, p_c6, p_hh_0..11, off_close :=
    p_close, prev_off := the previous kept session's p_close and r_cc. `daily_dir` is unused by
    the table itself (the controls read it); it is kept so the call matches Part 1's.
    """
    t = returns.daily_table(data_root, symbol, cal)
    t = t.join(part1._extra_bars(data_root, cal, symbol), on="session", how="left", validate="1:1")
    t = t.rename({"prev_close": "prev_off"}).with_columns(off_close=pl.col("p_close"))
    t = t.with_columns(r_cc=pl.col("off_close") / pl.col("prev_off") - 1).sort("session")
    assert t["session"].is_unique().all() and t["off_close"].null_count() == 0
    return t


def etf_frame(
    symbol: str, table: pl.DataFrame, ctrl: pl.DataFrame, data_root: Path, cal: pl.DataFrame
) -> pl.DataFrame:
    """Part 2's long frame for `symbol` (same columns) plus a `symbol` column."""
    f = part2_features.frame(table, ctrl, data_root, cal, symbol)
    return f.with_columns(symbol=pl.lit(symbol))


def coverage(tables: dict[str, pl.DataFrame], cal: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per ETF: first and last session, kept sessions, sessions dropped (a calendar session between
    its first and last that is not in its table; `cal` is the XNYS calendar, without it the union
    of the tables' sessions stands in), share of sessions with r13 exactly 0, share with r1 null."""
    days = (
        set(cal["session"].to_list())
        if cal is not None
        else {d for t in tables.values() for d in t["session"].to_list()}
    )
    rows = []
    for symbol, t in tables.items():
        first, last = t["session"].min(), t["session"].max()
        kept = set(t["session"].to_list())
        inside = {d for d in days if first <= d <= last}
        rows.append(
            {
                "symbol": symbol, "first_session": first, "last_session": last,
                "kept_sessions": t.height, "sessions_dropped": len(inside - kept),
                "share_r13_zero": float((t["r13"] == 0).mean()),
                "share_r1_null": float(t["r1"].is_null().mean()),
            }
        )  # fmt: skip
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ A: is the effect there
def _verdict(est: float, se: float) -> tuple[float, str]:
    """t against zero and the 2-SE verdict."""
    assert se > 0, f"non-positive standard error {se}"
    t = est / se
    z = config.PART1_VERDICT_Z
    return t, CLEARLY_ABOVE if t > z else CLEARLY_BELOW if t < -z else CANNOT_TELL


def _boot_sharpe_se(r: np.ndarray) -> float:
    """Stationary-bootstrap sd of the annualised Sharpe (PART3_BOOT_REPS draws, fixed seed)."""
    bs = StationaryBootstrap(timing._block(r), r, seed=config.BOOTSTRAP_SEED)
    draws = bs.apply(timing.sharpe, config.PART3_BOOT_REPS)
    return float(np.std(draws[:, 0], ddof=1))


def _daily_mean(long: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Equal-weight mean over the ETFs present that day (a null in `cols` leaves the ETF out)."""
    d = long.drop_nulls(cols)
    return d.group_by("session").agg(*[pl.col(c).mean() for c in cols], n=pl.len()).sort("session")


def _rows_for(long: pl.DataFrame, name: str, level: str, period: str) -> list[dict]:
    """The A rows of one class (or ETF) in one period. `long`: symbol, session, r1, rod, r13."""
    lo, hi = config.PART3_PERIODS[period]
    long = long.filter(pl.col("session").is_between(lo, hi))
    n_etfs = long.drop_nulls("r13")["symbol"].n_unique()
    out: list[dict] = []

    def add(item, metric, cost, n, est, se):
        t, verdict = _verdict(est, se)
        out.append(
            {"class": name, "period": period, "item": item, "metric": metric, "cost_bp": cost,
             "n_days": n, "n_etfs": n_etfs, "est": est, "se": se, "t": t, "verdict": verdict,
             "level": level}
        )  # fmt: skip

    for sig in _SIGNALS:
        # slope: class r13 on class r1 (or rod), over the ETFs with both values that day
        d = _daily_mean(long, [sig, "r13"])
        if d.height == 0:
            continue
        params, bse, _, n = part1._ols(d["r13"].to_numpy(), d[sig].to_numpy()[:, None])
        add(f"slope_{sig}", "slope", None, n, float(params[1]), float(bse[1]))
        # rule: per ETF long if signal >= 0 else short, earns r13; class return is the mean
        pos = long.drop_nulls([sig, "r13"]).with_columns(
            g=pl.when(pl.col(sig) >= 0).then(1.0).otherwise(-1.0) * pl.col("r13")
        )
        g = _daily_mean(pos, ["g"])
        for cost in config.PART3_RULE_COSTS_BP:
            r = g["g"].to_numpy() - cost / 1e4
            add(f"rule_{sig}", "ann_mean", cost, r.size, config.TRADING_DAYS * float(r.mean()),
                config.TRADING_DAYS * part1._nw_se_mean(r))  # fmt: skip
            add(f"rule_{sig}", "sharpe", cost, r.size, timing.sharpe(r), _boot_sharpe_se(r))
    return out


def _long(tables: dict[str, pl.DataFrame], symbols) -> pl.DataFrame:
    return pl.concat(
        [
            tables[s].select("session", "r1", "rod", "r13").with_columns(symbol=pl.lit(s))
            for s in symbols
        ]
    )


def effect_rows(tables: dict[str, pl.DataFrame], spy_official_table: pl.DataFrame) -> pl.DataFrame:
    """The A rows (COLUMNS): per class and period, per ETF and period (appendix), and SPY on the
    official-close table (level "spy_official", class "us_equity", beside the US class).

    `tables` maps every symbol of config.PART3_CLASSES to its etf_table; `spy_official_table` is
    part1.part1_table. Rows with no data in a period are not written (coverage shows it); every
    class has rows in every period.
    """
    members = {s for syms in config.PART3_CLASSES.values() for s in syms}
    assert members <= set(tables), f"missing tables: {sorted(members - set(tables))}"
    rows: list[dict] = []
    for period in config.PART3_PERIODS:
        for cls, syms in config.PART3_CLASSES.items():
            new = _rows_for(_long(tables, syms), cls, "class", period)
            assert new, f"class {cls} has no data in {period}"
            rows += new
            for s in syms:
                rows += _rows_for(_long(tables, [s]), s, "etf", period)
        spy = spy_official_table.select("session", "r1", "rod", "r13").with_columns(
            symbol=pl.lit("SPY")
        )
        rows += _rows_for(spy, "us_equity", "spy_official", period)
    return pl.DataFrame(rows, schema_overrides={"cost_bp": pl.Float64}).select(COLUMNS)
