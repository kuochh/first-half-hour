"""The ETF panel: r13 ~ r1 per ETF (Newey-West) with Holm correction, and pooled with date clusters.

Holm is applied within each group (equity, non-equity) and sample. Baselines are slope 0 and R2 0.
"""

from pathlib import Path

import polars as pl
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from first_half_hour import config, returns

POOLED = "POOLED"
GROUPS: dict[str, tuple[str, ...]] = {
    "equity": config.EQUITY_ETFS,
    "non_equity": config.NON_EQUITY_ETFS,
}
SCHEMA = [
    "group", "sample", "symbol", "slope", "t", "p", "p_holm", "r2", "T",
    "nw_lags", "n_clusters", "sessions_kept", "sessions_dropped",
    "baseline_slope", "baseline_r2",
]  # fmt: skip


def _in_sample(table: pl.DataFrame, sample: str) -> pl.DataFrame:
    start, end = config.SAMPLES[sample]
    return table.filter(pl.col("session").is_between(start, end)).drop_nulls(["r1", "r13"])


def _ols(d: pl.DataFrame, **cov):
    x = sm.add_constant(d["r1"].to_numpy())
    return sm.OLS(d["r13"].to_numpy(), x).fit(**cov)


def _row(group, sample, symbol, fit, n, **extra) -> dict:
    return {
        "group": group, "sample": sample, "symbol": symbol,
        "slope": float(fit.params[1]), "t": float(fit.tvalues[1]), "p": float(fit.pvalues[1]),
        "p_holm": None, "r2": float(fit.rsquared), "T": n, "nw_lags": None, "n_clusters": None,
        "sessions_kept": None, "sessions_dropped": None,
        "baseline_slope": 0.0, "baseline_r2": 0.0,
    } | extra  # fmt: skip


def panel_results(
    tables: dict[str, pl.DataFrame], n_calendar: int, groups: dict[str, tuple[str, ...]] = GROUPS
) -> pl.DataFrame:
    """Per-ETF and pooled rows for every group and sample (tidy, columns in SCHEMA).

    `tables` maps symbol -> daily_table; `n_calendar` is the XNYS session count of the window,
    so kept/dropped per symbol is carried in the rows. Sessions with a null r1 are excluded.
    """
    rows: list[dict] = []
    for group, symbols in groups.items():
        for sample in config.SAMPLES:
            part: list[dict] = []
            frames = []
            for sym in symbols:
                d = _in_sample(tables[sym], sample)
                assert d.height > 10, f"{sym} {sample}: only {d.height} usable sessions"
                lags = config.nw_lags(d.height)
                fit = _ols(d, cov_type="HAC", cov_kwds={"maxlags": lags})
                kept = tables[sym].height
                part.append(
                    _row(group, sample, sym, fit, d.height, nw_lags=lags,
                         sessions_kept=kept, sessions_dropped=n_calendar - kept)
                )  # fmt: skip
                frames.append(d.select("session", "r1", "r13"))
            holm = multipletests([r["p"] for r in part], method="holm")[1]
            for r, ph in zip(part, holm, strict=True):
                r["p_holm"] = float(ph)
            rows += part
            pooled = pl.concat(frames)
            codes = pooled["session"].rank("dense").to_numpy().astype(int)
            fit = _ols(pooled, cov_type="cluster", cov_kwds={"groups": codes})
            rows.append(
                _row(group, sample, POOLED, fit, pooled.height, n_clusters=int(codes.max()))
            )
    return pl.DataFrame(rows, schema_overrides={"p_holm": pl.Float64}).select(SCHEMA)


def run_panel(root: Path, cal: pl.DataFrame) -> pl.DataFrame:
    """daily_table for all 15 ETFs, then panel_results."""
    symbols = [s for g in GROUPS.values() for s in g]
    assert len(set(symbols)) == 15, "panel groups must hold 15 distinct ETFs"
    tables = {s: returns.daily_table(root, s, cal) for s in symbols}
    return panel_results(tables, cal.height)
