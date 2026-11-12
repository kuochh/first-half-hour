"""Out-of-sample forecasts of r13 (item 5): recursive monthly-refit OLS, R2_OS and Clark-West.

Each month's forecasts come from an OLS fitted on every kept session from the start of the table
up to the end of the PREVIOUS month, so nothing from the forecast month or later enters.
"""

import logging
import math
from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm

from first_half_hour import config

log = logging.getLogger(__name__)

SPECS: dict[str, tuple[str, ...]] = {
    "r13~r1": ("r1",),
    "r13~r12": ("r12",),
    "r13~r1+r12": ("r1", "r12"),
}


def recursive_forecasts(
    table: pl.DataFrame, start: date | None = None, end: date | None = None
) -> pl.DataFrame:
    """One row per (spec, forecast session): y, model forecast, expanding-mean forecast.

    Rows with a null r1, r12 or r13 are dropped first, so every spec and both benchmarks use the
    same sessions. The forecast window defaults to POST. For a session in month M the model and the
    mean are estimated on sessions strictly before the first day of M.
    """
    start = start or config.SAMPLES["POST"][0]
    end = end or config.SAMPLES["POST"][1]
    d = table.select("session", "r1", "r12", "r13").drop_nulls().sort("session")
    n_dropped_null = table.height - d.height
    log.info("oos: %d rows dropped for a null r1/r12/r13", n_dropped_null)
    assert d["session"].is_unique().all(), "repeated sessions"
    month = d["session"].dt.truncate("1mo")
    sess = d["session"].to_numpy()
    y = d["r13"].to_numpy()
    out = []
    for m in month.filter((d["session"] >= start) & (d["session"] <= end)).unique().sort():
        first = np.datetime64(m)
        train = sess < first
        test = (month == m).to_numpy() & (d["session"] >= start).to_numpy()
        test &= (d["session"] <= end).to_numpy()
        assert train.sum() >= 250, f"fewer than 250 training sessions before {m}"
        assert sess[train].max() < sess[test].min(), "training must end before the forecast month"
        mean_fc = float(y[train].mean())
        for spec, cols in SPECS.items():
            x = d.select(cols).to_numpy()
            fit = sm.OLS(y[train], sm.add_constant(x[train])).fit()
            yhat = sm.add_constant(x[test], has_constant="add") @ fit.params
            out.append(
                pl.DataFrame(
                    {
                        "spec": spec,
                        "session": sess[test],
                        "month": m,
                        "y": y[test],
                        "yhat": yhat,
                        "mean_fc": mean_fc,
                        "n_train": int(train.sum()),
                        "n_dropped_null": n_dropped_null,
                    }
                )
            )
    return pl.concat(out).with_columns(pl.col("session").cast(pl.Date))


def clark_west(
    y: np.ndarray, yhat: np.ndarray, bench: np.ndarray, lags: int | None = None
) -> tuple[float, float]:
    """Clark-West (2007) statistic and ONE-SIDED p (H1: the model beats the benchmark).

    f_t = (y - bench)^2 - [(y - yhat)^2 - (bench - yhat)^2]; regress f on a constant with
    Newey-West errors; t = mean(f)/se; p = 1 - Phi(t) (standard normal).
    """
    f = (y - bench) ** 2 - ((y - yhat) ** 2 - (bench - yhat) ** 2)
    lags = config.nw_lags(len(f)) if lags is None else lags
    fit = sm.OLS(f, np.ones_like(f)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues[0])
    return t, 0.5 * math.erfc(t / math.sqrt(2))


def oos_summary(fc: pl.DataFrame) -> pl.DataFrame:
    """Per spec: R2_OS vs the expanding mean and vs zero, Clark-West stats and one-sided p.

    No-skill value of every R2_OS is 0 (columns `baseline_r2`).
    """
    rows = []
    for (spec,), g in fc.group_by("spec", maintain_order=True):
        y, yh, mu = g["y"].to_numpy(), g["yhat"].to_numpy(), g["mean_fc"].to_numpy()
        sse = ((y - yh) ** 2).sum()
        t_mean, p_mean = clark_west(y, yh, mu)
        t_zero, p_zero = clark_west(y, yh, np.zeros_like(y))
        rows.append(
            {
                "spec": spec,
                "n_forecasts": g.height,
                "n_dropped_null": int(g["n_dropped_null"][0]),
                "first_session": g["session"].min(),
                "last_session": g["session"].max(),
                "r2_os_vs_mean": float(1 - sse / ((y - mu) ** 2).sum()),
                "r2_os_vs_zero": float(1 - sse / (y**2).sum()),
                "baseline_r2": 0.0,
                "cw_t_vs_mean": t_mean,
                "cw_p_vs_mean": p_mean,
                "cw_t_vs_zero": t_zero,
                "cw_p_vs_zero": p_zero,
                "nw_lags": config.nw_lags(g.height),
            }
        )
    return pl.DataFrame(rows)
