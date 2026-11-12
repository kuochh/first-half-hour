"""Chapter 1 regressions: items 1, 2, 3, 4, 6 and 7 of the analysis list.

Every function takes the daily table (and for item 7 the controls) and returns a tidy polars
frame, one row per coefficient (or per Wald test), with the no-skill values beside it:
baseline_coef = 0 (slope 0) and baseline_r2 = 0. OLS rows use Newey-West (HAC, Bartlett) errors
with floor(4 (T/100)^(2/9)) lags, T being the rows of that regression; p-values are normal-based.
Every row has `t` and `se_kind` ("nw", "bootstrap" or, in the panel, "cluster_date" and
"driscoll_kraay"): a t is only comparable to another of the same kind. Huber and median rows use a
stationary-bootstrap t (resampling (r1, r13) pairs jointly); nw_lags and r2 are null there.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from scipy import stats

from first_half_hour import config

log = logging.getLogger(__name__)

SCHEMA: dict[str, pl.DataType] = {
    "item": pl.Int64, "sample": pl.String, "variant": pl.String, "term": pl.String,
    "coef": pl.Float64, "t": pl.Float64, "se_kind": pl.String, "p": pl.Float64, "r2": pl.Float64, "T": pl.Int64,
    "nw_lags": pl.Int64, "chi2": pl.Float64, "n_dropped": pl.Int64, "note": pl.String,
    "baseline_coef": pl.Float64, "baseline_r2": pl.Float64,
}  # fmt: skip

CONTINUOUS = ("rv_fh", "rv_day_lag", "relvol", "vix_lag", "vix_ratio_lag")
"""Controls z-scored within each regression sample (units only; r1's slope is unaffected)."""
SINGLE_CONTROLS: dict[str, tuple[str, ...]] = {
    "rv_fh": ("rv_fh",), "rv_day_lag": ("rv_day_lag",), "relvol": ("relvol",),
    "weekday": ("dow_tue", "dow_wed", "dow_thu", "dow_fri"), "fomc": ("fomc",), "cpi": ("cpi",),
    "payroll": ("payroll",), "vix_lag": ("vix_lag",), "vix_ratio_lag": ("vix_ratio_lag",),
    "opex_monthly": ("opex_monthly",), "opex_quarterly": ("opex_quarterly",),
    "daily_expiry_era": ("daily_expiry_era",), "month_end": ("month_end",),
    "is_half_day": ("is_half_day",),
}  # fmt: skip


QUARTERLY_NOTE = (
    "SPY's 62 ex-dividend days are exactly the 62 quarterly expiry days "
    "(3rd Friday of Mar/Jun/Sep/Dec, holiday-shifted)"
)


def _row(
    item,
    sample,
    variant,
    term,
    *,
    coef,
    t,
    p,
    r2,
    n,
    lags,
    chi2=None,
    dropped=None,
    note=None,
    se_kind="nw",
):
    return {
        "item": item, "sample": sample, "variant": variant, "term": term, "coef": coef,
        "t": t, "se_kind": se_kind, "p": p, "r2": r2, "T": n, "nw_lags": lags, "chi2": chi2,
        "n_dropped": dropped, "note": note, "baseline_coef": 0.0, "baseline_r2": 0.0,
    }  # fmt: skip


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


def _in_sample(df: pl.DataFrame, sample: str) -> pl.DataFrame:
    lo, hi = config.SAMPLES[sample]
    return df.filter(pl.col("session").is_between(lo, hi))


def _ols_rows(item, sample, variant, y, X, names, *, dropped=None, note=None, skip=("const",)):
    """OLS of y on [const, X] with HAC errors; one row per term (const omitted by default)."""
    assert y.ndim == 1 and X.shape[0] == y.size, "shape mismatch"
    assert np.isfinite(y).all() and np.isfinite(X).all(), f"{item}/{sample}/{variant}: non-finite"
    n = y.size
    lags = config.nw_lags(n)
    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    terms = ["const", *names]
    return [
        _row(item, sample, variant, term, coef=float(fit.params[i]), t=float(fit.tvalues[i]),
             p=float(fit.pvalues[i]), r2=float(fit.rsquared), n=n, lags=lags, dropped=dropped,
             note=note)
        for i, term in enumerate(terms)
        if term not in skip
    ]  # fmt: skip


def _xy(df: pl.DataFrame, y: str, xs: Sequence[str]):
    sub = df.drop_nulls([y, *xs])
    return sub, sub[y].to_numpy().astype(float), sub.select(xs).to_numpy().astype(float)


def _fit(item, sample, variant, df, y, xs, **kw):
    _, yy, XX = _xy(df, y, xs)
    return _ols_rows(item, sample, variant, yy, XX, list(xs), **kw)


# ---------------------------------------------------------------- item 1
def item1(table: pl.DataFrame) -> pl.DataFrame:
    """r13 ~ r1, r13 ~ r12, r13 ~ r1 + r12, per sample."""
    specs = {"r13~r1": ["r1"], "r13~r12": ["r12"], "r13~r1+r12": ["r1", "r12"]}
    rows = []
    for sample in config.SAMPLES:
        df = _in_sample(table, sample)
        for name, xs in specs.items():
            rows += _fit(1, sample, name, df, "r13", xs)
    return _frame(rows)


# ---------------------------------------------------------------- item 2
def _bootstrap_row(sample, variant, y, x, estimator, reps):
    """Slope of `estimator(y, x)` with a stationary-bootstrap t (resampling (r1, r13) rows jointly).

    Block length: arch's optimal_block_length on the demeaned product r13*r1 (floored at 1);
    fixed seed. t = slope / sd(bootstrap slopes); p is normal-based.
    """
    prod = (y - y.mean()) * (x - x.mean())
    block = max(1.0, float(optimal_block_length(prod).to_numpy()[0, 0]))
    bs = StationaryBootstrap(block, x, y, seed=config.BOOTSTRAP_SEED)
    draws = bs.apply(lambda xx, yy: estimator(yy, xx), reps)[:, 0]
    slope = float(estimator(y, x)[0])
    t = slope / float(draws.std(ddof=1))
    return _row(
        2, sample, variant, "r1", coef=slope, t=t, p=float(2 * stats.norm.sf(abs(t))), r2=None,
        n=y.size, lags=None, se_kind="bootstrap",
        note=f"stationary bootstrap, {reps} reps, mean block {block:.2f}, seed {config.BOOTSTRAP_SEED}",
    )  # fmt: skip


def _huber(y, x):
    return sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit().params[1:]


def _median(y, x):
    return sm.QuantReg(y, sm.add_constant(x)).fit(q=0.5).params[1:]


def item2(table: pl.DataFrame, daily_dir: Path, reps: int = config.BOOTSTRAP_REPS) -> pl.DataFrame:
    """Robust variants of r13 ~ r1 per sample (Huber, median, trimmed, log, no half-days, no ex-div).

    The trimmed rows cut the largest |r1| (thresholds from that sample's own |r1| quantile): a
    descriptive robustness row, not a forecast. Huber and median rows carry a stationary-bootstrap
    t (`reps` replications; the count is in the row's note).
    """
    path = daily_dir / "spy_dividends.csv"
    assert path.exists(), f"missing dividends file {path}"
    ex_dates = pl.read_csv(path, try_parse_dates=True)["ex_date"]
    assert ex_dates.len() > 0, "empty dividends file"
    rows = []
    for sample in config.SAMPLES:
        df = _in_sample(table, sample).drop_nulls(["r13", "r1"])
        y, x = df["r13"].to_numpy(), df["r1"].to_numpy()
        rows.append(_bootstrap_row(sample, "huber", y, x, _huber, reps))
        rows.append(_bootstrap_row(sample, "median", y, x, _median, reps))
        absr1 = df["r1"].abs()
        for share in config.TRIM_SHARES:
            cut = absr1.quantile(1 - share, interpolation="linear")
            kept = df.filter(absr1 <= cut)
            rows += _fit(2, sample, f"trim_top_{share:.0%}", kept, "r13", ["r1"],
                         note=f"|r1| <= {cut:.6f}; {df.height - kept.height} rows cut")  # fmt: skip
        logs = df.with_columns(lr13=(1 + pl.col("r13")).log(), lr1=(1 + pl.col("r1")).log())
        rows += _fit(2, sample, "log_returns", logs, "lr13", ["lr1"])
        rows += _fit(2, sample, "no_half_days", df.filter(~pl.col("is_half_day")), "r13", ["r1"])
        exdiv = df.filter(pl.col("session").is_in(ex_dates.implode()))
        rows += _fit(2, sample, "no_ex_dividend", df.filter(~pl.col("session").is_in(ex_dates.implode())),
                     "r13", ["r1"], note=f"{exdiv.height} ex-dividend sessions dropped; {QUARTERLY_NOTE}")  # fmt: skip
    return _frame(rows)


# ---------------------------------------------------------------- item 3
def rolling_slope(table: pl.DataFrame) -> pl.DataFrame:
    """Rolling 252-session OLS slope of r13 ~ r1 on FULL, step 1, NW 95% band per window.

    Windows are 252 consecutive usable rows (non-null r1 and r13). Columns: end_session, slope,
    se, lo, hi, nw_t, nw_lags, r2, plus start_session (first session of the window).
    """
    df = _in_sample(table, "FULL").drop_nulls(["r13", "r1"])
    w = config.ROLLING_WINDOW
    assert df.height >= w, f"{df.height} usable rows, need {w}"
    y_all, x_all, days = df["r13"].to_numpy(), df["r1"].to_numpy(), df["session"].to_list()
    lags = config.nw_lags(w)
    out = []
    for end in range(w, df.height + 1):
        y, x = y_all[end - w : end], x_all[end - w : end]
        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        b, se = float(fit.params[1]), float(fit.bse[1])
        out.append((days[end - w], days[end - 1], b, se, b - 1.96 * se, b + 1.96 * se, b / se, lags,
                    float(fit.rsquared)))  # fmt: skip
    return pl.DataFrame(
        out,
        schema=["start_session", "end_session", "slope", "se", "lo", "hi", "nw_t", "nw_lags", "r2"],
        orient="row",
    )


def plot_rolling_slope(series: pl.DataFrame, path: Path, parquet_path: Path | None = None) -> None:
    """Figure: slope with NW 95% band, the zero line, and the publication date marked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = series["end_session"].to_list()
    ax.fill_between(x, series["lo"].to_numpy(), series["hi"].to_numpy(), alpha=0.25,
                    label="Newey-West 95% band")  # fmt: skip
    ax.plot(x, series["slope"].to_numpy(), lw=1.2, label="rolling slope")
    ax.axhline(0.0, color="black", lw=0.8, label="no skill (slope 0)")
    brk = config.BREAK_DATES[0]
    first_in = series.filter(pl.col("end_session") >= brk)["end_session"].min()
    first_after = series.filter(pl.col("start_session") >= brk)["end_session"].min()
    assert first_in is not None, "no window reaches the break date"
    ax.axvline(brk, color="firebrick", ls="--", lw=1, label=f"{brk} (first window containing it)")
    ax.axvspan(first_in, first_after or x[-1], color="firebrick", alpha=0.08,
               label="windows straddling the date")  # fmt: skip
    if first_after is not None:
        ax.axvline(first_after, color="black", ls=":", lw=1, label="first window wholly after")
    ax.set_xlabel(f"window ending on ({config.ROLLING_WINDOW}-session windows)")
    fig.text(0.01, 0.01, "Bands are per window (95% each), not joint.", fontsize=7)
    ax.set_ylabel("slope of r13 on r1")
    ax.set_title("Rolling slope of the last half-hour return on the first-half-hour return")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        series.write_parquet(parquet_path)


# ---------------------------------------------------------------- item 4
def item4(table: pl.DataFrame) -> pl.DataFrame:
    """HAC Wald (Chow) tests of intercept and slope shifts on FULL.

    r13 ~ r1 + D + D*r1, D = session >= break date. Rows: D alone (intercept shift), D*r1 alone
    (slope shift) and the joint test of both (term "D+D*r1", chi2 and p; coef/t null).
    """
    df = _in_sample(table, "FULL").drop_nulls(["r13", "r1"])
    y, r1 = df["r13"].to_numpy(), df["r1"].to_numpy()
    n, lags = y.size, config.nw_lags(df.height)
    rows = []
    for brk in config.BREAK_DATES:
        d = (df["session"] >= brk).to_numpy().astype(float)
        assert 0 < d.sum() < n, f"break {brk} outside the sample"
        X = sm.add_constant(np.column_stack([r1, d, d * r1]))
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        variant = f"break_{brk.isoformat()}"
        for i, term in ((2, "D"), (3, "D*r1")):
            rows.append(_row(4, "FULL", variant, term, coef=float(fit.params[i]),
                             t=float(fit.tvalues[i]), p=float(fit.pvalues[i]),
                             r2=float(fit.rsquared), n=n, lags=lags,
                             chi2=float(fit.tvalues[i] ** 2)))  # fmt: skip
        wald = fit.wald_test(np.array([[0, 0, 1, 0], [0, 0, 0, 1]]), use_f=False, scalar=True)
        rows.append(_row(4, "FULL", variant, "D+D*r1", coef=None, t=None, p=float(wald.pvalue),
                         r2=float(fit.rsquared), n=n, lags=lags, chi2=float(wald.statistic)))  # fmt: skip
    return _frame(rows)


# ---------------------------------------------------------------- item 6
def item6(table: pl.DataFrame) -> pl.DataFrame:
    """r13 ~ r_on + r_fh; r13 ~ rod; r13 ~ r1 + rod; per sample."""
    specs = {"r13~r_on+r_fh": ["r_on", "r_fh"], "r13~rod": ["rod"], "r13~r1+rod": ["r1", "rod"]}
    rows = []
    for sample in config.SAMPLES:
        df = _in_sample(table, sample)
        for name, xs in specs.items():
            rows += _fit(6, sample, name, df, "r13", xs)
    return _frame(rows)


# ---------------------------------------------------------------- item 7
def _prepare(df: pl.DataFrame, cols: Sequence[str]) -> tuple[pl.DataFrame, list[str]]:
    """Cast booleans to 0/1, z-score continuous controls within `df`, drop constant columns."""
    out = df.with_columns([pl.col(c).cast(pl.Float64) for c in cols if df.schema[c] == pl.Boolean])
    keep = []
    for c in cols:
        if out[c].std() == 0:
            log.info("dropping constant column %s (n=%d)", c, out.height)
            continue
        keep.append(c)
        if c in CONTINUOUS:
            out = out.with_columns((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
    return out, keep


def _run7(sample, usable, variant, data, cols, extra=(), reasons="", extra_note=""):
    """One item-7 regression of r13 on r1 and the prepared `cols` + `extra` columns."""
    data, keep = _prepare(data, [*cols, *extra]) if (cols or extra) else (data, [])
    gone = [c for c in [*cols, *extra] if c not in keep]
    reason = reasons if usable > data.height else ""
    note = "; ".join(
        s for s in (reason, f"constant, dropped: {gone}" if gone else "", extra_note) if s
    )
    return _fit(
        7, sample, variant, data, "r13", ["r1", *keep],
        dropped=usable - data.height, note=note or None,
    )  # fmt: skip


def item7(table: pl.DataFrame, ctrl: pl.DataFrame) -> pl.DataFrame:
    """Known-by-10:00 controls added to r13 ~ r1, FULL and per sample, plus three interactions.

    Variants per sample: `base_same_rows` (r13 ~ r1 on the rows of `all`), `+<control>` (each
    control or the weekday group on its own non-null rows), `all` (every control together), and
    the interactions `r1_x_top_rv_fh`, `r1_x_top_relvol` (terciles from earlier sessions only;
    rows in the tercile warm-up are dropped from those runs) and `r1_x_news` (FOMC, CPI or
    payroll day). Continuous controls are z-scored within each regression's rows; constant
    columns (e.g. the daily-expiry era in OVERLAP) are dropped and named in `note`. n_dropped is
    the count of rows with r1 and r13 present that the variant could not use.
    """
    assert table["session"].is_unique().all(), "repeated sessions"
    df_all = table.join(ctrl, on="session", how="left", validate="1:1", suffix="_ctrl")
    df_all = df_all.drop([c for c in df_all.columns if c.endswith("_ctrl")])
    all_cols = [c for cols in SINGLE_CONTROLS.values() for c in cols]
    assert not any("div" in c for c in all_cols), "ex-dividend dummy beside the quarterly expiry"
    rows = []
    for sample in config.SAMPLES:
        df = _in_sample(df_all, sample).drop_nulls(["r13", "r1"])
        usable = df.height

        full = df.drop_nulls(all_cols)
        reason = (
            "nulls in a control (lagged variance with previous session dropped, <20 prior volumes)"
        )
        rows += _run7(sample, usable, "base_same_rows", full, [], reasons=reason)
        for name, cols in SINGLE_CONTROLS.items():
            sub = df.drop_nulls(list(cols))
            rows += _run7(
                sample, usable, f"+{name}", sub, list(cols), reasons="null in the control",
                extra_note=QUARTERLY_NOTE if name == "opex_quarterly" else "",
            )  # fmt: skip
        rows += _run7(
            sample, usable, "all", full, all_cols, reasons=reason, extra_note=QUARTERLY_NOTE
        )
        for name, flag in (("rv_fh", "rv_fh_top"), ("relvol", "relvol_top")):
            sub = df.drop_nulls([flag])
            sub = sub.with_columns(inter=pl.col("r1") * pl.col(flag))
            rows += _run7(sample, usable, f"r1_x_top_{name}", sub, [], (flag, "inter"),
                        reasons=f"first {config.TERCILE_WARMUP} sessions (or volume history) have no tercile cut-off")  # fmt: skip
        sub = df.with_columns(inter=pl.col("r1") * pl.col("news").cast(pl.Float64))
        rows += _run7(sample, usable, "r1_x_news", sub, [], ("news", "inter"))
    return _frame(rows)
