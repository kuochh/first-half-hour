"""Part 1: replicate Gao et al. (2018) and Baltussen et al. (2021) on SPY, 2011-2026.

`part1_table` builds the daily table on the official-close basis; `part1_rows` turns it into one row
per (item, window, variant, term) with our estimate, its standard error, the paper's value (with
source and market), the no-skill value and a verdict. The spec is private/specs/p1-part1.md.

Past-only cut-offs (terciles, jump percentiles) at session i read sessions before i only.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap

from first_half_hour import bars, closes, config, oos, returns, timing

log = logging.getLogger(__name__)

SYMBOL = "SPY"
N_BOUNDARIES = 12
"""Half-hour boundaries 10:00 .. 15:30 on a regular day (a half-day has 6)."""
_FIRST_BOUNDARY_MOD = 10 * 60
CHECK_VARIANTS = ("check_1559", "cost_1bp")
COLUMNS = (
    "item", "paper", "window", "variant", "term", "metric", "n", "ours", "se", "se_kind", "lo",
    "hi", "paper_value", "paper_source", "paper_market", "null_value", "null_label", "se_null",
    "d_null", "d_paper", "verdict", "mde", "small_sample", "cw_p", "note",
)  # fmt: skip
_SCHEMA: dict[str, pl.DataType] = {
    **{c: pl.Float64 for c in ("ours", "se", "lo", "hi", "paper_value", "null_value", "se_null",
                               "d_null", "d_paper", "mde", "cw_p")},
    **{c: pl.String for c in ("item", "paper", "window", "variant", "term", "metric", "se_kind",
                              "paper_source", "paper_market", "null_label", "verdict", "note")},
    "n": pl.Int64, "small_sample": pl.Boolean,
}  # fmt: skip


# ------------------------------------------------------------------ the daily table
def _extra_bars(data_root: Path, cal: pl.DataFrame) -> pl.DataFrame:
    """Per kept-or-not session: p_c16, p_c6 and the half-hour boundary prices p_hh_0 .. p_hh_11."""
    rth = bars.rth_bars(data_root, SYMBOL, cal)
    mod = pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(
        pl.Int32
    )
    close_mod = pl.col("session_close").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "session_close"
    ).dt.minute().cast(pl.Int32)
    wanted = {
        "p_c16": close_mod - 16,
        "p_c6": close_mod - 6,
        **{f"p_hh_{k}": pl.lit(_FIRST_BOUNDARY_MOD + 30 * k - 1) for k in range(N_BOUNDARIES)},
    }
    out = (
        rth.group_by("session")
        .agg(
            [
                pl.col("close").filter(mod == target).first().alias(name)
                for name, target in wanted.items()
            ]
            + [close_mod.first().alias("_close_mod")]
        )
        .collect()
    )
    # a boundary j exists only up to C-30 (15:30 on a regular day, 12:30 on a half-day)
    return out.with_columns(
        pl.when(pl.col("_close_mod") - 30 >= _FIRST_BOUNDARY_MOD + 30 * k)
        .then(pl.col(f"p_hh_{k}"))
        .otherwise(None)
        .alias(f"p_hh_{k}")
        for k in range(N_BOUNDARIES)
    ).drop("_close_mod")


def part1_table(data_root: Path, daily_dir: Path, cal: pl.DataFrame) -> pl.DataFrame:
    """SPY's daily table on the official-close basis (columns as in the spec, plus chapter 1's).

    The chapter 1 columns r1, r13, rod are kept as r1_bar, r13_bar, rod_bar (15:59 bar basis);
    r1, r13, rod, r_last15, r_last5, r_cc, r_next_{1,2,3} use the adjusted official close.
    """
    t = returns.daily_table(data_root, SYMBOL, cal).rename(
        {"r1": "r1_bar", "r13": "r13_bar", "rod": "rod_bar"}
    )
    t = t.join(_extra_bars(data_root, cal), on="session", how="left", validate="1:1")
    t = closes.official_close(
        t, data_root / config.OFFICIAL_CLOSE_FILE, daily_dir / "spy_dividends.csv"
    )
    t = t.with_columns(
        r1=pl.col("p_10") / pl.col("prev_off") - 1,
        r13=pl.col("off_close") / pl.col("p_c31") - 1,
        rod=pl.col("p_c31") / pl.col("prev_off") - 1,
        r_last15=pl.col("off_close") / pl.col("p_c16") - 1,
        r_last5=pl.col("off_close") / pl.col("p_c6") - 1,
        r_cc=pl.col("off_close") / pl.col("prev_off") - 1,
    )
    g = next_k_returns(cal, t)
    t = t.join(g, on="session", how="left", validate="1:1").sort("session")
    for col in ("r1", "r13", "rod", "r_cc", "prev_off", "off_close"):
        assert t[col].null_count() == 0, f"nulls in {col}"
    return t


def next_k_returns(cal: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """(session, r_next_1, r_next_2, r_next_3) for every session of `cal`.

    r_next_k is the product of (1 + r_cc) over the next k CALENDAR sessions, minus 1; null if one of
    them is not in `t` (dropped) or lies past the calendar's end.
    """
    g = (
        cal.sort("session")
        .select("session")
        .join(t.select("session", g=1 + pl.col("r_cc")), on="session", how="left")
    )
    nxt = [pl.col("g").shift(-i) for i in (1, 2, 3)]
    return g.select(
        "session",
        r_next_1=nxt[0] - 1,
        r_next_2=nxt[0] * nxt[1] - 1,
        r_next_3=nxt[0] * nxt[1] * nxt[2] - 1,
    )


# ------------------------------------------------------------------ building blocks
def expanding_quantiles(values: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    """(n, len(qs)) quantiles of values[:i] (strictly earlier rows); NaN for the first TERCILE_WARMUP."""
    assert np.isfinite(values).all(), "non-finite values in an expanding cut-off"
    out = np.full((values.size, len(qs)), np.nan)
    for i in range(config.TERCILE_WARMUP, values.size):
        out[i] = np.quantile(values[:i], qs)
    return out


def tercile_labels(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """0 low (<= 1/3 cut), 1 mid, 2 high (> 2/3 cut); -1 where a cut is NaN. cuts is (n, 2)."""
    lab = np.where(values <= cuts[:, 0], 0, np.where(values > cuts[:, 1], 2, 1))
    return np.where(np.isnan(cuts).any(axis=1), -1, lab)


def first_jump(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per row, the first column of x (NaN = missing) below lo or above hi; NaN if none.

    x is (n, K), lo and hi are (n,) (NaN cut-offs give no jump).
    """
    with np.errstate(invalid="ignore"):
        hit = (x < lo[:, None]) | (x > hi[:, None])
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), x[np.arange(x.shape[0]), first], np.nan)


def _window(df: pl.DataFrame, name: str) -> pl.DataFrame:
    lo, hi = config.PART1_WINDOWS[name]
    return df.filter(pl.col("session").is_between(lo, hi))


def _ols(y: np.ndarray, X: np.ndarray, lags: int | None = None):
    """OLS of y on [const, X] with NW errors; (params, bse, r2, n). Index 0 is the constant."""
    assert np.isfinite(y).all() and np.isfinite(X).all(), "non-finite regression input"
    lags = config.nw_lags(y.size) if lags is None else lags
    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return fit.params, fit.bse, float(fit.rsquared), y.size


def _boot_sd(func: Callable, *arrays: np.ndarray, block_from: np.ndarray) -> float:
    """sd (ddof 1) of func over PART1_BOOT_REPS stationary-bootstrap draws of the arrays' rows."""
    bs = StationaryBootstrap(timing._block(block_from), *arrays, seed=config.BOOTSTRAP_SEED)
    draws = bs.apply(func, config.PART1_BOOT_REPS)
    return float(np.std(draws[:, 0], ddof=1))


def verdict(
    ours: float, se: float, paper: float | None, null: float, se_null: float
) -> dict[str, float | str | None]:
    """d_null, d_paper, verdict, mde and the +/- 2 se range. See the spec's Verdict section.

    d_paper treats the paper's value as fixed (our SE only). With no paper value, d_paper and the
    verdict are None.
    """
    assert se > 0 and se_null > 0, f"non-positive standard error {se}, {se_null}"
    z = config.PART1_VERDICT_Z
    d_null = abs(ours - null) / se_null
    out: dict[str, float | str | None] = {
        "d_null": d_null, "d_paper": None, "verdict": None,
        "mde": config.PART1_MDE_Z * se_null, "lo": ours - z * se, "hi": ours + z * se,
    }  # fmt: skip
    if paper is not None:
        d_paper = abs(ours - paper) / se
        out["d_paper"] = d_paper
        if d_paper > z:
            out["verdict"] = "not replicated"
        elif d_null > z:
            out["verdict"] = "replicated"
        else:
            out["verdict"] = "cannot tell"
    return out


class _Rows:
    """Collects rows with the verdict arithmetic applied."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, item, window, variant, term, metric, n, ours, se, se_kind, **kw) -> None:
        null_value = kw.pop("null_value", 0.0)
        null_label = kw.pop("null_label", "0")
        se_null = kw.pop("se_null", None) or se
        note = kw.pop("note", None)
        cw_p = kw.pop("cw_p", None)
        assert not kw, f"unknown row fields {kw}"
        paper = config.PAPER.get((item, term, metric)) if variant != "cost_1bp" else None
        pv = paper[0] if paper else None
        v = verdict(ours, se, pv, null_value, se_null)
        if variant in CHECK_VARIANTS:  # rows beside the main row: no verdict
            v["d_paper"] = v["verdict"] = None
        self.rows.append(
            {
                "item": item, "paper": "gao" if item.startswith("gao") else "baltussen",
                "window": window, "variant": variant, "term": term, "metric": metric, "n": n,
                "ours": ours, "se": se, "se_kind": se_kind, "lo": v["lo"], "hi": v["hi"],
                "paper_value": pv, "paper_source": paper[1] if paper else None,
                "paper_market": paper[2] if paper else None, "null_value": null_value,
                "null_label": null_label, "se_null": se_null, "d_null": v["d_null"],
                "d_paper": v["d_paper"], "verdict": v["verdict"], "mde": v["mde"],
                "small_sample": n < config.PART1_SMALL_N, "cw_p": cw_p, "note": note,
            }
        )  # fmt: skip

    def slopes(
        self, item, window, variant, df, y, xs, *, lags=None, note=None, tag=None, only=None
    ) -> None:
        """One slope row per x of y ~ xs (NW errors; `only` limits it); term <x> (or <tag>)."""
        d = df.drop_nulls([y, *xs])
        params, bse, _, n = _ols(d[y].to_numpy(), d.select(xs).to_numpy(), lags)
        for i, x in enumerate(xs, start=1):
            if only and x != only:
                continue
            term = tag or x
            self.add(item, window, variant, term, "slope", n, float(params[i]), float(bse[i]), "nw",
                     note=note)  # fmt: skip


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([np.ones(y.size), X])
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))


# ------------------------------------------------------------------ the rule rows (G6, B6)
def _success_paired_se(r: np.ndarray, long_r: np.ndarray) -> float:
    """Paired stationary-bootstrap SE of (success of r) - (success of always-long)."""
    assert r.shape == long_r.shape, "paired series need the same days"

    def diff(a: np.ndarray, b: np.ndarray) -> float:
        return float((a > 0).mean() - (b > 0).mean())

    return _boot_sd(diff, r, long_r, block_from=(r > 0).astype(float) - (long_r > 0))


def _rule_rows(rows: _Rows, item: str, window: str, df: pl.DataFrame, signal: str) -> None:
    """eta(signal): long if signal > 0 else short, earns r13; main, check_1559 and cost_1bp rows."""
    long_note = f"paper's always-long success {config.PAPER_ALWAYS_LONG_SUCCESS[item]:.2%}"
    for variant, sig, ret, cost in (
        ("main", signal, "r13", 0.0),
        ("check_1559", f"{signal}_bar", "r13_bar", 0.0),
        ("cost_1bp", signal, "r13", config.PART1_COST_BP),
    ):
        d = df.drop_nulls([sig, ret])
        pos = np.where(d[sig].to_numpy() > 0, 1.0, -1.0)
        x = d[ret].to_numpy()
        r = timing.net_returns(pos, x, cost)
        n = r.size
        nw_se = float(
            sm.OLS(r, np.ones(n))
            .fit(cov_type="HAC", cov_kwds={"maxlags": config.nw_lags(n)})
            .bse[0]
        )
        td = config.TRADING_DAYS
        rows.add(item, window, variant, "rule", "ann_mean", n, td * float(r.mean()), td * nw_se,
                 "nw")  # fmt: skip
        rows.add(item, window, variant, "rule", "sharpe", n, timing.sharpe(r),
                 _boot_sd(timing.sharpe, r, block_from=r), "bootstrap")  # fmt: skip
        if variant == "cost_1bp":
            continue
        ours = float((r > 0).mean())
        null = float((x > 0).mean())
        rows.add(
            item, window, variant, "rule", "success", n, ours,
            _boot_sd(lambda a: float((a > 0).mean()), r, block_from=(r > 0).astype(float)),
            "bootstrap", null_value=null, null_label="always-long success",
            se_null=_success_paired_se(r, x), note=long_note,
        )  # fmt: skip


# ------------------------------------------------------------------ the rows
def part1_rows(table: pl.DataFrame, ctrl: pl.DataFrame) -> pl.DataFrame:
    """All Part 1 rows (columns COLUMNS). `ctrl` is controls.controls(...) (fomc and cpi flags)."""
    t = table.sort("session").join(ctrl.select("session", "fomc", "cpi"), on="session", how="left")
    rows = _Rows()
    _gao(rows, t)
    _baltussen(rows, t)
    out = pl.DataFrame(rows.rows, schema=_SCHEMA).select(COLUMNS)
    assert out.height == len(rows.rows)
    return out


def _gao(rows: _Rows, t: pl.DataFrame) -> None:
    rv = t["rv_fh"].to_numpy()
    past_lab = tercile_labels(rv, expanding_quantiles(rv, (1 / 3, 2 / 3)))
    t = t.with_columns(past_tercile=pl.Series(past_lab))
    fc = oos.recursive_forecasts(
        t.select("session", "r1", "r12", "r13"),
        start=config.PART1_OOS_START,
        end=config.PART1_WINDOWS["gao_after"][1],
    ).filter(pl.col("spec") == "r13~r1")
    for w in ("gao_overlap", "gao_after"):
        d = _window(t, w)
        rows.slopes("gao_slope", w, "main", d, "r13", ["r1"])
        rows.slopes("gao_slope", w, "check_1559", d, "r13_bar", ["r1_bar"], tag="r1")
        rows.slopes("gao_r12", w, "main", d, "r13", ["r12"])
        rows.slopes("gao_joint", w, "main", d, "r13", ["r1", "r12"])
        # G4: terciles of the first-half-hour variance
        q = np.quantile(d["rv_fh"].to_numpy(), [1 / 3, 2 / 3])
        full_lab = tercile_labels(d["rv_fh"].to_numpy(), np.tile(q, (d.height, 1)))
        for variant, lab, note in (
            ("full_window", full_lab, "terciles of the whole window (the paper's way; uses the future)"),
            ("past_only", d["past_tercile"].to_numpy(), "cut-offs from earlier sessions only"),
        ):  # fmt: skip
            for k, name in enumerate(("low", "mid", "high")):
                g = d.filter(pl.Series(lab == k))
                rows.slopes("gao_vol_tercile", w, variant, g, "r13", ["r1", "r12"],
                            tag=name, only="r1", note=note)  # fmt: skip
        _oos_row(rows, w, _window(fc, w))
        _rule_rows(rows, "gao_rule", w, d, "r1")
        # G7: news days
        for flag, label in (("fomc", "fomc"), ("cpi", "cpi")):
            for val, name in ((True, label.upper()), (False, f"non-{label.upper()}")):
                g = d.filter(pl.col(flag) == val).drop_nulls(["r13", "r1", "r12"])
                y, X = g["r13"].to_numpy(), g.select("r1", "r12").to_numpy()
                note = (
                    "FOMC = scheduled statement days; Gao's table says 'FOMC minutes release at "
                    "2:15 pm', which matches statement days before 2013"
                    if flag == "fomc"
                    else None
                )
                rows.add("gao_news", w, "main", name, "r2", y.size, _r2(y, X),
                         _boot_sd(_r2, y, X, block_from=y), "bootstrap", note=note)  # fmt: skip


def _oos_row(rows: _Rows, window: str, fc: pl.DataFrame) -> None:
    y, yh, mu = fc["y"].to_numpy(), fc["yhat"].to_numpy(), fc["mean_fc"].to_numpy()

    def r2_os(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(1 - ((a - b) ** 2).sum() / ((a - c) ** 2).sum())

    se = _boot_sd(r2_os, y, yh, mu, block_from=(y - mu) ** 2 - (y - yh) ** 2)
    _, p = oos.clark_west(y, yh, mu)
    rows.add("gao_oos_r2", window, "main", "r1", "r2_os", y.size, r2_os(y, yh, mu), se,
             "bootstrap", cw_p=p,
             note=f"recursive monthly refit vs the expanding mean; forecasts start "
                  f"{config.PART1_OOS_START} (earlier sessions are the warm-up)")  # fmt: skip


def _baltussen(rows: _Rows, t: pl.DataFrame) -> None:
    cc = t["r_cc"].to_numpy()
    cuts = expanding_quantiles(cc, config.PART1_JUMP_PCT)
    x_all = (
        t.select(pl.col(f"p_hh_{k}") / pl.col("prev_off") - 1 for k in range(N_BOUNDARIES))
        .to_numpy()
        .astype(float)
    )
    past_jump = first_jump(x_all, cuts[:, 0], cuts[:, 1])
    t = t.with_columns(past_x_jump=pl.Series(past_jump, nan_to_null=True))
    for w in ("balt_overlap", "balt_after"):
        d = _window(t, w)
        rows.slopes("balt_slope", w, "main", d, "r13", ["rod"])
        rows.slopes("balt_slope", w, "check_1559", d, "r13_bar", ["rod_bar"], tag="rod")
        rows.slopes("balt_horse", w, "main", d, "r13", ["r1", "rod"])
        # B3: first half-hour boundary beyond the 10th / 90th percentile of r_cc
        first, last = config.PART1_WINDOWS[w]
        mask = t["session"].is_between(first, last).to_numpy()
        xw = x_all[mask]
        lo, hi = np.quantile(cc[mask], config.PART1_JUMP_PCT)
        full = first_jump(xw, np.full(xw.shape[0], lo), np.full(xw.shape[0], hi))
        for variant, xj, note in (
            ("full_window", full, "percentiles of the whole window (the paper's way); first jump"),
            ("past_only", past_jump[mask], "percentiles of earlier sessions only; first jump"),
        ):  # fmt: skip
            g = d.with_columns(x_jump=pl.Series(xj, nan_to_null=True))
            rows.slopes("balt_large_move", w, variant, g, "r13", ["x_jump"], note=note)
        for k in (1, 2, 3):
            g = d.drop_nulls(["r13", f"r_next_{k}"])
            rows.slopes("balt_reversal", w, "main", g, f"r_next_{k}", ["r13"],
                        lags=max(config.nw_lags(g.height), k), tag=f"k{k}")  # fmt: skip
        for y in ("r13", "r_last15", "r_last5"):
            rows.slopes("balt_last_k", w, "main", d, y, ["rod"], tag=y)
        _rule_rows(rows, "balt_rule", w, d, "rod")


# ------------------------------------------------------------------ series for the figures
def cumulative(table: pl.DataFrame) -> pl.DataFrame:
    """(session, rule, cost_bp, wealth): wealth after each session's last-half-hour trade, from 1.

    Rules: eta_r1, eta_rod (long if the signal > 0 else short) and always_long, at 0 and 1 bp per
    side, over every session with a non-null r1, rod and r13.
    """
    d = table.sort("session").drop_nulls(["r1", "rod", "r13"])
    r13 = d["r13"].to_numpy()
    pos = {
        "eta_r1": np.where(d["r1"].to_numpy() > 0, 1.0, -1.0),
        "eta_rod": np.where(d["rod"].to_numpy() > 0, 1.0, -1.0),
        "always_long": np.ones(d.height),
    }
    parts = []
    for rule, p in pos.items():
        for cost in (0.0, config.PART1_COST_BP):
            wealth = np.cumprod(1 + timing.net_returns(p, r13, cost))
            parts.append(
                pl.DataFrame(
                    {"session": d["session"], "rule": rule, "cost_bp": cost, "wealth": wealth}
                )
            )
    return pl.concat(parts)


def rolling(table: pl.DataFrame, window: int = 756) -> pl.DataFrame:
    """(session, signal, slope, se): rolling `window`-session NW slope of r13 on r1 and on rod.

    `session` is the last session of the window; se is the NW standard error of the slope.
    """
    d = table.sort("session").drop_nulls(["r1", "rod", "r13"])
    assert d.height >= window, f"{d.height} usable rows, need {window}"
    y_all, days = d["r13"].to_numpy(), d["session"].to_list()
    lags = config.nw_lags(window)
    out = []
    for signal in ("r1", "rod"):
        x_all = d[signal].to_numpy()
        for end in range(window, d.height + 1):
            params, bse, _, _ = _ols(y_all[end - window : end], x_all[end - window : end], lags)
            out.append((days[end - 1], signal, float(params[1]), float(bse[1])))
    return pl.DataFrame(out, schema=["session", "signal", "slope", "se"], orient="row")
