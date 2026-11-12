"""`fhh reproduce`: chapter 1, items 1-9, end to end, every sample/variant logged to MLflow.

Each item is one MLflow parent run; each row of its tidy table (a sample / variant / spec) is a
nested child run with params and metrics. Tables go to `<out>/chapter1/item<N>.parquet`, the
rolling-slope figure to `<out>/figures/`, and H1-H5 with their baselines to
`<out>/chapter1_summary.json`.

`--synthetic` runs the same code on synthetic bars in a temp dir. A full 2011-2026 synthetic set
(16 symbols x ~3,900 sessions x 390 bars) is far too slow for CI, so synthetic mode shrinks the
study window to 2018-2020 and moves the sample and break dates inside it (`_synthetic_config`).
It is the only place those config values are overridden, and they are restored afterwards.
"""

import json
import math
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from first_half_hour import (
    calendar,
    config,
    controls,
    oos,
    panel,
    regress,
    report,
    returns,
    synthetic,
    timing,
)

SYNTHETIC_BETA = 0.2
SYNTHETIC_SEED = 20260511
SYNTHETIC_REPS = 200
SYNTHETIC_OVERRIDES = {
    "DATA_START": date(2018, 1, 1),
    "DATA_END": date(2020, 12, 31),
    "SAMPLES": {
        "OVERLAP": (date(2018, 1, 2), date(2019, 6, 28)),
        "POST": (date(2019, 8, 1), date(2020, 12, 31)),
        "FULL": (date(2018, 1, 2), date(2020, 12, 31)),
    },
    "BREAK_DATES": (date(2019, 8, 1), date(2020, 3, 2), date(2020, 10, 19)),
}
"""Synthetic analogues: OVERLAP = 1.5 years (past the 252-session tercile warm-up), POST = Aug 2019 on."""


@contextmanager
def _synthetic_config():
    saved = {k: getattr(config, k) for k in SYNTHETIC_OVERRIDES}
    for k, v in SYNTHETIC_OVERRIDES.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def _write_stub_daily(daily: Path) -> None:
    """Stub daily files in the formats the real loaders read (see private/DATA.md)."""
    daily.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    days = pl.date_range(date(2017, 12, 1), date(2020, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    for name, base in (("VIXCLS", 15.0), ("VXVCLS", 18.0)):
        vals = base + np.cumsum(rng.normal(0, 0.2, days.len())) * 0.1
        pl.DataFrame(
            {config.FRED_DATE_COLUMN: days.cast(pl.String), name: vals.round(2)}
        ).write_csv(daily / f"{name}.csv")
    pl.DataFrame({"date": [date(2018, 3, 20), date(2018, 9, 25), date(2019, 6, 18)]}).write_csv(
        daily / "fomc_meetings.csv"
    )
    pl.DataFrame({"date": [date(2020, 3, 3)]}).write_csv(daily / "fomc_emergency.csv")
    pl.DataFrame({"date": [date(2020, 3, 17)], "cancelled_on": [date(2020, 3, 15)]}).write_csv(
        daily / "fomc_cancelled.csv"
    )
    pl.DataFrame(
        {
            "date": [date(2018, 2, 14), date(2018, 3, 9), date(2019, 4, 10), date(2019, 6, 7)],
            "release": ["cpi", "payroll", "cpi", "payroll"],
        }
    ).write_csv(daily / "release_dates.csv")
    pl.DataFrame(
        {"ex_date": [date(2018, 3, 16), date(2018, 6, 15), date(2019, 3, 15), date(2019, 6, 21)],
         "amount": [1.0, 1.1, 1.2, 1.3]}
    ).write_csv(daily / "spy_dividends.csv")  # fmt: skip


def _metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", name)


class _Logger:
    """Counts every MLflow run it creates so the printed count can be checked against MLflow."""

    def __init__(self, parents: list[str], children: list[str]):
        self.parents, self.children = parents, children

    def item(self, n: int, title: str, table: pl.DataFrame, rows, out: Path, params: dict) -> None:
        """One parent run for item `n`; one child per row (`rows(row) -> sample, variant, spec`)."""
        (out / "chapter1").mkdir(parents=True, exist_ok=True)
        table.write_parquet(out / "chapter1" / f"item{n}.parquet")
        with mlflow.start_run(run_name=f"item{n}: {title}") as parent:
            self.parents.append(parent.info.run_id)
            mlflow.log_params({"item": n, "rows": table.height, **params})
            for row in table.iter_rows(named=True):
                sample, variant, spec = rows(row)
                with mlflow.start_run(run_name=f"{sample}|{variant}|{spec}", nested=True) as child:
                    self.children.append(child.info.run_id)
                    texts = {  # string columns (se_kind, note, ...) are logged beside the metrics
                        k: v
                        for k, v in row.items()
                        if isinstance(v, str) and k not in ("sample", "variant", "term", "group")
                    }
                    mlflow.log_params(
                        {**texts, "item": n, "sample": sample, "variant": variant, "spec": spec}
                    )
                    metrics = {
                        _metric_name(k): float(v)
                        for k, v in row.items()
                        if isinstance(v, int | float)
                        and not isinstance(v, bool)
                        and math.isfinite(v)
                    }
                    mlflow.log_metrics(metrics)


def _one(table: pl.DataFrame, **where) -> dict:
    """The single row matching `where` (fails loud if there are zero or several)."""
    sub = table.filter(*[pl.col(k) == v for k, v in where.items()])
    assert sub.height == 1, f"expected one row for {where}, got {sub.height}"
    return sub.row(0, named=True)


def _summary(t: dict[int, pl.DataFrame]) -> dict:
    """H1-H5, each beside its baselines."""
    brk = config.BREAK_DATES[0].isoformat()
    h1 = _one(t[1], sample="OVERLAP", variant="r13~r1")
    h2 = _one(t[1], sample="POST", variant="r13~r1")

    def slope(r: dict) -> dict:
        assert r["se_kind"] == "nw", "H1/H2 t must be Newey-West"
        return {
            "slope": r["coef"], "nw_t": r["t"], "t_kind": "Newey-West (HAC, Bartlett)", "p": r["p"], "r2": r["r2"], "T": r["T"],
            "baseline_slope": 0.0, "baseline_r2": 0.0,
        }  # fmt: skip

    h3 = {
        r["spec"]: {
            "r2_os_vs_mean": r["r2_os_vs_mean"], "r2_os_vs_zero": r["r2_os_vs_zero"],
            "cw_p_vs_mean": r["cw_p_vs_mean"], "cw_p_vs_zero": r["cw_p_vs_zero"],
            "n_forecasts": r["n_forecasts"], "baseline_r2_os": 0.0,
        }
        for r in t[5].iter_rows(named=True)
    }  # fmt: skip
    h4 = _one(t[4], variant=f"break_{brk}", term="D+D*r1")
    h5 = _one(t[9], sample="POST", rule="one_signal", cost_bp=1.0)
    long_ = _one(t[9], sample="POST", rule="always_long", cost_bp=1.0)
    return {
        "H1": {"sample": "OVERLAP", **slope(h1), "paper_sign": "positive"},
        "H2": {"sample": "POST", **slope(h2)},
        "H3": {"sample": "POST", "specs": h3},
        "H4": {
            "break_date": brk, "joint_chi2": h4["chi2"], "joint_p": h4["p"],
            "baseline": "none (no-break null)",
        },
        "H5": {
            "sample": "POST", "rule": "one_signal", "cost_bp_per_side": 1.0,
            "sharpe_net": h5["sharpe"], "ci_lo": h5["sharpe_ci_lo"], "ci_hi": h5["sharpe_ci_hi"],
            "break_even_bp_per_side": h5["break_even_bp"],
            "baseline_always_long_sharpe": h5["sharpe_always_long"],
            "baseline_always_long_sharpe_check": long_["sharpe"], "baseline_zero_sharpe": 0.0,
        },
    }  # fmt: skip


def _run(data: Path, daily: Path, out: Path, synth: bool, reps: int) -> dict:
    cal = calendar.sessions(config.DATA_START, config.DATA_END)
    spy = returns.daily_table(data, "SPY", cal)
    ctrl = controls.controls(spy, cal, daily)
    fig = out / "figures"
    log = _Logger([], [])
    t: dict[int, pl.DataFrame] = {}
    reg_rows = lambda r: (r["sample"], r["variant"], r["term"])
    p0 = {"synthetic": synth}

    t[1] = regress.item1(spy)
    log.item(1, "baseline regressions", t[1], reg_rows, out, p0)
    if synth:
        r = _one(t[1], sample="OVERLAP", variant="r13~r1")
        assert r["se_kind"] == "nw"
        se = abs(r["coef"] / r["t"])
        assert abs(r["coef"] - SYNTHETIC_BETA) < 4 * se, (
            f"planted slope {SYNTHETIC_BETA} not recovered: {r['coef']:.4f} +- {se:.4f}"
        )
    t[2] = regress.item2(spy, daily, reps=reps)
    log.item(2, "robust variants", t[2], reg_rows, out, {**p0, "bootstrap_reps": reps})

    t[3] = regress.rolling_slope(spy)
    regress.plot_rolling_slope(t[3], fig / "rolling_slope.png")
    t3 = t[3].select(
        pl.lit("FULL").alias("sample"), pl.lit("rolling_252").alias("variant"),
        pl.lit("slope").alias("term"), pl.col("slope").mean().alias("mean_slope"),
        pl.col("slope").min().alias("min_slope"), pl.col("slope").max().alias("max_slope"),
        ((pl.col("lo") > 0) | (pl.col("hi") < 0)).mean().alias("share_band_excludes_0"),
        pl.len().alias("n_windows"), pl.lit(0.0).alias("baseline_slope"),
    ).head(1)  # fmt: skip
    log.item(3, "rolling slope", t3, reg_rows, out, p0)
    t[3].write_parquet(out / "chapter1" / "item3.parquet")  # the full series, not the summary

    t[4] = regress.item4(spy)
    log.item(4, "structural breaks", t[4], reg_rows, out, p0)

    fc = oos.recursive_forecasts(spy)
    t[5] = oos.oos_summary(fc)
    log.item(5, "out of sample", t[5], lambda r: ("POST", r["spec"], "oos"), out, p0)
    t[6] = regress.item6(spy)
    log.item(6, "decomposition", t[6], reg_rows, out, p0)
    t[7] = regress.item7(spy, ctrl)
    log.item(7, "controls known by 10:00", t[7], reg_rows, out, p0)

    t[8] = panel.run_panel(data, cal)
    log.item(8, "ETF panel", t[8], lambda r: (r["sample"], r["group"], r["symbol"]), out, p0)
    t[9] = timing.timing_results(spy, reps=reps)
    log.item(
        9, "timing rules", t[9],
        lambda r: (r["sample"], r["rule"], f"cost_{r['cost_bp']}bp"), out,
        {**p0, "bootstrap_reps": reps, "bootstrap_seed": config.BOOTSTRAP_SEED},
    )  # fmt: skip

    summary = {
        "synthetic": synth, "bootstrap_reps": reps,
        "bootstrap_reps_note": "item 2 Huber/median t and item 9 Sharpe intervals use this many reps",
        "samples": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in config.SAMPLES.items()},
        **_summary(t),
    }  # fmt: skip
    (out / "chapter1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"parents": log.parents, "children": log.children}


def reproduce(data: Path, daily: Path, out: Path, synth: bool = False) -> str:
    """Run chapter 1 end to end and return the count line (also asserts it against MLflow)."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A local SQLite tracking store (MLflow's file store is deprecated).
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlruns.db').resolve()}")
    mlflow.set_experiment("chapter1-synthetic" if synth else "chapter1")
    started_ms = int(time.time() * 1000)
    if synth:
        with tempfile.TemporaryDirectory() as tmp, _synthetic_config():
            root = Path(tmp)
            symbols = list(config.SYMBOLS)
            synthetic.write_synthetic(
                root / "raw", symbols, date(2018, 1, 2), date(2020, 12, 31),
                SYNTHETIC_BETA, SYNTHETIC_SEED,
            )  # fmt: skip
            _write_stub_daily(root / "daily")
            ids = _run(root / "raw", root / "daily", out, True, SYNTHETIC_REPS)
    else:
        ids = _run(data, daily, out, False, config.BOOTSTRAP_REPS)
    n_parent, n_child = len(ids["parents"]), len(ids["children"])
    assert n_parent == 9, f"expected 9 item runs, made {n_parent}"
    exp = mlflow.get_experiment_by_name("chapter1-synthetic" if synth else "chapter1")
    found = MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.start_time >= {started_ms}",
        max_results=50_000,
    )
    assert len(found) == n_parent + n_child, (
        f"MLflow holds {len(found)}, logged {n_parent + n_child}"
    )
    count_line = f"chapter 1: {n_parent} items, {n_parent + n_child} runs ({n_child} nested)"
    report.write_chapter1_report(out, count_line)
    return count_line
