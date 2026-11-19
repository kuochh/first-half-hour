"""Part 3: per-ETF walk-forward (Part 2's logit, unchanged), class pooling, the class x time grid.

Each ETF runs Part 2's `_walk` (logit, 2 bp, feature sets "all" and "vol_only") at every decision time and
gets Part 2's past-only trade flags (`_flags`). A class pools its ETFs by day: for rule r on test day t,
x_t = mean over the ETFs with a forecast that day of (net if traded else 0) and cov_t = mean of the
traded flags; net per trade = sum x_t / sum cov_t; hit rate = sum h_t / sum cov_t with h_t = mean of
(traded and net > 0); Sharpe = annualised Sharpe of x_t. The bootstrap resamples whole days, jointly for
all ETFs and rules, so differences (model - always-follow / big-move / vol-only) are paired per draw.
A grid cell (class, time) is a finding if d_af > 2 on all test days and the difference is positive in
each robustness split. With 20 cells, one or two passing the first test by chance is expected.
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from first_half_hour import config, part2

_KEY = ["symbol", "tau", "feature_set"]


def _boot_reps() -> int:
    return getattr(config, "PART3_BOOT_REPS", config.PART2_BOOT_REPS)


# ---------------------------------------------------------------- one ETF
def _etf_job(sym, frame, taus, feats, vol_feats, first, reps, opts):
    """Walk-forward of one ETF at every tau; returns forecasts, validation losses, flags, ETF rules."""
    c = config.PART3_C_BP
    qs = config.PART3_QS
    boot = part2._Boot(reps)
    fcs, vals, flags, rows = [], [], {}, []
    with threadpool_limits(limits=1):  # BLAS sum order must not depend on the worker count
        for t in taus:
            sub = frame.filter(pl.col("tau") == t)
            fa, va = part2._walk(sub, t, c, config.PART3_MODEL, feats, "all", first, opts)
            fv, vv = part2._walk(sub, t, c, config.PART3_MODEL, vol_feats, "vol_only", first, opts)
            key = (t, c, config.PART3_MODEL, "all")
            te, fl = part2._flags(fa, fv, sub, qs, min_pre=config.PART3_MIN_PRE)
            flags[t] = {"session": te["session"].to_numpy(), "g": te["g"].to_numpy(), "flags": fl}
            r = part2._rule_rows(
                fa,
                fv,
                sub,
                key,
                boot,
                qs,
                config.PART2_EVAL_COSTS,
                ("all",),
                min_pre=config.PART3_MIN_PRE,
            )
            auc = {}
            for nm, f in (("all", fa), ("vol_only", fv)):
                x = f.filter(pl.col("kind") == "test")
                auc[nm] = float(roc_auc_score(x["y"].to_numpy(), x["p"].to_numpy()))
            r = r.with_columns(
                symbol=pl.lit(sym), tau=pl.lit(t),
                auc=pl.col("rule").replace_strict(
                    {"model": auc["all"], "vol_only": auc["vol_only"], "always_follow": None,
                     "big_move": None}, return_dtype=pl.Float64),
            )  # fmt: skip
            rows.append(r)
            fcs += [fa, fv]
            vals += [va, vv]
    fc = pl.concat(fcs).with_columns(symbol=pl.lit(sym))
    val = pl.concat(vals).with_columns(symbol=pl.lit(sym))
    return fc, val, flags, pl.concat(rows)


# ---------------------------------------------------------------- class pooling
def _pool(members: list[dict], names, days: np.ndarray):
    """Day x ETF arrays of a class: g (0 where absent), present mask, flags per rule."""
    k, n = len(members), days.size
    present = np.zeros((n, k), bool)
    g = np.zeros((n, k))
    flags = {name: np.zeros((n, k)) for name in names}
    for j, m in enumerate(members):
        pos = np.searchsorted(days, m["session"])
        assert (days[pos] == m["session"]).all(), "ETF session outside the class days"
        present[pos, j] = True
        g[pos, j] = m["g"]
        for name in names:
            flags[name][pos, j] = m["flags"][name]
    assert present.any(1).all(), "a class day without any ETF forecast"
    return g, present, flags


def _sd(a: np.ndarray) -> float | None:
    """part2's bootstrap sd, or None when over 1% of the draws have no trade (a thin rule in a short
    split: the SE, and so its d, is not reported rather than failing the whole run)."""
    return part2._sd(a) if np.isnan(a).mean() < 0.01 else None


def _class_rows(members, cls, tau, boot, c_evals, splits) -> pl.DataFrame:
    """Rule rows of one (class, tau): every rule x q x c_eval x split."""
    days = np.unique(np.concatenate([m["session"] for m in members]))
    names = list(members[0]["flags"])
    assert all(list(m["flags"]) == names for m in members), "ETFs have different rule sets"
    g, present, flags = _pool(members, names, days)
    npres = present.sum(1)
    gbar = g.sum(1) / npres  # absent entries are 0 (block-length series of the bootstrap)
    masks = part2._split_masks(days)
    idx = {k: i for i, k in enumerate(names)}
    rows = []
    for ce in c_evals:
        net = g - ce
        cols, ntr = [], []
        for nm in names:
            f = flags[nm]
            x = (f * net).sum(1) / npres  # net of traded ETFs, 0 if not traded, mean over present
            h = (f * (net > 0)).sum(1) / npres
            cols.append(np.column_stack([x, x * x, f.sum(1) / npres, h]))
            ntr.append(f.sum(1))
        for split in splits:
            m = masks[split]
            if split != "all" and m.sum() < config.PART2_SPLIT_MIN_DAYS:
                continue
            point, draws, ssum = part2._boot_stats([c[m] for c in cols], boot.counts(gbar[m]))
            for i, (rule, q) in enumerate(names):
                row = dict(
                    {"class": cls, "tau": tau}, rule=rule, q=q, c_eval_bp=round(ce * 1e4),
                    split=split, n_days=int(m.sum()), n_etf_trades=int(ntr[i][m].sum()),
                    coverage=float(ssum[i] / m.sum()), mean_net=point[i, 0],
                    se=_sd(draws[:, i, 0]), hit=point[i, 1], hit_se=_sd(draws[:, i, 1]), sharpe=point[i, 2],
                    sharpe_se=_sd(draws[:, i, 2]),
                )  # fmt: skip
                for nm, ref in (("af", ("always_follow", 1.0)), ("bm", ("big_move", q)),
                                ("vol", ("vol_only", q))):  # fmt: skip
                    if ref in idx and ref != (rule, q) and rule != "always_follow":
                        r = idx[ref]
                        row[f"diff_vs_{nm}"] = point[i, 0] - point[r, 0]
                        row[f"diff_vs_{nm}_se"] = _sd(draws[:, i, 0] - draws[:, r, 0])
                    else:
                        row[f"diff_vs_{nm}"] = row[f"diff_vs_{nm}_se"] = None
                rows.append(row)
    diff_cols = [f"diff_vs_{a}{b}" for a in ("af", "bm", "vol") for b in ("", "_se")]
    return pl.DataFrame(
        rows, schema_overrides={c: pl.Float64 for c in [*diff_cols, "se", "hit_se", "sharpe_se"]}
    )


# ---------------------------------------------------------------- grid and findings
def finding(d_af: float, diffs: list) -> bool:
    """d_af > 2 on all test days AND the difference (model - always-follow) positive in every robustness
    version (None, i.e. a split too short to report, fails)."""
    return bool(d_af > config.PART3_FINDING_D and all(d is not None and d > 0 for d in diffs))


def _grid(class_rules: pl.DataFrame, aucs: pl.DataFrame) -> pl.DataFrame:
    main = (
        (pl.col("rule") == "model") & (pl.col("q") == config.PART3_MAIN_Q)
        & (pl.col("c_eval_bp") == config.PART3_C_BP)
    )  # fmt: skip
    a = class_rules.filter(main & (pl.col("split") == "all"))
    out = a.select(
        "class", "tau", top20_minus_af="diff_vs_af", se="diff_vs_af_se",
        d_af=pl.col("diff_vs_af") / pl.col("diff_vs_af_se"),
        d_bm=pl.col("diff_vs_bm") / pl.col("diff_vs_bm_se"),
        d_vol=pl.col("diff_vs_vol") / pl.col("diff_vs_vol_se"), coverage="coverage",
    )  # fmt: skip
    mean_auc = aucs.group_by(["class", "tau"]).agg(mean_auc=pl.col("auc").mean())
    out = out.join(mean_auc, on=["class", "tau"], how="left")
    for split, suf in config.PART3_ROBUST.items():
        r = class_rules.filter(main & (pl.col("split") == split)).select(
            "class", "tau", **{f"d_af_{suf}": pl.col("diff_vs_af") / pl.col("diff_vs_af_se"),
                               f"diff_{suf}": pl.col("diff_vs_af")},
        )  # fmt: skip
        out = out.join(r, on=["class", "tau"], how="left")
    diff_cols = [f"diff_{s}" for s in config.PART3_ROBUST.values()]
    out = out.with_columns(
        finding=pl.struct("d_af", *diff_cols).map_elements(
            lambda r: finding(r["d_af"], [r[c] for c in diff_cols]), return_dtype=pl.Boolean
        )
    )
    cols = ["class", "tau", "top20_minus_af", "se", "d_af", "d_bm", "d_vol", "coverage", "mean_auc",
            *[f"d_af_{s}" for s in config.PART3_ROBUST.values()], *diff_cols, "finding"]  # fmt: skip
    return out.select(cols)


# ---------------------------------------------------------------- the whole run
def _jobs() -> int:
    return int(os.environ.get("FHH_JOBS", "4"))


def run_all(frames: dict, *, groups=None, features=None, times=None, first_test_year=None,
            reps=None, n_jobs=None) -> dict:  # fmt: skip
    """Per-ETF walk-forward, class pooling, rules, the class x time grid, findings."""
    classes = config.PART3_CLASSES
    periods = config.PART3_PERIODS  # noqa: F841  (read so a missing config fails loud)
    groups, feats, taus = part2._settings(groups, features, times)
    first = first_test_year or part2._first_test_year()
    reps = reps or _boot_reps()
    vol_feats = list(groups["volatility"])
    symbols = [s for syms in classes.values() for s in syms]
    assert len(set(symbols)) == len(symbols), "an ETF is in two classes"
    assert set(symbols) <= set(frames), f"frames missing for {set(symbols) - set(frames)}"
    cls_of = {s: c for c, syms in classes.items() for s in syms}
    opts = part2.options(config.PART3_MODEL)  # resolved here: workers would miss run-time overrides
    out = Parallel(n_jobs=n_jobs or _jobs())(
        delayed(_etf_job)(s, frames[s], taus, feats, vol_feats, first, reps, opts) for s in symbols
    )
    res = dict(zip(symbols, out, strict=True))

    forecasts = pl.concat([res[s][0] for s in symbols]).with_columns(
        **{"class": pl.col("symbol").replace_strict(cls_of, return_dtype=pl.Utf8)}
    )
    val = pl.concat([res[s][1] for s in symbols])
    etf_rules = pl.concat(
        [res[s][3].with_columns(**{"class": pl.lit(cls_of[s])}) for s in symbols]
    ).select(
        "symbol", "class", "tau", "rule", "q", "c_eval_bp", "n_trades", "coverage", "mean_net",
        "se", "auc", "diff_vs_af", "diff_vs_af_se",
    )  # fmt: skip
    aucs = (
        etf_rules.filter((pl.col("rule") == "model") & (pl.col("q") == config.PART3_MAIN_Q)
                         & (pl.col("c_eval_bp") == config.PART3_C_BP))
        .select("symbol", "class", "tau", "auc")
    )  # fmt: skip

    boot = part2._Boot(reps)
    crules = []
    class_days_missing: dict[str, int] = {}
    for cls, syms in classes.items():
        for t in taus:
            members = [res[s][2][t] for s in syms]
            days = np.unique(np.concatenate([m["session"] for m in members]))
            # days other ETFs have but no ETF of this class has (dropped sessions): counted, not
            # filled; the class series simply has no row that day
            others = np.unique(np.concatenate([res[s][2][t]["session"] for s in symbols]))
            span = others[(others >= days.min()) & (others <= days.max())]
            class_days_missing[f"{cls} {t}"] = int((~np.isin(span, days)).sum())
            crules.append(
                _class_rows(members, cls, t, boot, config.PART2_EVAL_COSTS, part2.SPLIT_NAMES)
            )
    rules = pl.concat(crules)
    grid = _grid(rules, aucs)
    assert grid.height == len(classes) * len(taus), "grid lost cells"

    cell_years = val.select([*_KEY, "year"]).unique().height
    n_cells = val.select(_KEY).unique().height
    found = grid.filter(pl.col("finding"))
    summary = {
        "n_etfs": len(symbols), "n_cells": int(grid.height),
        "n_pass_full": int((grid["d_af"] > config.PART3_FINDING_D).sum()),
        "n_findings": int(found.height), "class_days_missing": class_days_missing,
        "findings": found.select("class", "tau", "top20_minus_af", "d_af", "diff_pre", "diff_post",
                                 "diff_nocovid", "d_af_pre", "d_af_post", "d_af_nocovid").to_dicts(),
        "n_fits": int(val.height + cell_years), "n_runs": int(n_cells + cell_years),
        "n_parent": int(n_cells), "n_child": int(cell_years), "boot_reps": reps,
        "first_test_year": first, "max_test_auc": float(etf_rules["auc"].max()),
    }  # fmt: skip
    return {"forecasts": forecasts, "rules": rules, "grid": grid, "etf_rules": etf_rules,
            "val": val, "summary": summary}  # fmt: skip


# ---------------------------------------------------------------- logging and outputs
def log_runs(results: dict) -> tuple[int, int]:
    """One MLflow parent per (symbol, tau, feature_set), one nested run per test year."""
    assert mlflow.get_tracking_uri().startswith("sqlite:"), "set the MLflow store first"
    test = results["forecasts"].filter(pl.col("kind") == "test")
    n_parent = n_child = 0
    for key, v in results["val"].group_by(_KEY, maintain_order=True):
        t = test.filter(
            (pl.col("symbol") == key[0]) & (pl.col("tau") == key[1]) & (pl.col("feature_set") == key[2])
        )  # fmt: skip
        auc = float(roc_auc_score(t["y"].to_numpy(), t["p"].to_numpy()))
        with mlflow.start_run(run_name="|".join(map(str, key))):
            n_parent += 1
            mlflow.log_params(dict(zip(_KEY, key, strict=True))
                              | {"model": config.PART3_MODEL, "c_bp": config.PART3_C_BP})  # fmt: skip
            mlflow.log_metrics({"test_auc": auc, "n_test": t.height})
            for (yr,), vy in v.group_by("year", maintain_order=True):
                ty = t.filter(pl.col("year") == yr)
                with mlflow.start_run(run_name=f"{key[0]}|{key[1]}|{key[2]}|{yr}", nested=True):
                    n_child += 1
                    mlflow.log_params({"test_year": yr, "option": ty["option"][0]})
                    mlflow.log_metrics(
                        {f"val_logloss_{o}": ll for o, ll in zip(vy["option"], vy["val_logloss"], strict=True)}
                        | {"test_logloss": float(log_loss(ty["y"].to_numpy(), ty["p"].to_numpy(), labels=[0, 1]))}
                    )  # fmt: skip
    s = results["summary"]
    assert (n_parent, n_child) == (s["n_parent"], s["n_child"]), "logged runs differ from the count"
    return n_parent, n_child


def write_outputs(results: dict, out_root) -> None:
    out = Path(out_root)
    (out / "part3").mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if name != "summary":
            df.write_parquet(out / "part3" / f"{name}.parquet")
    (out / "part3_summary.json").write_text(json.dumps(results["summary"], indent=2, default=float))
