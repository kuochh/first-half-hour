"""Part 2: per-day classifier walk-forward, top-q rules, baselines, paired bootstrap, verdict.

Input `frame`: the long frame of `part2_features.frame` (session, year, tau, s, d, g, r13, y_0, y_2,
y_4 and the features). A "cell" is (tau, c_bp, model, feature_set). For test year Y a cell fits each
option on years < Y-1, scores log loss on year Y-1, picks the best (ties: first), refits the pick on
years < Y and predicts Y; the picked option's year Y-1 forecasts are kept as "pre" forecasts.

Rule cut-offs use only earlier days. The top-q pool is the cell's "test" forecasts before the day plus
the "pre" forecasts of the year before the first test year (later "pre" rows re-forecast days that
already have a "test" forecast, so they are not pooled). Big-move uses abs(s) of every earlier session.
Bootstrap: stationary (arch's indices), whole days; the traded flag of a day is fixed by its past-only
cut-off and the per-trade statistics are recomputed in every draw (so the traded count varies by draw).
"""

import json
import os
from bisect import insort
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from arch.bootstrap import StationaryBootstrap
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from first_half_hour import config, timing

MODELS = ("logit", "hgb")
GROUP_ORDER = ("volatility", "signal", "volume", "calendar", "path")
SPLIT_NAMES = ("all", "pre_2020_05", "post_2020_05", "no_covid")
_KEY = ["tau", "c_bp", "model", "feature_set"]


# ---------------------------------------------------------------- settings read at call time
def _settings(groups=None, features=None, times=None):
    groups = groups if groups is not None else config.PART2_GROUPS
    features = list(features if features is not None else config.PART2_FEATURES)
    times = times if times is not None else config.PART2_TIMES
    taus = [t[0] if isinstance(t, tuple) else t for t in times]
    assert set(groups["volatility"]) <= set(features), "volatility group not in the features"
    return groups, features, taus


def _first_test_year() -> int:
    return getattr(config, "PART2_TEST_START", date(2014, 1, 2)).year


def _cost_bps() -> list[int]:
    costs = getattr(config, "PART2_COSTS_RT", (0.0, 0.0002, 0.0004))
    return [round(c * 1e4) for c in costs]


# ---------------------------------------------------------------- models
def options(model: str) -> list[tuple[str, dict]]:
    """(label, params) of the settings options, in tie-break order."""
    if model == "logit":
        return [(f"C{c}", {"C": c}) for c in config.PART2_LOGIT_C]
    assert model == "hgb", model
    return [(f"d{d}_i{i}", {"max_depth": d, "max_iter": i}) for d, i in config.PART2_HGB_GRID]


def _fit(model: str, params: dict, x: np.ndarray, y: np.ndarray):
    if model == "logit":
        est = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=params["C"], solver="lbfgs", max_iter=config.PART2_MAX_ITER
                    ),
                ),
            ]
        ).fit(x, y)
        n_iter = int(est.named_steps["clf"].n_iter_[0])
        assert n_iter < config.PART2_MAX_ITER, (
            f"logit {params}: {n_iter} iterations, no convergence"
        )
        return est
    # one thread per fit: OpenMP summation order would otherwise change the last digits with the
    # worker count; the parallelism is over cells (joblib), so results do not depend on n_jobs
    with threadpool_limits(limits=1):
        return HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            max_iter=params["max_iter"],
            learning_rate=config.PART2_HGB_LR,
            min_samples_leaf=config.PART2_HGB_MIN_LEAF,
            early_stopping=False,
            random_state=config.PART2_SEED,
        ).fit(x, y)


def _walk(sub: pl.DataFrame, tau, c_bp, model, feats, feature_set, first_year, opts=None):
    """One cell. `sub` is that tau's rows. Returns (forecasts, validation log losses)."""
    ycol = f"y_{c_bp}"
    sub = sub.sort("session")
    assert sub["session"].is_unique().all(), "sessions repeat within a tau"
    x = sub.select(pl.col(feats).cast(pl.Float64)).to_numpy()
    y = sub[ycol].to_numpy().astype(int)
    years = sub["year"].to_numpy()
    # `opts` is resolved in the parent: joblib workers re-import config and would miss run-time overrides
    opts = opts or options(model)
    fcs, vals = [], []
    for yr in range(first_year, int(years.max()) + 1):
        tr, va, te = years < yr - 1, years == yr - 1, years == yr
        assert tr.any() and va.any() and te.any(), (
            f"{tau} {c_bp} {model}: no rows for test year {yr}"
        )
        assert years[tr].max() < yr - 1 < yr, "training must end before the validation year"
        assert years[va].max() < years[te].min(), "validation year must precede the test year"
        fits = [_fit(model, prm, x[tr], y[tr]) for _, prm in opts]
        pv = [f.predict_proba(x[va])[:, 1] for f in fits]
        losses = [float(log_loss(y[va], p, labels=[0, 1])) for p in pv]
        k = int(np.argmin(losses))
        for (lab, _), ls in zip(opts, losses, strict=True):
            vals.append((tau, c_bp, model, feature_set, yr, lab, ls, lab == opts[k][0]))
        fin = _fit(model, opts[k][1], x[years < yr], y[years < yr])
        for kind, msk, p in (("pre", va, pv[k]), ("test", te, fin.predict_proba(x[te])[:, 1])):
            fcs.append(
                sub.filter(pl.Series(msk)).select("session", "year", "g", "d", "s")
                .with_columns(
                    tau=pl.lit(tau), c_bp=pl.lit(c_bp, pl.Int64), model=pl.lit(model),
                    feature_set=pl.lit(feature_set), p=pl.Series(p),
                    y=pl.Series(y[msk]), option=pl.lit(opts[k][0]), kind=pl.lit(kind),
                )
            )  # fmt: skip
    fc = pl.concat(fcs).select(
        "tau", "c_bp", "model", "feature_set", "session", "year", "p", "y", "g", "d", "s",
        "option", "kind",
    )  # fmt: skip
    t = fc.filter(pl.col("kind") == "test")
    auc = float(roc_auc_score(t["y"].to_numpy(), t["p"].to_numpy()))
    assert auc < config.PART2_AUC_ALARM, (
        f"leak alarm: {tau} c={c_bp} {model} {feature_set} test AUC {auc:.3f}"
    )
    val = pl.DataFrame(
        vals,
        schema=["tau", "c_bp", "model", "feature_set", "year", "option", "val_logloss", "picked"],
        orient="row",
    )
    return fc, val


def walk_forward(frame, tau, c_bp, model, features, feature_set, first_test_year=None):
    """Forecasts frame of one cell (kind "pre" / "test"); asserts the test AUC is below the alarm."""
    first = first_test_year or _first_test_year()
    return _walk(frame.filter(pl.col("tau") == tau), tau, c_bp, model, list(features),
                 feature_set, first)[0]  # fmt: skip


# ---------------------------------------------------------------- cut-offs
def past_cuts(base: np.ndarray, seq: np.ndarray, qs) -> np.ndarray:
    """Row i: the (1 - q) quantile (linear) of base + seq[:i], for each q. Past-only by construction."""
    assert base.size > 0, "empty cut-off pool"
    pool = sorted(base.tolist())
    out = np.empty((seq.size, len(qs)))
    for i, v in enumerate(seq.tolist()):
        m = len(pool)
        for j, q in enumerate(qs):
            pos = (m - 1) * (1 - q)
            lo = int(pos)
            hi = min(lo + 1, m - 1)
            out[i, j] = pool[lo] + (pos - lo) * (pool[hi] - pool[lo])
        insort(pool, v)
    return out


# ---------------------------------------------------------------- bootstrap
class _Boot:
    """Stationary-bootstrap draw counts (reps x n), cached by (n, block); sums are C @ X."""

    def __init__(self, reps: int):
        self.reps, self.cache = reps, {}

    def counts(self, g: np.ndarray) -> np.ndarray:
        n, block = g.size, round(timing._block(g), 3)
        if (n, block) not in self.cache:
            if len(self.cache) > 6:
                self.cache.clear()
            bs = StationaryBootstrap(block, np.arange(n), seed=config.BOOTSTRAP_SEED)
            c = np.zeros((self.reps, n), dtype=np.float32)
            for i, (pos, _) in enumerate(bs.bootstrap(self.reps)):
                c[i] = np.bincount(pos[0], minlength=n)
            self.cache[(n, block)] = c
        return self.cache[(n, block)]


def _cols(flag: np.ndarray, net: np.ndarray) -> np.ndarray:
    x = flag * net
    return np.column_stack([x, x * x, flag, flag * (net > 0)])


def _stats(s: np.ndarray, n: int) -> np.ndarray:
    """Sums (..., rules, 4) -> (..., rules, 3): mean net per trade, hit rate, annualised Sharpe."""
    sx, sxx, nf, nh = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sx / nf
        hit = nh / nf
        mx = sx / n
        sharpe = mx / np.sqrt((sxx - n * mx**2) / (n - 1)) * np.sqrt(252)
    return np.stack([mean, hit, sharpe], axis=-1)


def _boot_stats(cols: list[np.ndarray], boot_c: np.ndarray):
    """(point (rules,3), draws (reps, rules, 3), trades (rules,)) for the rules' column blocks."""
    n, k = cols[0].shape[0], len(cols)
    x = np.hstack(cols)
    point = _stats(x.sum(0).reshape(k, 4), n)
    draws = _stats((boot_c @ x).reshape(-1, k, 4), n)
    return point, draws, x.sum(0).reshape(k, 4)[:, 2]


def _sd(a: np.ndarray) -> float:
    assert np.isnan(a).mean() < 0.01, "bootstrap draws with no trades"
    return float(np.nanstd(a, ddof=1))


# ---------------------------------------------------------------- rules
def _split_masks(sessions: np.ndarray) -> dict[str, np.ndarray]:
    d = sessions
    out = {"all": np.ones(d.size, bool)}
    for name, (lo, hi) in config.PART2_SPLITS.items():
        out[name] = (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))
    lo, hi = config.PART2_COVID_OUT
    out["no_covid"] = ~((d >= np.datetime64(lo)) & (d <= np.datetime64(hi)))
    return out


def _flags(fc: pl.DataFrame, fcv: pl.DataFrame | None, tf: pl.DataFrame, qs):
    """Test sessions, g, and the past-only trade flags {(rule, q): bool array} of one cell."""
    te = fc.filter(pl.col("kind") == "test").sort("session")
    first = int(te["year"].min())
    pre = fc.filter((pl.col("kind") == "pre") & (pl.col("year") < first))
    assert pre.height > 0, "no pre forecasts to seed the cut-off"
    assert pre["session"].max() < te["session"].min(), "cut-off pool must precede the test days"
    cut = past_cuts(pre["p"].to_numpy(), te["p"].to_numpy(), qs)
    flags = {("always_follow", 1.0): np.ones(te.height, bool)}
    for j, q in enumerate(qs):
        flags[("model", q)] = te["p"].to_numpy() >= cut[:, j]
    tf = tf.sort("session")
    base = tf.filter(pl.col("session") < te["session"].min())["s"].abs().to_numpy()
    seq = te["s"].abs().to_numpy()
    assert (tf.filter(pl.col("session") >= te["session"].min())["session"].to_numpy()
            == te["session"].to_numpy()).all(), "test sessions differ from the tau's sessions"  # fmt: skip
    cb = past_cuts(base, seq, qs)
    for j, q in enumerate(qs):
        flags[("big_move", q)] = seq >= cb[:, j]
    if fcv is not None:
        tv = fcv.filter(pl.col("kind") == "test").sort("session")
        assert (tv["session"].to_numpy() == te["session"].to_numpy()).all(), "vol-only days differ"
        pv = fcv.filter((pl.col("kind") == "pre") & (pl.col("year") < first))
        cv = past_cuts(pv["p"].to_numpy(), tv["p"].to_numpy(), qs)
        for j, q in enumerate(qs):
            flags[("vol_only", q)] = tv["p"].to_numpy() >= cv[:, j]
    return te, flags


def _rule_rows(fc, fcv, tf, key, boot, qs, c_evals, splits) -> pl.DataFrame:
    te, flags = _flags(fc, fcv, tf, qs)
    sess, g = te["session"].to_numpy(), te["g"].to_numpy()
    masks = _split_masks(sess)
    names = list(flags)
    rows = []
    for split in splits:
        m = masks[split]
        if split != "all" and m.sum() < config.PART2_SPLIT_MIN_DAYS:
            continue
        gm = g[m]
        c_boot = boot.counts(gm)
        for ce in c_evals:
            net = gm - ce
            point, draws, ntr = _boot_stats(
                [_cols(flags[k][m].astype(float), net) for k in names], c_boot
            )
            idx = {k: i for i, k in enumerate(names)}
            for i, (rule, q) in enumerate(names):
                row = dict(zip(["tau", "c_bp", "model", "feature_set"], key, strict=True))
                row.update(
                    feature_set={"vol_only": "vol_only", "model": row["feature_set"]}.get(rule, "none"),
                    rule=rule, q=q, c_eval_bp=round(ce * 1e4), split=split, n_trades=int(ntr[i]),
                    mean_net=point[i, 0], se=_sd(draws[:, i, 0]), hit=point[i, 1],
                    hit_se=_sd(draws[:, i, 1]), sharpe=point[i, 2], sharpe_se=_sd(draws[:, i, 2]),
                    coverage=ntr[i] / m.sum(),
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
    return pl.DataFrame(rows, schema_overrides={c: pl.Float64 for c in diff_cols})


def _quintiles(fc, boot, key) -> pl.DataFrame:
    te = fc.filter(pl.col("kind") == "test").sort("session")
    g, p = te["g"].to_numpy(), te["p"].to_numpy()
    n = g.size
    grp = (np.argsort(np.argsort(p)) * 5) // n
    net = g - config.PART2_EVAL_COSTS[1]
    cols = [_cols((grp == k).astype(float), net) for k in range(5)]
    cols.append(_cols(np.ones(n), net))
    point, draws, ntr = _boot_stats(cols, boot.counts(g))
    tb = draws[:, 4, 0] - draws[:, 0, 0]
    rows = []
    for k in range(5):
        rows.append(
            dict(zip(["tau", "c_bp", "model", "feature_set"], key, strict=True),
                 group=k + 1, n=int(ntr[k]), mean_net=point[k, 0], se=_sd(draws[:, k, 0]),
                 hit=point[k, 1], hit_se=_sd(draws[:, k, 1]),
                 af_mean_net=point[5, 0], af_se=_sd(draws[:, 5, 0]),
                 top_minus_bottom=point[4, 0] - point[0, 0], top_minus_bottom_se=_sd(tb))
        )  # fmt: skip
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- verdict
def verdict(d_af: float, d_bm: float, d_0: float, tb: float, tb_se: float) -> str:
    """amplified / partly amplified / not amplified (see the spec's Verdict section)."""
    if d_af > 2 and d_bm > 2 and d_0 > 2:
        return "amplified"
    if abs(tb) <= 2 * tb_se and d_af <= 2:
        return "not amplified"
    return "partly amplified" if d_af > 2 else "not amplified"


def _d(rules: pl.DataFrame, **flt) -> dict:
    """The main-q, 2 bp, all-days model row of a cell as (d_af, d_bm, d_0, diff_af, se_af)."""
    r = rules.filter(
        (pl.col("rule") == "model") & (pl.col("q") == config.PART2_MAIN_Q)
        & (pl.col("c_eval_bp") == config.PART2_MAIN_C_BP) & (pl.col("split") == "all"),
        *[pl.col(k) == v for k, v in flt.items()],
    )  # fmt: skip
    assert r.height == 1, f"expected one headline row, found {r.height}"
    r = r.row(0, named=True)
    return {
        "d_af": r["diff_vs_af"] / r["diff_vs_af_se"], "d_bm": r["diff_vs_bm"] / r["diff_vs_bm_se"],
        "d_0": r["mean_net"] / r["se"], "diff_af": r["diff_vs_af"], "se_af": r["diff_vs_af_se"],
    }  # fmt: skip


# ---------------------------------------------------------------- the whole run
def _jobs() -> int:
    return int(os.environ.get("FHH_JOBS", "4"))


def run_all(frame, *, groups=None, features=None, times=None, first_test_year=None, reps=None,
            n_jobs=None) -> dict:  # fmt: skip
    """Every cell, rules, quintiles, grid, input-group steps, picked days, verdict."""
    groups, feats, taus = _settings(groups, features, times)
    first = first_test_year or _first_test_year()
    reps = reps or config.PART2_BOOT_REPS
    bps = _cost_bps()
    vol_feats = list(groups["volatility"])
    qs = config.PART2_QS
    head_c, head_tau = config.PART2_MAIN_C_BP, config.PART2_HEADLINE_TAU
    assert head_tau in taus and head_c in bps, "the headline cell is not in the run"
    by_tau = {t: frame.filter(pl.col("tau") == t) for t in taus}

    def walk(cells):
        out = Parallel(n_jobs=n_jobs or _jobs())(
            delayed(_walk)(by_tau[t], t, c, m, fs_feats, fs, first, options(m))
            for t, c, m, fs, fs_feats in cells
        )
        return [o[0] for o in out], [o[1] for o in out]

    base = [(t, c, m, fs, f) for t in taus for c in bps for m in MODELS
            for fs, f in (("all", feats), ("vol_only", vol_feats))]  # fmt: skip
    fcs, vals = walk(base)
    fcs = {(f["tau"][0], f["c_bp"][0], f["model"][0], f["feature_set"][0]): f for f in fcs}

    # headline model: lower pooled log loss of the pre forecasts at the headline cell
    pooled = {}
    for m in MODELS:
        pre = fcs[(head_tau, head_c, m, "all")].filter(pl.col("kind") == "pre")
        pooled[m] = float(log_loss(pre["y"].to_numpy(), pre["p"].to_numpy(), labels=[0, 1]))
    head_model = min(MODELS, key=lambda m: pooled[m])  # ties: logit (listed first)

    boot = _Boot(reps)
    rules, quints = [], []
    for t in taus:
        for c in bps:
            for m in MODELS:
                key = (t, c, m, "all")
                is_head = t == head_tau and c == head_c
                rules.append(_rule_rows(
                    fcs[key], fcs[(t, c, m, "vol_only")], by_tau[t], key, boot, qs,
                    config.PART2_EVAL_COSTS, SPLIT_NAMES if is_head else ("all",),
                ))  # fmt: skip
                quints.append(_quintiles(fcs[key], boot, key))
    rules, quints = pl.concat(rules), pl.concat(quints)

    hd = _d(rules, tau=head_tau, c_bp=head_c, model=head_model, feature_set="all")
    q_head = quints.filter(
        (pl.col("tau") == head_tau) & (pl.col("c_bp") == head_c) & (pl.col("model") == head_model)
    ).row(0, named=True)
    verd = verdict(hd["d_af"], hd["d_bm"], hd["d_0"], q_head["top_minus_bottom"],
                   q_head["top_minus_bottom_se"])  # fmt: skip
    skill = hd["d_af"] > 2

    # input-group steps (headline cell and model only, only when the skill gate passes)
    steps = _group_steps(groups, head_model)
    group_rows = []
    if skill:
        order = [g for g in GROUP_ORDER if g in groups]
        new = []
        for i in range(1, len(order) - 1):  # vol and "all" already exist
            fs = "+".join(order[: i + 1])
            new.append(
                (head_tau, head_c, head_model, fs, [f for g in order[: i + 1] for f in groups[g]])
            )
        nfc, nvals = walk(new)
        vals += nvals
        for f in nfc:
            fcs[(f["tau"][0], f["c_bp"][0], f["model"][0], f["feature_set"][0])] = f
        for fs in steps:
            key = (head_tau, head_c, head_model, fs)
            fc = fcs[key]
            r = _rule_rows(fc, None, by_tau[head_tau], key, boot, (config.PART2_MAIN_Q,),
                           (config.PART2_EVAL_COSTS[1],), ("all",))  # fmt: skip
            added = _step_group(fs, order)
            r = r.filter(pl.col("rule") == "model").with_columns(
                step=pl.lit(added if fs == "vol_only" else f"+{added}"),
                features_added=pl.lit(",".join(groups[added])),
            )
            te = fc.filter(pl.col("kind") == "test")
            group_rows.append(
                r.with_columns(
                    auc=pl.lit(float(roc_auc_score(te["y"].to_numpy(), te["p"].to_numpy()))),
                    d_af=pl.col("diff_vs_af") / pl.col("diff_vs_af_se"),
                    top20_mean_net=pl.col("mean_net"),
                )
            )
    groups_df = pl.concat(group_rows) if group_rows else pl.DataFrame(
        schema={"step": pl.Utf8, "features_added": pl.Utf8, "auc": pl.Float64,
                "top20_mean_net": pl.Float64, "se": pl.Float64, "d_af": pl.Float64}
    )  # fmt: skip

    forecasts = pl.concat(list(fcs.values()))
    val = pl.concat(vals)
    grid = _grid(rules, forecasts)
    picked, per_year, cumulative = _headline_frames(
        fcs[(head_tau, head_c, head_model, "all")], by_tau[head_tau], feats, groups, rules, boot,
        head_model, qs,
    )  # fmt: skip
    cells = forecasts.select(_KEY).unique().height
    n_cell_years = val.select([*_KEY, "year"]).unique().height
    n_fits = val.height + n_cell_years
    other = next(m for m in MODELS if m != head_model)
    ho = _d(rules, tau=head_tau, c_bp=head_c, model=other, feature_set="all")
    summary = {
        "verdict": verd, "d_af": hd["d_af"], "d_bm": hd["d_bm"], "d_0": hd["d_0"],
        "diff_af": hd["diff_af"], "se_af": hd["se_af"],
        "top_minus_bottom": q_head["top_minus_bottom"],
        "top_minus_bottom_se": q_head["top_minus_bottom_se"],
        "skill": bool(skill), "groups_skipped": not skill, "headline_model": head_model,
        "headline_tau": head_tau, "headline_c_bp": head_c,
        "pre_logloss": pooled, "other_model": other, "other_d_af": ho["d_af"],
        "other_d_bm": ho["d_bm"], "other_d_0": ho["d_0"],
        "n_fits": int(n_fits), "n_cells": int(cells), "n_runs": int(cells + n_cell_years),
        "boot_reps": reps, "first_test_year": first,
        "max_test_auc": float(grid["auc"].max()),
    }  # fmt: skip
    return {
        "forecasts": forecasts, "rules": rules, "quintiles": quints, "grid": grid,
        "groups": groups_df, "picked": picked, "per_year": per_year, "cumulative": cumulative,
        "val": val, "summary": summary,
    }  # fmt: skip


def _step_group(fs: str, order: list[str]) -> str:
    """The group a feature-set step adds: vol_only -> volatility, a+b+c -> c, all -> the last."""
    if fs == "vol_only":
        return order[0]
    if fs == "all":
        return order[-1]
    return fs.split("+")[-1]


def _group_steps(groups, head_model) -> list[str]:
    order = [g for g in GROUP_ORDER if g in groups]
    return ["vol_only"] + ["+".join(order[: i + 1]) for i in range(1, len(order) - 1)] + ["all"]


def _grid(rules: pl.DataFrame, forecasts: pl.DataFrame) -> pl.DataFrame:
    te = forecasts.filter((pl.col("kind") == "test") & (pl.col("feature_set") == "all"))
    auc = pl.DataFrame(
        [
            {**dict(zip(["tau", "c_bp", "model"], k, strict=True)),
             "auc": float(roc_auc_score(f["y"].to_numpy(), f["p"].to_numpy()))}
            for k, f in te.group_by(["tau", "c_bp", "model"], maintain_order=True)
        ]
    )  # fmt: skip
    r = rules.filter(
        (pl.col("rule") == "model")
        & (pl.col("q") == config.PART2_MAIN_Q)
        & (pl.col("c_eval_bp") == config.PART2_MAIN_C_BP)
        & (pl.col("split") == "all")
    ).select(
        "tau",
        "c_bp",
        "model",
        top20_minus_af="diff_vs_af",
        se="diff_vs_af_se",
        top20_mean_net="mean_net",
    )
    out = auc.join(r, on=["tau", "c_bp", "model"]).with_columns(
        d_af=pl.col("top20_minus_af") / pl.col("se")
    )
    assert out.height == auc.height, "grid lost cells"
    return out


def _headline_frames(fc, tf, feats, groups, rules, boot, head_model, qs):
    """picked (feature description), per_year and cumulative of the headline top-20% rule."""
    te, flags = _flags(fc, None, tf, qs)
    pick = flags[("model", config.PART2_MAIN_Q)]
    net = te["g"].to_numpy() - config.PART2_EVAL_COSTS[1]
    sess = te["session"]
    tfs = tf.join(te.select("session"), on="session", how="inner").sort("session")
    assert tfs.height == te.height
    rows = []
    for g, names in groups.items():
        for f in names:
            v = tfs[f].cast(pl.Float64)
            sd = float(v.std())
            m1, m0 = (
                float(v.filter(pl.Series(pick)).mean()),
                float(v.filter(pl.Series(~pick)).mean()),
            )
            rows.append({"feature": f, "group": g, "mean_picked": m1, "mean_other": m0,
                         "std_diff": (m1 - m0) / sd if sd > 0 else 0.0})  # fmt: skip
    picked = pl.DataFrame(rows).sort(pl.col("std_diff").abs(), descending=True)
    per_year = (
        te.select("year").with_columns(picked=pl.Series(pick), net=pl.Series(net * pick))
        .group_by("year", maintain_order=True)
        .agg(n_days=pl.len(), n_picked=pl.col("picked").sum(), net_sum=pl.col("net").sum())
        .with_columns(share=pl.col("n_picked") / pl.col("n_days"),
                      mean_net_picked=pl.col("net_sum") / pl.col("n_picked"))
        .drop("net_sum")
    )  # fmt: skip
    cum = []
    for rule, fl in (("model", pick), ("always_follow", flags[("always_follow", 1.0)]),
                     ("big_move", flags[("big_move", config.PART2_MAIN_Q)])):  # fmt: skip
        wealth = np.cumprod(1 + np.where(fl, net, 0.0))
        cum.append(pl.DataFrame({"session": sess, "rule": rule, "wealth": wealth}))
    return picked, per_year, pl.concat(cum)


# ---------------------------------------------------------------- logging and outputs
def log_runs(results: dict) -> tuple[int, int]:
    """One MLflow parent per cell, one nested run per test year. Returns (n_parent, n_child)."""
    assert mlflow.get_tracking_uri().startswith("sqlite:"), "set the MLflow store first"
    fc, val = results["forecasts"], results["val"]
    test = fc.filter(pl.col("kind") == "test")
    n_parent = n_child = 0
    for key, v in val.group_by(_KEY, maintain_order=True):
        t = test.filter(
            (pl.col("tau") == key[0]) & (pl.col("c_bp") == key[1])
            & (pl.col("model") == key[2]) & (pl.col("feature_set") == key[3])
        )  # fmt: skip
        auc = float(roc_auc_score(t["y"].to_numpy(), t["p"].to_numpy()))
        with mlflow.start_run(run_name="|".join(map(str, key))):
            n_parent += 1
            mlflow.log_params(dict(zip(_KEY, key, strict=True)))
            mlflow.log_metrics({"test_auc": auc, "n_test": t.height})
            for (yr,), vy in v.group_by("year", maintain_order=True):
                ty = t.filter(pl.col("year") == yr)
                with mlflow.start_run(run_name=f"{key[0]}|{key[2]}|{yr}", nested=True):
                    n_child += 1
                    mlflow.log_params({"test_year": yr, "option": ty["option"][0]})
                    mlflow.log_metrics(
                        {f"val_logloss_{o}": ll for o, ll in zip(vy["option"], vy["val_logloss"], strict=True)}
                        | {"test_logloss": float(log_loss(ty["y"].to_numpy(), ty["p"].to_numpy(), labels=[0, 1]))}
                    )  # fmt: skip
    assert n_parent + n_child == results["summary"]["n_runs"], "logged runs differ from the count"
    return n_parent, n_child


def write_outputs(results: dict, out_dir) -> None:
    out = Path(out_dir)
    (out / "part2").mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if name != "summary":
            df.write_parquet(out / "part2" / f"{name}.parquet")
    (out / "part2_summary.json").write_text(json.dumps(results["summary"], indent=2, default=float))
