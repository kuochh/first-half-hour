"""Chapter 2 evaluation: metrics, the choice rule, the primary criterion (HAC Diebold-Mariano),
the reliability diagram, SPA, the MLflow run log and the model-registry helper.

All of it reads the long `forecasts` frame (session, year, target, y, model, p).
"""

from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import statsmodels.api as sm
from arch.bootstrap import SPA
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from first_half_hour import config, walkforward

PRIMARY_TARGET = "up_0.4"


def _window(fc: pl.DataFrame, target: str, start: date, end: date) -> pl.DataFrame:
    return fc.filter((pl.col("target") == target) & pl.col("session").is_between(start, end)).sort(
        "model", "session"
    )


def _aligned(win: pl.DataFrame, models: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """(y, {model: p}) with every model on the identical sessions (asserted)."""
    models = sorted(win["model"].unique().to_list()) if models is None else models
    sessions = win.filter(pl.col("model") == models[0])["session"]
    out = {}
    for m in models:
        sub = win.filter(pl.col("model") == m)
        assert sub.height == sessions.len() and (sub["session"] == sessions).all(), (
            f"model {m} is not on the same sessions as {models[0]}"
        )
        out[m] = sub["p"].to_numpy()
    y = win.filter(pl.col("model") == models[0])["y"].to_numpy().astype(float)
    return y, out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _metrics(win: pl.DataFrame) -> pl.DataFrame:
    y, ps = _aligned(win)
    assert 0 < y.sum() < len(y), "window holds one class only; AUCs undefined"
    b_base, b_cal = brier(y, ps["base_rate"]), brier(y, ps["formula_cal"])
    clip = config.PROB_CLIP_EPS
    ll_base = float(log_loss(y, np.clip(ps["base_rate"], clip, 1 - clip)))
    ll_cal = float(log_loss(y, np.clip(ps["formula_cal"], clip, 1 - clip)))
    rows = []
    for m, p in ps.items():
        b = brier(y, p)
        rows.append(
            {
                "model": m,
                "n": len(y),
                "brier": b,
                "bss_base": 1 - b / b_base,
                "bss_cal": 1 - b / b_cal,
                "log_loss": float(log_loss(y, np.clip(p, clip, 1 - clip))),
                "log_loss_base": ll_base,
                "log_loss_cal": ll_cal,
                "pr_auc": float(average_precision_score(y, p)),
                "prevalence": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, p)),
                "roc_auc_null": 0.5,
            }
        )
    return pl.DataFrame(rows)


def window_metrics(fc: pl.DataFrame, target: str, window: tuple[date, date]) -> pl.DataFrame:
    """One row per model: Brier, BSS vs base rate and vs calibrated formula, log loss, PR-AUC
    beside the prevalence (its no-skill value), ROC-AUC. Pooled over the window."""
    return _metrics(_window(fc, target, *window))


def year_metrics(fc: pl.DataFrame, target: str, window: tuple[date, date]) -> pl.DataFrame:
    win = _window(fc, target, *window)
    return pl.concat(
        [_metrics(win.filter(pl.col("year") == yr)).with_columns(year=pl.lit(yr, pl.Int32))
         for yr in sorted(win["year"].unique().to_list())]
    )  # fmt: skip


def choose_tier(fc: pl.DataFrame, target: str = PRIMARY_TARGET) -> tuple[str, pl.DataFrame]:
    """The tuned tier with the highest pooled validation-window BSS vs the calibrated formula.

    Reads only the validation window (asserted to end before the judging window). Ties go to
    the simpler tier (config.CH2_TIERS order).
    """
    v0, v1 = config.CH2_VALID
    assert v1 < config.CH2_JUDGE[0], "validation window must end before the judging window"
    tiers = [f"tier:{t}" for t in config.CH2_TIERS]
    win = _window(fc, target, v0, v1).filter(pl.col("model").is_in([*tiers, "formula_cal"]))
    y, ps = _aligned(win)
    b_cal = brier(y, ps["formula_cal"])
    table = pl.DataFrame(
        {"model": tiers, "val_bss_cal": [1 - brier(y, ps[t]) / b_cal for t in tiers]}
    )
    best = tiers[0]
    for t, s in zip(tiers, table["val_bss_cal"].to_list(), strict=True):
        if s > table.filter(pl.col("model") == best)["val_bss_cal"].item():
            best = t  # strictly greater: a tie keeps the simpler tier
    return best.removeprefix("tier:"), table


def dm_test(d: np.ndarray) -> dict:
    """One-sided Diebold-Mariano on loss differences d_t: t = mean(d) / Newey-West se, p = 1 - Phi(t).

    HAC from an OLS of d on a constant, lags = config.nw_lags(T) (Bartlett, no small-sample term).
    """
    d = np.asarray(d, dtype=float)
    lags = config.nw_lags(len(d))
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(res.tvalues[0])
    return {
        "mean_d": float(d.mean()),
        "t": t,
        "p": float(1 - norm.cdf(t)),
        "lags": lags,
        "n": len(d),
    }


def primary_criterion(fc: pl.DataFrame, chosen: str, target: str = PRIMARY_TARGET) -> dict:
    """D19 on the judging window: BSS vs calibrated > 0 AND one-sided DM p < 0.05."""
    win = _window(fc, target, *config.CH2_JUDGE).filter(
        pl.col("model").is_in([f"tier:{chosen}", "formula_cal"])
    )
    y, ps = _aligned(win)
    p_cal, p_tier = ps["formula_cal"], ps[f"tier:{chosen}"]
    dm = dm_test((p_cal - y) ** 2 - (p_tier - y) ** 2)
    assert np.isfinite(dm["t"]), "DM t is not finite (zero-variance loss difference?)"
    bss = 1 - brier(y, p_tier) / brier(y, p_cal)
    return {
        "chosen": chosen,
        "bss_cal": float(bss),
        **{f"dm_{k}": v for k, v in dm.items()},
        "primary_pass": bool(bss > 0 and dm["p"] < config.CH2_PRIMARY_ALPHA),
    }


def reliability_figure(fc: pl.DataFrame, chosen: str, path, target: str = PRIMARY_TARGET) -> None:
    """Chosen tier vs calibrated formula on the judging window: 10 quantile bins."""
    models = [f"tier:{chosen}", "formula_cal"]
    y, ps = _aligned(_window(fc, target, *config.CH2_JUDGE).filter(pl.col("model").is_in(models)))
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    for m in models:
        frac, mean_p = calibration_curve(y, ps[m], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", label=m)
    ax.axhline(y.mean(), color="grey", lw=0.5, label=f"base rate {y.mean():.3f}")
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title="Reliability")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def spa_test(fc: pl.DataFrame, target: str = PRIMARY_TARGET) -> dict:
    """Hansen's SPA: benchmark = calibrated formula; models = every other forecast (18
    configurations, 3 tuned tiers, formula_raw, gao_r1, base_rate) on the judging window.
    Squared-error losses, stationary bootstrap (arch default block sqrt(T)), reps/seed in config."""
    win = _window(fc, target, *config.CH2_JUDGE)
    y, ps = _aligned(win)
    others = [m for m in ps if m != "formula_cal"]
    assert len(others) == len(walkforward.configs()) + len(config.CH2_TIERS) + 3, others
    loss = {m: (p - y) ** 2 for m, p in ps.items()}
    spa = SPA(
        loss["formula_cal"],
        np.column_stack([loss[m] for m in others]),
        reps=config.CH2_SPA_REPS,
        bootstrap="stationary",
        seed=config.CH2_SPA_SEED,
    )
    spa.compute()
    return {
        "benchmark": "formula_cal",
        "p_consistent": float(spa.pvalues["consistent"]),
        "n_models": len(others),
    }


def summary(fc: pl.DataFrame, sel: pl.DataFrame, target: str = PRIMARY_TARGET) -> dict:
    """The primary result with its baselines (json-ready); the caller adds economics and counts."""
    chosen, val = choose_tier(fc, target)
    keep = [
        "base_rate",
        "formula_raw",
        "formula_cal",
        "gao_r1",
        *(f"tier:{t}" for t in config.CH2_TIERS),
    ]
    return {
        "target": target,
        "chosen_tier": chosen,
        "validation_window": [d.isoformat() for d in config.CH2_VALID],
        "judging_window": [d.isoformat() for d in config.CH2_JUDGE],
        "validation_bss_cal_by_tier": dict(val.iter_rows()),
        "validation": window_metrics(fc, target, config.CH2_VALID).filter(
            pl.col("model").is_in(keep)).to_dicts(),
        "judging": window_metrics(fc, target, config.CH2_JUDGE).filter(
            pl.col("model").is_in(keep)).to_dicts(),
        "primary": primary_criterion(fc, chosen, target),
        "spa": spa_test(fc, target),
        "tier_configs": sel.filter(pl.col("target") == target).sort("year", "tier").to_dicts(),
    }  # fmt: skip


def write_forecasts(fc: pl.DataFrame, sel: pl.DataFrame, out: Path) -> tuple[Path, Path]:
    """Write `forecasts.parquet` (the spec's interface) and `tier_configs.parquet` (the tuned
    configuration per target, year and tier) into `out`. Asserts the schema and the model sets:
    the primary target 4 + 18 + 3 = 25 models, a descriptive target 4 + its tier's configurations + 1.
    """
    schema = {
        "session": pl.Date, "year": pl.Int32, "target": pl.Utf8, "y": pl.Int8,
        "model": pl.Utf8, "p": pl.Float64,
    }  # fmt: skip
    assert dict(fc.schema) == schema, f"forecasts schema {dict(fc.schema)}"
    assert fc.group_by("session", "target", "model").len()["len"].max() == 1
    assert fc["p"].is_finite().all() and fc["p"].is_between(0, 1).all()
    assert PRIMARY_TARGET in fc["target"].unique().to_list(), "no primary target"
    for tgt in fc["target"].unique().to_list():
        models = set(fc.filter(pl.col("target") == tgt)["model"].unique())
        tiers = tuple(t for t in config.CH2_TIERS if f"tier:{t}" in models)
        assert tiers, f"{tgt}: no tuned tier"
        assert len(tiers) == (len(config.CH2_TIERS) if tgt == PRIMARY_TARGET else 1), (tgt, tiers)
        assert models == walkforward.expected_models(tiers), f"{tgt}: unexpected model set"
    assert {"year", "tier", "model", "target"} <= set(sel.columns)
    out.mkdir(parents=True, exist_ok=True)
    fc_path, sel_path = out / "forecasts.parquet", out / "tier_configs.parquet"
    fc.sort("target", "model", "session").write_parquet(fc_path)
    sel.sort("target", "year", "tier").write_parquet(sel_path)
    return fc_path, sel_path


def log_runs(metrics: pl.DataFrame, target: str, sel: pl.DataFrame | None = None) -> int:
    """One nested MLflow child run per model row, under the open parent run "chapter2".

    Returns the number of children logged by this call (asserted against MLflow). A tuned-tier
    child also logs its configuration per year (from `sel`). The caller (T4) adds the parent run
    and the count line.
    """
    parent = mlflow.active_run()
    assert parent is not None and parent.data.tags.get("mlflow.runName") == "chapter2", (
        "log_runs needs the open parent run 'chapter2'"
    )
    client = MlflowClient()

    def nested() -> int:
        return len(
            client.search_runs(
                [parent.info.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
                max_results=50_000,
            )
        )

    before = nested()
    for row in metrics.iter_rows(named=True):
        with mlflow.start_run(run_name=f"{target}|{row['model']}", nested=True):
            mlflow.log_params({"target": target, "model": row["model"]})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, int | float)})
            if sel is not None and row["model"].startswith("tier:"):
                tier = row["model"].removeprefix("tier:")
                chosen = sel.filter((pl.col("target") == target) & (pl.col("tier") == tier))
                mlflow.log_param(
                    "configuration_by_year",
                    "; ".join(f"{r['year']}: {r['model']}" for r in chosen.sort("year").to_dicts()),
                )
    assert nested() - before == metrics.height, "MLflow child count differs from the logged rows"
    return metrics.height


def register_chosen(
    frame: pl.DataFrame, feats: list[str], fc: pl.DataFrame, sel: pl.DataFrame, primary: dict
) -> str:
    """Register the chosen tier's final model as `fhh-chapter2-hit` and return the version.

    The model is the chosen tier's configuration tuned for the last year, fit on all rows before
    that year (`walkforward.fit_final`). Asserted equal to the logged `tier:<chosen>` forecasts of
    the last year, both as fitted and after loading it back from the registry.
    """
    chosen, last = primary["chosen"], int(fc["year"].max())
    name = sel.filter(
        (pl.col("target") == PRIMARY_TARGET) & (pl.col("year") == last) & (pl.col("tier") == chosen)
    )["model"].item()
    model = walkforward.fit_final(frame, feats, name, last)
    test = frame.filter(pl.col("session").dt.year() == last)
    logged = fc.filter(
        (pl.col("target") == PRIMARY_TARGET) & (pl.col("model") == f"tier:{chosen}")
        & (pl.col("year") == last)
    ).sort("session")  # fmt: skip
    assert logged["session"].to_list() == test["session"].to_list()
    x_test = test.select(feats).to_numpy()
    np.testing.assert_allclose(
        model.predict_proba(x_test)[:, 1], logged["p"].to_numpy(), rtol=1e-12
    )
    version = register_final_model(
        model,
        {c: test[c].head(5).to_numpy() for c in feats},
        {
            "chosen_on": f"{config.CH2_VALID[0].year}-{config.CH2_VALID[1].year}",
            "judging_bss_vs_calibrated": f"{primary['bss_cal']:.6f}",
            "primary_pass": str(primary["primary_pass"]),
            "configuration": name,
        },
    )
    back = mlflow.sklearn.load_model(f"models:/{config.CH2_REGISTERED_MODEL}/{version}")
    np.testing.assert_allclose(back.predict_proba(x_test)[:, 1], logged["p"].to_numpy(), rtol=1e-12)
    return version


def register_final_model(model, x_example: dict[str, np.ndarray], tags: dict[str, str]) -> str:
    """Log `model` (sklearn) with an input signature and register it as fhh-chapter2-hit.

    `x_example`: {feature name: column} for a few rows. Returns the new model version.
    Tags (chosen_on, judging BSS, primary_pass, ...) go on the model version.
    """
    cols = list(x_example)
    x = np.column_stack([x_example[c] for c in cols])
    sig = infer_signature(
        {c: np.asarray(v, dtype=float) for c, v in x_example.items()}, model.predict_proba(x)
    )
    with mlflow.start_run(run_name="register final model"):
        info = mlflow.sklearn.log_model(
            model, name="model", signature=sig, pyfunc_predict_fn="predict_proba",
            registered_model_name=config.CH2_REGISTERED_MODEL,
        )  # fmt: skip
    client = MlflowClient()
    version = info.registered_model_version
    for k, v in tags.items():
        client.set_model_version_tag(config.CH2_REGISTERED_MODEL, version, k, str(v))
    return str(version)
