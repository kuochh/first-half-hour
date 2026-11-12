"""Chapter 2 end to end: forecasts, metrics, economics, interpretability, MLflow, summary.

`run` is what `fhh reproduce` calls for chapter 2. Everything is computed first (nothing here
reads MLflow), then logged under ONE parent run "chapter2"; the model registry run is made after
the parent closes, so it is not counted among the nested runs. Outputs: `<out>/chapter2/*.parquet`,
`<out>/chapter2_summary.json`, figures in `<out>/figures/`.
"""

import json
import time
from pathlib import Path

import mlflow
import polars as pl
from mlflow.tracking import MlflowClient

from first_half_hour import (
    calendar,
    config,
    controls,
    economics,
    evaluate,
    features,
    interpret,
    report_ch2,
    returns,
    walkforward,
)

PRIMARY = evaluate.PRIMARY_TARGET
STATED_FAIL = (
    "this question is (almost) all volatility; a model that beats chance would have looked "
    "skilful but adds nothing beyond a one-line formula."
)
"""Plan 02's stated result when the primary criterion fails (verbatim)."""


def targets() -> dict[str, tuple[float, str]]:
    """{target name: (b, side)}: the primary first, then the three descriptive-only targets."""
    out = {f"up_{config.TARGET_B * 100:g}": (config.TARGET_B, "up")}
    out |= {f"up_{b * 100:g}": (b, "up") for b in config.DESCRIPTIVE_BS}
    out[f"down_{config.MIRROR_B * 100:g}"] = (config.MIRROR_B, "down")
    assert next(iter(out)) == PRIMARY, "the primary target must be first"
    return out


def _metrics_for_log(fc: pl.DataFrame, target: str) -> pl.DataFrame:
    """Judging-window metrics per model, with the validation Brier skill columns beside them."""
    judge = evaluate.window_metrics(fc, target, config.CH2_JUDGE)
    valid = evaluate.window_metrics(fc, target, config.CH2_VALID).select(
        "model", val_brier="brier", val_bss_base="bss_base", val_bss_cal="bss_cal"
    )
    out = judge.join(valid, on="model", how="left", validate="1:1")
    assert out["val_brier"].null_count() == 0, "a model has no validation-window forecasts"
    return out


def _economics(fc, prices, chosen: str, reps: int) -> tuple[pl.DataFrame, dict]:
    tier = f"tier:{chosen}"
    table = economics.economics_table(
        fc, prices, {tier: tier, "formula_cal": "formula_cal"}, config.CH2_JUDGE, reps=reps
    )
    table = table.with_columns(
        note=pl.when((pl.col("rule") != "zero") & pl.col("break_even_bp").is_null())
        .then(pl.lit("no break-even: gross mean <= 0"))
        .otherwise(None)
    )
    trials = economics.trial_set(
        fc, prices, [f"tier:{t}" for t in config.CH2_TIERS] + ["formula_cal"], config.CH2_JUDGE
    )
    r = economics.rule_returns(fc, prices, tier, config.CH2_JUDGE)
    dsr = economics.deflated_sharpe(r, trials) | {"n_days": len(r)}
    return table, dsr


def _interpret(frame, sel, feats: list[str], fig: Path) -> tuple[pl.DataFrame, list[str]]:
    """Permutation importance of the tuned HGB tier per judging year, then the figures."""
    data = frame.with_columns(year=pl.col("session").dt.year().cast(pl.Int32))
    judge = data.filter(pl.col("session").is_between(*config.CH2_JUDGE))
    cfg = sel.filter((pl.col("target") == PRIMARY) & (pl.col("tier") == "hgb"))
    models = {
        yr: walkforward.fit_final(
            frame, feats, cfg.filter(pl.col("year") == yr)["model"].item(), yr
        )
        for yr in sorted(judge["year"].unique().to_list())
    }
    imp = interpret.permutation_importance_by_year(models, judge, feats)
    interpret.write_figures(models[max(models)], judge, feats, imp, fig)
    return imp, interpret.top_other_features(imp)


def _prevalence(fc: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for key, (lo, hi) in (("valid", config.CH2_VALID), ("judge", config.CH2_JUDGE)):
        w = fc.filter(
            (pl.col("target") == PRIMARY) & (pl.col("model") == "base_rate")
            & pl.col("session").is_between(lo, hi)
        )  # fmt: skip
        rows.append(
            {"window": key, "start": lo.isoformat(), "end": hi.isoformat(), "n": w.height,
             "prevalence": float(w["y"].mean())}
        )  # fmt: skip
    return pl.DataFrame(rows)


def run(data: Path, daily: Path, out: Path, synth: bool, reps: int) -> str:
    """Run chapter 2 end to end and return its count line (asserted against MLflow)."""
    assert mlflow.get_tracking_uri().startswith("sqlite:"), "set the MLflow store first"
    out_dir, fig = out / "chapter2", out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    cal = calendar.sessions(config.DATA_START, config.DATA_END)
    spy = returns.daily_table(data, "SPY", cal)
    ctrl = controls.controls(spy, cal, daily)
    feats = list(config.FEATURE_COLUMNS)
    complete = features.complete_rows(features.features(spy, ctrl, data, "SPY", cal))
    dropped = spy.height - complete.height

    frames, fcs, sels = {}, [], []
    for name, (b, side) in targets().items():
        t = features.target(data, "SPY", cal, b, side)
        frames[name] = complete.join(t.select("session", "y"), on="session", how="inner")
        assert frames[name].height == complete.height, f"{name}: sessions without a target"
        if name == PRIMARY:
            fc, sel = walkforward.walk_forward(frames[name], name, feats, b=b, side=side)
            chosen, _ = evaluate.choose_tier(fc)
            prices = economics.prices(
                spy, t.filter(pl.col("session").is_in(complete["session"].implode()))
            )
        else:
            fc, sel = walkforward.walk_forward(frames[name], name, feats, (chosen,), b=b, side=side)
        fcs.append(fc)
        sels.append(sel)
    fc, sel = pl.concat(fcs), pl.concat(sels)
    evaluate.write_forecasts(fc, sel, out_dir)

    summ = evaluate.summary(fc, sel)
    assert summ["chosen_tier"] == chosen
    prim = summ["primary"]
    met = {name: _metrics_for_log(fc, name) for name in frames}
    for key, f in (("judging", config.CH2_JUDGE), ("validation", config.CH2_VALID)):
        pl.concat(
            [evaluate.window_metrics(fc, n, f).with_columns(target=pl.lit(n)) for n in frames]
        ).write_parquet(out_dir / f"metrics_{key}.parquet")
    evaluate.year_metrics(fc, PRIMARY, config.CH2_JUDGE).write_parquet(
        out_dir / "metrics_by_year.parquet"
    )
    prev = _prevalence(fc)
    prev.write_parquet(out_dir / "prevalence.parquet")
    evaluate.reliability_figure(fc, chosen, fig / "reliability.png")
    eco, dsr = _economics(fc, prices, chosen, reps)
    eco.write_parquet(out_dir / "economics.parquet")
    imp, others = _interpret(frames[PRIMARY], sel, feats, fig)
    imp.write_parquet(out_dir / "importance.parquet")

    # One parent run "chapter2"; the registry run comes after it closes.
    started_ms = int(time.time() * 1000)
    n_children = 0
    with mlflow.start_run(run_name="chapter2") as parent:
        mlflow.log_params(
            {"synthetic": synth, "chosen_tier": chosen, "bootstrap_reps": reps,
             "spa_reps": config.CH2_SPA_REPS, "dropped_sessions": dropped, "features": len(feats)}
        )  # fmt: skip
        mlflow.log_metrics(
            {"judging_bss_vs_calibrated": prim["bss_cal"], "dm_p": prim["dm_p"],
             "primary_pass": float(prim["primary_pass"]), "spa_p": summ["spa"]["p_consistent"]}
        )  # fmt: skip
        for name in frames:
            n_children += evaluate.log_runs(met[name], name, sel)
    n_expected = len(walkforward.expected_models(config.CH2_TIERS)) + sum(
        len(walkforward.expected_models((chosen,))) for _ in list(frames)[1:]
    )
    assert n_children == n_expected, f"logged {n_children} child runs, expected {n_expected}"
    version = evaluate.register_chosen(frames[PRIMARY], feats, fc, sel, prim)

    exp = mlflow.get_experiment(parent.info.experiment_id)
    found = MlflowClient().search_runs(
        [exp.experiment_id], filter_string=f"attributes.start_time >= {started_ms}",
        max_results=50_000,
    )  # fmt: skip
    nested = [r for r in found if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
    assert len(nested) == n_children, f"MLflow holds {len(nested)} nested runs, logged {n_children}"
    assert len(found) == n_children + 2, "expected the parent, its children and one registry run"
    count_line = f"chapter 2: {n_children + 1} runs ({n_children} nested)"

    judge_rows = {r["model"]: r for r in summ["judging"]}
    one_bp = eco.filter(
        (pl.col("cost_bp") == config.THRESHOLD_COST_BP) | (pl.col("rule") == "zero")
    )
    summary = {
        "synthetic": synth,
        "target": PRIMARY,
        "data_window": [config.DATA_START.isoformat(), config.DATA_END.isoformat()],
        "chosen_tier": chosen,
        "windows": {"validation": summ["validation_window"], "judging": summ["judging_window"]},
        "primary": {
            "judging_bss_vs_calibrated": prim["bss_cal"],
            "baseline_bss": 0.0,
            "dm_t": prim["dm_t"],
            "dm_p_one_sided": prim["dm_p"],
            "dm_lags": prim["dm_lags"],
            "dm_n": prim["dm_n"],
            "alpha": config.CH2_PRIMARY_ALPHA,
            "primary_pass": prim["primary_pass"],
        },
        "stated_result": report_ch2.stated_result(prim, summ["spa"], count_line),
        "stated_result_fail_sentence": STATED_FAIL,
        "validation_bss_cal_by_tier": summ["validation_bss_cal_by_tier"],
        "judging_by_model": judge_rows,
        "prevalence": prev.to_dicts(),
        "spa": {
            **summ["spa"],
            "note": "tier:logit duplicates cfg:logit:none, so the 24 models hold 23 distinct forecasts",
        },
        "economics_1bp": one_bp.to_dicts(),
        "deflated_sharpe": dsr,
        "top_features_beyond_sigma": others,
        "volatility_group_importance": imp.filter(pl.col("feature").str.starts_with("[group]")).row(
            0, named=True
        ),
        "runs": {
            "total_excluding_registry_run": n_children + 1,
            "nested": n_children,
            "registry_run": 1,
            "count_line": count_line,
        },
        "dropped_sessions_null_feature": dropped,
        "registered_model": {"name": config.CH2_REGISTERED_MODEL, "version": version},
        "bootstrap_reps": reps,
        "tried": {
            "runs": n_children + 1,
            "targets": len(frames),
            "models_primary": len(walkforward.expected_models(config.CH2_TIERS)),
            "configs": len(walkforward.configs()),
            "thresholds": len(config.TRADE_THRESHOLDS),
            "costs": len(config.COSTS_BP),
            "trials": config.N_TRIALS,
            "spa_models": summ["spa"]["n_models"],
            "test_years": fc["year"].n_unique(),
        },
    }
    (out / "chapter2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report_ch2.write_chapter2_report(out, count_line)
    return count_line
