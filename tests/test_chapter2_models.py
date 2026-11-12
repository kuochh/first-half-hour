"""Chapter 2 walk-forward, tuning, choice rule, Diebold-Mariano and the model registry."""

import math
from datetime import date

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from first_half_hour import config, evaluate, walkforward

FEATS = ["f0", "f1", "f2", "f3"]


def _frame(seed=0, shuffle_y=False) -> pl.DataFrame:
    """Business days 2011-2020; y is Bernoulli(logistic(-1.6 + 0.9 z)) with z = f0 (known truth)."""
    days = pl.date_range(date(2011, 1, 3), date(2020, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.6 + 0.9 * x[:, 0])))).astype(int)
    if shuffle_y:
        y = rng.permutation(y)
    z = x[:, 0]
    p_raw = np.array([0.5 * math.erfc(v / math.sqrt(2)) for v in z])  # 2 * (1 - Phi(z))/2, a stub
    return pl.DataFrame(
        {"session": days, "y": y, "z": z, "r1": rng.normal(size=n), "p_raw": p_raw}
        | {f: x[:, i] for i, f in enumerate(FEATS)}
    )


def _perturb(frame: pl.DataFrame, cut: date, seed=1) -> pl.DataFrame:
    """Every row on or after `cut`: features scaled and shifted, labels redrawn."""
    rng = np.random.default_rng(seed)
    late = (frame["session"] >= cut).to_numpy()
    cols = {}
    for c in [*FEATS, "z", "r1", "p_raw"]:
        v = frame[c].to_numpy().copy()
        v[late] = v[late] * 3 + 1.5 if c != "p_raw" else rng.random(late.sum())
        cols[c] = v
    y = frame["y"].to_numpy().copy()
    y[late] = rng.integers(0, 2, late.sum())
    return frame.with_columns(**{k: pl.Series(k, v) for k, v in cols.items()}, y=pl.Series("y", y))


@pytest.fixture(scope="module")
def small_grid():
    """Smaller boosted-tree grid keeps the fast set fast; the logic under test is unchanged."""
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "CH2_HGB_ITER", (20, 40))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def base(small_grid):
    frame = _frame()
    fc, sel = walkforward.walk_forward(frame, FEATS, "up_0.4")
    return frame, fc, sel


def test_folds_never_train_on_the_test_year(small_grid, monkeypatch):
    frame = _frame()
    seen = []
    real = walkforward.fold

    def spy(f, year):
        train, test = real(f, year)
        seen.append((year, train["session"].max(), test["session"].min(), test["session"].max()))
        return train, test

    monkeypatch.setattr(walkforward, "fold", spy)
    walkforward.fixed_forecasts(frame, FEATS, [2015, 2016], ("logit",))
    assert [s[0] for s in seen] == [2015, 2016]
    for year, train_max, test_min, test_max in seen:
        assert train_max < date(year, 1, 1) <= test_min and test_max.year == year


def test_choice_and_tuning_ignore_rows_from_2019(base):
    frame, fc, sel = base
    bad_fc, bad_sel = walkforward.walk_forward(_perturb(frame, date(2019, 1, 1)), FEATS, "up_0.4")
    # Tuning for year Y reads year Y-1 only: choices for <= 2019 cannot see 2019 rows' effect.
    assert sel.filter(pl.col("year") <= 2019).equals(bad_sel.filter(pl.col("year") <= 2019))
    assert fc.filter(pl.col("year") <= 2018).equals(bad_fc.filter(pl.col("year") <= 2018))
    assert evaluate.choose_tier(fc)[0] == evaluate.choose_tier(bad_fc)[0]
    assert evaluate.choose_tier(fc)[1].equals(evaluate.choose_tier(bad_fc)[1])
    # the perturbation is real: the later forecasts did move
    assert not fc.filter(pl.col("year") == 2020).equals(bad_fc.filter(pl.col("year") == 2020))


def test_year_forecasts_unchanged_when_later_rows_change(base):
    frame, fc, _ = base
    # Y = 2016 (all models, tuned tiers too): change everything from 2017-01-01 on.
    got, _ = walkforward.walk_forward(
        _perturb(frame, date(2017, 1, 1)), FEATS, "up_0.4", years=[2015, 2016]
    )
    assert fc.filter(pl.col("year") <= 2016).equals(got)
    assert got.filter(pl.col("year") == 2016)["model"].n_unique() == 4 + 18 + 3


def test_shuffled_labels_give_near_zero_skill_against_the_base_rate(small_grid):
    # Labels permuted across time: no model can have skill. Tolerance: pooled 2016-2018 BSS vs the
    # base rate within [-0.06, +0.02] (flexible trees overfit noise, so the downside is wider).
    # A control that the planted dependence IS found: calibrated formula beats base rate there.
    fc, _ = walkforward.walk_forward(_frame(shuffle_y=True), FEATS, "up_0.4")
    m = evaluate.window_metrics(fc, "up_0.4", config.CH2_VALID)
    for model in ("formula_cal", "gao_r1", "tier:logit", "tier:enet", "tier:hgb"):
        bss = m.filter(pl.col("model") == model)["bss_base"].item()
        assert -0.06 < bss < 0.02, (model, bss)
    real, _ = walkforward.walk_forward(_frame(), FEATS, "up_0.4", tiers=("logit",))
    cal = evaluate.window_metrics(real, "up_0.4", config.CH2_VALID)
    assert cal.filter(pl.col("model") == "formula_cal")["bss_base"].item() > 0.03


def test_diebold_mariano_by_hand():
    d = np.array([0.3, -0.1, 0.2, 0.4, -0.2, 0.1, 0.3, 0.0])
    t_obs = len(d)
    lags = config.nw_lags(t_obs)
    assert lags == 2
    # By hand: Bartlett long-run variance of the demeaned series, se = sqrt(S) / T.
    mean = sum(d) / t_obs
    e = [v - mean for v in d]
    gamma = [sum(e[i] * e[i - j] for i in range(j, t_obs)) for j in range(lags + 1)]
    s = gamma[0] + 2 * sum((1 - j / (lags + 1)) * gamma[j] for j in range(1, lags + 1))
    t_hand = mean / (math.sqrt(s) / t_obs)
    p_hand = 0.5 * math.erfc(t_hand / math.sqrt(2))  # 1 - Phi(t)
    got = evaluate.dm_test(d)
    assert got["t"] == pytest.approx(t_hand, rel=1e-9)
    assert got["p"] == pytest.approx(p_hand, rel=1e-9)


def test_registry_round_trip_and_run_count(tmp_path, base):
    frame, _, _ = base
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    exp = MlflowClient().create_experiment("t2-test", artifact_location=str(tmp_path / "art"))
    mlflow.set_experiment(experiment_id=exp)
    model = walkforward.fit_final(frame, FEATS, "cfg:logit:none", 2019)
    x = frame.head(5).select(FEATS)
    version = evaluate.register_final_model(
        model, {c: x[c].to_numpy() for c in FEATS},
        {"chosen_on": "2016-2018", "judging_bss": "0.01", "primary_pass": "False"},
    )  # fmt: skip
    uri = f"models:/{config.CH2_REGISTERED_MODEL}/{version}"
    back = mlflow.sklearn.load_model(uri)
    xs = frame.select(FEATS).to_numpy()
    np.testing.assert_allclose(back.predict_proba(xs), model.predict_proba(xs))
    mv = MlflowClient().get_model_version(config.CH2_REGISTERED_MODEL, version)
    assert mv.tags["primary_pass"] == "False" and mv.tags["chosen_on"] == "2016-2018"
    info = mlflow.models.get_model_info(uri)
    assert [c.name for c in info.signature.inputs.inputs] == FEATS
    # run log: one parent, one child per model row, asserted against MLflow inside
    metrics = pl.DataFrame({"model": ["a", "b", "c"], "brier": [0.1, 0.2, 0.3]})
    assert evaluate.log_runs(metrics, "up_0.4") == (1, 3)


def test_summary_figure_and_spa_run_end_to_end(base, tmp_path, monkeypatch):
    _, fc, _ = base
    monkeypatch.setattr(config, "CH2_SPA_REPS", 200)
    s = evaluate.summary(fc)
    assert s["chosen_tier"] in config.CH2_TIERS
    assert s["spa"]["n_models"] == 18 + 3 + 3 and 0 <= s["spa"]["p_consistent"] <= 1
    prim = s["primary"]
    assert prim["primary_pass"] == (prim["bss_cal"] > 0 and prim["dm_p"] < 0.05)
    base_row = next(r for r in s["judging"] if r["model"] == "base_rate")
    assert base_row["bss_base"] == 0 and base_row["pr_auc"] == pytest.approx(
        base_row["prevalence"], rel=0.05
    )  # base rate differs a little by year
    evaluate.reliability_figure(fc, s["chosen_tier"], tmp_path / "rel.png")
    assert (tmp_path / "rel.png").stat().st_size > 1000
