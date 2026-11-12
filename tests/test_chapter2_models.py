"""Chapter 2 walk-forward, tuning, choice rule, Diebold-Mariano and the model registry."""

import math
from datetime import date

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from first_half_hour import baselines, config, evaluate, walkforward

FEATS = ["f0", "f1", "f2", "z"]
CUT = date(2016, 7, 1)


def _frame(seed=0, shuffle_y=False) -> pl.DataFrame:
    """Business days 2011-2020; y is Bernoulli(logistic(-3 + 4 z)), z the formula's z (known truth)."""
    days = pl.date_range(date(2011, 1, 3), date(2020, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(2e-4, 8e-4, n)
    tau = np.full(n, 360.0)
    z = baselines.formula_z(sigma, tau, config.TARGET_B)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-3 + 4 * z)))).astype(int)
    if shuffle_y:
        y = rng.permutation(y)
    return pl.DataFrame(
        {"session": days, "y": y, "sigma": sigma, "tau": tau, "r1": rng.normal(size=n), "z": z}
        | {f"f{i}": rng.normal(size=n) for i in range(3)}
    )


def _perturb(frame: pl.DataFrame, cut: date, seed=1) -> pl.DataFrame:
    """Every row on or after `cut`: features scaled and shifted, volatility rescaled, labels redrawn."""
    rng = np.random.default_rng(seed)
    late = (frame["session"] >= cut).to_numpy()
    cols = {}
    for c in ["f0", "f1", "f2", "z", "r1", "sigma"]:
        v = frame[c].to_numpy().copy()
        v[late] = v[late] * 3 + (0 if c == "sigma" else 1.5)
        cols[c] = v
    y = frame["y"].to_numpy().copy()
    y[late] = rng.integers(0, 2, late.sum())
    return frame.with_columns(**{k: pl.Series(k, v) for k, v in cols.items()}, y=pl.Series("y", y))


def _wf(frame, **kw):
    return walkforward.walk_forward(frame, "up_0.4", FEATS, **kw)


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
    fc, sel = _wf(frame)
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
    bad_fc, bad_sel = _wf(_perturb(frame, date(2019, 1, 1)))
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
    got, _ = _wf(_perturb(frame, date(2017, 1, 1)), years=[2015, 2016])
    assert fc.filter(pl.col("year") <= 2016).equals(got)
    assert got.filter(pl.col("year") == 2016)["model"].n_unique() == 4 + 18 + 3


def test_mid_year_cut_leaves_earlier_forecasts_bit_identical(small_grid):
    # A fit that read its own test year (base rate, calibrated formula, scaler, labels) would move
    # forecasts dated before a cut inside the test year when rows after the cut change. Year-boundary
    # cuts cannot see that. Five mutants (fit on train+test; scaler on train+test; tuning scores
    # year Y; base rate from year Y; calibrated formula on train+test) all fail this test.
    frame = _frame()
    a, sel_a = _wf(frame, years=[2015, 2016])
    b, sel_b = _wf(_perturb(frame, CUT), years=[2015, 2016])
    early = lambda f: f.filter(pl.col("session") < CUT)
    assert early(a).height > 0 and early(a).equals(early(b)) and sel_a.equals(sel_b)
    assert not a.equals(b), "the perturbation must move the later 2016 forecasts"


def test_shuffled_labels_give_near_zero_skill_against_the_base_rate(small_grid, monkeypatch):
    # Labels permuted across time: no model can have skill. Band: pooled 2016-2018 BSS vs the base
    # rate within (-0.02, +0.02). Derivation: over 5 shuffle seeds (0-4) of this frame every model's BSS lay
    # in [-0.0089, +0.0087], so the band is about 2.3x the observed extreme on each side.
    fc, _ = _wf(_frame(shuffle_y=True))
    m = evaluate.window_metrics(fc, "up_0.4", config.CH2_VALID)
    for model in ("formula_cal", "gao_r1", "tier:logit", "tier:enet", "tier:hgb"):
        bss = m.filter(pl.col("model") == model)["bss_base"].item()
        assert -0.02 < bss < 0.02, (model, bss)
    # A control that the planted dependence IS found: calibrated formula beats base rate there.
    real, _ = _wf(_frame(), tiers=("logit",))
    cal = evaluate.window_metrics(real, "up_0.4", config.CH2_VALID)
    assert cal.filter(pl.col("model") == "formula_cal")["bss_base"].item() > 0.03

    # Planted leak: a fold that trains on the test year too must push the boosted trees out of the band.
    def leaky(f, year):
        train = f.filter(pl.col("session").dt.year() <= year)
        return train, f.filter(pl.col("session").dt.year() == year)

    monkeypatch.setattr(walkforward, "fold", leaky)
    bad, _ = _wf(_frame(shuffle_y=True), tiers=("hgb",), years=[2015, 2016, 2017, 2018])
    bad_m = evaluate.window_metrics(bad, "up_0.4", config.CH2_VALID)
    assert bad_m.filter(pl.col("model") == "tier:hgb")["bss_base"].item() > 0.02


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


def _fc(p_by_model: dict, y, start=date(2019, 1, 2), extra_pre=True) -> pl.DataFrame:
    """Hand-built forecasts: consecutive calendar days from `start`; one 2018 row per model that
    must not enter any judging-window number (tier perfect there, formula_cal worst)."""
    days = [date.fromordinal(start.toordinal() + i) for i in range(len(y))]
    parts = []
    for m, p in p_by_model.items():
        parts.append(
            pl.DataFrame(
                {
                    "session": days,
                    "year": [d.year for d in days],
                    "target": "up_0.4",
                    "y": y,
                    "model": m,
                    "p": p,
                }
            )
        )
        if extra_pre:
            good = 1.0 if m != "formula_cal" else 0.0
            parts.append(
                pl.DataFrame(
                    {
                        "session": [date(2018, 6, 1)],
                        "year": [2018],
                        "target": "up_0.4",
                        "y": [1],
                        "model": [m],
                        "p": [good],
                    }
                )
            )
    return pl.concat(parts).with_columns(pl.col("year").cast(pl.Int32), pl.col("y").cast(pl.Int8))


def test_primary_by_hand():
    # Checked by hand formulas on the judging rows only: BSS = 1 - Brier(tier)/Brier(calibrated)
    # (not vs the base rate), d = loss(cal) - loss(tier) so a better tier gives d > 0 and a small p.
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(int)
    p_cal, p_base = np.full(400, 0.4), np.full(400, 0.6)
    p_tier = np.clip(0.4 + 0.15 * (2 * y - 1) + rng.normal(0, 0.2, 400), 0, 1)
    models = {"tier:logit": p_tier, "formula_cal": p_cal, "base_rate": p_base}
    got = evaluate.primary_criterion(_fc(models, y), "logit")
    d = (p_cal - y) ** 2 - (p_tier - y) ** 2
    bss = 1 - np.mean((p_tier - y) ** 2) / np.mean((p_cal - y) ** 2)
    assert abs(bss - (1 - np.mean((p_tier - y) ** 2) / np.mean((p_base - y) ** 2))) > 0.05
    assert got["dm_n"] == 400 and got["bss_cal"] == pytest.approx(bss)
    assert got["dm_mean_d"] == pytest.approx(d.mean()) and got["dm_mean_d"] > 0
    assert got["dm_p"] < 0.05 and got["primary_pass"] is True
    # a worse tier: BSS < 0, DM p near 1, no pass
    worse = evaluate.primary_criterion(
        _fc({**models, "tier:logit": p_cal + 0.0 * y + (1 - 2 * y) * 0.2}, y), "logit"
    )
    assert worse["bss_cal"] < 0 and worse["dm_p"] > 0.95 and worse["primary_pass"] is False


def test_tune_argmin_and_ties_go_to_the_first_configuration():
    # Year 2015's labels; enet configuration 3 and 5 forecast it perfectly (a tie), the rest are 0.5.
    days = [date(2015, 1, 1 + i) for i in range(10)] + [date(2016, 1, 4)]
    y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    frame = pl.DataFrame({"session": days, "y": y})
    cfgs = walkforward.configs(("enet",))
    y15 = np.array(y[:10])
    preds = {(2015, c.name): np.full(10, 0.5) for c in cfgs}
    preds[2015, cfgs[3].name] = y15.astype(float)
    preds[2015, cfgs[5].name] = y15.astype(float)
    sel = walkforward.tune(frame, preds, [2015, 2016], ("enet",))
    assert sel["model"].to_list() == [cfgs[3].name]  # argmin, not argmax; first of the tie
    assert sel["year"].to_list() == [2016]


def _spa_fc(p_cal_fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    names = [*walkforward.expected_models(config.CH2_TIERS)]
    p = {m: np.clip(0.5 + rng.normal(0, 0.05, n), 0, 1) for m in names}
    p["formula_cal"] = p_cal_fn(y)
    return _fc(p, y, extra_pre=False)


def test_spa_benchmark_is_the_calibrated_formula(monkeypatch):
    monkeypatch.setattr(config, "CH2_SPA_REPS", 200)
    # Benchmark perfect: none beats it, p not small. Benchmark anti-informative: others beat it, p small.
    # With the base rate as benchmark instead (a mutant), the second case would not give a small p.
    good = evaluate.spa_test(_spa_fc(lambda y: 0.05 + 0.9 * y))
    bad = evaluate.spa_test(_spa_fc(lambda y: 0.9 - 0.8 * y))
    assert good["benchmark"] == bad["benchmark"] == "formula_cal"
    assert good["n_models"] == 18 + 3 + 3
    assert good["p_consistent"] > 0.2 and bad["p_consistent"] < 0.05


def test_check_frame_rejects_nan_and_null(small_grid):
    frame = _frame()
    with pytest.raises(AssertionError, match="NaN"):
        walkforward.fixed_forecasts(
            frame.with_columns(
                f0=pl.when(pl.col("f0") > 2).then(float("nan")).otherwise(pl.col("f0"))
            ),
            FEATS,
            [2015],
            ("hgb",),
        )
    with pytest.raises(AssertionError, match="nulls"):
        walkforward.fixed_forecasts(
            frame.with_columns(f1=pl.when(pl.col("f1") > 2).then(None).otherwise(pl.col("f1"))),
            FEATS,
            [2015],
            ("hgb",),
        )


def test_non_convergence_fails_loud(small_grid, monkeypatch):
    frame = _frame()
    monkeypatch.setattr(config, "CH2_MAX_ITER", 2)
    with pytest.raises(AssertionError, match="not converged"):
        walkforward.fixed_forecasts(frame, FEATS, [2015], ("enet",))


def test_write_forecasts_counts_and_schema(base, tmp_path, small_grid):
    frame, fc, sel = base
    assert fc.filter(pl.col("year") == 2020)["model"].n_unique() == 25
    # a descriptive target: 4 baselines + the logit tier's 1 configuration + 1 tuned row = 6 models
    desc, dsel = walkforward.walk_forward(
        frame, "up_0.2", FEATS, tiers=("logit",), years=[2015, 2016, 2017], b=0.002
    )
    assert desc["model"].n_unique() == 4 + 1 + 1
    fc_path, sel_path = evaluate.write_forecasts(
        pl.concat([fc, desc]), pl.concat([sel, dsel]), tmp_path / "chapter2"
    )
    back = pl.read_parquet(fc_path)
    assert back.columns == ["session", "year", "target", "y", "model", "p"]
    assert back.filter(pl.col("target") == "up_0.4")["model"].n_unique() == 25
    tc = pl.read_parquet(sel_path)
    assert set(tc.columns) == {"year", "tier", "model", "target"}
    assert tc.filter(pl.col("target") == "up_0.4").height == 3 * len(range(2016, 2021))
    with pytest.raises(AssertionError, match="unexpected model set"):  # a missing configuration
        drop = fc.filter(pl.col("model") != "cfg:logit:none")
        evaluate.write_forecasts(drop, sel, tmp_path / "bad")


def test_registry_is_the_model_behind_the_last_forecasts_and_runs_nest(tmp_path, base, monkeypatch):
    frame, fc, sel = base
    monkeypatch.setattr(config, "CH2_SPA_REPS", 100)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    exp = MlflowClient().create_experiment("t2-test", artifact_location=str(tmp_path / "art"))
    mlflow.set_experiment(experiment_id=exp)
    summ = evaluate.summary(fc, sel)
    prim = summ["primary"]
    judging = evaluate.window_metrics(fc, "up_0.4", config.CH2_JUDGE)
    with mlflow.start_run(run_name="chapter2") as parent:
        n1 = evaluate.log_runs(judging, "up_0.4", sel)
        n2 = evaluate.log_runs(judging.head(3), "up_0.2")
    assert (n1, n2) == (judging.height, 3)
    client = MlflowClient()
    kids = client.search_runs(
        [exp], filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'", max_results=100
    )
    assert len(kids) == n1 + n2
    assert len(client.search_runs([exp], max_results=100)) == 1 + n1 + n2
    tier_run = next(r for r in kids if r.data.params["model"] == f"tier:{prim['chosen']}"
                    and r.data.params["target"] == "up_0.4")  # fmt: skip
    assert tier_run.data.params["configuration_by_year"].count("cfg:") == 5  # 2016..2020
    with pytest.raises(AssertionError, match="open parent"):  # no parent, no children
        evaluate.log_runs(judging, "up_0.4")
    version = evaluate.register_chosen(frame, FEATS, fc, sel, prim)  # asserts the identity inside
    mv = client.get_model_version(config.CH2_REGISTERED_MODEL, version)
    assert (
        mv.tags["primary_pass"] == str(prim["primary_pass"]) and mv.tags["chosen_on"] == "2016-2018"
    )
    info = mlflow.models.get_model_info(f"models:/{config.CH2_REGISTERED_MODEL}/{version}")
    assert [c.name for c in info.signature.inputs.inputs] == FEATS


def test_summary_figure_and_spa_run_end_to_end(base, tmp_path, monkeypatch):
    _, fc, sel = base
    monkeypatch.setattr(config, "CH2_SPA_REPS", 200)
    s = evaluate.summary(fc, sel)
    assert s["chosen_tier"] in config.CH2_TIERS
    assert s["spa"]["n_models"] == 18 + 3 + 3 and 0 <= s["spa"]["p_consistent"] <= 1
    assert len(s["tier_configs"]) == 3 * 5
    prim = s["primary"]
    assert prim["primary_pass"] == (prim["bss_cal"] > 0 and prim["dm_p"] < 0.05)
    base_row = next(r for r in s["judging"] if r["model"] == "base_rate")
    assert base_row["bss_base"] == 0 and base_row["pr_auc"] == pytest.approx(
        base_row["prevalence"], rel=0.05
    )  # base rate differs a little by year
    assert base_row["roc_auc_null"] == 0.5 and base_row["log_loss"] == base_row["log_loss_base"]
    cal = next(r for r in s["judging"] if r["model"] == "formula_cal")
    assert cal["log_loss"] == cal["log_loss_cal"]
    evaluate.reliability_figure(fc, s["chosen_tier"], tmp_path / "rel.png")
    assert (tmp_path / "rel.png").stat().st_size > 1000
