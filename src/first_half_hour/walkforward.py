"""Chapter 2 walk-forward: folds, fixed-configuration forecasts, tuning and the tuned tiers.

Input `frame`: one row per kept session, sorted, no nulls or NaN, with `session` (Date), `y` (0/1),
`sigma`, `tau`, `r1` and the feature columns (default `config.FEATURE_COLUMNS`), as `features.features`
joined to `features.target` gives. The baselines are `baselines.py`'s (one implementation).
For test year Y every configuration and baseline is fit on ALL rows with session < Y-01-01 and
predicts year Y. A tier's setting for Y is the configuration with the lowest Brier score on year
Y-1's fixed-configuration forecasts (fit before Y-1, so out of sample there and blind to Y).
"""

from dataclasses import dataclass
from datetime import date
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from first_half_hour import baselines, config

BASELINES: tuple[str, ...] = ("base_rate", "formula_raw", "formula_cal", "gao_r1")
_UNPENALISED = np.inf  # sklearn >= 1.8: C=inf is the unpenalised fit (penalty=None is deprecated)
_UNPENALISED_TOL = 1e-8  # the default 1e-4 stops early: up to 7e-4 off the MLE in probability


@dataclass(frozen=True)
class Cfg:
    tier: str
    params: tuple[tuple[str, float | int], ...]

    @property
    def name(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params) or "none"
        return f"cfg:{self.tier}:{p}"


def configs(tiers: tuple[str, ...] | None = None) -> list[Cfg]:
    """The fixed configurations in the spec's order: logit, enet (C-major), hgb (lr, depth, iter)."""
    out = [Cfg("logit", ())]
    out += [
        Cfg("enet", (("C", c), ("l1_ratio", r)))
        for c, r in product(config.CH2_ENET_C, config.CH2_ENET_L1)
    ]
    out += [
        Cfg("hgb", (("learning_rate", lr), ("max_depth", d), ("max_iter", it)))
        for lr, d, it in product(config.CH2_HGB_LR, config.CH2_HGB_DEPTH, config.CH2_HGB_ITER)
    ]
    return [c for c in out if tiers is None or c.tier in tiers]


def build_model(cfg: Cfg):
    """An unfitted estimator for one configuration."""
    p = dict(cfg.params)
    if cfg.tier == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=_UNPENALISED, tol=_UNPENALISED_TOL, max_iter=config.CH2_MAX_ITER
                    ),
                ),
            ]
        )
    if cfg.tier == "enet":
        clf = LogisticRegression(
            C=p["C"], l1_ratio=p["l1_ratio"], solver="saga",
            max_iter=config.CH2_MAX_ITER,
            random_state=config.CH2_SEED,
        )  # fmt: skip
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    assert cfg.tier == "hgb", cfg.tier
    return HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_depth=p["max_depth"], max_iter=p["max_iter"],
        min_samples_leaf=config.CH2_HGB_MIN_LEAF, early_stopping=False,
        random_state=config.CH2_SEED,
    )  # fmt: skip


def config_by_name(name: str) -> Cfg:
    found = [c for c in configs() if c.name == name]
    assert len(found) == 1, f"no single configuration named {name!r}"
    return found[0]


def fit_model(cfg: Cfg, x: np.ndarray, y: np.ndarray):
    """Fit one configuration; a logistic / elastic-net fit must converge before max_iter."""
    model = build_model(cfg).fit(x, y)
    if cfg.tier != "hgb":
        n_iter = int(model.named_steps["clf"].n_iter_[0])
        assert n_iter < config.CH2_MAX_ITER, f"{cfg.name}: {n_iter} iterations, not converged"
    return model


def _check_frame(frame: pl.DataFrame, feats: list[str]) -> None:
    need = ["session", "y", "sigma", "tau", "r1", *feats]
    missing = [c for c in need if c not in frame.columns]
    assert not missing, f"frame lacks columns {missing}"
    assert frame["session"].dtype == pl.Date, frame["session"].dtype
    assert frame["session"].is_sorted() and frame["session"].is_unique().all(), (
        "sessions must be sorted and unique"
    )
    assert frame.select(need).null_count().sum_horizontal().item() == 0, "nulls in the frame"
    floats = [c for c in need if frame.schema[c].is_float()]
    assert frame.select(pl.col(floats).is_finite().all()).row(0) == (True,) * len(floats), (
        "NaN or infinite values in the frame (polars null_count does not see NaN)"
    )
    assert set(frame["y"].unique().to_list()) <= {0, 1}, "y must be 0/1"


def years_for(frame: pl.DataFrame) -> list[int]:
    return list(range(config.CH2_FIRST_TEST_YEAR, frame["session"].max().year + 1))


def fold(frame: pl.DataFrame, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(train, test): train = every row before Y-01-01, test = rows of year Y."""
    train = frame.filter(pl.col("session") < date(year, 1, 1))
    test = frame.filter(pl.col("session").dt.year() == year)
    assert train.height > 0 and test.height > 0, f"empty fold for {year}"
    assert train["session"].max() < test["session"].min()
    assert train["y"].n_unique() == 2, f"training rows before {year} hold one class only"
    return train, test


def fixed_forecasts(
    frame: pl.DataFrame,
    feats: list[str],
    years: list[int],
    tiers: tuple[str, ...] | None = None,
    b: float = config.TARGET_B,
    side: str = "up",
) -> dict[tuple[int, str], np.ndarray]:
    """{(year, model): probabilities for that year's rows} for the baselines and configurations."""
    _check_frame(frame, feats)
    out: dict[tuple[int, str], np.ndarray] = {}
    for yr in years:
        train, test = fold(frame, yr)
        y_tr = train["y"].to_numpy()
        sg_tr, tau_tr = train["sigma"].to_numpy(), train["tau"].to_numpy()
        sg_te, tau_te = test["sigma"].to_numpy(), test["tau"].to_numpy()
        out[yr, "base_rate"] = np.full(test.height, baselines.base_rate(y_tr))
        out[yr, "formula_raw"] = baselines.formula_raw(sg_te, tau_te, b, side)
        out[yr, "formula_cal"] = baselines.fit_calibrated(sg_tr, tau_tr, y_tr, b, side)(
            sg_te, tau_te
        )
        out[yr, "gao_r1"] = baselines.fit_gao(train["r1"].to_numpy(), y_tr)(test["r1"].to_numpy())
        x_tr, x_te = train.select(feats).to_numpy(), test.select(feats).to_numpy()
        for cfg in configs(tiers):
            out[yr, cfg.name] = fit_model(cfg, x_tr, y_tr).predict_proba(x_te)[:, 1]
    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def tune(
    frame: pl.DataFrame,
    preds: dict[tuple[int, str], np.ndarray],
    years: list[int],
    tiers: tuple[str, ...],
) -> pl.DataFrame:
    """For each year Y >= first+1 and tier: the configuration with the lowest Brier on year Y-1.

    Reads only year Y-1's forecasts and labels. Ties go to the first configuration listed.
    """
    rows = []
    for yr in years[1:]:
        y_prev = frame.filter(pl.col("session").dt.year() == yr - 1)["y"].to_numpy()
        for tier in tiers:
            cands = configs((tier,))
            scores = [_brier(y_prev, preds[yr - 1, c.name]) for c in cands]
            rows.append({"year": yr, "tier": tier, "model": cands[int(np.argmin(scores))].name})
    return pl.DataFrame(rows, schema={"year": pl.Int32, "tier": pl.Utf8, "model": pl.Utf8})


def expected_models(tiers: tuple[str, ...]) -> set[str]:
    """Model names of one target: 4 baselines + the tiers' configurations + one tuned row per tier."""
    return {*BASELINES, *(c.name for c in configs(tiers)), *(f"tier:{t}" for t in tiers)}


def walk_forward(
    frame: pl.DataFrame,
    target: str,
    feats: list[str] | None = None,
    tiers: tuple[str, ...] | None = None,
    years: list[int] | None = None,
    b: float = config.TARGET_B,
    side: str = "up",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(forecasts, selections). `forecasts` is the long frame of the spec's interface;
    `selections` is the per-year tuned configuration per tier (year, tier, model, target).

    `tiers` limits which tiers are fitted (descriptive targets re-tune the chosen tier only).
    Tuned-tier rows exist for years after the first test year. The model set is asserted:
    primary 4 + 18 + 3 = 25, a descriptive target 4 + the tier's configurations + 1.
    """
    tiers = config.CH2_TIERS if tiers is None else tiers
    years = years_for(frame) if years is None else years
    feats = list(config.FEATURE_COLUMNS) if feats is None else feats
    preds = fixed_forecasts(frame, feats, years, tiers, b, side)
    sel = tune(frame, preds, years, tiers)
    for r in sel.iter_rows(named=True):
        preds[r["year"], f"tier:{r['tier']}"] = preds[r["year"], r["model"]]
    parts = []
    for (yr, model), p in preds.items():
        test = frame.filter(pl.col("session").dt.year() == yr)
        parts.append(
            pl.DataFrame({"session": test["session"], "y": test["y"], "p": p}).with_columns(
                year=pl.lit(yr, pl.Int32), target=pl.lit(target), model=pl.lit(model)
            )
        )
    fc = pl.concat(parts).select(
        "session",
        "year",
        "target",
        pl.col("y").cast(pl.Int8),
        "model",
        pl.col("p").cast(pl.Float64),
    )
    assert fc.group_by("session", "model").len()["len"].max() == 1
    if len(years) > 1:
        assert set(fc["model"].unique()) == expected_models(tiers), "unexpected model set"
    return fc.sort("model", "session"), sel.with_columns(target=pl.lit(target))


def fit_final(frame: pl.DataFrame, feats: list[str], model_name: str, year: int):
    """The configuration `model_name` fit on all rows before `year`-01-01 (as for that year's forecasts)."""
    train, _ = fold(frame, year)
    return fit_model(
        config_by_name(model_name), train.select(feats).to_numpy(), train["y"].to_numpy()
    )
