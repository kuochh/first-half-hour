"""Chapter 2 walk-forward: folds, fixed-configuration forecasts, tuning and the tuned tiers.

Input `frame`: one row per kept session, sorted, no nulls, with `session` (Date), `y` (0/1),
`z` (formula z), `r1`, `p_raw` (raw-formula probability) and the feature columns (names passed in).
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

from first_half_hour import config

BASELINES: tuple[str, ...] = ("base_rate", "formula_raw", "formula_cal", "gao_r1")
_UNPENALISED = np.inf  # sklearn >= 1.8: C=inf is the unpenalised fit (penalty=None is deprecated)


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
                ("clf", LogisticRegression(C=_UNPENALISED, max_iter=2000)),
            ]
        )
    if cfg.tier == "enet":
        clf = LogisticRegression(
            C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=5000,
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


def _logit1(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """Unpenalised logistic regression of y on one variable (standardised; same fit, easier solve).

    Private so T1's baselines.fit_calibrated / fit_gao can replace it at merge.
    """
    m = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(C=_UNPENALISED, max_iter=2000))]
    )
    m.fit(x_tr.reshape(-1, 1), y_tr)
    return m.predict_proba(x_te.reshape(-1, 1))[:, 1]


def _check_frame(frame: pl.DataFrame, feats: list[str]) -> None:
    need = ["session", "y", "z", "r1", "p_raw", *feats]
    missing = [c for c in need if c not in frame.columns]
    assert not missing, f"frame lacks columns {missing}"
    assert frame["session"].dtype == pl.Date, frame["session"].dtype
    assert frame["session"].is_sorted() and frame["session"].is_unique().all(), (
        "sessions must be sorted and unique"
    )
    assert frame.select(need).null_count().sum_horizontal().item() == 0, "nulls in the frame"
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
    frame: pl.DataFrame, feats: list[str], years: list[int], tiers: tuple[str, ...] | None = None
) -> dict[tuple[int, str], np.ndarray]:
    """{(year, model): probabilities for that year's rows} for the baselines and configurations."""
    _check_frame(frame, feats)
    out: dict[tuple[int, str], np.ndarray] = {}
    for yr in years:
        train, test = fold(frame, yr)
        y_tr = train["y"].to_numpy()
        out[yr, "base_rate"] = np.full(test.height, y_tr.mean())
        out[yr, "formula_raw"] = test["p_raw"].to_numpy()
        out[yr, "formula_cal"] = _logit1(train["z"].to_numpy(), y_tr, test["z"].to_numpy())
        out[yr, "gao_r1"] = _logit1(train["r1"].to_numpy(), y_tr, test["r1"].to_numpy())
        x_tr, x_te = train.select(feats).to_numpy(), test.select(feats).to_numpy()
        for cfg in configs(tiers):
            out[yr, cfg.name] = build_model(cfg).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
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


def walk_forward(
    frame: pl.DataFrame,
    feats: list[str],
    target: str,
    tiers: tuple[str, ...] | None = None,
    years: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(forecasts, selections). `forecasts` is the long frame of the spec's interface.

    `tiers` limits which tiers are fitted (descriptive targets re-tune the chosen tier only).
    Tuned-tier rows exist for years after the first test year.
    """
    tiers = config.CH2_TIERS if tiers is None else tiers
    years = years_for(frame) if years is None else years
    preds = fixed_forecasts(frame, feats, years, tiers)
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
    return fc.sort("model", "session"), sel


def fit_final(frame: pl.DataFrame, feats: list[str], model_name: str, year: int):
    """The configuration `model_name` fit on all rows before `year`-01-01 (as for that year's forecasts)."""
    train, _ = fold(frame, year)
    return build_model(config_by_name(model_name)).fit(
        train.select(feats).to_numpy(), train["y"].to_numpy()
    )
