"""Chapter 2 interpretability: permutation importance and partial dependence of the HGB tier.

Takes fitted sklearn estimators (one per judging year) and the feature frame; writes figures to a
given folder. Brier is the score, matching the chapter's metric.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import brier_score_loss

from first_half_hour import config

log = logging.getLogger(__name__)


VOLATILITY_GROUP: tuple[str, ...] = (
    "sigma", "z", "abs_r_on", "abs_r_fh", "or_high", "or_low", "rv_day_lag", "vix_lag",
    "vix_ratio_lag",
)  # fmt: skip
"""The volatility features permuted as one block; names absent from the feature list are skipped
(the caller sees which were used in the output)."""


def _check_order(est: object, features: list[str]) -> None:
    assert est.n_features_in_ == len(features), "estimator and feature list differ in length"
    if hasattr(est, "feature_names_in_"):
        assert list(est.feature_names_in_) == features, "feature order differs from the fit"


def _group_importance(
    est: object, x: np.ndarray, y: np.ndarray, cols: list[int], repeats: int, seed: int
) -> np.ndarray:
    """Rise in Brier score when the columns `cols` are shuffled together (one row permutation).

    Own few lines: sklearn permutes one column at a time. Returns one value per repeat.
    """
    base = brier_score_loss(y, est.predict_proba(x)[:, 1])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        xp = x.copy()
        xp[:, cols] = x[rng.permutation(len(x))][:, cols]
        out.append(brier_score_loss(y, est.predict_proba(xp)[:, 1]) - base)
    return np.asarray(out)


def permutation_importance_by_year(
    models: dict[int, object], data: pl.DataFrame, features: list[str],
    group: tuple[str, ...] = VOLATILITY_GROUP,
) -> pl.DataFrame:  # fmt: skip
    """Rise in Brier score when a feature is shuffled, per judging year, then averaged.

    `models` maps year -> fitted estimator (the tuned HGB model that forecast that year). `data`
    has session, year, y and the feature columns. Each year's model is scored on that year's rows
    only. Returns one row per feature plus one row "[group]" for the features of `group` shuffled
    as ONE block (the same row permutation): feature, importance (mean over years and repeats),
    std (sd over all year-repeat values), n_years_positive. Most important first.
    """
    cols = [features.index(f) for f in group if f in features]
    assert cols, "none of the group's features are in the feature list"
    vals = []  # one array (n_features + 1, repeats) per year
    for year, est in sorted(models.items()):
        _check_order(est, features)
        d = data.filter(pl.col("year") == year)
        assert d.height > 0, f"no rows in {year}"
        x, y = d.select(features).to_numpy(), d["y"].to_numpy()
        res = permutation_importance(
            est, x, y, scoring="neg_brier_score", n_repeats=config.PERMUTATION_REPEATS,
            random_state=config.PERMUTATION_SEED,
        )  # fmt: skip
        g = _group_importance(est, x, y, cols, config.PERMUTATION_REPEATS, config.PERMUTATION_SEED)
        vals.append(np.vstack([res.importances, g]))
    allv = np.concatenate(vals, axis=1)  # years x repeats along axis 1
    per_year = np.stack([v.mean(axis=1) for v in vals])
    names = [*features, "[group] " + "+".join(f for f in group if f in features)]
    return pl.DataFrame(
        {"feature": names, "importance": allv.mean(axis=1), "std": allv.std(axis=1, ddof=1),
         "n_years_positive": (per_year > 0).sum(axis=0).astype(int)}
    ).sort("importance", descending=True)  # fmt: skip


def top_other_features(importance: pl.DataFrame, k: int = 3, always: str = "sigma") -> list[str]:
    """Up to k most important single features other than `always`, only with importance > 0.

    Fewer than k are returned when fewer have a positive mean importance (the caller notes it).
    """
    assert always in importance["feature"].to_list(), f"{always} is not a feature"
    top = importance.filter(
        (pl.col("feature") != always) & ~pl.col("feature").str.starts_with("[group]")
        & (pl.col("importance") > 0)
    )  # fmt: skip
    return top["feature"].head(k).to_list()


def write_figures(
    final_model: object, data: pl.DataFrame, features: list[str], importance: pl.DataFrame,
    folder: Path,
) -> list[Path]:  # fmt: skip
    """Importance bar chart and one partial-dependence figure (sigma + up to three others with importance > 0)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance.head(15).reverse()
    ax.barh(top["feature"].to_list(), top["importance"].to_list(), xerr=top["std"].to_list())
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("mean increase in Brier score when shuffled (judging years)")
    fig.tight_layout()
    paths.append(folder / "permutation_importance.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    _check_order(final_model, features)
    others = top_other_features(importance)
    if len(others) < 3:
        log.warning("only %d features beyond sigma have importance > 0", len(others))
    chosen = ["sigma", *others]
    disp = PartialDependenceDisplay.from_estimator(
        final_model, data.select(features).to_numpy(), [features.index(f) for f in chosen],
        feature_names=features, kind="average", response_method="predict_proba",
        method="brute",
    )  # fmt: skip
    disp.figure_.suptitle("Partial dependence of P(hit) on the HGB tier's features")
    disp.figure_.tight_layout()
    paths.append(folder / "partial_dependence.png")
    disp.figure_.savefig(paths[-1], dpi=150)
    plt.close(disp.figure_)
    return paths
