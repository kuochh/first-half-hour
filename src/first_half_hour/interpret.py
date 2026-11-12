"""Chapter 2 interpretability: permutation importance and partial dependence of the HGB tier.

Takes fitted sklearn estimators (one per judging year) and the feature frame; writes figures to a
given folder. Brier is the score, matching the chapter's metric.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from first_half_hour import config


def permutation_importance_by_year(
    models: dict[int, object], data: pl.DataFrame, features: list[str]
) -> pl.DataFrame:
    """Mean drop in negative Brier when a feature is shuffled, per judging year, then averaged.

    `models` maps year -> fitted estimator (the tuned HGB model that forecast that year). `data`
    has session, year, y and the feature columns. Each year's model is scored on that year's rows
    only. Returns one row per feature (feature, importance = mean over years), most important first.
    """
    per_year = []
    for year, est in sorted(models.items()):
        d = data.filter(pl.col("year") == year)
        assert d.height > 0, f"no rows in {year}"
        res = permutation_importance(
            est, d.select(features).to_numpy(), d["y"].to_numpy(),
            scoring="neg_brier_score", n_repeats=config.PERMUTATION_REPEATS,
            random_state=config.PERMUTATION_SEED,
        )  # fmt: skip
        per_year.append(res.importances_mean)
    mean = np.mean(per_year, axis=0)
    return pl.DataFrame({"feature": features, "importance": mean}).sort(
        "importance", descending=True
    )


def top_other_features(importance: pl.DataFrame, k: int = 3, always: str = "sigma") -> list[str]:
    """The k most important features other than `always`."""
    assert always in importance["feature"].to_list(), f"{always} is not a feature"
    return importance.filter(pl.col("feature") != always)["feature"].head(k).to_list()


def write_figures(
    final_model: object, data: pl.DataFrame, features: list[str], importance: pl.DataFrame,
    folder: Path,
) -> list[Path]:  # fmt: skip
    """Importance bar chart and one partial-dependence figure (sigma + top three others)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance.head(15).reverse()
    ax.barh(top["feature"].to_list(), top["importance"].to_list())
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("mean increase in Brier score when shuffled (judging years)")
    fig.tight_layout()
    paths.append(folder / "permutation_importance.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    chosen = ["sigma", *top_other_features(importance)]
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
