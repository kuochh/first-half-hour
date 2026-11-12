"""Chapter 2 baselines: base rate, the volatility-only formula (raw and calibrated), Gao's sign.

The formula treats the log price from 10:00 as a driftless Brownian motion with per-minute
volatility sigma, so P(max over tau minutes >= ln(1 + b)) = 2 * (1 - Phi(z)), z = ln(1+b) /
(sigma * sqrt(tau)) (reflection principle). Fitted baselines return plain callables.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def formula_z(sigma: np.ndarray, tau: np.ndarray, b: float) -> np.ndarray:
    sigma, tau = np.asarray(sigma, dtype=float), np.asarray(tau, dtype=float)
    assert (sigma > 0).all() and (tau > 0).all(), "sigma and tau must be positive"
    return np.log1p(b) / (sigma * np.sqrt(tau))


def formula_raw(sigma: np.ndarray, tau: np.ndarray, b: float) -> np.ndarray:
    """p = 2 * (1 - Phi(z)). No fitting."""
    return 2 * norm.sf(formula_z(sigma, tau, b))


def _logit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    y = np.asarray(y)
    assert set(np.unique(y)) <= {0, 1} and len(np.unique(y)) == 2, "y must hold both 0 and 1"
    assert x.shape[0] == y.shape[0] and np.isfinite(x).all(), "bad regressor"
    return LogisticRegression(penalty=None, max_iter=1000).fit(x, y)


def fit_calibrated(
    sigma: np.ndarray, tau: np.ndarray, y: np.ndarray, b: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Logistic regression of y on z (with an intercept, unpenalised). Returns predict(sigma, tau)."""
    model = _logit(formula_z(sigma, tau, b).reshape(-1, 1), y)

    def predict(sigma: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return model.predict_proba(formula_z(sigma, tau, b).reshape(-1, 1))[:, 1]

    return predict


def fit_gao(r1: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Unpenalised logistic regression of y on r1. Returns predict(r1). Reported, not a candidate."""
    model = _logit(np.asarray(r1, dtype=float).reshape(-1, 1), y)
    return lambda r1: model.predict_proba(np.asarray(r1, dtype=float).reshape(-1, 1))[:, 1]


def base_rate(y: np.ndarray) -> float:
    """Mean of y over the training rows."""
    y = np.asarray(y)
    assert y.size > 0, "no training rows"
    return float(y.mean())
