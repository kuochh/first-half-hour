"""ETF panel: Holm by hand, date clustering, and per-ETF rows against plain OLS fits."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from first_half_hour import config, panel


def test_holm_matches_hand_worked_case():
    # p = (0.01, 0.04, 0.03), m = 3. Sorted: 0.01, 0.03, 0.04.
    # Holm: 3*0.01 = 0.03; max(0.03, 2*0.03) = 0.06; max(0.06, 1*0.04) = 0.06.
    adj = multipletests([0.01, 0.04, 0.03], method="holm")[1]
    assert np.allclose(adj, [0.03, 0.06, 0.06], atol=1e-12)


def _tables(n=1400, seed=3, shock_sd=0.0, symbols=("A", "B")):
    rng = np.random.default_rng(seed)
    days = [date(2011, 1, 3) + timedelta(days=4 * i) for i in range(n)]
    shock = rng.normal(0, shock_sd, n) if shock_sd else np.zeros(n)
    out = {}
    for s in symbols:
        r1 = rng.normal(0, 0.01, n)
        # a date shock common to both ETFs enters r13 AND r1's partner via the same date
        r13 = 0.3 * r1 + shock + rng.normal(0, 0.005, n)
        out[s] = pl.DataFrame({"session": days, "r1": r1 + shock, "r13": r13}).with_columns(
            pl.col("session").cast(pl.Date)
        )
    return out


def test_pooled_cluster_se_is_larger_when_etfs_share_a_date_shock():
    tables = _tables(shock_sd=0.01)
    res = panel.panel_results(tables, 1400, {"g": ("A", "B")})
    pooled = res.filter((pl.col("symbol") == panel.POOLED) & (pl.col("sample") == "FULL")).row(
        0, named=True
    )
    both = pl.concat([t.select("session", "r1", "r13") for t in tables.values()])
    plain = sm.OLS(both["r13"].to_numpy(), sm.add_constant(both["r1"].to_numpy())).fit()
    plain_t = plain.tvalues[1]
    assert abs(pooled["slope"] - plain.params[1]) < 1e-12
    # The two series of each date are positively correlated through the shock, so the
    # effective sample is smaller than 2*1400 and the clustered |t| must be lower.
    assert abs(pooled["t"]) < abs(plain_t)
    assert pooled["n_clusters"] == 1400 and pooled["T"] == 2800


def test_per_etf_rows_equal_individual_ols_and_baselines():
    tables = _tables(n=1400, symbols=("A", "B", "C"))
    res = panel.panel_results(tables, 1400, {"g": ("A", "B", "C")})
    for sym, t in tables.items():
        d = t.filter(pl.col("session") >= config.SAMPLES["FULL"][0])
        lags = int(np.floor(4 * (1400 / 100) ** (2 / 9)))
        fit = sm.OLS(d["r13"].to_numpy(), sm.add_constant(d["r1"].to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        row = res.filter((pl.col("symbol") == sym) & (pl.col("sample") == "FULL")).row(
            0, named=True
        )
        assert abs(row["slope"] - fit.params[1]) < 1e-12
        assert abs(row["t"] - fit.tvalues[1]) < 1e-9
        assert abs(row["r2"] - fit.rsquared) < 1e-12
        assert row["nw_lags"] == lags and row["T"] == 1400
        assert row["baseline_slope"] == 0.0 and row["baseline_r2"] == 0.0
    holm = res.filter((pl.col("sample") == "FULL") & (pl.col("symbol") != panel.POOLED))
    expect = multipletests(holm["p"].to_numpy(), method="holm")[1]
    assert np.allclose(holm["p_holm"].to_numpy(), expect, atol=1e-12)
