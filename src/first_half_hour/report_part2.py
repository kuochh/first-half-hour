"""Part 2 figures and the two-language report (a per-day classifier that picks the days to follow).

Reads `<out>/part2/*.parquet` (quintiles, rules, grid, groups, picked, per_year and, if present,
cumulative) and `<out>/part2_summary.json` (all written by `part2.write_outputs`, the summary
completed by `reproduce._part2`); writes four figures under `<out>/figures/` and
`<out>/part2_report.html` and `<out>/part2_report_zh.html`. Nothing is recomputed: both pages lay
out the same numbers, and a code label with no entry here (a feature, a rule, a split) fails an
assertion instead of showing raw. Return names are looked up in `report_part1.RET`.

The cumulative figure needs the per-day wealth series (`cumulative.parquet`: session, rule, wealth;
rules model / always_follow / big_move). It is private (D7) and used only by the figure.
"""

import base64
import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from first_half_hour import config
from first_half_hour.report import CSS
from first_half_hour.report_part1 import _lab

TAU, C_MAIN = config.PART2_REPORT_HEADLINE
MODELS = ("logit", "hgb")
SPLITS = ("all", "pre_2020_05", "post_2020_05", "no_covid")
COSTS = (C_MAIN, 0, 4)  # evaluation costs shown, main first (bp round trip)
FRAMES = ("quintiles", "rules", "grid", "groups", "picked", "per_year")
RULE_NAMES = ("model", "big_move", "vol_only")

MODEL = {
    "logit": ("logistic regression", "邏輯迴歸"),
    "hgb": ("boosted trees", "梯度提升樹"),
}
RULE = {  # a "{q}" is the share of days
    "model": ("model, top {q}% of days", "模型，分數最高的 {q}% 的日子"),
    "always_follow": ("always follow the signal", "一律跟著訊號做"),
    "big_move": ("big-move rule, {q}% of days", "大幅移動規則，{q}% 的日子"),
    "vol_only": ("volatility-only model, {q}% of days", "只用波動的模型，{q}% 的日子"),
}
SPLIT = {
    "all": ("all test days", "全部測試日"),
    "pre_2020_05": ("2014-01-02 to 2020-05-01", "2014-01-02 到 2020-05-01"),
    "post_2020_05": ("2020-05-04 to 2026-08-31", "2020-05-04 到 2026-08-31"),
    "no_covid": ("all test days except 2020-02-01 to 2020-04-30", "全部測試日，但不含 2020-02-01 到 2020-04-30"),
}  # fmt: skip
VERDICT = {
    "amplified": ("amplified", "放得大"),
    "partly amplified": ("partly amplified", "部分放大"),
    "not amplified": ("not amplified", "放不大"),
}
FEAT = {  # features, their groups, and the input-group steps (a leading "+" is stripped)
    "abs_s": ("size of the signal", "訊號的大小"),
    "s_rank60": ("rank of that size over the last 60 sessions", "這個大小在過去 60 個交易日中的排名"),
    "on_aligned": ("previous close to the open, in the signal's direction", "前一天收盤到開盤，朝訊號方向"),
    "intra_aligned": ("open to the decision time, in the signal's direction", "開盤到決策時間，朝訊號方向"),
    "d_long": ("signal is up", "訊號為正"),
    "rv_sofar": ("volatility so far today", "今天到目前為止的波動"),
    "rv_fh": ("09:30 to 10:00 volatility", "09:30 到 10:00 的波動"),
    "vix_lag": ("VIX, previous session", "前一個交易日的 VIX"),
    "range_prev": ("high-low range, previous session", "前一個交易日的高低價差"),
    "vol_rel": ("volume so far relative to the last 20 sessions", "到目前為止的成交量相對過去 20 個交易日"),
    "fomc": ("FOMC day", "FOMC 日"),
    "cpi": ("CPI day", "CPI 日"),
    "payroll": ("payroll day", "非農就業日"),
    "opex_monthly": ("monthly options expiry", "月選擇權到期日"),
    "month_end": ("month end", "月底"),
    "quarter_end": ("quarter end", "季底"),
    "dow_tue": ("Tuesday", "星期二"), "dow_wed": ("Wednesday", "星期三"),
    "dow_thu": ("Thursday", "星期四"), "dow_fri": ("Friday", "星期五"),
    "last30_aligned": ("last 30 minutes before the decision, in the signal's direction", "決策前最後 30 分鐘，朝訊號方向"),
    "last60_share": ("share of the move made in the last 60 minutes", "最後 60 分鐘完成的走勢占比"),
    "hit20": ("hit rate of the signal over the last 20 sessions", "訊號在過去 20 個交易日的命中率"),
    "yest_s_aligned": ("previous session's signal, in today's direction", "前一個交易日的訊號，朝今天的方向"),
    "yest_r13_aligned": ("previous session's 15:30 to close, in today's direction", "前一個交易日 15:30 到收盤，朝今天的方向"),
    "signal": ("signal", "訊號"), "volatility": ("volatility", "波動"), "vol": ("volatility", "波動"),
    "volume": ("volume", "成交量"), "calendar": ("calendar", "日曆"), "path": ("path", "走勢"),
}  # fmt: skip


# ---------------------------------------------------------------- formats
def _bp(v: float) -> str:
    assert v is not None and np.isfinite(v), f"missing number {v!r}"
    text = f"{v * 1e4:.1f}"
    return "0.0" if text == "-0.0" else text


def _pm(v: float, se: float, kind: str = "bp") -> str:
    """`value ± 2 SE` in bp, or a percent (`pct`) or plain (`plain`, Sharpe)."""
    assert se is not None and np.isfinite(se), f"missing SE {se!r}"
    if kind == "bp":
        return f"{_bp(v)} ± {_bp(2 * se)}"
    if kind == "pct":
        return f"{v * 100:.1f}% ± {2 * se * 100:.1f}"
    return f"{v:.2f} ± {2 * se:.2f}"


def _pct(share: float) -> str:
    assert share is not None and np.isfinite(share) and 0 <= share <= 1, f"bad share {share!r}"
    return f"{share * 100:.1f}%"


def _sig3(v: float) -> str:
    """3 significant figures, never scientific notation."""
    assert v is not None and np.isfinite(v), f"missing number {v!r}"
    if v == 0:
        return "0"
    dec = max(0, 2 - int(np.floor(np.log10(abs(v)))))
    text = f"{v:.{dec}f}"
    return "0" if float(text) == 0 else text


def _q(q: float) -> str:
    return f"{q * 100:.0f}"


# ---------------------------------------------------------------- figures
def _cell(df: pl.DataFrame, model: str, c: float = C_MAIN, tau: str = TAU) -> pl.DataFrame:
    sub = df.filter((pl.col("tau") == tau) & (pl.col("c_bp") == c) & (pl.col("model") == model))
    assert sub.height, f"no rows for {tau}, {c} bp, {model}"
    return sub


def plot_quintiles(quintiles: pl.DataFrame, path: Path, model: str) -> None:
    """Five bars of mean net per trade at 2 bp, +/- 2 SE, always-follow as a line."""
    q = _cell(quintiles, model).sort("group")
    assert q["group"].to_list() == [1, 2, 3, 4, 5], "quintile groups are not 1..5"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(q["group"].to_numpy(), q["mean_net"].to_numpy() * 1e4,
           yerr=2 * q["se"].to_numpy() * 1e4, capsize=4, color="C0",
           label="mean net per trade, +/- 2 SE")  # fmt: skip
    af = q["af_mean_net"].unique().to_list()
    assert len(af) == 1, "always-follow reference differs across groups"
    ax.axhline(af[0] * 1e4, color="firebrick", ls="--", label="always follow")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("group of test days by model score (1 = lowest)")
    ax.set_ylabel(f"net per trade after {C_MAIN} bp (bp)")
    ax.set_title(f"SPY, {TAU}: five groups of days by model score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cumulative(cum: pl.DataFrame, path: Path) -> None:
    """Wealth from 1 over the test days: top-20% rule, always follow, big-move rule, at 2 bp."""
    names = {
        "model": f"model, top {_q(config.PART2_REPORT_Q)}% of days",
        "always_follow": "always follow the signal",
        "big_move": f"big-move rule, {_q(config.PART2_REPORT_Q)}% of days",
    }
    assert set(cum["rule"].unique()) == set(names), (
        f"cumulative rules differ: {cum['rule'].unique()}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (rule,), s in cum.sort("session").group_by("rule", maintain_order=True):
        ax.plot(s["session"].to_list(), s["wealth"].to_numpy(), lw=1.2, label=names[rule])
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel("wealth (start = 1)")
    ax.set_title(f"SPY, {TAU}: cumulative wealth after {C_MAIN} bp, test days")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _times_series(grid: pl.DataFrame, rules: pl.DataFrame, model: str) -> dict[str, pl.DataFrame]:
    """name -> frame(tau, v, se): the top-20% rule (both models) and the two baselines minus always-follow."""
    out = {m: grid.filter((pl.col("c_bp") == C_MAIN) & (pl.col("model") == m)).sort("tau")
           .select("tau", v=pl.col("top20_minus_af"), se=pl.col("se")) for m in MODELS}  # fmt: skip
    for rule in ("big_move", "vol_only"):
        r = rules.filter(
            (pl.col("rule") == rule) & (pl.col("c_bp") == C_MAIN) & (pl.col("c_eval_bp") == C_MAIN)
            & (pl.col("model") == model) & (pl.col("split") == "all")
            & ((pl.col("q") - config.PART2_REPORT_Q).abs() < 1e-9)
        ).sort("tau")  # fmt: skip
        out[rule] = r.select("tau", v=pl.col("diff_vs_af"), se=pl.col("diff_vs_af_se"))
    return out


def plot_times(grid: pl.DataFrame, rules: pl.DataFrame, path: Path, model: str) -> None:
    series = _times_series(grid, rules, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = {"logit": MODEL["logit"][0], "hgb": MODEL["hgb"][0],
              "big_move": "big-move rule", "vol_only": "volatility-only model"}  # fmt: skip
    for i, (name, s) in enumerate(series.items()):
        assert s.height, f"no rows for {name} in the times figure"
        x = np.arange(s.height) + (i - 1.5) * 0.12
        ax.errorbar(x, s["v"].to_numpy() * 1e4, yerr=2 * s["se"].to_numpy() * 1e4, fmt="o",
                    capsize=3, label=labels[name])  # fmt: skip
    taus = series["logit"]["tau"].to_list()
    ax.set_xticks(np.arange(len(taus)), taus)
    ax.axhline(0, color="black", lw=0.8, label="always follow (zero)")
    ax.set_xlabel("decision time")
    ax.set_ylabel(f"top {_q(config.PART2_REPORT_Q)}% of days minus always follow (bp)")
    ax.set_title(
        f"Top-{_q(config.PART2_REPORT_Q)}% days against always follow, by decision time (after {C_MAIN} bp)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(grid: pl.DataFrame, path: Path) -> None:
    """Heat-table, one panel per model: decision time x cost, each cell with value and SE (bp)."""
    taus = sorted(grid["tau"].unique().to_list())
    costs = sorted(grid["c_bp"].unique().to_list())
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        g = grid.filter(pl.col("model") == model)
        assert g.height == len(taus) * len(costs), f"grid for {model} is not complete"
        z, se = np.zeros((len(taus), len(costs))), np.zeros((len(taus), len(costs)))
        for i, t in enumerate(taus):
            for j, c in enumerate(costs):
                r = g.filter((pl.col("tau") == t) & (pl.col("c_bp") == c)).row(0, named=True)
                z[i, j], se[i, j] = r["top20_minus_af"] * 1e4, r["se"] * 1e4
        lim = max(abs(z).max(), 1e-9)
        for i, j in np.ndindex(z.shape):
            ax.text(j, i, f"{z[i, j]:.1f}\n(SE {se[i, j]:.1f})", ha="center", va="center",
                    fontsize=8, color="white" if abs(z[i, j]) > 0.6 * lim else "black")  # fmt: skip
        ax.imshow(z, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(costs)), [f"{c:g} bp" for c in costs])
        ax.set_yticks(range(len(taus)), taus)
        ax.set_xlabel("threshold of the target, round trip")
        ax.set_title(MODEL[model][0])
    axes[0].set_ylabel("decision time")
    fig.suptitle(
        f"Top-{_q(config.PART2_REPORT_Q)}% days minus always follow (bp, after {C_MAIN} bp): value and SE per cell"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(out: Path) -> None:
    """Draw the four figures from `<out>/part2/` into `<out>/figures/`; the cumulative one only if its series exists."""
    d, fig = out / "part2", out / "figures"
    s = json.loads((out / "part2_summary.json").read_text())
    model = s["headline_model"]
    assert model in MODELS, f"unknown headline model {model!r}"
    f = {k: pl.read_parquet(d / f"{k}.parquet") for k in ("quintiles", "rules", "grid")}
    plot_quintiles(f["quintiles"], fig / "part2_quintiles.png", model)
    plot_times(f["grid"], f["rules"], fig / "part2_times.png", model)
    plot_grid(f["grid"], fig / "part2_grid.png")
    if (d / "cumulative.parquet").exists():
        plot_cumulative(pl.read_parquet(d / "cumulative.parquet"), fig / "part2_cumulative.png")


# ---------------------------------------------------------------- text
TEXT = {
    "en": {
        "lang": "en",
        "title": "Part 2: can a per-day classifier pick the days when the signal predicts 15:30 to close?",
        "h1": "Part 2: following the signal from 15:30 to close, only on the days a model picks ({kind})",
        "synthetic": "SYNTHETIC data (a pipeline check, not a result)",
        "warn": "Data window {start} to {end}. Test days 2014-01-02 to 2026-08-31 (half-days left out). Not yet fact-checked.",
        "count": "<p><b>{count_line}</b></p>",
        "verdict": "Verdict at the headline cell (SPY, {tau}, {c} bp round trip, {model}): <b>{v}</b>.",
        "numbers": (
            "The top {q}% of days, by the model, earn {d_af:.1f} paired standard errors more per trade than "
            "always following, {d_bm:.1f} more than the big-move rule, and {d_0:.1f} above zero "
            "(the model traded {cov:.1f}% of test days). The verdict rule: amplified = d_af, d_bm and d_0 "
            "all above 2; partly amplified = d_af above 2 but not all three; not amplified = d_af at or "
            "below 2. All rules use the same past-only cut-off method; realised shares of days traded are shown."
        ),
        "groups": (
            "The five groups of days are descriptive: they are ranked on the whole test period, so they "
            "cannot be traded. Top group minus bottom group: {tb:.1f} bp (+/- 2 SE {tb2:.1f})."
        ),
        "other": (
            "The other model ({model}): d_af {d_af:.1f}, d_bm {d_bm:.1f}, d_0 {d_0:.1f} "
            "(share of test days traded {cov:.1f}%)."
        ),
        "fragile": (
            "How firm is d_af? 2014 to 2020-05-01: {pre:.2f}; 2020-05-04 on: {post:.2f}; all test days "
            "without 2020-02 to 2020-04: {nocovid:.2f}; with the expanding cut-off pool: {exp:.2f}; "
            "the other model: {other:.2f}. Only the full period with the rolling pool is above 2."
        ),
        "skillline": (
            "Out-of-sample AUC at the headline cell: {model1} {auc1:.3f}, {model2} {auc2:.3f} (0.5 = no skill). "
            "Log loss: {model1} {ll1:.4f}, {model2} {ll2:.4f}, against {base:.4f} for a constant forecast of "
            "the training-span base rate."
        ),
        "fits": "Model fits behind this page: {n_fits} across {n_cells} cells (each cell is a decision time, a threshold, a model and a feature set).",
        "sec": "{n}. {title}",
        "t1": "Five groups of days by model score",
        "x1": (
            "Test days split into fifths by the model's score (headline cell). Mean net payoff per trade of "
            "following the signal from {tau} to the close, after {c} bp round trip, in basis points; "
            "+/- 2 bootstrap standard errors. The dashed line is always following."
        ),
        "t2": "The rule against its baselines",
        "x2": (
            "Headline cell (target threshold {c} bp), all test days. Mean net per trade in basis points, "
            "annualised Sharpe of the daily series (zero on days not traded), hit rate (net above zero on "
            "traded days), each +/- 2 standard errors from a day-resampling bootstrap that recomputes the "
            "traded days. The last column is the rule minus always following, paired in the same draws. "
            "All rules use the same past-only cut-off method; realised shares are shown in the share "
            "column. Cost rows: {costs} bp round trip. The target threshold of 0 bp (= sign agreement) is "
            "in the grid below."
        ),
        "t3": "Splits and days picked per year",
        "x3": "The headline rule on three sub-periods of the test days, with the share of days it traded.",
        "t4": "Cumulative wealth",
        "t5": "By decision time and by cell",
        "x5": (
            "Top-{q}% days minus always following, after {c} bp, by decision time (left to right: "
            "the signal measured from the previous close to each time). Every cell of the grid follows: "
            "value and standard error (SE) in basis points, realised share of test days traded, "
            "out-of-sample AUC (0.5 = no skill), and test log loss with the log loss of a constant "
            "base-rate forecast in brackets. Rows use the target threshold shown in the column heading."
        ),
        "t6": "The picked days",
        "x6": (
            "Headline rule, days it picks against days it does not: the {k} features with the largest "
            "standardised difference (difference of means over the standard deviation)."
        ),
        "t7": "Which inputs matter",
        "x7": (
            "Input groups added one at a time at the headline cell: out-of-sample AUC and the top-{q}% mean "
            "net per trade (basis points, +/- 2 standard errors)."
        ),
        "skipped": (
            "The input-group comparison was skipped because the headline cell has no predictive power "
            "(the top {q}% of days do not beat always following by more than 2 standard errors)."
        ),
        "t8": "Notes",
        "notes": [
            "Half-days (early closes) are left out at every decision time. A session with no price at the decision time is dropped for that time only.",
            "Sign agreement is the threshold 0 target: the signal's sign equals the sign of the 15:30 to close return. It is one cell, reported under both names.",
            "The signal at each decision time is the return from the previous close to that time: previous close to 10:00, previous close to 12:00, previous close to 14:00, previous close to 15:00, previous close to 15:30. The payoff is always the 15:30 to close return in the signal's direction (zero signal counts as up).",
            "Model settings (one penalty strength for the logistic regression, a depth and tree count for boosted trees) are chosen each test year by log loss on the year before, then refit on all earlier years. Training and validation always end before the test year.",
            "The top-share cut-off at a day is the (1 - share) quantile of the most recent {pool} out-of-sample scores of that cell on earlier days (a rolling window). The big-move cut-off is the same quantile of the size of the signal over the {pool} sessions before the day. All rules use the same past-only cut-off method; realised shares are shown.",
            "The model (logistic regression or boosted trees) shown as the headline is the one with the lower log loss on the validation forecasts for the year before the first test year; no test-period label is read.",
            "AUC alarm: the run stops if any cell's out-of-sample AUC reaches 0.75, which would point to a leak.",
            "Costs are per trade, round trip, subtracted from each traded day; the threshold of the target and the evaluation cost are separate settings, equal in the main table.",
        ],
        "cols": {
            "rule": "rule", "cost": "cost (bp)", "trades": "trades", "mean": "net per trade (bp)",
            "sharpe": "Sharpe", "hit": "hit rate", "diff": "minus always follow (bp)",
            "cov": "share of days traded",
            "split": "period", "year": "year", "picked": "days picked", "days": "test days", "ypick": "net per picked trade (bp)", "yaf": "always follow, net per trade (bp)",
            "feature": "feature", "mp": "mean on picked days", "mo": "mean on other days",
            "sd": "standardised difference", "step": "inputs", "added": "added", "auc": "AUC (0.5 = no skill)",
            "top": "top-{q}% net per trade (bp)", "tau": "decision time", "cell": "{c}: value (SE)", "ccov": "{c}: share of days",
            "cauc": "{c}: AUC (0.5 = no skill)", "cll": "{c}: log loss (constant)",
            "c0": "0 bp (= sign agreement)", "cn": "{c} bp",
        },
        "alt": {"q": "Mean net per trade by group of days", "c": "Cumulative wealth",
                "t": "Top days minus always follow by decision time", "g": "Value and SE in every cell"},
        "nocum": "The cumulative series was not available for this run.",
        "expanding": (
            "Sensitivity, expanding cut-off pools (all earlier out-of-sample scores; all earlier sessions "
            "for the big-move rule), headline cell: logistic regression minus always follow "
            "{diff_af_logit:.1f} ± {se_af_logit:.1f} (2 SE) bp, minus big-move {diff_bm_logit:.1f} ± {se_bm_logit:.1f} bp, "
            "share of days traded {coverage_logit:.1f}%; boosted trees {diff_af_hgb:.1f} ± {se_af_hgb:.1f} bp, "
            "{diff_bm_hgb:.1f} ± {se_bm_hgb:.1f} bp, {coverage_hgb:.1f}%."
        ),
    },
    "zh": {
        "lang": "zh-Hant-TW",
        "title": "第二部分：逐日分類器能挑出訊號預測 15:30 到收盤的日子嗎？",
        "h1": "第二部分：只在模型挑出的日子，順著訊號做 15:30 到收盤（{kind}）",
        "synthetic": "合成資料（只檢查流程，不是結果）",
        "warn": "資料期間 {start} 到 {end}。測試日 2014-01-02 到 2026-08-31（不含提早收盤日）。尚未查核。",
        "count": "<p><b>{count_line}</b></p>",
        "verdict": "主要格子（SPY、{tau}、來回成本 {c} bp、{model}）的判定：<b>{v}</b>。",
        "numbers": (
            "模型分數最高的 {q}% 的日子，每筆比一律跟著做多 {d_af:.1f} 個配對標準誤，比大幅移動規則多 "
            "{d_bm:.1f} 個標準誤，高於零 {d_0:.1f} 個標準誤（模型交易了 {cov:.1f}% 的測試日）。判定規則：放得大＝d_af、d_bm、d_0 "
            "三者都大於 2；部分放大＝d_af 大於 2，但不是三者都大於 2；放不大＝d_af 小於或等於 2。"
            "所有規則都用同樣只看過去的門檻方法；實際交易日比例一併列出。"
        ),
        "groups": (
            "五組日子只是描述：它們依整段測試期間的分數排序，所以不能拿來交易。最高組減最低組："
            "{tb:.1f} bp（± 2 個標準誤 {tb2:.1f}）。"
        ),
        "other": (
            "另一個模型（{model}）：d_af {d_af:.1f}、d_bm {d_bm:.1f}、d_0 {d_0:.1f}（交易了 {cov:.1f}% 的測試日）。"
        ),
        "fragile": (
            "d_af 有多穩？2014 到 2020-05-01：{pre:.2f}；2020-05-04 之後：{post:.2f}；全部測試日但拿掉 2020 年 2 到 4 月："
            "{nocovid:.2f}；改用一路累積的門檻資料：{exp:.2f}；另一個模型：{other:.2f}。只有「全期、滾動門檻」超過 2。"
        ),
        "skillline": (
            "主要格子的樣本外 AUC：{model1} {auc1:.3f}、{model2} {auc2:.3f}（0.5＝沒有預測力）。"
            "對數損失：{model1} {ll1:.4f}、{model2} {ll2:.4f}；固定用訓練期間基準比例的預測為 {base:.4f}。"
        ),
        "fits": "這一頁背後的模型擬合：{n_fits} 次，共 {n_cells} 個格子（每個格子是一個決策時間、一個門檻、一個模型和一組輸入）。",
        "sec": "{n}. {title}",
        "t1": "依模型分數分成五組的日子",
        "x1": (
            "測試日依模型分數分成五等分（主要格子）。從 {tau} 順著訊號做到收盤，扣除來回 {c} bp 成本後每筆的平均淨報酬，"
            "單位 bp；± 2 個自助法標準誤。虛線是一律跟著做。"
        ),
        "t2": "規則與基準的比較",
        "x2": (
            "主要格子（目標門檻 {c} bp）、全部測試日。每筆平均淨報酬（bp）、日報酬序列的年化 Sharpe（沒交易的日子算 0）、"
            "命中率（有交易的日子淨報酬大於 0），各附 ± 2 個標準誤（以日為單位重抽樣，每次重抽都重新決定交易日）。"
            "最後一欄是規則減一律跟著做，在同樣的重抽中配對。所有規則都用同樣只看過去的門檻方法；"
            "實際交易日比例列在比例欄。成本列：來回 {costs} bp。門檻 0 bp（＝方向相同）見下方的格子表。"
        ),
        "t3": "分期結果與每年挑出的日子",
        "x3": "主要規則在測試日的三個子期間，並列出它交易的日子比例。",
        "t4": "累積財富",
        "t5": "依決策時間與格子",
        "x5": (
            "前 {q}% 的日子減一律跟著做，扣除 {c} bp 後，依決策時間（訊號是前一天收盤到各個時間）。"
            "下表是格子中的每一格：數值與標準誤（SE），單位 bp；實際交易的測試日比例；樣本外 AUC（0.5＝沒有預測力）；"
            "測試對數損失，括號內是固定用基準比例預測的對數損失。各欄依標題的目標門檻。"
        ),
        "t6": "被挑出的日子",
        "x6": "主要規則挑出的日子與沒挑出的日子相比，標準化差距最大的 {k} 個特徵（平均數的差除以標準差）。",
        "t7": "哪些輸入有用",
        "x7": "在主要格子逐組加入輸入：樣本外 AUC 與前 {q}% 的每筆平均淨報酬（bp，± 2 個標準誤）。",
        "skipped": "輸入群組的比較被略過，因為主要格子沒有預測力（前 {q}% 的日子沒有比一律跟著做多出 2 個標準誤以上）。",
        "t8": "附註",
        "notes": [
            "每個決策時間都排除提早收盤日。某個決策時間沒有價格的交易日，只在該時間被排除。",
            "正負號一致＝門檻 0 的目標：訊號的正負號等於 15:30 到收盤報酬的正負號。這是同一個格子，兩個名稱都報告。",
            "各決策時間的訊號是前一天收盤到該時間的報酬：前一天收盤到 10:00、前一天收盤到 12:00、前一天收盤到 14:00、前一天收盤到 15:00、前一天收盤到 15:30。報酬一律是 15:30 到收盤，順著訊號的方向（訊號為零算做多）。",
            "模型設定（邏輯迴歸的懲罰強度，梯度提升樹的深度與樹數）每個測試年用前一年的對數損失挑選，再用之前所有年份重新擬合。訓練與驗證一律在測試年之前結束。",
            "某一天的前幾名門檻，取自該格子更早日子最近 {pool} 個樣本外分數的 (1 - 比例) 分位數（滾動視窗）。大幅移動規則的門檻是這天之前 {pool} 個交易日訊號大小的同一個分位數。所有規則都用同樣只看過去的門檻方法；實際交易比例一併列出。",
            "標示為主要的模型（邏輯迴歸或梯度提升樹），是用第一個測試年之前一年的驗證預測、對數損失較低的那個；不讀任何測試期間的標籤。",
            "AUC 警報：任何格子的樣本外 AUC 達到 0.75 就停止，因為那代表有洩漏。",
            "成本按每筆交易、來回，從每個有交易的日子扣除；目標的門檻與評估成本是兩個獨立設定，主表中相等。",
        ],
        "cols": {
            "rule": "規則", "cost": "成本（bp）", "trades": "交易筆數", "mean": "每筆淨報酬（bp）",
            "sharpe": "Sharpe", "hit": "命中率", "diff": "減一律跟著做（bp）",
            "cov": "交易日比例",
            "split": "期間", "year": "年", "picked": "挑出的日子", "days": "測試日", "ypick": "挑出的日子每筆淨報酬（bp）", "yaf": "一律跟著做每筆淨報酬（bp）",
            "feature": "特徵", "mp": "挑出日的平均", "mo": "其他日的平均",
            "sd": "標準化差距", "step": "輸入", "added": "新增", "auc": "AUC（0.5＝沒有預測力）",
            "top": "前 {q}% 每筆淨報酬（bp）", "tau": "決策時間", "cell": "{c}：數值（SE）", "ccov": "{c}：日子比例",
            "cauc": "{c}：AUC（0.5＝沒有預測力）", "cll": "{c}：對數損失（固定預測）",
            "c0": "0 bp（＝方向相同）", "cn": "{c} bp",
        },
        "alt": {"q": "各組日子的每筆平均淨報酬", "c": "累積財富",
                "t": "依決策時間，前幾名日子減一律跟著做", "g": "每個格子的數值與 SE"},
        "nocum": "這次執行沒有累積序列可用。",
        "expanding": (
            "敏感度檢查，門檻改用擴展式（所有更早的樣本外分數；大幅移動規則用所有更早的交易日），主要格子："
            "邏輯迴歸減一律跟著做 {diff_af_logit:.1f} ± {se_af_logit:.1f}（2 個標準誤）bp，減大幅移動規則 "
            "{diff_bm_logit:.1f} ± {se_bm_logit:.1f} bp，交易日比例 {coverage_logit:.1f}%；梯度提升樹 "
            "{diff_af_hgb:.1f} ± {se_af_hgb:.1f} bp、{diff_bm_hgb:.1f} ± {se_bm_hgb:.1f} bp、{coverage_hgb:.1f}%。"
        ),
    },
}  # fmt: skip


def _table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="t" style="overflow-x:auto"><table><tr>{h}</tr>{b}</table></div>'


def _feat_list(v, lang: str) -> str:
    items = (
        v if isinstance(v, list) else [x for x in str(v).replace("+", ",").split(",") if x.strip()]
    )
    return ", ".join(_lab(FEAT, x.strip(), lang, "feature") for x in items)


def _rule_label(rule: str, q: float | None, lang: str) -> str:
    return _lab(RULE, rule, lang, "rule").format(q="" if q is None else _q(q))


def _rules_table(rules: pl.DataFrame, model: str, lang: str) -> str:
    c = TEXT[lang]["cols"]
    base = rules.filter(
        (pl.col("tau") == TAU) & (pl.col("c_bp") == C_MAIN) & (pl.col("model") == model)
        & (pl.col("split") == "all")
    )  # fmt: skip
    assert base.height, "no headline rule rows"
    qs = sorted(base.filter(pl.col("rule") == "model")["q"].unique().to_list())
    rows = []
    for ce in COSTS:
        at = base.filter(pl.col("c_eval_bp") == ce)
        af = at.filter(pl.col("rule") == "always_follow")
        assert af.height and af["mean_net"].round(12).n_unique() == 1, (
            f"always-follow rows differ at {ce} bp"
        )
        lines = [("always_follow", None, af.row(0, named=True))]
        for q in qs:
            for rule in RULE_NAMES:
                r = at.filter((pl.col("rule") == rule) & ((pl.col("q") - q).abs() < 1e-9))
                assert r.height == 1, f"expected one {rule} row at q={q}, {ce} bp, got {r.height}"
                lines.append((rule, q, r.row(0, named=True)))
        for rule, q, r in lines:
            diff = "" if rule == "always_follow" else _pm(r["diff_vs_af"], r["diff_vs_af_se"])
            rows.append([
                _rule_label(rule, q, lang), f"{ce:g}", str(r["n_trades"]), _pct(r["coverage"]),
                _pm(r["mean_net"], r["se"]),
                _pm(r["sharpe"], r["sharpe_se"], "plain"), _pm(r["hit"], r["hit_se"], "pct"), diff,
            ])  # fmt: skip
    head = [c[k] for k in ("rule", "cost", "trades", "cov", "mean", "sharpe", "hit", "diff")]
    return _table(head, rows)


def _splits_table(rules: pl.DataFrame, model: str, lang: str) -> str:
    c = TEXT[lang]["cols"]
    r = rules.filter(
        (pl.col("tau") == TAU) & (pl.col("c_bp") == C_MAIN) & (pl.col("model") == model)
        & (pl.col("rule") == "model") & (pl.col("c_eval_bp") == C_MAIN)
        & ((pl.col("q") - config.PART2_REPORT_Q).abs() < 1e-9)
    )  # fmt: skip
    rows = []
    for sp in SPLITS:
        x = r.filter(pl.col("split") == sp)
        assert x.height == 1, f"expected one headline row for split {sp!r}, got {x.height}"
        x = x.row(0, named=True)
        rows.append([_lab(SPLIT, sp, lang, "split"), str(x["n_trades"]), _pct(x["coverage"]),
                     _pm(x["mean_net"], x["se"]), _pm(x["diff_vs_af"], x["diff_vs_af_se"])])  # fmt: skip
    return _table([c["split"], c["trades"], c["cov"], c["mean"], c["diff"]], rows)


def _per_year_table(per_year: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    rows = [[str(r["year"]), str(r["n_picked"]), str(r["n_days"]), _bp(r["mean_net_picked"]),
             _bp(r["mean_net_af"])] for r in per_year.sort("year").iter_rows(named=True)]  # fmt: skip
    return _table([c["year"], c["picked"], c["days"], c["ypick"], c["yaf"]], rows)


def _quintile_table(quintiles: pl.DataFrame, model: str, lang: str) -> str:
    c = TEXT[lang]["cols"]
    q = _cell(quintiles, model).sort("group")
    rows = [[str(r["group"]), str(r["n"]), _pm(r["mean_net"], r["se"]), _pm(r["hit"], r["hit_se"], "pct")]
            for r in q.iter_rows(named=True)]  # fmt: skip
    rows.append([_rule_label("always_follow", None, lang), str(int(q["n"].sum())),
                 _bp(float(q["af_mean_net"][0])), ""])  # fmt: skip
    return _table(["", c["trades"], c["mean"], c["hit"]], rows)


def _grid_table(grid: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    costs = sorted(grid["c_bp"].unique().to_list())
    label = lambda x: c["c0"] if x == 0 else c["cn"].format(c=f"{x:g}")
    head = [c["rule"], c["tau"]]
    for x in costs:
        head += [
            c["cell"].format(c=label(x)), c["ccov"].format(c=label(x)),
            c["cauc"].format(c=label(x)), c["cll"].format(c=label(x)),
        ]  # fmt: skip
    rows = []
    for m in MODELS:
        for t in sorted(grid["tau"].unique().to_list()):
            line = [_lab(MODEL, m, lang, "model"), t]
            for x in costs:
                r = grid.filter(
                    (pl.col("model") == m) & (pl.col("tau") == t) & (pl.col("c_bp") == x)
                )
                assert r.height == 1, f"grid cell {m}, {t}, {x} bp: {r.height} rows"
                r = r.row(0, named=True)
                line += [
                    f"{_bp(r['top20_minus_af'])} ({_bp(r['se'])})", _pct(r["coverage"]),
                    f"{r['auc']:.3f}", f"{r['logloss']:.4f} ({r['logloss_base']:.4f})",
                ]  # fmt: skip
            rows.append(line)
    return _table(head, rows)


BP_FEATURES = ("abs_s", "on_aligned", "intra_aligned", "rv_sofar", "rv_fh", "range_prev",
               "last30_aligned", "yest_s_aligned", "yest_r13_aligned")  # fmt: skip


PER_MINUTE = ("rv_sofar", "rv_fh")
"""Volatility features measured per one-minute return."""


def _feat_value(feature: str, v: float, lang: str) -> str:
    """Return and volatility features in bp (x 1e4), the rest plain; 3 significant figures."""
    if feature in PER_MINUTE:
        return f"{_sig3(v * 1e4)} " + ("bp per minute" if lang == "en" else "bp／分鐘")
    return f"{_sig3(v * 1e4)} bp" if feature in BP_FEATURES else _sig3(v)


def _picked_table(picked: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    top = (
        picked.with_columns(a=pl.col("std_diff").abs())
        .sort("a", descending=True)
        .head(config.PART2_REPORT_TOP_FEATURES)
    )
    rows = [[_lab(FEAT, r["feature"], lang, "feature"), _feat_value(r["feature"], r["mean_picked"], lang),
             _feat_value(r["feature"], r["mean_other"], lang), f"{r['std_diff']:.2f}"] for r in top.iter_rows(named=True)]  # fmt: skip
    return _table([c["feature"], c["mp"], c["mo"], c["sd"]], rows)


def _groups_table(groups: pl.DataFrame, af_mean_net: float, lang: str) -> str:
    c = TEXT[lang]["cols"]
    c_top = c["top"].format(q=_q(config.PART2_REPORT_Q))
    rows = [[_lab(FEAT, r["step"].lstrip("+"), lang, "input step"), _feat_list(r["features_added"], lang),
             f"{r['auc']:.3f}", _pm(r["top20_mean_net"], r["se"])] for r in groups.iter_rows(named=True)]  # fmt: skip
    rows.append([_rule_label("always_follow", None, lang), "", "0.5", _bp(af_mean_net)])
    return _table([c["step"], c["added"], c["auc"], c_top], rows)


def _render(lang: str, s: dict, f: dict[str, pl.DataFrame], imgs: dict[str, str]) -> str:
    tx = TEXT[lang]
    missing = [k for k in FRAMES if k not in f]
    assert not missing, f"part2 frames missing: {missing}"
    model = s["headline_model"]
    q = _q(config.PART2_REPORT_Q)
    img = lambda k, alt: (
        f'<img src="data:image/png;base64,{imgs[k]}" alt="{escape(tx["alt"][alt])}">'
    )
    sec = lambda n, *parts: (
        f'<h2 id="s{n}">{escape(tx["sec"].format(n=n, title=tx["t" + str(n)]))}</h2>'
        + "".join(parts)
    )
    p = lambda text: f"<p>{escape(text)}</p>"
    verdict = tx["verdict"].format(
        tau=TAU, c=C_MAIN, model=_lab(MODEL, model, lang, "model"),
        v=escape(_lab(VERDICT, s["verdict"], lang, "verdict")),
    )  # fmt: skip
    pool = s["cut_pool"]
    numbers = tx["numbers"].format(q=q, d_af=s["d_af"], d_bm=s["d_bm"], d_0=s["d_0"],
                                   cov=s["headline_coverage"] * 100)  # fmt: skip
    other = _lab(MODEL, s["other_model"], lang, "model")
    other_line = tx["other"].format(model=other, d_af=s["other_d_af"], d_bm=s["other_d_bm"],
                                    d_0=s["other_d_0"], cov=s["other_coverage"] * 100)  # fmt: skip
    hr = f["rules"].filter(
        (pl.col("tau") == TAU) & (pl.col("c_bp") == C_MAIN) & (pl.col("model") == model)
        & (pl.col("rule") == "model") & (pl.col("q") == config.PART2_REPORT_Q)
        & (pl.col("c_eval_bp") == C_MAIN)
    )  # fmt: skip
    d_split = {r["split"]: r["diff_vs_af"] / r["diff_vs_af_se"] for r in hr.iter_rows(named=True)}
    ep0 = s["expanding_pool"][model]
    fragile_line = tx["fragile"].format(
        pre=d_split.get("pre_2020_05", float("nan")), post=d_split.get("post_2020_05", float("nan")),
        nocovid=d_split.get("no_covid", float("nan")),
        exp=ep0["diff_af"] / ep0["se_af"], other=s["other_d_af"],
    )  # fmt: skip
    groups_line = tx["groups"].format(
        tb=s["top_minus_bottom"] * 1e4, tb2=2 * s["top_minus_bottom_se"] * 1e4
    )
    skill_line = tx["skillline"].format(
        model1=_lab(MODEL, model, lang, "model"), auc1=s["headline_auc"], ll1=s["headline_logloss"],
        model2=other, auc2=s["other_auc"], ll2=s["other_logloss"], base=s["headline_logloss_base"],
    )  # fmt: skip
    assert abs(s["headline_logloss_base"] - s["other_logloss_base"]) < 1e-9, (
        "base-rate log loss differs"
    )
    ep = s["expanding_pool"]
    exp_note = tx["expanding"].format(
        **{f"{k}_{m}": ep[m][k] * (100 if k == "coverage" else 1e4) * (2 if k.startswith("se") else 1)
           for m in MODELS
           for k in ("diff_af", "se_af", "diff_bm", "se_bm", "coverage")}
    )  # fmt: skip
    costs = ", ".join(f"{x:g}" for x in COSTS)
    cum = img("part2_cumulative", "c") if "part2_cumulative" in imgs else p(tx["nocum"])
    groups = f["groups"]
    inputs = (
        p(tx["skipped"].format(q=q)) if groups.height == 0
        else p(tx["x7"].format(q=q)) + _groups_table(groups, float(_cell(f["quintiles"], model)["af_mean_net"][0]), lang)
    )  # fmt: skip
    assert (groups.height == 0) == (not s["skill"]), "groups frame and the skill gate disagree"
    body = [
        sec(
            1,
            p(tx["x1"].format(tau=TAU, c=C_MAIN)),
            img("part2_quintiles", "q"),
            _quintile_table(f["quintiles"], model, lang),
        ),
        sec(2, p(tx["x2"].format(costs=costs, c=C_MAIN)), _rules_table(f["rules"], model, lang)),
        sec(
            3,
            p(tx["x3"]),
            _splits_table(f["rules"], model, lang),
            _per_year_table(f["per_year"], lang),
        ),
        sec(4, cum),
        sec(
            5,
            p(tx["x5"].format(q=q, c=C_MAIN)),
            img("part2_times", "t"),
            img("part2_grid", "g"),
            _grid_table(f["grid"], lang),
        ),
        sec(
            6,
            p(tx["x6"].format(k=config.PART2_REPORT_TOP_FEATURES)),
            _picked_table(f["picked"], lang),
        ),
        sec(7, inputs),
        sec(
            8,
            "<ul>"
            + "".join(f"<li>{escape(n.format(pool=pool))}</li>" for n in [*tx["notes"], exp_note])
            + "</ul>",
        ),
    ]
    kind = tx["synthetic"] if s["synthetic"] else "SPY"
    warn = tx["warn"].format(start=s["data_window"][0], end=s["data_window"][1])
    fits = tx["fits"].format(n_fits=s["n_fits"], n_cells=s["n_cells"])
    return f"""<!doctype html><html lang="{tx["lang"]}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tx["title"])}</title><style>{CSS}</style></head><body>
<h1>{escape(tx["h1"].format(kind=kind))}</h1>
<p class="warn">{escape(warn)}</p>
<div class="box">{tx["count"].format(count_line=escape(s["count_line"]))}<p>{verdict}</p>{p(numbers)}{p(other_line)}{p(fragile_line)}{p(groups_line)}{p(skill_line)}{p(fits)}</div>
{"".join(body)}
</body></html>"""


def write_part2_report(out: Path) -> tuple[Path, Path]:
    """Write both pages from `<out>/part2/`, `part2_summary.json` and the figures (see `make_figures`)."""
    s = json.loads((out / "part2_summary.json").read_text())
    f = {k: pl.read_parquet(out / "part2" / f"{k}.parquet") for k in FRAMES}
    imgs = {
        k: base64.b64encode((out / "figures" / f"{k}.png").read_bytes()).decode()
        for k in config.PART2_FIGURES
        if (out / "figures" / f"{k}.png").exists()
    }
    required = set(config.PART2_FIGURES) - {"part2_cumulative"}
    assert required <= set(imgs), f"figures missing: {sorted(required - set(imgs))}"
    paths = (out / "part2_report.html", out / "part2_report_zh.html")
    for path, lang in zip(paths, ("en", "zh"), strict=True):
        path.write_text(_render(lang, s, f, imgs))
    return paths
