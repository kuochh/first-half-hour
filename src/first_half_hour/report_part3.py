"""Part 3 figures and the two-language report (the same signal on other ETFs, in four classes).

Reads `<out>/part3/*.parquet` (coverage, effect, grid, rules, etf_rules; written by
`part3_data` and `part3.write_outputs`) and `<out>/part3_summary.json` (completed by
`reproduce._part3`); writes two figures under `<out>/figures/` and `<out>/part3_report.html` and
`<out>/part3_report_zh.html`. Nothing is recomputed except counts that are checked against the
summary: both pages lay out the same numbers, and a code label with no entry here fails an
assertion instead of showing raw. Return names are looked up in `report_part1.RET`.

Column contracts: `effect` has `class` (or `symbol` for ETF rows),
period, item, metric, cost_bp, n_days, n_etfs, est, se, t, verdict, level; `grid` the columns of `part3._grid`;
`rules` and `etf_rules` the Part 2 `rules` columns with `class` / `symbol` instead of `model`,
and `etf_rules` also `auc`.
"""

import base64
import json
from datetime import date, timedelta
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from first_half_hour import config
from first_half_hour.report import CSS
from first_half_hour.report_part1 import RET, _lab
from first_half_hour.report_part2 import _bp, _pct, _pm, _q

FRAMES = ("coverage", "effect", "grid", "rules", "etf_rules")
C_MAIN = config.PART3_REPORT_COST_BP
TAU = config.PART3_REPORT_TAU
Q = config.PART3_REPORT_Q
LATE_START_DAYS = 7
"""A first session more than this many days after the window start is called a late start."""

CLASS = {
    "us_equity": ("US equity", "美國股票"),
    "intl_equity": ("international equity", "海外股票"),
    "bonds": ("bonds", "債券"),
    "gold": ("gold", "黃金"),
}
SPY_OFFICIAL = ("SPY, official close", "SPY，官方收盤價")
VERDICT = {
    "clearly above": ("clearly above 0", "明顯大於 0"),
    "cannot tell": ("cannot tell", "分辨不出來"),
    "clearly below": ("clearly below 0", "明顯小於 0"),
}
YESNO = {True: ("yes", "是"), False: ("no", "否")}
RULE = {  # a "{q}" is the share of days
    "model": ("model, top {q}% of days", "模型，分數最高的 {q}% 的日子"),
    "always_follow": ("always follow the signal", "一律跟著訊號做"),
    "big_move": ("big-move rule, {q}% of days", "大幅移動規則，{q}% 的日子"),
    "vol_only": ("volatility-only model, {q}% of days", "只用波動的模型，{q}% 的日子"),
}
RULE_NAMES = ("model", "big_move", "vol_only")
NOTE = {
    "late start": ("late start", "較晚開始"),
    "gaps": ("sessions dropped (missing a key bar)", "有交易日被排除（缺關鍵分鐘棒）"),
    "": ("", ""),
}
PERIODS = ("p1", "p2", "p3")
SLOPES = ("slope_r1", "slope_rod")
RULES = ("rule_r1", "rule_rod")
ITEM_RET = {"slope_r1": "r1", "slope_rod": "rod", "rule_r1": "r1", "rule_rod": "rod"}


# ---------------------------------------------------------------- helpers
def _cls(code: str, lang: str) -> str:
    return _lab(CLASS, code, lang, "class")


def _class_of(symbol: str) -> str:
    for c, syms in config.PART3_CLASSES.items():
        if symbol in syms:
            return c
    raise AssertionError(f"symbol {symbol!r} is in no class of config.PART3_CLASSES")


def _period_dates(p: str) -> tuple[date, date]:
    assert p in config.PART3_PERIODS, f"unknown period {p!r}"
    lo, hi = config.PART3_PERIODS[p]
    assert isinstance(lo, date) and isinstance(hi, date), f"period {p!r} is not a date pair"
    return lo, hi


def _period(p: str, lang: str) -> str:
    lo, hi = _period_dates(p)
    return f"{lo} {'to' if lang == 'en' else '到'} {hi}"


def _slope(v: float, se: float) -> str:
    assert v is not None and se is not None and np.isfinite(v) and np.isfinite(se), (
        f"missing slope {v!r} / SE {se!r}"
    )
    text = f"{v:.4f}"
    return f"{'0.0000' if text == '-0.0000' else text} ± {2 * se:.4f}"


def _ann(v: float, se: float) -> str:
    """Annual mean in percent, `value ± 2 SE`."""
    assert np.isfinite(v) and np.isfinite(se), f"missing annual mean {v!r} / SE {se!r}"
    text = f"{v * 100:.2f}"
    return f"{'0.00' if text == '-0.00' else text} ± {2 * se * 100:.2f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="t" style="overflow-x:auto"><table><tr>{h}</tr>{b}</table></div>'


def _key(df: pl.DataFrame) -> str:
    return "symbol" if "symbol" in df.columns else "class"


def _one(df: pl.DataFrame, **where) -> dict:
    sub = df.filter(*[pl.col(k) == v for k, v in where.items()])
    assert sub.height == 1, f"expected one row for {where}, got {sub.height}"
    return sub.row(0, named=True)


def _eff(
    effect: pl.DataFrame, level: str, who: str | None, p: str, item: str, metric: str, cost=None
):
    """The one effect row; `who` None for spy_official (its key is not relied on)."""
    sub = effect.filter(
        (pl.col("level") == level) & (pl.col("period") == p) & (pl.col("item") == item)
        & (pl.col("metric") == metric)
    )  # fmt: skip
    if who is not None:
        sub = sub.filter(pl.col(_key(effect)) == who)
    if cost is not None:
        sub = sub.filter(pl.col("cost_bp") == cost)
    assert sub.height == 1, f"expected one effect row for {level}/{who}/{p}/{item}/{metric}/{cost}, got {sub.height}"  # fmt: skip
    return sub.row(0, named=True)


# ---------------------------------------------------------------- figures
def plot_effect(effect: pl.DataFrame, path: Path) -> None:
    """One panel per class: the two slopes in each period, +/- 2 SE; the US panel adds official-close SPY."""
    classes = list(config.PART3_CLASSES)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, len(classes), figsize=(14, 4.2), sharey=True)
    ret_en = {k: v[0] for k, v in RET.items()}
    for ax, c in zip(axes, classes, strict=True):
        series = [(f"slope on {ret_en['r1']}", "level_class", "slope_r1", "C0"),
                  (f"slope on {ret_en['rod']}", "level_class", "slope_rod", "C1")]  # fmt: skip
        if c == "us_equity":
            series.append(
                (f"SPY official close, on {ret_en['r1']}", "spy_official", "slope_r1", "grey")
            )
        for i, (label, level, item, color) in enumerate(series):
            rows = [
                _eff(effect, "class" if level == "level_class" else level,
                     c if level == "level_class" else None, p, item, "slope")
                for p in PERIODS
            ]  # fmt: skip
            x = np.arange(len(PERIODS)) + (i - (len(series) - 1) / 2) * 0.2
            ax.errorbar(x, [r["est"] for r in rows], yerr=[2 * r["se"] for r in rows], fmt="o",
                        capsize=3, color=color, label=label)  # fmt: skip
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(PERIODS)), [f"{p[-1]}" for p in PERIODS])
        ax.set_xlabel("period (1 earliest)")
        ax.set_title(CLASS[c][0])
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"slope of {ret_en['r13']} (+/- 2 SE)")
    fig.suptitle(
        f"Slope of the {ret_en['r13']} return on the earlier signal, by class and period (zero = no effect)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(grid: pl.DataFrame, path: Path) -> None:
    """4 classes x 5 decision times: top-20% minus always-follow (bp) +/- 2 SE; findings starred, cells that
    beat always-follow only marked with an open diamond."""
    classes = list(config.PART3_CLASSES)
    taus = [t for t, _ in config.PART2_TIMES]
    assert grid.height == len(classes) * len(taus), f"grid has {grid.height} cells"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, len(classes), figsize=(14, 4.2), sharey=True)
    for ax, c in zip(axes, classes, strict=True):
        g = grid.filter(pl.col("class") == c)
        assert g.height == len(taus), f"class {c} has {g.height} cells"
        g = g.with_columns(pl.col("tau").replace_strict({t: i for i, t in enumerate(taus)}).alias("_i")).sort("_i")  # fmt: skip
        ax.errorbar(g["_i"].to_numpy(), g["top20_minus_af"].to_numpy() * 1e4,
                    yerr=2 * g["se"].to_numpy() * 1e4, fmt="o", capsize=3, color="C0")  # fmt: skip
        only = g.filter(pl.col("beats_af") & ~pl.col("finding"))
        ax.plot(only["_i"].to_numpy(), only["top20_minus_af"].to_numpy() * 1e4, "D", ms=9, mfc="none",
                mec="C1", mew=1.5, label="beats always follow only")  # fmt: skip
        hit = g.filter(pl.col("finding"))
        ax.plot(hit["_i"].to_numpy(), hit["top20_minus_af"].to_numpy() * 1e4, "*", ms=16,
                color="gold", mec="black", label="finding")  # fmt: skip
        hit = g.filter(pl.col("beats_af"))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(taus)), taus, fontsize=8)
        ax.set_xlabel("decision time")
        ax.set_title(CLASS[c][0])
        if hit.height:
            ax.legend(fontsize=8)
    axes[0].set_ylabel(f"top {_q(Q)}% of days minus always follow (bp, after {C_MAIN} bp)")
    fig.suptitle(
        "Part 3: 20 cells, top-20% days minus always follow (+/- 2 SE; zero = no gain from picking days)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(out: Path) -> None:
    """Draw both figures from `<out>/part3/` into `<out>/figures/`."""
    d, fig = out / "part3", out / "figures"
    plot_effect(pl.read_parquet(d / "effect.parquet"), fig / "part3_effect.png")
    plot_grid(pl.read_parquet(d / "grid.parquet"), fig / "part3_grid.png")


# ---------------------------------------------------------------- text
TEXT = {
    "en": {
        "lang": "en",
        "title": "Part 3: do the same signals work on other ETFs, and can a classifier pick the days?",
        "h1": "Part 3: other markets, four ETF classes ({kind})",
        "synthetic": "SYNTHETIC data (a pipeline check, not a result)",
        "warn": "Data window {start} to {end}.",
        "count": "<p><b>{count_line}</b></p>",
        "question": (
            "Two questions, asked on {n_etfs} ETFs grouped into four classes (US equity, international equity, "
            "bonds, gold). A: is there an effect, i.e. does the {r13} return follow the {r1} return "
            "(or the {rod} return) with a slope above zero? B: does the Part 2 classifier, which picks "
            "the days to follow the signal, beat always following it? Every number sits beside zero: "
            "a slope of 0 and a payoff of 0 are the no-effect values."
        ),
        "findings": (
            "{n_pass} of {n_cells} cells (class x decision time) have the top-{q}% days beating always "
            "following by more than 2 paired standard errors; {n_af} of them also keep a positive gain in "
            "2014 to 2020-05-01, in 2020-05-04 on, and over all test days without 2020-02 to "
            "2020-04{which_af}.{extra} With {n_cells} cells, one or two passing the first test by chance "
            "is expected. A cell counts as a finding only if, in addition, the model beats the big-move "
            "rule by more than 2 paired standard errors and its net per trade after {c} bp is above zero "
            "by more than 2 standard errors (this rule was tightened after seeing the results): "
            "{n_find} findings{which}."
        ),
        "extra": (
            " Of these {n_af}, in {n_neg} the model's net per trade after {c} bp is still below zero, and "
            "in {n_bm} the model beats the big-move rule by more than 2 paired standard errors."
        ),
        "none": "",
        "fits": "Model fits behind this page: {n_fits}, for {n_etfs} ETFs.",
        "sec": "{n}. {title}",
        "t1": "Data coverage",
        "x1": (
            "Every ETF keeps what it has; nothing is dropped silently. Sessions dropped are those missing "
            "a key bar. The zero-return share is the share of sessions where the {r13} return is exactly zero "
            "(thin trading shows up here); the missing-signal share is the share where the {r1} return is missing."
        ),
        "x1b": (
            "Sessions that other ETFs have a forecast for but no ETF of the class has are left out of that "
            "class's series (not filled). Per decision time: {missing}."
        ),
        "x1c": "none in any class",
        "t2": "A: is the effect there?",
        "x2": (
            "Class series are equal-weight means of the class's ETFs present that day. Slope of the {r13} "
            "return on the {r1} return and on the {rod} return, each +/- 2 Newey-West standard errors, "
            "against 0, in three periods. The US class is on the 15:59-bar close; the SPY row beside it uses "
            "the official close."
        ),
        "x2r": (
            "The trading rule: follow the sign of the signal over {r13}, equal weight across the class's "
            "ETFs, after {c} bp round trip. Annual mean in percent (zero = no gain) and Sharpe, each +/- 2 "
            "standard errors; the word is the verdict against 0."
        ),
        "x2c": "Same rules at {cost} bp round trip.",
        "t3": "B: does picking days help?",
        "x3": (
            "Each cell: the model's top-{q}% days minus always following, in basis points per trade after "
            "{c} bp, +/- standard error in brackets. The gain-over-always-follow column is that difference over its paired standard error "
            "(the same against the big-move rule and the volatility-only model in the next two columns). The three "
            "split columns repeat the first in 2014 to 2020-05-01, in 2020-05-04 on, and over all test days without "
            "2020-02 to 2020-04. Finding = gain over always follow above 2 and the gain positive in all three splits."
        ),
        "t4": "The rule against its baselines, by class, at {tau}",
        "x4": (
            "Target threshold {c} bp, all test days. Net per trade in basis points, Sharpe and hit rate, "
            "each +/- 2 standard errors from a bootstrap that resamples whole days for all ETFs; the last "
            "column is the rule minus always following, paired in the same draws. Share = the share of the class's ETFs traded, averaged over days."
        ),
        "t5": "Limits",
        "limits": [
            "The close is the 15:59 bar for every ETF, including SPY in the US class; official-close SPY is shown beside it in section 2.",
            "The bond ETFs trade thinly in the {r13} window; the zero-return share in section 1 shows it. Costs may be higher for bonds and international ETFs than the {c} bp used here.",
            "20 cells are tested: one or two passing the first test by chance is expected, which is why the finding rule also asks for a positive gain in three robustness splits, a win over the big-move rule and a net payoff above zero.",
            "ETFs with late starts or gaps keep what they have (section 1); the class mean uses the ETFs present each day.",
            "Half-days are left out of the model tests (B) at every decision time.",
            "A session is kept only if the ETF traded in the 09:59, 14:59, 15:29 and 15:59 minutes. The later of these are known only after the 10:00, 12:00, 14:00 and 15:00 decisions (and 15:59 after 15:30), so at those times days that later turn out to have no trade in those minutes are missing from the sample. ETFs with the most sessions dropped for any reason (section 1): {drop}.",
        ],
        "t6": "Appendix: every ETF",
        "x6": "Slopes and the rule after {c} bp, per ETF and period (no pooling), each +/- 2 standard errors.",
        "x6b": (
            "The model's top-{q}% days minus always following, per ETF and decision time (basis points, "
            "after {c} bp, +/- 2 standard errors), the net per trade of the model and of always following (beside 0), "
            "the share of days traded and the out-of-sample AUC (0.5 = no skill)."
        ),
        "alt": {"e": "Slopes by class and period", "g": "Top-20% days minus always follow, 20 cells"},
        "cols": {
            "etf": "ETF", "class": "class", "first": "first session", "last": "last session",
            "kept": "sessions kept", "dropped": "sessions dropped", "zero": "share of sessions with a zero {r13} return",
            "null": "share of sessions with no signal", "note": "note", "period": "period", "days": "days",
            "etfs": "ETFs", "tau": "decision time", "cell": "top-{q}% minus always follow (SE)",
            "cov": "share of days traded", "auc": "AUC (0.5 = no skill)", "finding": "finding", "beats_af": "beats always follow (earlier rule)",
            "net_model": "model net per trade (bp, ± 2 SE)", "net_af": "always follow net per trade (bp, ± 2 SE)",
            "rule": "rule", "trades": "trades", "mean": "net per trade (bp)", "sharpe": "Sharpe",
            "hit": "hit rate", "diff": "minus always follow (bp)", "verdict": "verdict",
            "cost": "cost (bp)", "ann": "annual mean (%)", "slope": "slope",
            "d_af": "gain over always follow, in standard errors",
            "d_bm": "gain over the big-move rule, in standard errors",
            "d_vol": "gain over the volatility-only model, in standard errors",
            "d_pre": "same gain, 2014 to 2020-05-01", "d_post": "same gain, 2020-05-04 on",
            "d_nocovid": "same gain, without 2020-02 to 2020-04",
        },
    },
    "zh": {
        "lang": "zh-Hant-TW",
        "title": "第三部分：同樣的訊號在其他 ETF 也有效嗎？分類器能挑出日子嗎？",
        "h1": "第三部分：其他市場，四類 ETF（{kind}）",
        "synthetic": "合成資料（只檢查流程，不是結果）",
        "warn": "資料期間 {start} 到 {end}。",
        "count": "<p><b>{count_line}</b></p>",
        "question": (
            "兩個問題，用分成四類的 {n_etfs} 檔 ETF（美國股票、海外股票、債券、黃金）來問。A：效應存在嗎？也就是 {r13} "
            "報酬是否跟著 {r1} 報酬（或 {rod} 報酬）走，斜率大於 0？B：第二部分挑日子的分類器，"
            "比一律跟著訊號做更好嗎？每個數字都放在 0 旁邊：斜率 0 與報酬 0 是「沒有效應」的值。"
        ),
        "findings": (
            "{n_cells} 個格子（類別 x 決策時間）中，有 {n_pass} 個格子的前 {q}% 日子比一律跟著做多出 2 個配對標準誤以上；"
            "其中 {n_af} 個在 2014 到 2020-05-01、2020-05-04 之後、以及拿掉 2020 年 2 到 4 月的全部測試日"
            "三種版本中，差距也都是正的{which_af}。{extra}有 {n_cells} 個格子時，"
            "光靠運氣就有一兩個通過第一關是預期之內的。要算「發現」，模型還必須比大幅移動規則多出 2 個配對標準誤以上，"
            "而且扣除 {c} bp 後每筆淨報酬比 0 高出 2 個標準誤以上（這條規則是看到結果之後才收緊的）：共 {n_find} 個{which}。"
        ),
        "extra": (
            "這 {n_af} 個之中，有 {n_neg} 個的模型扣除 {c} bp 後每筆淨報酬仍小於 0，"
            "有 {n_bm} 個的模型比大幅移動規則多出 2 個配對標準誤以上。"
        ),
        "none": "",
        "fits": "這一頁背後的模型擬合：{n_fits} 次，涵蓋 {n_etfs} 檔 ETF。",
        "sec": "{n}. {title}",
        "t1": "資料涵蓋",
        "x1": (
            "每檔 ETF 保留它有的資料，沒有任何資料被默默丟掉。被排除的交易日是缺關鍵分鐘棒的日子。"
            "報酬為零的比例是 {r13} 報酬剛好為 0 的交易日占比（交易清淡會顯示在這裡）；訊號缺值比例是 {r1} 報酬缺值的占比。"
        ),
        "x1b": (
            "其他 ETF 有預測、但該類沒有任何 ETF 有預測的交易日，會從該類的序列中排除（不補值）。每個決策時間：{missing}。"
        ),
        "x1c": "每一類都沒有",
        "t2": "A：效應存在嗎？",
        "x2": (
            "類別序列是當天有資料的該類 ETF 的等權平均。{r13} 報酬對 {r1} 報酬、對 {rod} 報酬的斜率，"
            "各附 ± 2 個 Newey-West 標準誤，與 0 比較，分三個期間。美國類用 15:59 分鐘棒的收盤；旁邊的 SPY 列用官方收盤價。"
        ),
        "x2r": (
            "交易規則：順著訊號的正負號做 {r13}，該類 ETF 等權，扣除來回 {c} bp 成本。"
            "年化平均報酬（%，0＝沒有獲利）與 Sharpe，各附 ± 2 個標準誤；文字是與 0 比較的判定。"
        ),
        "x2c": "同樣的規則，來回成本 {cost} bp。",
        "t3": "B：挑日子有幫助嗎？",
        "x3": (
            "每個格子：模型分數最高的 {q}% 日子減一律跟著做，單位 bp／每筆，扣除 {c} bp 後，括號內是標準誤。"
            "「比一律跟著做多出幾個標準誤」是這個差距除以配對標準誤（接下來兩欄是對大幅移動規則、只用波動的模型做同樣的比較）。"
            "三個分期欄是同樣的數字，分別在 2014 到 2020-05-01、2020-05-04 之後、以及拿掉 2020 年 2 到 4 月的全部測試日。"
            "發現＝比一律跟著做多出的標準誤數大於 2，且三種分期的差距都是正的。"
        ),
        "t4": "各類別在 {tau} 的規則與基準比較",
        "x4": (
            "目標門檻 {c} bp、全部測試日。每筆淨報酬（bp）、Sharpe、命中率，各附 ± 2 個標準誤（自助法以整天為單位，"
            "所有 ETF 一起重抽）；最後一欄是規則減一律跟著做，在同樣的重抽中配對。比例＝每天該類 ETF 中有交易的比例，再對天數平均。"
        ),
        "t5": "限制",
        "limits": [
            "所有 ETF 的收盤都用 15:59 分鐘棒，包括美國類中的 SPY；第 2 節旁邊列出用官方收盤價的 SPY。",
            "債券 ETF 在 {r13} 這段時間交易清淡，第 1 節的報酬為零比例顯示了這點。債券與海外 ETF 的成本可能比這裡用的 {c} bp 高。",
            "共檢定 20 個格子：光靠運氣就有一兩個通過第一關是預期之內的，所以「發現」的規則還要求三種分期的差距都是正的、贏過大幅移動規則、而且扣除成本後淨報酬大於 0。",
            "較晚開始或有缺漏的 ETF 保留它有的資料（第 1 節）；類別平均用當天有資料的 ETF。",
            "模型檢定（B）在每個決策時間都排除提早收盤日。",
            "一個交易日要保留，該 ETF 在 09:59、14:59、15:29 與 15:59 這幾分鐘都必須有成交。較晚的幾分鐘要到 10:00、12:00、14:00、15:00 的決策之後才知道（15:59 則在 15:30 之後），所以在這些時間，事後發現沒有成交的日子不在樣本內。被排除交易日最多的 ETF（任何原因，見第 1 節）：{drop}。",
        ],
        "t6": "附錄：每檔 ETF",
        "x6": "每檔 ETF、每個期間的斜率與扣除 {c} bp 後的規則（不合併），各附 ± 2 個標準誤。",
        "x6b": (
            "模型分數最高的 {q}% 日子減一律跟著做，每檔 ETF、每個決策時間（bp，扣除 {c} bp，± 2 個標準誤），"
            "模型與一律跟著做的每筆淨報酬（與 0 比較）、交易日比例與樣本外 AUC（0.5＝沒有預測力）。"
        ),
        "alt": {"e": "各類別與期間的斜率", "g": "前 20% 日子減一律跟著做，20 個格子"},
        "cols": {
            "etf": "ETF", "class": "類別", "first": "第一個交易日", "last": "最後一個交易日",
            "kept": "保留的交易日", "dropped": "被排除的交易日", "zero": "{r13} 報酬為零的比例",
            "null": "訊號缺值比例", "note": "備註", "period": "期間", "days": "天數",
            "etfs": "ETF 數", "tau": "決策時間", "cell": "前 {q}% 減一律跟著做（SE）",
            "cov": "交易日比例", "auc": "AUC（0.5＝沒有預測力）", "finding": "發現", "beats_af": "贏過一律跟著做（先前的規則）",
            "net_model": "模型每筆淨報酬（bp，± 2 SE）", "net_af": "一律跟著做每筆淨報酬（bp，± 2 SE）",
            "rule": "規則", "trades": "交易筆數", "mean": "每筆淨報酬（bp）", "sharpe": "Sharpe",
            "hit": "命中率", "diff": "減一律跟著做（bp）", "verdict": "判定",
            "cost": "成本（bp）", "ann": "年化平均報酬（%）", "slope": "斜率",
            "d_af": "比一律跟著做多出幾個標準誤",
            "d_bm": "比大幅移動規則多出幾個標準誤",
            "d_vol": "比只用波動的模型多出幾個標準誤",
            "d_pre": "同樣的差距，2014 到 2020-05-01", "d_post": "同樣的差距，2020-05-04 之後",
            "d_nocovid": "同樣的差距，不含 2020 年 2 到 4 月",
        },
    },
}  # fmt: skip


def _ret(lang: str) -> dict[str, str]:
    return {k: v[0 if lang == "en" else 1] for k, v in RET.items()}


# ---------------------------------------------------------------- tables
def _coverage_table(cov: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    start = _period_dates("p1")[0]
    rows = []
    for r in cov.iter_rows(named=True):
        note = [k for k, on in (("late start", r["first_session"] > start + timedelta(days=LATE_START_DAYS)),
                                ("gaps", r["sessions_dropped"] > 0)) if on]  # fmt: skip
        rows.append([
            r["symbol"], _cls(_class_of(r["symbol"]), lang), str(r["first_session"]),
            str(r["last_session"]), str(r["kept_sessions"]), str(r["sessions_dropped"]), _pct(r["share_r13_zero"]),
            _pct(r["share_r1_null"]), "; ".join(_lab(NOTE, k, lang, "coverage note") for k in note),
        ])  # fmt: skip
    head = [
        c[k].format(**_ret(lang))
        for k in ("etf", "class", "first", "last", "kept", "dropped", "zero", "null", "note")
    ]
    return _table(head, rows)


def _owners(effect: pl.DataFrame, lang: str) -> list[tuple[str, str, str | None]]:
    """(label, level, key) rows of the A tables: the four classes, SPY official after the US class."""
    out = []
    for c in config.PART3_CLASSES:
        out.append((_cls(c, lang), "class", c))
        if c == "us_equity":
            out.append((_lab({"x": SPY_OFFICIAL}, "x", lang, "label"), "spy_official", None))
    return out


def _slopes_table(effect: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    ret = _ret(lang)
    rows = []
    for label, level, who in _owners(effect, lang):
        for p in PERIODS:
            line = [label, _period(p, lang)]
            first = _eff(effect, level, who, p, SLOPES[0], "slope")
            line += [str(first["n_days"]), str(first["n_etfs"])]
            for item in SLOPES:
                r = _eff(effect, level, who, p, item, "slope")
                line += [_slope(r["est"], r["se"]), _lab(VERDICT, r["verdict"], lang, "verdict")]
            rows.append(line)
    head = [c["class"], c["period"], c["days"], c["etfs"]]
    for item in SLOPES:
        head += [f"{c['slope']}: {ret['r13']} / {ret[ITEM_RET[item]]}", c["verdict"]]
    return _table(head, rows)


def _rules_effect_table(effect: pl.DataFrame, lang: str, cost: int) -> str:
    c = TEXT[lang]["cols"]
    ret = _ret(lang)
    rows = []
    for label, level, who in _owners(effect, lang):
        for p in PERIODS:
            line = [label, _period(p, lang)]
            for item in RULES:
                a = _eff(effect, level, who, p, item, "ann_mean", cost)
                s = _eff(effect, level, who, p, item, "sharpe", cost)
                line += [_ann(a["est"], a["se"]), _lab(VERDICT, a["verdict"], lang, "verdict"),
                         _pm(s["est"], s["se"], "plain")]  # fmt: skip
            rows.append(line)
    head = [c["class"], c["period"]]
    for item in RULES:
        n = ret[ITEM_RET[item]]
        head += [f"{c['ann']}: {n}", c["verdict"], f"{c['sharpe']}: {n}"]
    return _table(head, rows)


def _grid_rows(grid: pl.DataFrame) -> list[dict]:
    order = {t: i for i, (t, _) in enumerate(config.PART2_TIMES)}
    assert grid.height == len(config.PART3_CLASSES) * len(order), f"grid has {grid.height} cells"
    cls = {k: i for i, k in enumerate(config.PART3_CLASSES)}
    return sorted(grid.iter_rows(named=True), key=lambda r: (cls[r["class"]], order[r["tau"]]))


def _d(v: float) -> str:
    assert np.isfinite(v), "missing d"
    return f"{v:.2f}"


def _model_net(rules: pl.DataFrame, cls: str, tau: str) -> dict:
    """The class's model row (top share Q, cost C_MAIN, all test days) at one decision time."""
    sub = rules.filter(
        (pl.col("class") == cls) & (pl.col("tau") == tau) & (pl.col("rule") == "model")
        & ((pl.col("q") - Q).abs() < 1e-9) & (pl.col("c_eval_bp") == C_MAIN)
        & (pl.col("split") == "all")
    )  # fmt: skip
    assert sub.height == 1, f"expected one model row for {cls} {tau}, got {sub.height}"
    return sub.row(0, named=True)


def _grid_table(grid: pl.DataFrame, rules: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    rows = []
    for r in _grid_rows(grid):
        m = _model_net(rules, r["class"], r["tau"])
        rows.append([
            _cls(r["class"], lang), r["tau"], f"{_bp(r['top20_minus_af'])} ({_bp(r['se'])})",
            _pm(m["mean_net"], m["se"]), _d(r["d_af"]), _d(r["d_bm"]), _d(r["d_vol"]), _d(r["d_af_pre"]), _d(r["d_af_post"]),
            _d(r["d_af_nocovid"]), _pct(r["coverage"]), f"{r['mean_auc']:.3f}",
            _lab(YESNO, bool(r["beats_af"]), lang, "flag"), _lab(YESNO, bool(r["finding"]), lang, "flag"),
        ])  # fmt: skip
    head = [c["class"], c["tau"], c["cell"].format(q=_q(Q)), c["net_model"], c["d_af"], c["d_bm"], c["d_vol"],
            c["d_pre"], c["d_post"], c["d_nocovid"], c["cov"], c["auc"], c["beats_af"], c["finding"]]  # fmt: skip
    return _table(head, rows)


def _findings_text(grid: pl.DataFrame, rules: pl.DataFrame, s: dict, lang: str) -> str:
    rows = _grid_rows(grid)
    n_pass = sum(r["d_af"] > config.PART3_FINDING_D for r in rows)
    found = [r for r in rows if r["finding"]]
    af = [r for r in rows if r["beats_af"]]
    assert n_pass == s["n_pass_full"], f"grid passes {n_pass}, summary says {s['n_pass_full']}"
    assert len(af) == s["n_beats_af"], f"grid beats-af {len(af)}, summary {s['n_beats_af']}"
    assert all(r["beats_af"] for r in found), "a finding fails the earlier rule"
    assert len(found) == s["n_findings"], f"grid findings {len(found)}, summary {s['n_findings']}"
    assert all(r["d_af"] > config.PART3_FINDING_D for r in found), "a finding fails the d_af test"
    assert s["n_cells"] == len(rows), "summary and grid cell counts differ"

    def names(cells):
        if not cells:
            return ""
        s_ = "; ".join(f"{_cls(r['class'], lang)}, {r['tau']}" for r in cells)
        return f" ({s_})" if lang == "en" else f"（{s_}）"

    extra = ""
    if af:
        n_neg = sum(_model_net(rules, r["class"], r["tau"])["mean_net"] < 0 for r in af)
        n_bm = sum(r["d_bm"] > config.PART3_FINDING_D for r in af)
        extra = TEXT[lang]["extra"].format(n_af=len(af), n_neg=n_neg, n_bm=n_bm, c=C_MAIN)
    return TEXT[lang]["findings"].format(
        n_pass=n_pass,
        n_cells=len(rows),
        q=_q(Q),
        n_af=len(af),
        which_af=names(af),
        extra=extra,
        c=C_MAIN,
        n_find=len(found),
        which=names(found),
    )


def _missing_text(s: dict, lang: str) -> str:
    """Per class: fewest to most class-days missing across decision times (from the summary)."""
    per: dict[str, list[int]] = {}
    for key, n in s["class_days_missing"].items():
        cls, tau = key.rsplit(" ", 1)
        assert cls in config.PART3_CLASSES and tau in dict(config.PART2_TIMES), f"bad key {key!r}"
        per.setdefault(cls, []).append(int(n))
    assert set(per) == set(config.PART3_CLASSES), "class_days_missing lacks a class"
    parts = []
    for cls in config.PART3_CLASSES:
        lo, hi = min(per[cls]), max(per[cls])
        if hi:
            parts.append(
                f"{_cls(cls, lang)} {lo}"
                + (f"{'-' if lang == 'en' else '–'}{hi}" if hi != lo else "")
            )
    sep = "; " if lang == "en" else "；"
    return sep.join(parts) if parts else TEXT[lang]["x1c"]


def _drop_text(cov: pl.DataFrame, top: int = 3) -> str:
    """The ETFs with most sessions dropped, with the counts."""
    worst = (
        cov.filter(pl.col("sessions_dropped") > 0)
        .sort("sessions_dropped", descending=True)
        .head(top)
    )
    if not worst.height:
        return "none"
    return ", ".join(f"{r['symbol']} {r['sessions_dropped']}" for r in worst.iter_rows(named=True))


def _rule_label(rule: str, q: float | None, lang: str) -> str:
    return _lab(RULE, rule, lang, "rule").format(q="" if q is None else _q(q))


def _class_rules_table(rules: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    base = rules.filter(
        (pl.col("tau") == TAU) & (pl.col("c_eval_bp") == C_MAIN) & (pl.col("split") == "all")
    )  # fmt: skip
    rows = []
    for cl in config.PART3_CLASSES:
        sub = base.filter(pl.col("class") == cl)
        lines = [("always_follow", None, _one(sub, rule="always_follow"))]
        for q in config.PART3_REPORT_QS:
            for rule in RULE_NAMES:
                x = sub.filter((pl.col("rule") == rule) & ((pl.col("q") - q).abs() < 1e-9))
                assert x.height == 1, f"expected one {rule} row for {cl}, q={q}, got {x.height}"
                lines.append((rule, q, x.row(0, named=True)))
        for rule, q, r in lines:
            diff = "" if rule == "always_follow" else _pm(r["diff_vs_af"], r["diff_vs_af_se"])
            rows.append([
                _cls(cl, lang), _rule_label(rule, q, lang), str(r["n_etf_trades"]), _pct(r["coverage"]),
                _pm(r["mean_net"], r["se"]), _pm(r["sharpe"], r["sharpe_se"], "plain"),
                _pm(r["hit"], r["hit_se"], "pct"), diff,
            ])  # fmt: skip
    head = [c[k] for k in ("class", "rule", "trades", "cov", "mean", "sharpe", "hit", "diff")]
    return _table(head, rows)


def _etf_effect_table(effect: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    ret = _ret(lang)
    etf = effect.filter(pl.col("level") == "etf")
    assert etf.height, "no per-ETF effect rows"
    rows = []
    for sym in [s for syms in config.PART3_CLASSES.values() for s in syms]:
        for p in PERIODS:
            line = [sym, _period(p, lang)]
            for item in SLOPES:
                r = _eff(effect, "etf", sym, p, item, "slope")
                line.append(_slope(r["est"], r["se"]))
            for item in RULES:
                a = _eff(effect, "etf", sym, p, item, "ann_mean", C_MAIN)
                line.append(_ann(a["est"], a["se"]))
            rows.append(line)
    head = [c["etf"], c["period"]]
    head += [f"{c['slope']}: {ret['r13']} / {ret[ITEM_RET[i]]}" for i in SLOPES]
    head += [f"{c['ann']}: {ret[ITEM_RET[i]]}" for i in RULES]
    return _table(head, rows)


def _etf_cells_table(etf_rules: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    base = etf_rules.filter(
        (pl.col("c_eval_bp") == C_MAIN)
        & (pl.col("rule") == "model")
        & ((pl.col("q") - Q).abs() < 1e-9)
    )  # etf_rules covers all test days only (no splits)
    af = etf_rules.filter((pl.col("c_eval_bp") == C_MAIN) & (pl.col("rule") == "always_follow"))
    rows = []
    for sym in [s for syms in config.PART3_CLASSES.values() for s in syms]:
        for t, _ in config.PART2_TIMES:
            r = _one(base, symbol=sym, tau=t)
            a = _one(af, symbol=sym, tau=t)
            rows.append([sym, t, _pm(r["diff_vs_af"], r["diff_vs_af_se"]),
                         _pm(r["mean_net"], r["se"]), _pm(a["mean_net"], a["se"]),
                         _pct(r["coverage"]), f"{r['auc']:.3f}"])  # fmt: skip
    head = [c[k] for k in ("etf", "tau", "diff", "net_model", "net_af", "cov", "auc")]
    return _table(head, rows)


def _render(lang: str, s: dict, f: dict[str, pl.DataFrame], imgs: dict[str, str]) -> str:
    tx = TEXT[lang]
    missing = [k for k in FRAMES if k not in f]
    assert not missing, f"part3 frames missing: {missing}"
    ret = _ret(lang)
    img = lambda k, alt: (
        f'<img src="data:image/png;base64,{imgs[k]}" alt="{escape(tx["alt"][alt])}">'
    )
    sec = lambda n, *parts: (
        f'<h2 id="s{n}">{escape(tx["sec"].format(n=n, title=tx["t" + str(n)].format(tau=TAU)))}</h2>'
        + "".join(parts)
    )
    p = lambda text: f"<p>{escape(text)}</p>"
    fmt = lambda key, **kw: tx[key].format(c=C_MAIN, q=_q(Q), tau=TAU, **ret, **kw)
    findings = _findings_text(f["grid"], f["rules"], s, lang)
    missing = _missing_text(s, lang)
    drop = _drop_text(f["coverage"])
    body = [
        sec(1, p(fmt("x1")), _coverage_table(f["coverage"], lang), p(fmt("x1b", missing=missing))),
        sec(
            2,
            p(fmt("x2")),
            img("part3_effect", "e"),
            _slopes_table(f["effect"], lang),
            p(fmt("x2r")),
            _rules_effect_table(f["effect"], lang, C_MAIN),
            "".join(
                p(fmt("x2c", cost=x)) + _rules_effect_table(f["effect"], lang, x) for x in (0, 4)
            ),
        ),
        sec(3, p(fmt("x3")), img("part3_grid", "g"), _grid_table(f["grid"], f["rules"], lang)),
        sec(4, p(fmt("x4")), _class_rules_table(f["rules"], lang)),
        sec(
            5,
            "<ul>"
            + "".join(
                f"<li>{escape(n.format(c=C_MAIN, drop=drop, **ret))}</li>" for n in tx["limits"]
            )
            + "</ul>",
        ),
        sec(
            6,
            p(fmt("x6")),
            _etf_effect_table(f["effect"], lang),
            p(fmt("x6b")),
            _etf_cells_table(f["etf_rules"], lang),
        ),
    ]
    kind = tx["synthetic"] if s["synthetic"] else f"{s['n_etfs']} ETFs"
    warn = tx["warn"].format(start=s["data_window"][0], end=s["data_window"][1])
    fits = tx["fits"].format(n_fits=s["n_fits"], n_etfs=s["n_etfs"])
    return f"""<!doctype html><html lang="{tx["lang"]}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tx["title"])}</title><style>{CSS}</style></head><body>
<h1>{escape(tx["h1"].format(kind=kind))}</h1>
<p class="warn">{escape(warn)}</p>
<div class="box">{tx["count"].format(count_line=escape(s["count_line"]))}{p(tx["question"].format(n_etfs=s["n_etfs"], **ret))}{p(findings)}{p(fits)}</div>
{"".join(body)}
</body></html>"""


def write_part3_report(out: Path) -> tuple[Path, Path]:
    """Write both pages from `<out>/part3/`, `part3_summary.json` and the figures (see `make_figures`)."""
    s = json.loads((out / "part3_summary.json").read_text())
    f = {k: pl.read_parquet(out / "part3" / f"{k}.parquet") for k in FRAMES}
    imgs = {
        k: base64.b64encode((out / "figures" / f"{k}.png").read_bytes()).decode()
        for k in config.PART3_FIGURES
    }
    paths = (out / "part3_report.html", out / "part3_report_zh.html")
    for path, lang in zip(paths, ("en", "zh"), strict=True):
        path.write_text(_render(lang, s, f, imgs))
    return paths
