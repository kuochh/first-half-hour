"""Part 1 figures and the two-language report (replicating Gao et al. and Baltussen et al. on SPY).

Reads `<out>/part1/rows.parquet` (the tidy rows of `part1.part1_rows`) and `<out>/part1_summary.json`
and the two figures; writes `<out>/part1_report.html` and `<out>/part1_report_zh.html`. The CSS and
cell formatters are chapter 1's (`report.py`). Nothing is recomputed: both pages lay out the same
numbers, and a code label with no Chinese entry fails an assertion instead of showing raw.

The figures read the private per-day series (`cumulative.parquet`, `rolling.parquet`) and are the
only place those series are used; the report itself shows aggregates only.
"""

import base64
import json
from datetime import date
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from first_half_hour import config
from first_half_hour.report import CSS, _sig

GAO = "Gao WP Oct 2014"
BALT = "Baltussen SSRN Aug 2020"

RET = {  # the four returns, named once (D50); every reader-facing text looks them up here
    "r1": ("previous close to 10:00", "前一天收盤到 10:00"),
    "r12": ("15:00 to 15:30", "15:00 到 15:30"),
    "r13": ("15:30 to close", "15:30 到收盤"),
    "rod": ("previous close to 15:30", "前一天收盤到 15:30"),
}
RET_EN = {k: v[0] for k, v in RET.items()}
RET_ZH = {k: v[1] for k, v in RET.items()}

ITEMS: dict[str, tuple[str, str, str]] = {
    # item: (paper, English title, Chinese title)
    "gao_slope": ("gao", "G1 slope of {r13} on {r1}", "G1 {r13} 對 {r1} 的斜率"),
    "gao_r12": ("gao", "G2 slope of {r13} on {r12}", "G2 {r13} 對 {r12} 的斜率"),
    "gao_joint": ("gao", "G3 {r13} on {r1} and {r12}", "G3 {r13} 對 {r1} 與 {r12}"),
    "gao_vol_tercile": (
        "gao", "G4 slope by tercile of 9:30 to 10:00 volatility", "G4 依 9:30 到 10:00 波動三分位的斜率",
    ),
    "gao_oos_r2": ("gao", "G5 out-of-sample R2", "G5 樣本外 R²"),
    "gao_rule": ("gao", "G6 trade {r13} on the sign of {r1}", "G6 依 {r1} 的正負號交易 {r13}"),
    "gao_news": ("gao", "G7 R2 on FOMC and CPI days", "G7 FOMC 與 CPI 日的 R²"),
    "balt_slope": ("baltussen", "B1 slope of {r13} on {rod}", "B1 {r13} 對 {rod} 的斜率"),
    "balt_horse": ("baltussen", "B2 {r13} on {r1} and {rod}", "B2 {r13} 對 {r1} 與 {rod}"),
    "balt_large_move": (
        "baltussen", "B3 after a large first jump", "B3 第一次大幅跳動之後",
    ),
    "balt_reversal": ("baltussen", "B4 next-day reversal", "B4 隔日反轉"),
    "balt_last_k": ("baltussen", "B5 last 30, 15 and 5 minutes", "B5 最後 30、15、5 分鐘"),
    "balt_rule": ("baltussen", "B6 trade {r13} on the sign of {rod}", "B6 依 {rod} 的正負號交易 {r13}"),
}  # fmt: skip

VARIANT = {
    "main": ("main", "主要設定"),
    "check_1559": ("15:59 bar check", "15:59 分鐘棒檢查"),
    "cost_1bp": ("net of 1 bp per side", "扣除單邊 1 bp 成本"),
    "full_window": ("cut-offs from the whole window", "用整段期間定界線"),
    "past_only": ("cut-offs from earlier sessions only", "只用之前的日子定界線"),
    "always_long": ("always long", "一直做多"),
}  # fmt: skip
METRIC = {
    "slope": ("slope", "斜率"),
    "r2": ("R2 (%)", "R²（%）"),
    "r2_os": ("out-of-sample R2 (%)", "樣本外 R²（%）"),
    "ann_mean": ("annual mean (%)", "年化平均報酬（%）"),
    "sharpe": ("Sharpe", "Sharpe"),
    "success": ("success rate (%)", "勝率（%）"),
    "success_excess": ("success rate minus always-long (pp)", "勝率減一直做多的勝率（百分點）"),
}  # fmt: skip
TERM = {
    **{k: v for k, v in RET.items()},
    "r1+r12": (f"{RET['r1'][0]} + {RET['r12'][0]}", f"{RET['r1'][1]} + {RET['r12'][1]}"), "rule": ("rule", "規則"),
    "low": ("low volatility", "低波動"), "mid": ("mid volatility", "中波動"),
    "high": ("high volatility", "高波動"),
    "FOMC": ("FOMC days", "FOMC 日"), "non-FOMC": ("non-FOMC days", "非 Fed 日"),
    "CPI": ("CPI days", "CPI 日"), "non-CPI": ("non-CPI days", "非 CPI 日"),
    "x_jump": ("first jump", "第一次跳動"),
    "k1": ("next 1 day", "隔 1 日"), "k2": ("next 2 days", "隔 2 日"), "k3": ("next 3 days", "隔 3 日"),
    "r_last15": ("last 15 min", "最後 15 分鐘"), "r_last5": ("last 5 min", "最後 5 分鐘"),
}  # fmt: skip
MARKET = {
    "SPY": ("SPY", "SPY"),
    "S&P 500 E-mini futures": ("S&P 500 E-mini futures", "S&P 500 小型期貨"),
    "pooled equity-index futures": ("pooled equity-index futures", "股價指數期貨合併樣本"),
    "S&P 500 cash index": ("S&P 500 cash index", "S&P 500 現貨指數"),
    "equity-index futures, 1/N portfolio": ("equity-index futures, 1/N portfolio", "股價指數期貨 1/N 投組"),
}  # fmt: skip
VERDICT = {
    "replicated": ("replicated", "重現"),
    "not replicated": ("not replicated", "沒重現"),
    "cannot tell": ("cannot tell", "分辨不出來"),
    "sample too small": ("sample too small", "樣本太少"),
}
WINDOW = {
    "gao_overlap": ("Gao overlap", "Gao 重疊期"),
    "gao_after": ("after Gao's data", "Gao 資料之後"),
    "balt_overlap": ("Baltussen overlap", "Baltussen 重疊期"),
    "balt_after": ("after Baltussen's data", "Baltussen 資料之後"),
}
NULL_LABEL = {
    "0": ("0", "0"),
    "always-long success": ("always-long success", "一直做多的勝率"),
    "no-skill R² k/(n-1)": ("no-skill R² k/(n-1)", "無技能 R² k/(n-1)"),
}
NOTE = {  # the analysis' row notes (part1.py), in both languages; an unknown note fails loud
    (
        "FOMC = scheduled statement days; Gao's table says 'FOMC minutes release at 2:15 pm', "
        "which matches statement days before 2013"
    ): (
        (
            "FOMC = scheduled statement days (Gao's table says minutes at 2:15 pm, which matches "
            "statement days before 2013)"
        ),
        "FOMC＝排定的聲明日（Gao 的表寫會議紀錄 2:15 pm，和 2013 年以前的聲明日吻合）",
    ),
    "terciles of the whole window (the paper's way; uses the future)": (
        "terciles of the whole window (the paper's way; uses the future)",
        "用整段期間的三分位（論文的做法；用到未來）",
    ),
    "cut-offs from earlier sessions only": (
        "cut-offs from earlier sessions only",
        "只用之前的日子定界線",
    ),
    "percentiles of the whole window (the paper's way); first jump": (
        "percentiles of the whole window (the paper's way); first jump",
        "用整段期間的百分位（論文的做法）；第一次跳動",
    ),
    "percentiles of earlier sessions only; first jump": (
        "percentiles of earlier sessions only; first jump",
        "只用之前日子的百分位；第一次跳動",
    ),
    "recursive monthly refit vs the expanding mean; earlier sessions are the warm-up": (
        "recursive monthly refit against the expanding mean; earlier sessions are the warm-up",
        "每月遞迴重新估計，對照擴張平均；更早的交易日是暖機期",
    ),
    "same regression as the 0.062 row": (
        "same regression as the 0.062 row (B1)",
        "和 0.062 那一列（B1）是同一個迴歸",
    ),
    (
        "the paper's figure is for a 1/N portfolio of 17 futures, whose Sharpe and success rate differ from "
        "a single market's (diversification): no verdict"
    ): (
        (
            "the paper's figure is for a 1/N portfolio of 17 futures, whose Sharpe and success rate differ from "
            "a single market's (diversification): no verdict"
        ),
        "論文的數字是 17 種期貨的 1/N 投組，分散效果讓 Sharpe 和勝率都和單一市場不同：不做判定",
    ),
}
ROW_COLS = (
    "item", "paper", "window", "variant", "term", "metric", "n", "ours", "se", "se_kind", "lo",
    "hi", "paper_value", "paper_source", "paper_market", "null_value", "null_label", "se_null",
    "d_null", "d_paper", "verdict", "mde", "small_sample", "cw_p", "loo_min", "note",
)  # fmt: skip


def _lab(table: dict, key, lang: str, what: str) -> str:
    assert key in table, f"no {lang} label for {what} {key!r}: add it to report_part1"
    return table[key][0 if lang == "en" else 1]


def _kind(window: str) -> str:
    assert window.endswith(("_overlap", "_after")), f"unexpected window name {window!r}"
    return "overlap" if window.endswith("_overlap") else "after"


def _num(metric: str, v: float | None) -> str:
    """One number in its metric's unit: plain for slopes, 2 decimals for Sharpe, a percent else."""
    if v is None:
        return ""
    if metric == "slope":
        text = _sig(v)
    elif metric == "sharpe":
        text = f"{v:.2f}"
    else:
        assert metric in ("r2", "r2_os", "ann_mean", "success", "success_excess"), (
            f"unknown metric {metric!r}"
        )
        text = f"{v * 100:.2f}"
    return "0.00" if text == "-0.00" else text


# ---------------------------------------------------------------- figures
def _marks(ax, gao_end: date, balt_end: date) -> None:
    ax.axvline(gao_end, color="firebrick", ls="--", lw=1, label=f"{gao_end} Gao data ends")
    ax.axvline(balt_end, color="darkgreen", ls="--", lw=1, label=f"{balt_end} Baltussen data ends")


def plot_cumulative(cum: pl.DataFrame, path: Path, gao_end: date, balt_end: date) -> None:
    """Wealth from 1, compounded: the two rules and always-long (dashed), at 0 and 1 bp."""
    assert set(cum["rule"].unique()) == {"eta_r1", "eta_rod", "always_long"}, "rules differ"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = {
        "eta_r1": f"trade {RET['r13'][0]} on the sign of {RET['r1'][0]}",
        "eta_rod": f"trade {RET['r13'][0]} on the sign of {RET['rod'][0]}",
        "always_long": "always long",
    }
    colors = {"eta_r1": "C0", "eta_rod": "C1", "always_long": "black"}
    always_long_grey = "grey"  # at 1 bp: dotted grey, so the two always-long lines differ
    for rule, label in names.items():
        for cost in sorted(cum["cost_bp"].unique()):
            s = cum.filter((pl.col("rule") == rule) & (pl.col("cost_bp") == cost)).sort("session")
            assert s.height, f"no series for {rule} at {cost} bp"
            if rule == "always_long":  # 0 bp dashed black, 1 bp dotted grey
                color, ls = ("black", "--") if cost == 0 else (always_long_grey, ":")
            else:
                color, ls = colors[rule], "-" if cost == 0 else ":"
            ax.plot(
                s["session"].to_list(), s["wealth"].to_numpy(), color=color, lw=1.1, ls=ls,
                label=f"{label}, {cost:g} bp per side",
            )  # fmt: skip
    _marks(ax, gao_end, balt_end)
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel(f"wealth (start = 1), {RET['r13'][0]} only")
    ax.set_title(
        f"Cumulative wealth of trading {RET['r13'][0]}: the two rules and always long (dashed, dotted)"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rolling(rolling: pl.DataFrame, path: Path, gao_end: date, balt_end: date) -> None:
    """Rolling slope with +/- 2 NW SE bands for the two signals, the papers' slopes, zero, data ends."""
    assert set(rolling["signal"].unique()) == {"r1", "rod"}, "signals differ"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    papers = {"r1": GAO, "rod": BALT}
    for ax, sig in zip(axes, ("r1", "rod"), strict=True):
        s = rolling.filter(pl.col("signal") == sig).sort("session")
        x = s["session"].to_list()
        slope, se = s["slope"].to_numpy(), s["se"].to_numpy()
        ax.fill_between(x, slope - 2 * se, slope + 2 * se, alpha=0.25, label="+/- 2 Newey-West SE")
        ax.plot(x, slope, lw=1.2, label=f"rolling slope of {RET['r13'][0]} on {RET[sig][0]}")
        ax.axhline(config.PART1_PAPER_SLOPES[sig], color="C3", ls="-.", lw=1,
                   label=f"{papers[sig]}: {config.PART1_PAPER_SLOPES[sig]}")  # fmt: skip
        ax.axhline(0.0, color="black", lw=0.8, label="no skill (slope 0)")
        _marks(ax, gao_end, balt_end)
        ax.set_ylabel(f"slope on {RET[sig][0]}", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel(f"window ending on ({config.PART1_REPORT_ROLLING_WINDOW}-session windows)")
    axes[0].set_title(
        f"Rolling slope of {RET['r13'][0]} on the signals (bands per window, not joint)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report text
TEXT = {
    "en": {
        "lang": "en",
        "title": "Part 1: replicating Gao et al. and Baltussen et al. on SPY",
        "h1": "Part 1: does {r1}, or {rod}, predict {r13}? ({kind})",
        "synthetic": "SYNTHETIC data (a pipeline check, not a result)",
        "warn": "Data window {start} to {end}. Fact-checked 2026-11-16: every number recomputed independently from the raw data.",
        "count": "<p><b>{count_line}</b></p>",
        "papers": (
            "Paper versions held: Gao, Han, Li and Zhou, working paper of October 2014 "
            "(\"Gao WP Oct 2014\"); Baltussen, Da, Lammers and Martens, SSRN version of 26 August "
            "2020 (\"Baltussen SSRN Aug 2020\"). The journal versions were not read. Baltussen's "
            "tables print coefficients times 100; they are divided by 100 here."
        ),
        "sec": "{n}. {title}",
        "t1": "Comparison with the papers",
        "x1": (
            "One line per item, term and variant. The columns are our sample inside the paper's "
            "data window (\"overlap\") and after the paper's data ended (\"after\"): our "
            "number, the no-skill value it is compared with, our number +/- 2 standard errors, the "
            "verdict, and the smallest effect the sample could detect (mde, 80% power under the "
            "2-SE rule). R2 rows also show the smallest R2 after dropping any one day (leave-one-out "
            "minimum). Always-long rows are the baseline: the paper's own always-long figure sits "
            "beside them, with no verdict. The paper's number is beside its source and market. Slopes are plain numbers; R2, annual mean and success "
            "rate are percents. Verdict: replicated = different from the no-skill value "
            "(more than 2 SE) and within 2 SE of the paper; not replicated = more than 2 SE "
            "from the paper; cannot tell = neither; sample too small = fewer than 250 days (run and "
            "shown, not judged). The success rate itself carries no verdict: the verdict is on "
            "the success rate minus always-long's, against the paper's difference. Rows without a "
            "paper number, the 15:59 check, the cost rows and the always-long rows carry no verdict."
        ),
        "t2": "Verdicts by window",
        "x2": (
            "Rows that carry a verdict, counted per window. Only the main and the earlier-sessions-only "
            "variants are counted; the whole-window cut-off rows show their verdict in the table but are "
            "not counted (they use later data, and each has an earlier-sessions-only twin). Rows "
            "with fewer than 250 days are counted as sample too small."
        ),
        "t3": "Out-of-sample R2: Clark-West p",
        "x3": "One-sided Clark-West p of the recursive forecast against the expanding mean.",
        "t4": "Figures",
        "alt_cum": "Cumulative wealth of the two rules and always-long",
        "alt_roll": "Rolling slope with bands and the papers' slopes",
        "t5": "Notes",
        "notes": [
            ("Official close: the 16:00 price is Yahoo's official close, moved onto the minute bars' "
            "split- and dividend-adjusted basis with one constant per dividend period (the median "
            "gap to the 15:59 bar stays within about 3 bp of zero in every period; the day-to-day gap, "
            "which the closing auction adds, is kept). The 15:59 bar versions are the check rows."),
            ("Gao's data runs 1993-02 to 2013-12 and Baltussen's ends 2020-05-01; our data starts "
            "2011-01. \"Overlap\" is therefore short for Gao (three years)."),
            ("Official close, known by the session: each dividend period is adjusted with the "
            "previous period's constant, which uses earlier sessions only. The one exception is the "
            "first period (2011-01-03 to the first ex-date), which uses its own median."),
            ("No-skill values: R2 rows are compared with k/(n-1), the in-sample R2 expected with "
            "k = 2 regressors and no skill, and tested by the Newey-West Wald test that both "
            "slopes are 0 (more than 2 means p below 0.0455). Success rows are compared with "
            "always-long's success rate."),
            ("The past-only tercile rows in the Gao overlap cover 2012-13 only (2011 is the "
            "warm-up for the cut-offs), and their classes are uneven."),
            ("The B1 and B5 (15:30 to close) rows are the same regression; B5's carries no verdict."),
            ("B6 compares a single ETF with the paper's 1/N portfolio of 17 futures, whose Sharpe "
            "is raised by diversification, so its Sharpe and success-excess rows carry no verdict."),
            ("The paper's number is treated as fixed: the error range is ours only, because the "
            "overlap samples share data with the papers."),
            ("Out-of-sample forecasts start 2012-01-03 (2011 is the warm-up), so the Gao overlap "
            "has about 500 forecasts."),
            ("FOMC days are scheduled statement days. Gao's table says \"FOMC minutes release at "
            "2:15 pm\", which matches statement days before 2013."),
            ("Large-move rows (B3): the jump is the first 30-minute boundary (10:00, 10:30, ...) at which the return "
            "from the previous close is below the 10th or above the 90th percentile of "
            "close-to-close returns. The paper does not say which jump it uses when there are "
            "several; this is our choice."),
            ("\"Terciles from the whole window\" is the paper's way and uses later data; the "
            "\"earlier sessions only\" rows are the version a trader could have used."),
        ],
        "t6": "What this report could not do",
        "cant": [
            "Crisis against non-crisis split: our data starts in 2011.",
            "Market-maker gamma split: no dealer position data.",
            "The 60+ futures: ETFs only here (Part 3).",
            "Gao's MCSI and GDP news rows: no release-date files; out of the plan.",
            "The journal versions of both papers were not read.",
        ],
        "cols": {
            "what": "item / term / variant", "paper": "paper", "source": "source (market)",
            "n": "n", "ours": "ours", "range": "+/- 2 SE", "verdict": "verdict", "mde": "mde",
            "null": "no-skill", "loo": "leave-one-out min", "note": "note",
            "overlap": "overlap with the paper's data", "after": "after the paper's data ended",
            "window": "window", "count": "rows",
        },
    },
    "zh": {
        "lang": "zh-Hant-TW",
        "title": "第一部分：用 SPY 重現 Gao 等人與 Baltussen 等人",
        "h1": "第一部分：{r1}，或{rod}，能預測{r13}嗎？（{kind}）",
        "synthetic": "合成資料（只檢查流程，不是結果）",
        "warn": "資料期間 {start} 到 {end}。2026-11-16 已查核：每個數字都由獨立的方法從原始資料重算。",
        "count": "<p><b>{count_line}</b></p>",
        "papers": (
            "使用的論文版本：Gao、Han、Li、Zhou，2014 年 10 月工作論文（「Gao WP Oct 2014」）；"
            "Baltussen、Da、Lammers、Martens，SSRN 2020 年 8 月 26 日版（「Baltussen SSRN Aug 2020」）。"
            "期刊版沒有讀。Baltussen 的表格係數印的是乘以 100 的值，這裡已除以 100。"
        ),
        "sec": "{n}. {title}",
        "t1": "和論文的比較",
        "x1": (
            "每個項目、變數、設定一行。兩組欄位分別是我們的樣本在論文資料期間內（「重疊」）與論文資料結束之後"
            "（「之後」）：我們的數字、拿來比較的無技能值、我們的數字 ± 2 個標準誤、判定，以及樣本能偵測的最小效應"
            "（mde，在 2 倍標準誤規則下檢定力 80%）。R² 的列另外顯示拿掉任何一天後最小的 R²（留一法最小值）。"
            "一直做多的列是基準，論文自己的一直做多數字和它並列，不做判定。論文的數字和它的出處、市場並列。斜率是一般數字；"
            "R²、年化平均報酬、勝率以百分比表示。判定：重現＝和無技能值差超過 2 個標準誤，"
            "且和論文數字差不到 2 個標準誤；沒重現＝和論文數字差超過 2 個標準誤；分辨不出來＝兩者皆非。"
            "樣本太少＝少於 250 天（照做並顯示，不判定）。勝率本身不做判定：判定看的是勝率減一直做多的勝率，"
            "對照論文的差距。沒有論文數字的列、15:59 檢查列、成本列和一直做多的列不做判定。"
        ),
        "t2": "各期間的判定",
        "x2": (
            "只算有判定的列，依期間計數。只計主要設定與「只用之前的日子定界線」的列；"
            "「用整段期間定界線」的列在表中仍顯示判定，但不計入（它們用到之後的資料，且各有一列只用之前日子的對照）。"
            "少於 250 天的列計入「樣本太少」。"
        ),
        "t3": "樣本外 R²：Clark-West p",
        "x3": "遞迴預測對擴張平均的單邊 Clark-West p 值。",
        "t4": "圖",
        "alt_cum": "兩個規則與一直做多的累積財富",
        "alt_roll": "滾動斜率、區間與論文斜率",
        "t5": "附註",
        "notes": [
            ("官方收盤價：16:00 價格是 Yahoo 的官方收盤價，每個股息期間用一個常數調整到分鐘棒"
            "（已調整分割與股息）的基準上（各期間和 15:59 分鐘棒的差距中位數都在 0 上下約 3 bp 內；"
            "收盤拍賣造成的逐日差距保留）。用 15:59 分鐘棒算的版本是檢查列。"),
            ("Gao 的資料是 1993-02 到 2013-12，Baltussen 的資料到 2020-05-01 結束；我們的資料從 2011-01 開始。"
            "所以 Gao 的「重疊」期間很短（三年）。"),
            ("官方收盤價，只用當時已知的資訊：每個股息期間用前一個股息期間的常數調整，只讀更早的交易日。"
            "唯一例外是第一個期間（2011-01-03 到第一個除息日），用它自己的中位數。"),
            ("無技能值：R² 的列拿 k/(n-1) 比較（k = 2 個自變數、沒有技能時樣本內 R² 的期望值），並用 Newey-West "
            "Wald 檢定兩個斜率皆為 0（大於 2 即 p 小於 0.0455）。勝率的列拿一直做多的勝率比較。"),
            "Gao 重疊期間「只用之前的日子」的三分位列只涵蓋 2012-13 年（2011 年是定界線的暖機期），各類天數不均。",
            "B1 與 B5（15:30 到收盤）的列是同一個迴歸；B5 的那一列不做判定。",
            ("B6 拿單一 ETF 對照論文 17 種期貨的 1/N 投組，分散效果會拉高投組的 Sharpe，"
            "所以它的 Sharpe 與勝率差距列不做判定。"),
            "論文的數字視為固定：誤差範圍只含我們這邊的，因為重疊樣本和論文共用資料。",
            "樣本外預測從 2012-01-03 開始（2011 年是暖機期），所以 Gao 的重疊期間約 500 筆預測。",
            "FOMC 日指排定的聲明日。Gao 的表寫「FOMC 會議紀錄 2:15 pm 公布」，和 2013 年以前的聲明日吻合。",
            ("大幅跳動列（B3）：跳動是第一個整點或半點時點，其相對前一收盤的報酬低於收盤對收盤報酬的第 10 百分位"
            "或高於第 90 百分位。論文沒說有多次跳動時用哪一次；這是我們的選擇。"),
            "「用整個期間算分位」是論文的做法，會用到之後的資料；「只用更早的交易日」才是交易者當時做得到的版本。",
        ],
        "t6": "本報告做不到的事",
        "cant": [
            "危機與非危機的分割：我們的資料從 2011 年開始。",
            "造市商 gamma 分割：沒有造市商部位資料。",
            "60 多種期貨：這裡只做 ETF（第三部分）。",
            "Gao 的 MCSI 與 GDP 新聞列：沒有公布日檔案，不在計畫內。",
            "兩篇論文的期刊版都沒有讀。",
        ],
        "cols": {
            "what": "項目／變數／設定", "paper": "論文數字", "source": "出處（市場）",
            "n": "天數", "ours": "我們的數字", "range": "± 2 標準誤", "verdict": "判定",
            "mde": "mde", "null": "無技能值", "loo": "留一法最小值", "note": "附註", "overlap": "和論文資料重疊的期間", "after": "論文資料結束之後",
            "window": "期間", "count": "列數",
        },
    },
}  # fmt: skip


SUB = ("n", "null", "ours", "range", "loo", "verdict", "mde")


def _null_cell(r: dict, metric: str, lang: str) -> str:
    """The no-skill value with its label ("0" alone when the value is zero). Blank on the
    always-long rows: they are the baseline themselves."""
    if r["variant"] == "always_long":
        return ""
    label = _lab(NULL_LABEL, r["null_label"], lang, "no-skill label")
    if r["null_label"] == "0":
        return "0"
    return f"{_num(metric, r['null_value'])} ({label})"


SIZE_NOTE = {
    "larger": ("same direction, much larger, wide error", "方向相同、大很多、誤差很寬"),
    "smaller": ("same direction, much smaller, wide error", "方向相同、小很多、誤差很寬"),
}
"""Size note on a "replicated" row whose number is far from the paper's (PM working default, D52)."""


def _size_note(r: dict) -> str | None:
    """ "larger"/"smaller" when a replicated row is more than PART1_SIZE_RATIO times (or under
    1/PART1_SIZE_RATIO of) the paper's number with the same sign; None otherwise."""
    if r["verdict"] != "replicated" or not r["paper_value"]:
        return None
    ratio = r["ours"] / r["paper_value"]
    if ratio > config.PART1_SIZE_RATIO:
        return "larger"
    if 0 < ratio < 1 / config.PART1_SIZE_RATIO:
        return "smaller"
    return None


def _cell_group(r: dict | None, metric: str, lang: str) -> str:
    """The cells of one window (SUB); blank when it has no row."""
    if r is None:
        return "<td></td>" * len(SUB)
    verdict = "" if r["verdict"] is None else _lab(VERDICT, r["verdict"], lang, "verdict")
    size = _size_note(r)
    if size is not None:
        verdict = f"{verdict} ({_lab(SIZE_NOTE, size, lang, 'size note')})"
    loo = "" if r["loo_min"] is None else _num(metric, r["loo_min"])
    cells = [
        str(r["n"]), _null_cell(r, metric, lang), _num(metric, r["ours"]),
        f"[{_num(metric, r['lo'])}, {_num(metric, r['hi'])}]", loo, verdict, _num(metric, r["mde"]),
    ]  # fmt: skip
    return "".join(f"<td>{escape(c)}</td>" for c in cells)


def _comparison(rows: pl.DataFrame, lang: str) -> str:
    tx = TEXT[lang]
    c = tx["cols"]
    group = lambda k: f'<th class="g" colspan="{len(SUB)}">{escape(c[k])}</th>'
    head1 = f'<tr><th colspan="3"></th>{group("overlap")}{group("after")}<th></th></tr>'
    head2 = (
        "<tr>"
        + "".join(f"<th>{escape(c[k])}</th>" for k in ("what", "paper", "source"))
        + "".join(f"<th>{escape(c[k])}</th>" for k in SUB * 2)
        + f"<th>{escape(c['note'])}</th></tr>"
    )
    body = []
    n_cols = 3 + 2 * len(SUB) + 1
    for item, (paper, title_en, title_zh) in ITEMS.items():
        part = rows.filter(pl.col("item") == item)
        assert part.height, f"no rows for item {item!r}"
        assert part["paper"].unique().to_list() == [paper], f"{item}: paper column is not {paper}"
        title = (title_en if lang == "en" else title_zh).format(
            **(RET_EN if lang == "en" else RET_ZH)
        )
        body.append(f'<tr><th class="sub" colspan="{n_cols}">{escape(title)}</th></tr>')
        seen = []
        for r in part.iter_rows(named=True):
            key = (r["term"], r["variant"], r["metric"])
            if key in seen:
                continue
            seen.append(key)
            grp = {
                _kind(w["window"]): w
                for w in part.filter(
                    (pl.col("term") == key[0])
                    & (pl.col("variant") == key[1])
                    & (pl.col("metric") == key[2])
                ).iter_rows(named=True)
            }
            assert len(grp) == part.filter(
                (pl.col("term") == key[0]) & (pl.col("variant") == key[1])
                & (pl.col("metric") == key[2])
            ).height, f"{item} {key}: two rows in one window kind"  # fmt: skip
            any_r = next(iter(grp.values()))
            paper_v = any_r["paper_value"]
            src = (
                ""
                if paper_v is None
                else f"{any_r['paper_source']} ({_lab(MARKET, any_r['paper_market'], lang, 'market')})"
            )
            what = ", ".join(
                (
                    _lab(TERM, key[0], lang, "term"),
                    _lab(VARIANT, key[1], lang, "variant"),
                    _lab(METRIC, key[2], lang, "metric"),
                )
            )
            notes = {w["note"] for w in grp.values() if w["note"] is not None}
            assert len(notes) <= 1, f"{item} {key}: different notes in the two windows"
            note = "" if not notes else _lab(NOTE, notes.pop(), lang, "note")
            body.append(
                "<tr>"
                f'<td class="wrap">{escape(what)}</td>'
                f"<td>{escape(_num(key[2], paper_v))}</td>"
                f'<td class="wrap">{escape(src)}</td>'
                f"{_cell_group(grp.get('overlap'), key[2], lang)}"
                f"{_cell_group(grp.get('after'), key[2], lang)}"
                f'<td class="wrap">{escape(note)}</td></tr>'
            )
    return (
        f'<div class="t" style="overflow-x:auto"><table>{head1}{head2}{"".join(body)}</table></div>'
    )


COUNTED_VARIANTS = ("main", "past_only")


def _verdict_counts(rows: pl.DataFrame, lang: str) -> str:
    """Verdicts per window, main and past_only variants only (full_window rows are not counted)."""
    c = TEXT[lang]["cols"]
    rated = rows.filter(pl.col("verdict").is_not_null() & pl.col("variant").is_in(COUNTED_VARIANTS))
    assert rated.height, "no row carries a counted verdict"
    head = "".join(
        f"<th>{escape(h)}</th>"
        for h in (c["window"], *(_lab(VERDICT, v, lang, "verdict") for v in VERDICT))
    )
    body = ""
    for w in sorted(rated["window"].unique().to_list()):
        counts = [
            rated.filter((pl.col("window") == w) & (pl.col("verdict") == v)).height for v in VERDICT
        ]
        body += (
            f"<tr><td>{escape(_lab(WINDOW, w, lang, 'window'))}</td>"
            + "".join(f"<td>{n}</td>" for n in counts)
            + "</tr>"
        )
    return f'<div class="t" style="overflow-x:auto"><table><tr>{head}</tr>{body}</table></div>'


def _cw_table(rows: pl.DataFrame, lang: str) -> str:
    c = TEXT[lang]["cols"]
    oos = rows.filter((pl.col("item") == "gao_oos_r2") & pl.col("cw_p").is_not_null()).sort(
        "window"
    )
    assert oos.height, "no Clark-West p on the out-of-sample rows"
    head = "".join(
        f"<th>{escape(h)}</th>" for h in (c["window"], c["n"], c["ours"], "Clark-West p")
    )
    body = "".join(
        f"<tr><td>{escape(_lab(WINDOW, r['window'], lang, 'window'))}</td><td>{r['n']}</td><td>{_num('r2_os', r['ours'])}%</td>"
        f"<td>{'<0.001' if r['cw_p'] < 0.001 else f'{r["cw_p"]:.3f}'}</td></tr>"
        for r in oos.iter_rows(named=True)
    )
    return f'<div class="t" style="overflow-x:auto"><table><tr>{head}</tr>{body}</table></div>'


def _render(lang: str, s: dict, rows: pl.DataFrame, imgs: dict[str, str]) -> str:
    tx = TEXT[lang]
    missing = [c for c in ROW_COLS if c not in rows.columns]
    assert not missing, f"rows.parquet lacks columns {missing}"
    img = lambda k, alt: f'<img src="data:image/png;base64,{imgs[k]}" alt="{escape(tx[alt])}">'
    sec = lambda n, key, *parts: (
        f'<h2 id="s{n}">{escape(tx["sec"].format(n=n, title=tx["t" + str(key)]))}</h2>'
        + "".join(parts)
    )
    kind = tx["synthetic"] if s["synthetic"] else "SPY"
    warn = tx["warn"].format(start=s["data_window"][0], end=s["data_window"][1])
    notes = "".join(f"<li>{escape(n)}</li>" for n in tx["notes"])
    cant = "".join(f"<li>{escape(n)}</li>" for n in tx["cant"])
    body = [
        sec(1, 1, f"<p>{escape(tx['x1'])}</p>", _comparison(rows, lang)),
        sec(2, 2, f"<p>{escape(tx['x2'])}</p>", _verdict_counts(rows, lang)),
        sec(3, 3, f"<p>{escape(tx['x3'])}</p>", _cw_table(rows, lang)),
        sec(4, 4, img("part1_cumulative", "alt_cum"), img("part1_rolling_slope", "alt_roll")),
        sec(5, 5, f"<ul>{notes}</ul>"),
        sec(6, 6, f"<ul>{cant}</ul>"),
    ]
    return f"""<!doctype html><html lang="{tx["lang"]}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tx["title"])}</title><style>{CSS}</style></head><body>
<h1>{escape(tx["h1"].format(kind=kind, **(RET_EN if lang == "en" else RET_ZH)))}</h1>
<p class="warn">{escape(warn)}</p>
<div class="box">{tx["count"].format(count_line=escape(s["count_line"]))}<p>{escape(tx["papers"])}</p></div>
{"".join(body)}
</body></html>"""


def write_part1_report(out: Path) -> tuple[Path, Path]:
    """Write both pages from `<out>/part1/rows.parquet`, `part1_summary.json` and the figures."""
    s = json.loads((out / "part1_summary.json").read_text())
    rows = pl.read_parquet(out / "part1" / "rows.parquet")
    imgs = {
        k: base64.b64encode((out / "figures" / f"{k}.png").read_bytes()).decode()
        for k in config.PART1_FIGURES
    }
    paths = (out / "part1_report.html", out / "part1_report_zh.html")
    for path, lang in zip(paths, ("en", "zh"), strict=True):
        path.write_text(_render(lang, s, rows, imgs))
    return paths
