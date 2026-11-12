"""Self-contained HTML pages for chapter 2 (English and Taiwan Traditional Chinese).

Reads `<out>/chapter2_summary.json`, `<out>/chapter2/*.parquet` and the figures; writes
`<out>/chapter2_report.html` and `<out>/chapter2_report_zh.html`. The CSS, cell formatters, table
builders and label machinery are chapter 1's (`report.py`); this file only holds chapter 2's
words and layout. Nothing is recomputed: both pages lay out the same numbers.
"""

import base64
import json
from html import escape
from pathlib import Path

import polars as pl

from first_half_hour import config
from first_half_hour.report import (
    CSS,
    FILL_NOTE,
    _fmt,
    _label,
    _rule_table,
    _table,
)

PRIMARY = "up_0.4"
STATED_FAIL = (
    "this question is (almost) all volatility; a model that beats chance would have looked "
    "skilful but adds nothing beyond a one-line formula."
)
BASELINES = ("base_rate", "formula_raw", "formula_cal", "gao_r1")
JUDGE_COLS = [
    "model", "n", "brier", "bss_base", "bss_cal", "log_loss", "log_loss_base", "log_loss_cal",
    "pr_auc", "prevalence", "roc_auc", "roc_auc_null",
]  # fmt: skip
ECO_COLS = [
    "cost_bp", "n_days", "n_traded", "mean", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
    "break_even_bp",
]  # fmt: skip
EN_RULE = {
    "formula_cal": "calibrated formula, same rule",
    "always_buy": "always buy at 10:00 (same exit)",
    "zero": "no trade (zero)",
}
EN_WHAT = {
    "runs": "MLflow runs (parent included)",
    "targets": "targets (1 primary, 3 descriptive)",
    "models_primary": "forecasts of the primary target",
    "configs": "fixed configurations (18 over 3 tiers)",
    "thresholds": "candidate trade thresholds",
    "costs": "cost scenarios",
    "trials": "deflated-Sharpe trials N",
    "spa_models": "models compared in the SPA",
    "test_years": "walk-forward test years",
}
EN_WINDOW = {"valid": "validation", "judge": "judging"}


def stated_result(prim: dict, spa: dict, count_line: str) -> str:
    """The stated result: plan 02's sentence if the primary criterion fails, else the finding."""
    if not prim["primary_pass"]:
        return STATED_FAIL
    return (
        f"there is information beyond volatility (SPA p = {spa['p_consistent']:.3f} over "
        f"{spa['n_models']} models; {count_line})"
    )


TEXT = {
    "en": {
        "lang": "en",
        "title": "Chapter 2: can ML beat a volatility-only formula?",
        "h1": "Chapter 2: can ML beat a volatility-only formula at predicting a further +0.4%? ({kind})",
        "synthetic": "SYNTHETIC data (a pipeline check, not a result)",
        "warn": (
            "Data window {start} to {end}. The judging window ({jlo} to {jhi}) is data the "
            "owner has already seen: exploration, not a clean test. The volatility ratio z "
            "was added to the features on the plan auditor's advice before any real-data result "
            "was seen. Not yet fact-checked."
        ),
        "count": "<p><b>{count_line}</b> (plus one model-registry run outside the parent run)</p>",
        "box": (
            "<ul>\n"
            "<li><b>Primary criterion</b> (chosen tier: {tier}, chosen on {vlo} to {vhi}; "
            "judged on {jlo} to {jhi}, {n} days): Brier skill against the calibrated "
            "volatility formula {bss} (no-skill baseline 0), one-sided Diebold-Mariano t {t}, "
            "p {p} (the line to clear is 0.05). Result: <b>{verdict}</b>.</li>\n"
            "<li><b>Stated result</b>: {stated}</li>\n"
            "<li>Validation Brier skill against the calibrated formula, by tier: {tiers}.</li>\n"
            "<li>SPA p {spa_p} over {spa_n} models (24 models, 23 distinct: tier:logit repeats "
            "cfg:logit:none).</li>\n"
            "</ul>"
        ),
        "pass": "PASS",
        "fail": "FAIL",
        "stated_pass": (
            "there is information beyond volatility (SPA p = {p} over {n} models; {count_line})"
        ),
        "stated_fail": STATED_FAIL,
        "sec": "{n}. {title}",
        "t1": "Target and baselines",
        "x1": (
            "Target: y = 1 if the day's high from 10:00 to the close reaches the 10:00 price "
            "(close of the 09:59 bar) plus 0.4%. Each model forecasts P(y = 1) from information "
            "known by 10:00. Baselines: base_rate (the historical share of hits); formula_raw "
            "(2(1 - Phi(z)) from the first half-hour's volatility, no fitting); formula_cal "
            "(that z fed to a logistic regression, THE BAR ML MUST CLEAR); gao_r1 (a logistic "
            "regression on the first half-hour return, reported, not a candidate). Models: three "
            "tiers (logistic regression; elastic-net logistic; HistGradientBoosting, HGB), each "
            "tuned once a year on the previous year's forecasts, refit walk-forward (expanding "
            "window, each test year fitted on all earlier years only). Share of hit days "
            "(prevalence) by window:"
        ),
        "t2": "Validation choice",
        "x2": (
            "The tier is chosen on the validation window only, by the highest Brier skill "
            "against the calibrated formula (BSS = 1 - Brier(tier) / Brier(calibrated); the "
            "Brier score is the mean squared error of the probability, lower is better; 0 "
            "means no better than the baseline). Chosen tier: {tier}."
        ),
        "t3": "Judging metrics, all forecasts",
        "x3": (
            "Judging window, primary target, every forecast on identical days. Brier skill "
            "against the base rate and against the calibrated formula (no-skill value 0); log "
            "loss beside the base rate's and the calibrated formula's; PR-AUC beside the "
            "prevalence (its no-skill value); ROC-AUC beside 0.5."
        ),
        "t4": "Chosen tier by year",
        "x4": "Brier skill of the chosen tier ({tier}) in each judging year.",
        "t5": "Reliability",
        "x5": (
            "Chosen tier and calibrated formula on the judging window, 10 equal-count bins: "
            "predicted probability against observed frequency; the diagonal is perfect."
        ),
        "alt5": "reliability diagram",
        "t6": "Superior predictive ability (SPA)",
        "x6": (
            "Hansen's SPA test, benchmark the calibrated formula, squared-error loss, stationary "
            "bootstrap: consistent p = {p} over {n} models. {note}"
        ),
        "t7": "Economics",
        "x7": (
            "Buy at the 10:00 price when the forecast is above a threshold; exit at +0.4% if "
            "the target was hit, else at the close. The threshold for each year is chosen from "
            "0.3, 0.4, 0.5, 0.6 and 0.7 on the previous year's forecasts. Costs are per side "
            "(0, 0.5, 1, 2, 5 bp), two sides per trade. Beside the chosen tier and the "
            "calibrated formula: always buy (the same exit every day) and zero (no trade). "
            "break_even_bp is the per-side cost at which the mean return reaches zero. Judging "
            "window."
        ),
        "t8": "Deflated Sharpe",
        "x8": (
            "The chosen tier's rule at 1 bp, with its Sharpe deflated for the {n} variants "
            "tried (5 thresholds x 4 forecasts; Bailey and Lopez de Prado 2014). All Sharpe "
            "ratios here are daily, not annualised. dsr is a probability that the Sharpe beats "
            "what the best of {n} lucky variants would show; a value near 1 would be "
            "convincing, 0.5 or below is not.{none}"
        ),
        "none_dsr": " The rule never traded, so the deflated Sharpe is undefined.",
        "t9": "What matters beyond volatility",
        "x9": (
            "Permutation importance of the tuned HGB tier: the rise in Brier score when one "
            "feature is shuffled, on each judging year's rows, averaged over years. The "
            "volatility features (sigma, z, |r_on|, |r_fh|, or_high, or_low, rv_day_lag, "
            "vix_lag, vix_ratio_lag) are also shuffled as one group, so that skill spread over "
            "several of them is not understated. Top features beyond sigma: {top}. Partial "
            "dependence plots show the HGB model's average predicted probability as one "
            "feature varies."
        ),
        "alt_imp": "permutation importance",
        "alt_pdp": "partial dependence",
        "t10": "Descriptive targets",
        "x10": (
            "Not part of the primary criterion: a further +0.2% and +0.6%, and a further -0.4% "
            "drop, with the baselines and the chosen tier (re-tuned the same way), judging "
            "window."
        ),
        "t11": "What was tried",
        "x11": "Counts of everything run for this chapter.",
        "z_note": "",
    },
    "zh": {
        "lang": "zh-Hant-TW",
        "title": "第 2 章：機器學習能勝過只看波動率的公式嗎？",
        "h1": "第 2 章：機器學習能勝過只看波動率的公式、預測再漲 0.4% 嗎？（{kind}）",
        "synthetic": "合成資料（只是檢查流程，不是結果）",
        "warn": (
            "資料期間 {start} 到 {end}。評斷期（{jlo} 到 {jhi}）的資料擁有者已經看過，所以這是"
            "探索，不是乾淨的檢驗。公式的 z 值（價格要漲 0.4% 相對於波動的倍數）是在計畫審查員"
            "建議下，於看到任何真實資料結果之前加入特徵的。尚未經過事實查核。"
        ),
        "count": (
            "<p><b>{count_line}</b>（執行次數：本章在 MLflow 裡跑了幾次，含母執行；"
            "另有一次模型登錄執行不在母執行之內）</p>"
        ),
        "box": (
            "<ul>\n"
            "<li><b>主要判準</b>（選出的模型：{tier}，依 {vlo} 到 {vhi} 的驗證期選出；"
            "在 {jlo} 到 {jhi} 的評斷期共 {n} 天判斷）：對比「已校準的波動公式」的 Brier 技能"
            "分數 {bss}（Brier 分數是機率預測的平均平方誤差，越低越好；Brier 技能分數是 "
            "1 減去模型誤差除以基準誤差，0 代表沒有比基準好；基準值 0），單邊 Diebold-Mariano "
            "t 值 {t}、p 值 {p}（Diebold-Mariano 檢定是檢查兩個預測的誤差差距是否真的大於"
            "巧合；要通過必須低於 0.05）。結果：<b>{verdict}</b>。</li>\n"
            "<li><b>陳述的結論</b>：{stated}</li>\n"
            "<li>驗證期對比已校準公式的 Brier 技能分數，三層各自：{tiers}。</li>\n"
            "<li>SPA p 值 {spa_p}，比較 {spa_n} 個預測方法（24 個方法其中 23 個不重複：調整後的羅吉斯"
            "迴歸和固定設定的羅吉斯迴歸是同一個預測）。</li>\n"
            "</ul>"
        ),
        "pass": "通過",
        "fail": "未通過",
        "stated_pass": "有超出波動率的資訊（SPA p 值 {p}，比較 {n} 個預測方法；{count_line}）",
        "stated_fail": (
            "這個問題（幾乎）全是波動率；一個能勝過隨機猜測的模型看起來會很有本事，"
            "但比一行公式並沒有多出任何東西。"
        ),
        "sec": "{n}. {title}",
        "t1": "目標與基準",
        "x1": (
            "目標：10:00 到收盤之間的最高價，有沒有碰到 10:00 價格（09:59 那根 K 棒的收盤價）"
            "再加 0.4%；碰到為 1，沒碰到為 0。每個模型只用 10:00 以前已知的資訊，預測碰到的機率。"
            "基準有四個：基準率（永遠猜歷史上碰到的比例）；未校準的波動公式（只用第一個半小時的"
            "波動率算出機率，完全不估計參數）；已校準的波動公式（把同一個 z 值丟進羅吉斯迴歸，"
            "這是機器學習必須跨過的門檻）；Gao 訊號（只用第一個半小時報酬的羅吉斯迴歸，只報告，"
            "不是候選）。模型有三層：羅吉斯迴歸；彈性網路（elastic-net，在羅吉斯迴歸上加一個"
            "把用不到的係數壓小的懲罰）；梯度提升樹（HGB，一棵接一棵修正前面錯誤的決策樹）。每層"
            "每年用前一年的預測成績調一次設定，並用 walk-forward（往前推進：每個測試年只用更早的"
            "年份訓練，訓練區間逐年擴大）重新訓練。各期間碰到目標的天數占比（實際發生率）："
        ),
        "t2": "驗證期的選擇",
        "x2": (
            "只用驗證期選模型層級：對比已校準公式的 Brier 技能分數最高者勝出（Brier 技能分數 = "
            "1 − 模型的 Brier 分數 ÷ 校準公式的 Brier 分數；0 代表沒有比基準好）。選出的是："
            "{tier}。"
        ),
        "t3": "評斷期的全部指標",
        "x3": (
            "評斷期、主要目標，所有預測在完全相同的日子上比較。BSS 是 Brier 技能分數，分別對比"
            "基準率與已校準公式（無技能值 0）；對數損失（log loss，對機率猜錯又很有把握的預測"
            "懲罰特別重，越低越好）旁邊列出基準率與已校準公式的值；PR-AUC（精確率—召回率曲線下面積，"
            "專看抓出「碰到」的那些天抓得準不準）旁邊是實際發生率（它的無技能值）；ROC-AUC"
            "（ROC 曲線下面積，衡量能不能把會碰到的天排在前面）旁邊是 0.5（無技能值）。"
        ),
        "t4": "選出的模型逐年表現",
        "x4": "選出的模型（{tier}）在評斷期每一年的 Brier 技能分數。",
        "t5": "可靠度圖",
        "x5": (
            "選出的模型與已校準公式在評斷期的表現，分成 10 組、每組天數相同：橫軸是預測機率，"
            "縱軸是實際發生的頻率；對角線代表完全準確。"
        ),
        "alt5": "可靠度圖",
        "t6": "SPA 檢定",
        "x6": (
            "SPA 檢定（Hansen 的優越預測能力檢定：在一大堆候選預測裡，檢查最好的那個是否只是"
            "運氣好），基準是已校準公式，用平方誤差，並以 stationary bootstrap（隨機區塊重抽）"
            "計算：一致 p 值 {p}，比較 {n} 個預測方法。{note}"
        ),
        "t7": "交易報酬",
        "x7": (
            "預測機率高於門檻時，在 10:00 的價格買進；若目標有碰到就在 +0.4% 賣出，否則收盤"
            "賣出。每年的門檻，是用前一年的預測從 0.3、0.4、0.5、0.6、0.7 中挑的。成本是單邊"
            "（0、0.5、1、2、5 bp；bp 即基點，1 bp = 0.01%），一次交易有買賣兩邊。表中也列出"
            "選出的模型和已校準公式，旁邊是兩個基準：每天都買（同樣的出場規則）和不交易（零）。"
            "損益兩平成本是讓平均報酬降到零的單邊成本。評斷期。"
        ),
        "t8": "Deflated Sharpe",
        "x8": (
            "選出的模型在 1 bp 成本下的交易規則，其 Sharpe（平均報酬除以波動，衡量每單位風險"
            "賺多少）要扣掉「試過 {n} 種變化（5 個門檻 × 4 種預測）」造成的運氣成分，這叫 "
            "deflated Sharpe（Bailey 與 Lopez de Prado 2014）。這裡的 Sharpe 都是每日值，沒有年化。"
            "dsr 是這個 Sharpe 勝過「{n} 種變化裡最幸運者」的機率；接近 1 才有說服力，0.5 以下"
            "沒有。{none}"
        ),
        "none_dsr": "這個規則從未交易，所以 deflated Sharpe 沒有定義。",
        "t9": "除了波動率，還有什麼重要",
        "x9": (
            "調整後的 HGB 模型的排列重要度（permutation importance：把某個特徵的值隨機打亂，"
            "看 Brier 分數變差多少，變差越多代表模型越依賴它），用評斷期每一年的資料，再對年份"
            "平均。波動類特徵（波動 σ、z、隔夜與開盤半小時報酬的絕對值、開盤半小時最高最低點、"
            "前一日波動、前一日 VIX 與其比值）也會「整組一起打亂」，免得技能分散在好幾個特徵時被"
            "低估。σ 以外最重要的特徵：{top}。部分相依圖（PDP，partial dependence plot：讓一個"
            "特徵變動、其他不動，看 HGB 模型平均預測機率怎麼變）。"
        ),
        "alt_imp": "排列重要度",
        "alt_pdp": "部分相依圖",
        "t10": "描述用的其他目標",
        "x10": (
            "不屬於主要判準：再漲 0.2%、再漲 0.6%，以及再跌 0.4%，列出基準與選出的模型"
            "（以同樣方式重新調整），評斷期。"
        ),
        "t11": "試過什麼",
        "x11": "本章所有跑過的東西的數量。",
        "z_note": "",
    },
}


def _en(lang: str, df: pl.DataFrame, col: str, mapping: dict) -> pl.DataFrame:
    """English display names for a code column (Chinese goes through `_label` in the table)."""
    if lang != "en":
        return df
    assert set(df[col].unique()) <= set(mapping), f"no English name for {col} values"
    return df.with_columns(pl.col(col).replace(mapping))


def _order(models: list[str], chosen: str) -> list[str]:
    tiers = [f"tier:{t}" for t in config.CH2_TIERS]
    rest = sorted(m for m in models if m not in (*BASELINES, *tiers))
    return [*BASELINES, *tiers, *rest]


def _render(lang: str, s: dict, t: dict, imgs: dict, count_line: str) -> str:
    tx = TEXT[lang]
    chosen = f"tier:{s['chosen_tier']}"
    p = s["primary"]
    f = _fmt
    jlo, jhi = s["windows"]["judging"]
    vlo, vhi = s["windows"]["validation"]
    tier_name = _label(lang, "model", chosen) if lang == "zh" else chosen
    stated = (
        tx["stated_pass"].format(
            p=f(s["spa"]["p_consistent"], "p"), n=s["spa"]["n_models"], count_line=count_line
        )
        if p["primary_pass"]
        else tx["stated_fail"]
    )
    tiers = "; ".join(
        f"{_label(lang, 'model', k) if lang == 'zh' else k} {f(v, 'bss_cal')}"
        for k, v in s["validation_bss_cal_by_tier"].items()
    )
    judge_n = s["judging_by_model"][chosen]["n"]
    box = tx["box"].format(
        tier=escape(tier_name), vlo=vlo, vhi=vhi, jlo=jlo, jhi=jhi, n=judge_n,
        bss=f(p["judging_bss_vs_calibrated"], "bss_cal"), t=f(p["dm_t"], "t"),
        p=f(p["dm_p_one_sided"], "p"), verdict=tx["pass" if p["primary_pass"] else "fail"],
        stated=escape(stated), tiers=escape(tiers), spa_p=f(s["spa"]["p_consistent"], "p"),
        spa_n=s["spa"]["n_models"],
    )  # fmt: skip

    def tab(df: pl.DataFrame, cols: list[str]) -> str:
        return _table(df, cols, lang)

    def sec(num: int, *parts: str, **fmt) -> str:
        title = tx["sec"].format(n=num, title=tx[f"t{num}"])
        return (
            f'<h2 id="s{num}">{escape(title)}</h2><p>{tx[f"x{num}"].format(**fmt)}</p>'
            + "".join(parts)
        )

    def img(key: str, alt: str) -> str:
        return f'<img alt="{escape(tx[alt])}" src="data:image/png;base64,{imgs[key]}">'

    prim = lambda df: df.filter(pl.col("target") == PRIMARY)
    valid, judge = prim(t["valid"]), prim(t["judge"])
    order = _order(judge["model"].to_list(), chosen)
    rank = {m: i for i, m in enumerate(order)}

    def ranked(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(_r=pl.col("model").replace_strict(rank, return_dtype=pl.Int64))
            .sort("_r")
            .drop("_r")
        )

    prev = _en(lang, t["prevalence"], "window", EN_WINDOW)
    tier_models = [f"tier:{x}" for x in config.CH2_TIERS]
    v_rows = ranked(valid.filter(pl.col("model").is_in([*BASELINES, *tier_models])))
    by_year = t["by_year"].filter(pl.col("model") == chosen).sort("year")
    eco_order = {chosen: 0, "formula_cal": 1, "always_buy": 2, "zero": 3}
    eco = t["eco"].with_columns(
        _r=pl.col("rule").replace_strict(eco_order, return_dtype=pl.Int64)
    ).sort("_r", "cost_bp")  # fmt: skip

    def rule_name(rec: dict) -> str:
        if lang == "zh":
            return _label(lang, "model", rec["rule"])
        return EN_RULE.get(rec["rule"], f"chosen tier ({rec['rule']})")

    d = s["deflated_sharpe"]
    dsr_df = pl.DataFrame(
        {"n_trials": [d["n_trials"]], "n_days": [d["n_days"]], "sharpe_daily": [d["sharpe_daily"]],
         "sr0_daily": [d["sr0_daily"]], "dsr": [d["dsr"]]}
    )  # fmt: skip
    imp = t["imp"].head(15)
    desc = t["judge"].filter(
        pl.col("model").is_in([*BASELINES, chosen]) & (pl.col("target") != PRIMARY)
    )
    tgt_order = {n: i for i, n in enumerate(t["judge"]["target"].unique(maintain_order=True))}
    desc = desc.with_columns(_t=pl.col("target").replace_strict(tgt_order, return_dtype=pl.Int64))
    desc = desc.sort("_t", "model").drop("_t")
    tried = _en(
        lang, pl.DataFrame({"what": list(s["tried"]), "count": list(s["tried"].values())}),
        "what", EN_WHAT,
    )  # fmt: skip
    top = ", ".join(s["top_features_beyond_sigma"]) or "-"
    if lang == "zh":
        top = "、".join(_label(lang, "feature", x) for x in s["top_features_beyond_sigma"]) or "無"
    note6 = (
        escape(s["spa"]["note"])
        if lang == "en"
        else "24 個方法中有 23 個預測不重複（調整後的羅吉斯迴歸與固定設定的羅吉斯迴歸是同一個預測）。"
    )

    body = [
        sec(1, tab(prev, ["window", "start", "end", "n", "prevalence"])),
        sec(2, tab(v_rows, ["model", "n", "brier", "bss_base", "bss_cal"]), tier=escape(tier_name)),
        sec(3, tab(ranked(judge), JUDGE_COLS)),
        sec(4, tab(by_year, ["year", "n", "brier", "bss_base", "bss_cal", "prevalence"]),
            tier=escape(tier_name)),
        sec(5, img("reliability", "alt5")),
        sec(6, p=f(s["spa"]["p_consistent"], "p"), n=s["spa"]["n_models"], note=note6),
        sec(7, _rule_table(
            eco, ECO_COLS, ("rule",), rule_name, lang,
            lambda rec: FILL_NOTE if rec["rule"] != "zero" else None,
        )),
        sec(8, tab(dsr_df, ["n_trials", "n_days", "sharpe_daily", "sr0_daily", "dsr"]),
            n=d["n_trials"], none=" " + tx["none_dsr"] if d["dsr"] is None else ""),
        sec(9, tab(imp, ["feature", "importance", "std", "n_years_positive"]),
            img("permutation_importance", "alt_imp"), img("partial_dependence", "alt_pdp"),
            top=escape(top)),
        sec(10, tab(desc, ["target", "model", "n", "brier", "bss_base", "bss_cal", "pr_auc",
                           "prevalence", "roc_auc"])),
        sec(11, tab(tried, ["what", "count"])),
    ]  # fmt: skip
    kind = tx["synthetic"] if s["synthetic"] else "SPY"
    warn = tx["warn"].format(start=s["data_window"][0], end=s["data_window"][1], jlo=jlo, jhi=jhi)
    return f"""<!doctype html><html lang="{tx["lang"]}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tx["title"])}</title><style>{CSS}</style></head><body>
<h1>{escape(tx["h1"].format(kind=kind))}</h1>
<p class="warn">{escape(warn)}</p>
<div class="box">{tx["count"].format(count_line=escape(count_line))}{box}</div>
{"".join(body)}
</body></html>"""


def write_chapter2_report(out: Path, count_line: str) -> tuple[Path, Path]:
    """Write the English and Chinese pages from the files `fhh reproduce` already wrote."""
    s = json.loads((out / "chapter2_summary.json").read_text())
    d = out / "chapter2"
    t = {
        "prevalence": pl.read_parquet(d / "prevalence.parquet"),
        "valid": pl.read_parquet(d / "metrics_validation.parquet"),
        "judge": pl.read_parquet(d / "metrics_judging.parquet"),
        "by_year": pl.read_parquet(d / "metrics_by_year.parquet"),
        "eco": pl.read_parquet(d / "economics.parquet"),
        "imp": pl.read_parquet(d / "importance.parquet"),
    }
    imgs = {
        k: base64.b64encode((out / "figures" / f"{k}.png").read_bytes()).decode()
        for k in ("reliability", "permutation_importance", "partial_dependence")
    }
    paths = (out / "chapter2_report.html", out / "chapter2_report_zh.html")
    for path, lang in zip(paths, ("en", "zh"), strict=True):
        path.write_text(_render(lang, s, t, imgs, count_line))
    return paths
