"""Self-contained HTML pages for chapter 1: a table per analysis item, baselines beside the numbers.

Reads `<out>/chapter1/item1..9.parquet`, `<out>/chapter1_summary.json` and the rolling-slope
figure; writes `<out>/chapter1_report.html` (English) and `<out>/chapter1_report_zh.html`
(Taiwan Traditional Chinese). Inline CSS, the figure embedded as base64, no JS, no external
assets. Nothing is recomputed here: `_render` lays the numbers out once per language and every
word comes from the `TEXT` table, so both pages show identical numbers. A code label (sample,
variant, term, ...) with no entry in `LABELS_ZH` fails an assertion instead of showing raw.
"""

import base64
import json
import math
import re
from html import escape
from pathlib import Path

import polars as pl

CSS = """
body{font:15px/1.55 -apple-system,Helvetica,Arial,'PingFang TC','Noto Sans TC',sans-serif;
max-width:1100px;margin:20px auto;padding:0 14px;color:#222}
h1{font-size:1.5em}h2{margin-top:2em;border-bottom:1px solid #ddd;font-size:1.2em}
.t{overflow-x:auto;max-width:100%}table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 7px;text-align:right;white-space:nowrap}
th{background:#f4f4f4}.box{background:#f7f9fc;border:1px solid #dde5f0;padding:8px 16px;
border-radius:6px}.warn{color:#8a4b00}img{max-width:100%;height:auto}
.wrap{white-space:normal;min-width:16em;text-align:left}
th.g{text-align:center}th.sub{text-align:left;background:#eaeef5}
.fn{font-size:13px;margin:2px 0}
"""

REG = [
    "sample",
    "variant",
    "term",
    "coef",
    "baseline_coef",
    "t",
    "se_kind",
    "p",
    "r2",
    "baseline_r2",
    "T",
]

# Columns whose cells are code labels; the Chinese page maps them through LABELS_ZH.
LABEL_COLS = ("sample", "variant", "spec", "term", "se_kind", "rule", "group", "note", "unit")

HEADERS_ZH = {
    "sample": "樣本",
    "variant": "設定",
    "spec": "設定",
    "term": "變數",
    "coef": "係數",
    "baseline_coef": "基準係數",
    "unit": "單位",
    "t": "t 值",
    "se_kind": "標準誤算法",
    "p": "p 值",
    "r2": "R²（%）",
    "baseline_r2": "基準 R²（%）",
    "T": "交易日數 T",
    "n_dropped": "剔除列數",
    "chi2": "卡方值",
    "n_forecasts": "預測筆數",
    "r2_os_vs_mean": "R²_OS（對比擴張平均，%）",
    "r2_os_vs_zero": "R²_OS（對比零預測，%）",
    "cw_t_vs_mean": "Clark-West t（對比平均）",
    "cw_p_vs_mean": "Clark-West p（對比平均）",
    "cw_t_vs_zero": "Clark-West t（對比零）",
    "cw_p_vs_zero": "Clark-West p（對比零）",
    "group": "分組",
    "symbol": "代號",
    "slope": "斜率",
    "baseline_slope": "基準斜率",
    "p_holm": "Holm 校正 p",
    "t_dk": "Driscoll-Kraay t",
    "sessions_in_sample": "樣本內交易日數",
    "rows_excluded": "排除列數",
    "rule": "規則",
    "cost_bp": "單邊成本（bp）",
    "n_active": "有交易的天數",
    "sharpe": "Sharpe",
    "sharpe_ci_lo": "Sharpe 區間下限",
    "sharpe_ci_hi": "Sharpe 區間上限",
    "sharpe_always_long": "一直做多的 Sharpe",
    "hit_rate": "方向猜對率",
    "baseline_hit": "基準猜對率",
    "break_even_bp": "損益兩平成本（bp）",
    "note": "備註",
}

LABELS_ZH = {
    "sample": {
        "OVERLAP": "2011–13（和論文重疊）",
        "POST": "2014–26（發表後）",
        "FULL": "2011–26（全期）",
    },
    "variant": {
        "r13~r1": "最後半小時 ~ 第一個半小時",
        "r13~r12": "最後半小時 ~ 倒數第二個半小時",
        "r13~r1+r12": "最後半小時 ~ 第一個半小時 + 倒數第二個半小時",
        "huber": "Huber 穩健迴歸（降低極端值的權重）",
        "median": "中位數迴歸",
        "trim_top_1%": "去掉最極端 1%",
        "trim_top_3%": "去掉最極端 3%",
        "trim_top_5%": "去掉最極端 5%",
        "log_returns": "改用對數報酬",
        "no_half_days": "排除半天交易日",
        "no_ex_dividend": "排除除息日（也就是季度到期日）",
        "r13~r_on+r_fh": "最後半小時 ~ 隔夜 + 開盤半小時",
        "r13~rod": "最後半小時 ~ 前收到 15:30",
        "r13~r1+rod": "最後半小時 ~ 第一個半小時 + 前收到 15:30",
        "+cpi": "加入 CPI 公布日",
        "+daily_expiry_era": "加入每日到期期權時期",
        "+fomc": "加入聯準會利率決策日",
        "+is_half_day": "加入半天交易日",
        "+month_end": "加入月底",
        "+opex_monthly": "加入每月期權到期日",
        "+opex_quarterly": "加入季度期權到期日",
        "+payroll": "加入非農就業公布日",
        "+relvol": "加入第一個半小時的相對成交量",
        "+rv_day_lag": "加入前一日的波動",
        "+rv_fh": "加入第一個半小時的波動",
        "+vix_lag": "加入前一日的 VIX",
        "+vix_ratio_lag": "加入前一日 VIX 對三個月 VIX 的比值",
        "+weekday": "加入星期幾",
        "all": "全部控制變數一起加",
        "base_same_rows": "同一批交易日、不加控制變數的斜率",
        "r1_x_news": "第一個半小時報酬 × 重大消息日",
        "r1_x_top_relvol": "第一個半小時報酬 × 相對成交量最高三分位",
        "r1_x_top_rv_fh": "第一個半小時報酬 × 第一個半小時波動最高三分位",
    },
    "term": {
        "r1": "第一個半小時報酬",
        "r12": "倒數第二個半小時報酬",
        "lr1": "第一個半小時對數報酬",
        "r_on": "隔夜報酬",
        "r_fh": "開盤半小時報酬",
        "rod": "前收到 15:30 的報酬",
        "D": "斷點後的截距變化",
        "D*r1": "斷點後的斜率變化",
        "D+D*r1": "截距與斜率一起（聯合 Wald 檢定）",
        "const": "常數項",
        "cpi": "CPI 公布日",
        "daily_expiry_era": "每日到期期權時期",
        "dow_tue": "星期二",
        "dow_wed": "星期三",
        "dow_thu": "星期四",
        "dow_fri": "星期五",
        "fomc": "聯準會利率決策日",
        "inter": "交互項（第一個半小時報酬 × 該條件）",
        "is_half_day": "半天交易日",
        "month_end": "月底",
        "news": "重大消息日",
        "opex_monthly": "每月期權到期日",
        "opex_quarterly": "季度期權到期日",
        "payroll": "非農就業公布日",
        "relvol": "相對成交量",
        "relvol_top": "相對成交量最高三分位",
        "rv_day_lag": "前一日波動",
        "rv_fh": "第一個半小時波動",
        "rv_fh_top": "第一個半小時波動最高三分位",
        "vix_lag": "前一日 VIX",
        "vix_ratio_lag": "前一日 VIX 對三個月 VIX 的比值",
    },
    "se_kind": {
        "nw": "Newey-West",
        "bootstrap": "bootstrap 重抽",
        "cluster_date": "依日期分群",
        "driscoll_kraay": "Driscoll-Kraay",
    },
    "rule": {
        "always_long": "一直做多",
        "one_signal": "單一訊號（只看第一個半小時）",
        "one_signal_minus_always_long": "單一訊號比一直做多多出的 Sharpe",
        "two_signal": "兩個訊號（第一個與倒數第二個半小時）",
        "zero": "不交易（零）",
    },
    "group": {"equity": "股票型", "non_equity": "非股票型"},
    "unit": {"bp": "bp", "slope": "斜率"},
    "note": {
        "no break-even: gross mean <= 0": "沒有損益兩平點：未扣成本的平均報酬 ≤ 0",
        "Sharpe difference, paired bootstrap CI; the skill comparison. "
        "H5's own CI excludes 0 because of costs, not because of skill.": (
            "Sharpe 的差額，用配對 bootstrap 求區間；這是真正比較技能的地方。"
            "H5 自己的區間不含 0，是因為成本，不是因為有技能。"
        ),
    },
}
VARIANT_BREAK = re.compile(r"break_(\d{4}-\d{2}-\d{2})")

H4_BASELINE_ZH = {"none (no-break null)": "無斷點（虛無假設）"}

TEXT = {
    "en": {
        "lang": "en",
        "title": "Chapter 1: does the first half-hour predict the last?",
        "h1": "Chapter 1: does the first half-hour predict the last half-hour? ({kind})",
        "synthetic": "SYNTHETIC data (a pipeline check, not a result)",
        "warn": "Data window {start} to {end}; samples: {samples}.\n"
        "Bootstrap reps: {reps}. Not yet fact-checked.",
        "sample_to": "{name} {a} to {b}",
        "sample_join": "; ",
        "count": "<p><b>{count_line}</b></p>",
        "box": (
            "<ul>\n"
            "<li><b>H1</b> (2011-13 slope of last half-hour return on first half-hour return): "
            "slope {h1_slope},\nNewey-West t {h1_t}, p {h1_p}, R2 {h1_r2}%, "
            "T {h1_T}. Baseline: slope 0, R2 0.</li>\n"
            "<li><b>H2</b> (same slope, 2014-2026): slope {h2_slope}, t {h2_t}, "
            "p {h2_p},\nR2 {h2_r2}%, T {h2_T}. Baseline: slope 0, R2 0.</li>\n"
            "<li><b>H3</b> (out-of-sample, {spec}): R2_OS vs the expanding mean {os_mean}%, "
            "vs zero\n{os_zero}%, Clark-West p {cw_p}. Baseline R2_OS: 0.</li>\n"
            "<li><b>H4</b> (break at {break_date}, intercept and slope jointly): "
            "chi2 {chi2},\np {h4_p}. Baseline: {h4_base}.</li>\n"
            "<li><b>H5</b> (timing rule, {cost:g} bp per side): net Sharpe {sharpe},\n"
            "95% CI [{ci_lo}, {ci_hi}]. Baselines: always-long Sharpe\n"
            "{al}; zero (no trade) {zero:g}.</li>\n"
            "</ul>"
        ),
        "item": "Item {n}: {title}",
        "s3": (
            "{n} rolling windows of 252 sessions: mean slope {mean:.4g} "
            "(baseline 0), min {lo:.4g}, max {hi:.4g}; "
            "the confidence band excludes 0 in {excl:.1%} of windows."
        ),
        "t1": "Baseline regressions",
        "x1": (
            "Last half-hour return (r13) regressed on the first half-hour return (r1, previous "
            "close to 10:00) and on the second-to-last half-hour return (r12, 15:00 to 15:30), "
            "by sample. Read coef against baseline_coef (0) and r2 against baseline_r2 (0). "
            "t uses the standard error named in se_kind (nw = Newey-West); T is the number of "
            "sessions."
        ),
        "t2": "Robust variants",
        "x2": (
            "The same slope with outliers trimmed, half days or ex-dividend days dropped, log "
            "returns, and Huber and median regressions. Same reading as item 1; se_kind says "
            "which standard error each t uses."
        ),
        "t3": "Rolling slope",
        "x3": (
            "The slope of r13 on r1 in rolling windows of 252 sessions, with a 95% band. A slope "
            "that is really there should stay above the zero line. {s3}"
        ),
        "alt": "rolling slope",
        "t4": "Structural breaks",
        "x4": (
            "Tests whether intercept (D) and slope (D*r1) change after each candidate break date; "
            "the joint row (D+D*r1) gives a chi-square test. Baseline: no break (all zero)."
        ),
        "t5": "Out-of-sample forecasts",
        "x5": (
            "Monthly refit: the model is re-estimated once a month on all data up to the end of "
            "the previous month, and that month's coefficients forecast every trading day of the "
            "month, so no forecast uses data from its own month or later. R2_OS compares the "
            "squared error with an expanding-mean forecast and with a zero forecast; positive "
            "means better than that baseline, and the no-skill value is 0. Clark-West p tests "
            "the same comparison. R2_OS is in percent."
        ),
        "t6": "Decomposition of the signal",
        "x6": (
            "Splits the first-half-hour return into the overnight part (r_on) and the opening "
            "half-hour (r_fh), and uses the return from the previous close to 15:30 (rod). Same "
            "reading as item 1."
        ),
        "t7": "Controls known by 10:00",
        "x7": (
            "The r1 slope again after adding one control at a time, then all together; every "
            "control uses only information available by 10:00. Compare each r1 coef with "
            "base_same_rows (the baseline slope on the identical rows). One table per sample; "
            "t uses the standard error named in se_kind and T is the number of sessions. Second "
            "set of tables: the control coefficients in the all-controls regression, in basis "
            "points (bp, 0.01%) of last-half-hour return per one unit of the control (0 to 1 "
            "for yes/no controls)."
        ),
        "cap7": "<p>Control coefficients, all-controls regression:</p>",
        "cap8": (
            "<p>t of single-ETF rows: {one}; t of pooled rows: {pool}. DK t is the "
            "Driscoll-Kraay t, pooled rows only. Days = sessions used. Holm p is corrected for "
            "the number of ETFs tested.</p>"
        ),
        "pooled": "POOLED",
        "hd_slope": "slope",
        "hd_holm": "Holm p",
        "hd_days": "days",
        "hd_dk": "DK t",
        "hd_etf": "ETF",
        "t8": "ETF panel",
        "x8": (
            "The same slope for other ETFs, one row per ETF and one pooled row per group, with "
            "the three samples side by side. Baseline: slope 0."
        ),
        "t9": "Timing rules",
        "x9": (
            "Trade the last half-hour in the direction of the first half-hour, net of cost, "
            "against two baselines: always long (sharpe_always_long) and not trading at all "
            "(Sharpe 0, the zero rule). All five costs (0, 0.5, 1, 2 and 5 bp per side) are shown for each rule, "
            "2014-26 first. break_even_bp is the per-side cost at which the rule's mean return "
            "reaches zero; it is blank, with a footnote marker, when the gross mean is already not "
            "positive. Sharpe intervals are bootstrap intervals."
        ),
    },
    "zh": {
        "lang": "zh-Hant-TW",
        "title": "第 1 章：第一個半小時能預測最後半小時嗎？",
        "h1": "第 1 章：第一個半小時能預測最後半小時嗎？（{kind}）",
        "synthetic": "合成資料（只是檢查流程，不是結果）",
        "warn": "資料期間 {start} 到 {end}；樣本：{samples}。\n"
        "bootstrap 重抽次數：{reps}。尚未經過事實查核。",
        "sample_to": "{name}，{a} 至 {b}",
        "sample_join": "；",
        "count": "<p><b>{count_line}</b>（執行次數：本章 9 個項目共跑了幾次，含 MLflow 裡的子執行）</p>",
        "box": (
            "<ul>\n"
            "<li><b>H1</b>（2011–13 年，最後半小時報酬對第一個半小時報酬的斜率；斜率就是第一個"
            "半小時每多漲 1，最後半小時平均多漲多少）：斜率 {h1_slope}，"
            "Newey-West t 值 {h1_t}（t 值是估計值除以它的誤差，絕對值越大越不像巧合；"
            "Newey-West 是一種允許報酬前後相關、波動忽大忽小的誤差算法），"
            "p 值 {h1_p}（p 值是「若真實斜率為 0，仍看到這麼極端結果」的機率），"
            "R² {h1_r2}%（R² 是模型解釋掉的變動占比），交易日數 T {h1_T}。"
            "基準：斜率 0、R² 0。</li>\n"
            "<li><b>H2</b>（同一個斜率，2014–2026 年）：斜率 {h2_slope}，t 值 {h2_t}，"
            "p 值 {h2_p}，R² {h2_r2}%，交易日數 T {h2_T}。基準：斜率 0、R² 0。</li>\n"
            "<li><b>H3</b>（樣本外預測，{spec}）：R²_OS（每月用截至上月底的資料重新估計一次，預測當月每一天，再和基準預測比"
            "誤差，正數表示比基準好）對比擴張平均 {os_mean}%，對比零預測 {os_zero}%，"
            "Clark-West p 值 {cw_p}（Clark-West 是檢定預測是否真的比基準好的方法）。"
            "基準 R²_OS：0。</li>\n"
            "<li><b>H4</b>（在 {break_date} 斷點，截距與斜率一起檢定；斷點檢定看這一天前後"
            "關係有沒有變，這裡用 Wald 檢定，即檢查幾個係數是否同時為 0）：卡方值 {chi2}，"
            "p 值 {h4_p}。基準：{h4_base}。</li>\n"
            "<li><b>H5</b>（擇時規則，單邊成本 {cost:g} bp；bp 即基點，1 bp = 0.01%）："
            "扣成本後的 Sharpe {sharpe}（Sharpe 是平均報酬除以波動，再年化，衡量每單位風險"
            "賺多少），95% 信賴區間 [{ci_lo}, {ci_hi}]（用 bootstrap 重抽，也就是把"
            "資料反覆隨機重抽來估計不確定程度）。基準：一直做多的 Sharpe {al}；"
            "不交易（零）{zero:g}。</li>\n"
            "</ul>"
        ),
        "item": "項目 {n}：{title}",
        "s3": (
            "共 {n} 個滾動視窗（每個 252 個交易日）：平均斜率 {mean:.4g}（基準 0），"
            "最小 {lo:.4g}，最大 {hi:.4g}；信賴帶不含 0 的視窗占 {excl:.1%}。"
        ),
        "t1": "基本迴歸",
        "x1": (
            "用第一個半小時報酬（r1，前一天收盤到 10:00）和倒數第二個半小時報酬（r12，15:00 到"
            "15:30）來解釋最後半小時報酬（r13），依樣本分開看。係數要對照基準係數（0），R² 要"
            "對照基準 R²（0）。t 值用「標準誤算法」欄所列的標準誤（誤差大小的估計）算出；"
            "交易日數 T 是用到的交易日數。"
        ),
        "t2": "穩健性檢查",
        "x2": (
            "同一個斜率，但改成去掉極端值、排除半天交易日或除息日、改用對數報酬，以及 Huber 與"
            "中位數迴歸。讀法和項目 1 相同；「標準誤算法」欄說明每個 t 值用哪一種標準誤。"
        ),
        "t3": "滾動斜率",
        "x3": (
            "以 252 個交易日為一個滾動視窗，逐窗估計 r13 對 r1 的斜率，並畫出 95% 信賴帶。"
            "真的存在的斜率應該一直在零線之上。{s3}"
        ),
        "alt": "滾動斜率圖",
        "t4": "結構性斷點",
        "x4": (
            "檢查在每個候選斷點日之後，截距（D）和斜率（D×r1）有沒有改變；最後一列（D 與 D×r1"
            "一起）是聯合 Wald 檢定（同時檢查兩個係數是否為 0，結果以卡方值表示）。"
            "基準：沒有斷點（全為零）。"
        ),
        "t5": "樣本外預測",
        "x5": (
            "每個月重新估計一次模型（每月重估）：用截至上個月底的所有資料估計，該月份的係數再拿去"
            "預測這個月的每一個交易日，所以任何預測都沒用到當月或更晚的資料。R²_OS 把預測的平方誤差拿去和"
            "「擴張平均」預測（用至今所有日子的平均當預測）以及「零預測」比較；正數代表比該基準好，"
            "沒有技能時為 0；單位是百分比。Clark-West p 值檢定的是同一個比較。"
        ),
        "t6": "訊號拆解",
        "x6": (
            "把第一個半小時報酬拆成隔夜部分（r_on）和開盤半小時（r_fh），另外也用前一天收盤到"
            "15:30 的報酬（rod）。讀法和項目 1 相同。"
        ),
        "t7": "10:00 前就已知的控制變數",
        "x7": (
            "每次加入一個控制變數，最後再全部一起加，重新看 r1 的斜率；所有控制變數都只用 "
            "10:00 以前能取得的資訊。請把每個第一個半小時報酬的係數和「同一批交易日、不加控制變數的斜率」"
            "對照。每個樣本一張表；t 值用「標準誤算法」欄所列的標準誤，交易日數 T 是用到的交易日數。"
            "第二組表是「全部控制一起加」迴歸中各控制變數的係數，單位是 bp（1 bp = 0.01%）："
            "控制變數每增加 1 單位（是非題為 0 變 1），最後半小時報酬平均多多少 bp。"
        ),
        "cap7": "<p>各控制變數的係數（全部控制一起加的迴歸）：</p>",
        "cap8": (
            "<p>單檔 ETF 的 t 值用 {one}；合併列的 t 值用 {pool}。DK t 是 Driscoll-Kraay t，"
            "只有合併列才有。交易日數是用到的交易日數。Holm 校正 p 是依檢定的 ETF 檔數修正過的 "
            "p 值。</p>"
        ),
        "pooled": "合併",
        "hd_slope": "斜率",
        "hd_holm": "Holm 校正 p",
        "hd_days": "交易日數",
        "hd_dk": "DK t",
        "hd_etf": "ETF",
        "t8": "其他 ETF 面板",
        "x8": (
            "同一個斜率，拿去看其他 ETF：每檔一列，每個分組再有一列合併，三個樣本並排。"
            "Holm 是控制「檢定多次容易誤中」的方法；Driscoll-Kraay 是容許各檔同日相關、且前後日"
            "相關的誤差算法。基準：斜率 0。"
        ),
        "t9": "擇時規則",
        "x9": (
            "在最後半小時，順著第一個半小時的方向交易，扣掉成本，並和兩個基準比：一直做多"
            "（「一直做多的 Sharpe」欄）以及完全不交易（Sharpe 0，也就是零規則）。每個規則都列出"
            "全部五種單邊成本（0、0.5、1、2、5 bp），先列 2014–26。損益兩平成本是讓該規則平均報酬"
            "降到零的單邊成本；若未扣成本的平均報酬本來就不是正的，這格留空，並以腳註標記說明。Sharpe 的"
            "區間是 bootstrap 重抽區間。"
        ),
    },
}


def _label(lang: str, col: str, v):
    """Chinese label for a code cell; fails loud when a code has no translation."""
    if lang == "en" or v is None or col not in LABEL_COLS:
        return v
    kind = "variant" if col == "spec" else col
    table = LABELS_ZH[kind]
    if kind == "variant" and (m := VARIANT_BREAK.fullmatch(v)):
        return f"{m.group(1)} 斷點（此日之後）"
    assert v in table, f"no Chinese label for {kind} {v!r}: add it to LABELS_ZH"
    return table[v]


P_COLS = {"p", "p_holm", "cw_p_vs_mean", "cw_p_vs_zero"}
PCT_COLS = {"r2", "baseline_r2", "r2_os_vs_mean", "r2_os_vs_zero"}
FIXED2_COLS = {
    "t", "t_dk", "cw_t_vs_mean", "cw_t_vs_zero", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
    "sharpe_always_long",
}  # fmt: skip
BP_COLS = {"coef", "baseline_coef"}
BP_TERMS = {"const", "D"}  # return-type terms (an intercept or a shift in it), shown in bp
SAMPLE_ORDER = ("OVERLAP", "POST", "FULL")
SCROLL = '<div class="t" style="overflow-x:auto">'


def _sig(v: float, digits: int = 4) -> str:
    """`digits` significant digits in plain decimal notation, never e-notation."""
    if v == 0:
        return "0"
    places = max(digits - 1 - math.floor(math.log10(abs(v))), 0)
    text = f"{v:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt(v, col: str = "", bp: bool = False) -> str:
    """Cell text: None/NaN blank; p, %, t and Sharpe in fixed formats; the rest 4 significant
    digits; bp-unit coefficients scaled to basis points; all HTML-escaped, never e-notation."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if not isinstance(v, float):
        return escape(str(v))
    if col in P_COLS:
        text = "<0.001" if v < 0.001 else f"{v:.3f}"
    elif col in PCT_COLS:
        text = f"{v * 100:.2f}"
    elif col in FIXED2_COLS:
        text = f"{v:.2f}"
    elif bp and col in BP_COLS:
        text = _sig(v * 1e4)
    else:
        text = _sig(v)
    return escape("0.00" if text == "-0.00" else text)


def _head(lang: str, col: str, bp: bool = False) -> str:
    if lang == "en":
        name = f"{col} (bp)" if bp and col in BP_COLS else col
        return f"{name} (%)" if col in PCT_COLS else name
    assert col in HEADERS_ZH, f"no Chinese header for column {col!r}: add it to HEADERS_ZH"
    name = HEADERS_ZH[col]
    return name.replace("係數", "係數（單位：bp）") if bp and col in BP_COLS else name


def _table(df: pl.DataFrame, cols: list[str], lang: str, bp: bool = False) -> str:
    """One scrollable table. `bp`: every coef column is in basis points; otherwise rows whose
    `unit` column says "bp" are (so a table can mix intercept shifts and slopes, unit shown)."""
    if bp:
        assert "unit" not in df.columns
    row_bp = df["unit"].eq("bp").to_list() if "unit" in df.columns else [bp] * df.height
    head = "".join(f"<th>{escape(_head(lang, c, bp))}</th>" for c in cols)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{' class=wrap' if c == 'note' else ''}>{_fmt(_label(lang, c, v), c, row_bp[i])}</td>"
            for c, v in zip(cols, row, strict=True)
        )
        + "</tr>"
        for i, row in enumerate(df.select(cols).iter_rows())
    )
    return f"{SCROLL}<table><tr>{head}</tr>{body}</table></div>"


RULE_EN = {
    "always_long": "always long",
    "one_signal": "one signal (first half-hour only)",
    "one_signal_minus_always_long": "one-signal Sharpe minus always-long Sharpe",
    "two_signal": "two signals (first and 2nd-to-last)",
    "zero": "no trade (zero)",
}
MARKS = "¹²³⁴⁵⁶⁷⁸⁹"
T9_COLS = [
    "sample", "cost_bp", "n_active", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
    "sharpe_always_long", "hit_rate", "baseline_hit", "break_even_bp",
]  # fmt: skip


def _timing_table(df: pl.DataFrame, lang: str, tx: dict) -> str:
    """Item 9: one-line rows. A sub-heading row opens each rule's block; a row's note becomes a
    footnote marker in its break-even cell, the notes printed once under the table."""
    notes: list[str] = []
    head = "".join(f"<th>{escape(_head(lang, c))}</th>" for c in T9_COLS)
    rows, prev = "", None
    for rec in df.iter_rows(named=True):
        rule = rec["rule"]
        if (rec["sample"], rule) != prev:
            prev = (rec["sample"], rule)
            name = RULE_EN[rule] if lang == "en" else _label(lang, "rule", rule)
            rows += f'<tr><th colspan="{len(T9_COLS)}" class="sub">{escape(name)}</th></tr>'
        mark = ""
        if rec["note"] is not None:
            if rec["note"] not in notes:
                notes.append(rec["note"])
            mark = MARKS[notes.index(rec["note"])]
        cells = ""
        for c in T9_COLS:
            text = _fmt(_label(lang, c, rec[c]), c)
            cells += f"<td>{text + mark if c == 'break_even_bp' else text}</td>"
        rows += f"<tr>{cells}</tr>"
    foot = "".join(
        f'<p class="fn">{MARKS[i]} {escape(_label(lang, "note", n))}</p>'
        for i, n in enumerate(notes)
    )
    return f"{SCROLL}<table><tr>{head}</tr>{rows}</table></div>{foot}"


def _panel_table(df: pl.DataFrame, lang: str, tx: dict) -> str:
    """Item 8: one row per ETF (pooled row last) per group, the three samples side by side."""
    sub = [("slope", "hd_slope"), ("t", None), ("p_holm", "hd_holm"), ("T", "hd_days")]
    sub.append(("t_dk", "hd_dk"))
    top = f'<th rowspan="2">{_head(lang, "group")}</th><th rowspan="2">{escape(tx["hd_etf"])}</th>'
    low = ""
    for smp in SAMPLE_ORDER:
        top += f'<th class="g" colspan="{len(sub)}">{escape(_label(lang, "sample", smp))}</th>'
        low += "".join(f"<th>{escape(tx[h] if h else 't')}</th>" for _, h in sub)
    rows = ""
    for grp in df["group"].unique(maintain_order=True):
        g = df.filter(pl.col("group") == grp)
        syms = [x for x in g["symbol"].unique(maintain_order=True) if x != "POOLED"]
        for sym in [*syms, "POOLED"]:
            cells = ""
            for smp in SAMPLE_ORDER:
                r = g.filter((pl.col("symbol") == sym) & (pl.col("sample") == smp))
                assert r.height == 1, f"panel row missing or repeated: {grp} {sym} {smp}"
                rec = r.row(0, named=True)
                cells += "".join(f"<td>{_fmt(rec[c], c)}</td>" for c, _ in sub)
            name = tx["pooled"] if sym == "POOLED" else sym
            rows += f"<tr><td>{escape(_label(lang, 'group', grp))}</td><td>{escape(name)}</td>{cells}</tr>"
    return f"{SCROLL}<table><tr>{top}</tr><tr>{low}</tr>{rows}</table></div>"


def _se_caption(df: pl.DataFrame, lang: str, tx: dict) -> str:
    one = df.filter(pl.col("symbol") != "POOLED")["se_kind"].unique().to_list()
    pool = df.filter(pl.col("symbol") == "POOLED")["se_kind"].unique().to_list()
    assert len(one) == 1 and len(pool) == 1, f"item 8 se_kind not uniform: {one} {pool}"
    return tx["cap8"].format(
        one=_label(lang, "se_kind", one[0]), pool=_label(lang, "se_kind", pool[0])
    )


def _section(tx: dict, n: int, text: str, *parts: str) -> str:
    title = tx[f"t{n}"]
    return f'<h2 id="item{n}">{escape(tx["item"].format(n=n, title=title))}</h2><p>{text}</p>' + (
        "".join(parts)
    )


def _box(tx: dict, s: dict, count_line: str, lang: str) -> str:
    h1, h2, h3, h4, h5 = (s[k] for k in ("H1", "H2", "H3", "H4", "H5"))
    spec = "r13~r1"
    o = h3["specs"][spec]
    base = h4["baseline"]
    if lang == "zh":
        assert base in H4_BASELINE_ZH, f"no Chinese label for H4 baseline {base!r}"
        base = H4_BASELINE_ZH[base]
    f = _fmt  # the same cell formatters as the tables (percent R2, 3-decimal slope, ...)
    box = tx["box"].format(
        h1_slope=f"{h1['slope']:.3f}", h1_t=f(h1["nw_t"], "t"), h1_p=f(h1["p"], "p"),
        h1_r2=f(h1["r2"], "r2"), h1_T=h1["T"],
        h2_slope=f"{h2['slope']:.3f}", h2_t=f(h2["nw_t"], "t"), h2_p=f(h2["p"], "p"),
        h2_r2=f(h2["r2"], "r2"), h2_T=h2["T"],
        spec=escape(spec), os_mean=f(o["r2_os_vs_mean"], "r2"), os_zero=f(o["r2_os_vs_zero"], "r2"),
        cw_p=f(o["cw_p_vs_mean"], "p"), break_date=escape(h4["break_date"]),
        chi2=f(h4["joint_chi2"], "t"), h4_p=f(h4["joint_p"], "p"), h4_base=escape(base),
        cost=h5["cost_bp_per_side"], sharpe=f(h5["sharpe_net"], "sharpe"),
        ci_lo=f(h5["ci_lo"], "sharpe"), ci_hi=f(h5["ci_hi"], "sharpe"),
        al=f(h5["baseline_always_long_sharpe"], "sharpe"), zero=h5["baseline_zero_sharpe"],
    )  # fmt: skip
    return f'<div class="box">{tx["count"].format(count_line=escape(count_line))}{box}</div>'


def _render(lang: str, s: dict, t: dict, img: str, count_line: str) -> str:
    """One report page from the shared numbers; only words and code labels differ by language."""
    tx = TEXT[lang]
    starts = [v[0] for v in s["samples"].values()]
    ends = [v[1] for v in s["samples"].values()]
    samples = tx["sample_join"].join(
        tx["sample_to"].format(name=_label(lang, "sample", k), a=v[0], b=v[1])
        for k, v in s["samples"].items()
    )
    kind = tx["synthetic"] if s["synthetic"] else "SPY"
    r1 = pl.col("term") == "r1"

    r3 = t[3]
    excl = r3.select(((pl.col("lo") > 0) | (pl.col("hi") < 0)).mean()).item()
    s3 = tx["s3"].format(
        n=r3.height, mean=r3["slope"].mean(), lo=r3["slope"].min(), hi=r3["slope"].max(), excl=excl
    )
    ctrl = t[7].filter((pl.col("variant") == "all") & ~r1 & (pl.col("term") != "const"))
    s9 = {"POST": 0, "OVERLAP": 1, "FULL": 2}
    rule_rank = {r: i for i, r in enumerate(t[9]["rule"].unique(maintain_order=True))}
    t9 = (
        t[9]
        .with_columns(
            pl.col("sample").replace_strict(s9, return_dtype=pl.Int64).alias("_s"),
            pl.col("rule").replace_strict(rule_rank, return_dtype=pl.Int64).alias("_r"),
        )
        .sort("_s", "_r", "cost_bp")
    )
    t4 = t[4].with_columns(
        pl.when(pl.col("term").is_in(BP_TERMS))
        .then(pl.lit("bp"))
        .when(pl.col("term") == "D*r1")
        .then(pl.lit("slope"))
        .alias("unit")  # the joint row has no coefficient, so no unit
    )

    def per_sample(df: pl.DataFrame, cols: list[str], bp: bool = False) -> str:
        """One small table per sample (OVERLAP, POST, FULL), each under its own label."""
        out = ""
        for smp in SAMPLE_ORDER:
            part = df.filter(pl.col("sample") == smp)
            assert part.height, f"no rows for sample {smp}"
            out += f"<p><b>{escape(_label(lang, 'sample', smp))}</b></p>" + _table(
                part, cols, lang, bp
            )
        return out

    def tab(df: pl.DataFrame, cols: list[str]) -> str:
        return _table(df, cols, lang)

    body = [
        _section(tx, 1, tx["x1"], tab(t[1], REG)),
        _section(tx, 2, tx["x2"], tab(t[2], REG)),
        _section(
            tx,
            3,
            tx["x3"].format(s3=escape(s3)),
            f'<img alt="{escape(tx["alt"])}" src="data:image/png;base64,{img}">',
        ),
        _section(tx, 4, tx["x4"], tab(t4, [*REG[:4], "unit", *REG[4:], "chi2"])),
        _section(
            tx,
            5,
            tx["x5"],
            tab(
                t[5],
                [
                    "spec",
                    "n_forecasts",
                    "r2_os_vs_mean",
                    "r2_os_vs_zero",
                    "baseline_r2",
                    "cw_t_vs_mean",
                    "cw_p_vs_mean",
                    "cw_t_vs_zero",
                    "cw_p_vs_zero",
                ],
            ),
        ),
        _section(tx, 6, tx["x6"], tab(t[6], REG)),
        _section(
            tx,
            7,
            tx["x7"],
            per_sample(
                t[7].filter(r1),
                [c for c in REG if c not in ("sample", "term")] + ["n_dropped"],
            ),
            tx["cap7"],
            per_sample(
                ctrl,
                ["term", "coef", "baseline_coef", "t", "se_kind", "p", "T"],
                bp=True,
            ),
        ),
        _section(tx, 8, tx["x8"], _se_caption(t[8], lang, tx), _panel_table(t[8], lang, tx)),
        _section(
            tx,
            9,
            tx["x9"],
            _timing_table(t9, lang, tx),
        ),
    ]
    warn = tx["warn"].format(
        start=min(starts), end=max(ends), samples=escape(samples), reps=s["bootstrap_reps"]
    )
    return f"""<!doctype html><html lang="{tx["lang"]}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tx["title"])}</title><style>{CSS}</style></head><body>
<h1>{escape(tx["h1"].format(kind=kind))}</h1>
<p class="warn">{warn}</p>
{_box(tx, s, count_line, lang)}
{"".join(body)}
</body></html>"""


def write_chapter1_report(out: Path, count_line: str) -> tuple[Path, Path]:
    """Write the English and Chinese pages from the files `fhh reproduce` already wrote."""
    s = json.loads((out / "chapter1_summary.json").read_text())
    t = {n: pl.read_parquet(out / "chapter1" / f"item{n}.parquet") for n in range(1, 10)}
    img = base64.b64encode((out / "figures" / "rolling_slope.png").read_bytes()).decode()
    paths = (out / "chapter1_report.html", out / "chapter1_report_zh.html")
    for path, lang in zip(paths, ("en", "zh"), strict=True):
        path.write_text(_render(lang, s, t, img, count_line))
    return paths
