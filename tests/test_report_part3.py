"""Part 3 figures and report on fake frames (the column contract of `part3_data` / `part3`)."""

import json
import re
from datetime import date

import polars as pl
import pytest

from first_half_hour import config, report_part3, reproduce

CLASSES = {
    "us_equity": ("SPY", "QQQ"),
    "intl_equity": ("EFA",),
    "bonds": ("TLT",),
    "gold": ("GLD",),
}
PERIODS = {
    "p1": (date(2011, 1, 3), date(2013, 12, 31)),
    "p2": (date(2014, 1, 2), date(2020, 5, 1)),
    "p3": (date(2020, 5, 4), date(2026, 8, 31)),
}
TAUS = [t for t, _ in config.PART2_TIMES]
ANY_CELL = re.compile(r"<t[dh][^>]*>([^<]*)</t[dh]>")
BARE = re.compile(r"(?<![A-Za-z0-9_])(r1|r12|r13|rod)(?![A-Za-z0-9_])")
SYMBOLS = [s for v in CLASSES.values() for s in v]


@pytest.fixture(autouse=True)
def part3_config(monkeypatch):
    monkeypatch.setattr(config, "PART3_CLASSES", CLASSES, raising=False)
    monkeypatch.setattr(config, "PART3_PERIODS", PERIODS, raising=False)


def _effect() -> pl.DataFrame:
    rows = []
    owners = [("class", c) for c in CLASSES] + [("etf", s) for s in SYMBOLS]
    owners.append(("spy_official", "SPY"))
    for level, who in owners:
        for p in PERIODS:
            for item, metrics in (
                ("slope_r1", ("slope",)), ("slope_rod", ("slope",)),
                ("rule_r1", ("ann_mean", "sharpe")), ("rule_rod", ("ann_mean", "sharpe")),
            ):  # fmt: skip
                for metric in metrics:
                    for cost in (0, 2, 4) if item.startswith("rule") else (None,):
                        rows.append({
                            "class": who, "period": p, "item": item, "metric": metric,
                            "cost_bp": cost, "n_days": 700, "n_etfs": len(CLASSES.get(who, (0,))),
                            "est": 0.0123 if metric == "slope" else 0.034, "se": 0.004,
                            "t": 3.1, "verdict": "clearly above", "level": level,
                        })  # fmt: skip
    return pl.DataFrame(rows)


def _grid(finding_cells=(("bonds", "15:30"),), pass_cells=()) -> pl.DataFrame:
    rows = []
    for c in CLASSES:
        for i, t in enumerate(TAUS):
            hit = (c, t) in finding_cells
            rows.append({
                "class": c, "tau": t, "top20_minus_af": 0.0001 * (i + 1), "se": 0.00004,
                "d_af": 2.5 if hit or (c, t) in pass_cells else 0.7, "d_bm": 1.1, "d_vol": 0.9,
                "coverage": 0.19, "mean_auc": 0.512, "d_af_pre": 1.5, "d_af_post": 1.25,
                "d_af_nocovid": 2.0, "diff_pre": 0.0001, "diff_post": 0.0001,
                "diff_nocovid": 0.0001, "finding": hit,
            })  # fmt: skip
    return pl.DataFrame(rows)


def _rules(by_symbol=False) -> pl.DataFrame:
    rows = []
    for who in SYMBOLS if by_symbol else CLASSES:
        for t in TAUS if by_symbol else (config.PART3_REPORT_TAU,):
            for rule, q in [("always_follow", 1.0)] + [
                (r, q) for q in (0.1, 0.2, 0.5) for r in ("model", "big_move", "vol_only")
            ]:
                rows.append({
                    "symbol" if by_symbol else "class": who, "tau": t, "rule": rule, "q": q,
                    "c_eval_bp": 2, "split": "all", "n_trades": int(3000 * q),
                    "n_etf_trades": int(3000 * q),
                    "coverage": q, "mean_net": 0.0002 * (1 + q), "se": 0.00005, "hit": 0.52,
                    "hit_se": 0.01, "sharpe": 0.4, "sharpe_se": 0.2, "diff_vs_af": 0.0001,
                    "diff_vs_af_se": 0.00004, "auc": 0.512,
                })  # fmt: skip
    return pl.DataFrame(rows)


def _coverage() -> pl.DataFrame:
    return pl.DataFrame([
        {"symbol": s, "first_session": date(2011, 1, 3), "last_session": date(2026, 8, 31),
         "kept_sessions": 3900, "sessions_dropped": 0, "share_r13_zero": 0.012,
         "share_r1_null": 0.0}
        for s in SYMBOLS
    ]).with_columns(
        first_session=pl.when(pl.col("symbol") == "HYG").then(date(2012, 6, 1)).otherwise(pl.col("first_session")),
    )  # fmt: skip


def write_inputs(out, finding_cells=(("bonds", "15:30"),), n_pass=1):
    d = out / "part3"
    d.mkdir()
    frames = {"coverage": _coverage(), "effect": _effect(), "rules": _rules(),
              "etf_rules": _rules(by_symbol=True),
              "grid": _grid(finding_cells, (("gold", "10:00"),) if n_pass > len(finding_cells) else ())}  # fmt: skip
    for k, v in frames.items():
        v.write_parquet(d / f"{k}.parquet")
    summary = {
        "n_etfs": len(SYMBOLS), "n_cells": 20, "n_pass_full": n_pass,
        "n_findings": len(finding_cells), "findings": [], "n_fits": 777, "synthetic": True,
        "data_window": ["2011-01-03", "2026-08-31"], "count_line": "part 3: 5 ETFs, 99 runs",
    }  # fmt: skip
    (out / "part3_summary.json").write_text(json.dumps(summary))
    report_part3.make_figures(out)


def pages(out):
    en, zh = report_part3.write_part3_report(out)
    return en.read_text(), zh.read_text()


def test_report_renders_both_languages_with_d50_labels_only(tmp_path):
    write_inputs(tmp_path, n_pass=2)
    en, zh = pages(tmp_path)
    for page in (en, zh):
        assert "part 3: 5 ETFs, 99 runs" in page and "777" in page
        assert page.count("data:image/png;base64,") == 2
        assert "http://" not in page and "<script" not in page
        text = re.sub(r"<style>.*?</style>|<img[^>]*>", "", page, flags=re.DOTALL)
        assert not BARE.search(text), BARE.search(text)
        assert not re.search(r"replicat|重現|\d[eE][+-]?\d", text)
        assert not [c for c in ANY_CELL.findall(page) if re.search(r"[a-z]+_[a-z0-9_]+", c)]
    for word in ("US equity", "international equity", "bonds", "gold", "clearly above 0",
                 "previous close to 10:00", "15:30 to close", "previous close to 15:30",
                 "SPY, official close"):  # fmt: skip
        assert word in en
    for word in ("美國股票", "海外股票", "債券", "黃金", "明顯大於 0", "前一天收盤到 10:00",
                 "15:30 到收盤", "前一天收盤到 15:30", "SPY，官方收盤價"):  # fmt: skip
        assert word in zh
    num = re.compile(r"-?\d+(\.\d+)?")
    nums = lambda page: [c for c in ANY_CELL.findall(page) if num.fullmatch(c)]
    assert nums(en) == nums(zh) and len(nums(en)) > 50


def test_findings_sentence_counts_and_chance_remark(tmp_path):
    write_inputs(tmp_path, n_pass=2)
    en, zh = pages(tmp_path)
    assert "2 of 20 cells" in en and "1 are findings (bonds, 15:30)" in en
    assert "one or two passing the first test by chance is expected" in en
    assert "20 個格子" in zh and "有 2 個格子" in zh and "共 1 個（債券, 15:30）" in zh


def test_summary_disagreeing_with_grid_fails_loud(tmp_path):
    write_inputs(tmp_path)
    s = json.loads((tmp_path / "part3_summary.json").read_text())
    (tmp_path / "part3_summary.json").write_text(json.dumps(s | {"n_findings": 3}))
    with pytest.raises(AssertionError, match="grid findings 1, summary 3"):
        report_part3.write_part3_report(tmp_path)


def test_grid_table_shows_value_split_d_and_flag():
    html = report_part3._grid_table(_grid(), "en")
    assert html.count("<tr>") == 21
    # hand values: 0.0001 * 5 = 5.0 bp, SE 0.4 bp; split d's 1.50 / 1.25 / 2.00; AUC 0.512
    assert "5.0 (0.4)</td><td>2.50</td><td>1.10</td><td>0.90</td><td>1.50</td><td>1.25</td><td>2.00</td><td>19.0%</td><td>0.512</td><td>yes" in html  # fmt: skip
    assert html.count("<td>yes</td>") == 1


def test_coverage_table_flags_late_start_and_gaps():
    cov = _coverage().with_columns(
        sessions_dropped=pl.when(pl.col("symbol") == "TLT").then(4).otherwise(0)
    )
    cov = cov.with_columns(first_session=pl.when(pl.col("symbol") == "GLD").then(date(2012, 6, 1)).otherwise(pl.col("first_session")))  # fmt: skip
    html = report_part3._coverage_table(cov, "en")
    assert html.count("late start") == 1 and html.count("sessions dropped (missing a key bar)") == 1
    assert "1.2%" in html


def test_labels_fail_loud():
    with pytest.raises(AssertionError, match="no zh label for class 'crypto'"):
        report_part3._cls("crypto", "zh")
    bad = _effect().with_columns(verdict=pl.lit("maybe"))
    with pytest.raises(AssertionError, match="no en label for verdict 'maybe'"):
        report_part3._slopes_table(bad, "en")
    with pytest.raises(AssertionError, match="is in no class"):
        report_part3._class_of("IVV")


def test_class_rules_table_numbers():
    html = report_part3._class_rules_table(_rules(), "en")
    assert html.count("<tr>") == 1 + 4 * 10
    # mean_net 0.0002 * 1.2 = 2.4 bp, SE 0.5 bp -> "2.4 ± 1.0" (by hand)
    assert "model, top 20% of days</td><td>600</td><td>20.0%</td><td>2.4 ± 1.0" in html


def test_synthetic_overrides_give_every_class_an_etf():
    o = reproduce.SYNTHETIC_PART3_OVERRIDES["PART3_CLASSES"]
    assert list(o) == list(report_part3.CLASS)
    assert all(len(v) >= 1 for v in o.values())
