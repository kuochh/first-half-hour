"""Part 1 report and figures on a fake rows frame (the column contract of `part1.part1_rows`)."""

import json
import re
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from first_half_hour import config, report_part1

COLS = [
    "item", "paper", "window", "variant", "term", "metric", "n", "ours", "se", "se_kind", "lo",
    "hi", "paper_value", "paper_source", "paper_market", "null_value", "null_label", "se_null",
    "d_null", "d_paper", "verdict", "mde", "small_sample", "cw_p", "loo_min", "note",
]  # fmt: skip
ANY_CELL = re.compile(r"<t[dh][^>]*>([^<]*)</t[dh]>")
CODE = re.compile(r"[a-z]+_[a-z0-9_]+")
SCI = re.compile(r"\d[eE][+-]?\d")
VERDICTS = ["replicated", "not replicated", "cannot tell", "sample too small", None]
# item: (metric, term, market, windows' paper); one row per window kind below
SPEC = {
    "gao_slope": ("slope", "r1", "SPY"),
    "gao_r12": ("slope", "r12", "SPY"),
    "gao_joint": ("slope", "r1", "SPY"),
    "gao_vol_tercile": ("slope", "low", "SPY"),
    "gao_oos_r2": ("r2_os", "r1", "SPY"),
    "gao_rule": ("sharpe", "rule", "SPY"),
    "gao_news": ("r2", "FOMC", "SPY"),
    "balt_slope": ("slope", "rod", "S&P 500 E-mini futures"),
    "balt_horse": ("slope", "rod", "pooled equity-index futures"),
    "balt_large_move": ("slope", "x_jump", "pooled equity-index futures"),
    "balt_reversal": ("slope", "k1", "pooled equity-index futures"),
    "balt_last_k": ("slope", "r_last15", "S&P 500 cash index"),
    "balt_rule": ("success", "rule", "equity-index futures, 1/N portfolio"),
}


def fake_rows() -> pl.DataFrame:
    rows = []
    for i, (item, (metric, term, market)) in enumerate(SPEC.items()):
        gao = item.startswith("gao")
        for w, kind in enumerate(("overlap", "after")):
            ours = 0.0123 * (i + 1) * (1 + w)
            se = 0.004
            rows.append({
                "item": item, "paper": "gao" if gao else "baltussen",
                "window": f"{'gao' if gao else 'balt'}_{kind}", "variant": "main", "term": term,
                "metric": metric, "n": 700 + i, "ours": ours, "se": se, "se_kind": "nw",
                "lo": ours - 2 * se, "hi": ours + 2 * se, "paper_value": 0.069,
                "paper_source": f"{'Gao WP Oct 2014' if gao else 'Baltussen SSRN Aug 2020'}, Table 1",
                "paper_market": market, "null_value": 0.0, "null_label": "0", "se_null": se,
                "d_null": 3.0, "d_paper": 1.0, "verdict": VERDICTS[(i + w) % 5],
                "mde": 2.84 * se, "small_sample": item == "gao_news" and w == 0,
                "cw_p": 0.0004 if item == "gao_oos_r2" and w == 0 else (
                    0.2 if item == "gao_oos_r2" else None),
                "loo_min": 0.01 if metric == "r2" else None,
                "note": "same regression as the 0.062 row" if item == "balt_last_k" else None,
            })  # fmt: skip
    return pl.DataFrame(rows).select(COLS)


def series(signals, n=40):
    days = [date(2011, 1, 3) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(1)
    return days, rng


def figures(tmp_path):
    days, rng = series(None)
    cum = pl.DataFrame([
        {"session": d, "rule": r, "cost_bp": c, "wealth": float(1 + 0.001 * k)}
        for r in ("eta_r1", "eta_rod", "always_long") for c in (0.0, 1.0)
        for k, d in enumerate(days)
    ])  # fmt: skip
    roll = pl.DataFrame([
        {"session": d, "signal": s, "slope": float(rng.normal(0.05, 0.01)), "se": 0.02}
        for s in ("r1", "rod") for d in days
    ])  # fmt: skip
    ends = (date(2011, 1, 20), date(2011, 2, 5))
    report_part1.plot_cumulative(cum, tmp_path / "figures" / "part1_cumulative.png", *ends)
    report_part1.plot_rolling(roll, tmp_path / "figures" / "part1_rolling_slope.png", *ends)


def write_inputs(tmp_path):
    figures(tmp_path)
    (tmp_path / "part1").mkdir()
    fake_rows().write_parquet(tmp_path / "part1" / "rows.parquet")
    summary = {
        "synthetic": True, "data_window": ["2011-01-03", "2026-08-31"],
        "count_line": "part 1: 13 items, 39 runs (26 nested)",
    }  # fmt: skip
    (tmp_path / "part1_summary.json").write_text(json.dumps(summary))


def test_report_renders_in_both_languages_with_same_numbers(tmp_path):
    write_inputs(tmp_path)
    en_path, zh_path = report_part1.write_part1_report(tmp_path)
    en, zh = en_path.read_text(), zh_path.read_text()
    for page in (en, zh):
        assert "part 1: 13 items, 39 runs (26 nested)" in page
        assert page.count("data:image/png;base64,") == 2
        assert "Gao WP Oct 2014" in page and "Baltussen SSRN Aug 2020" in page
        assert "http://" not in page and "<script" not in page
        assert page.count("<table>") == page.count('style="overflow-x:auto"><table>') == 3
        assert not [c for c in ANY_CELL.findall(page) if SCI.search(c)]
        assert "after publication" not in page
        for gone in ("first half-hour", "first-half-hour", "overnight", "last half-hour"):
            assert gone not in page.lower(), gone
    bare = re.compile(r"(?<![A-Za-z0-9_])(r1|r12|r13|rod)(?![A-Za-z0-9_])")
    for page in (en, zh):  # the four returns are named in words, never by code
        text = re.sub(r"<style>.*?</style>|<img[^>]*>", "", page, flags=re.DOTALL)
        assert not bare.search(text), bare.search(text)
        assert "previous close to 10:00" in page or "前一天收盤到 10:00" in page
    assert "15:30 to close" in en and "15:30 到收盤" in zh
    assert 'lang="zh-Hant-TW"' in zh
    for word in ("重現", "沒重現", "分辨不出來", "樣本太少"):
        assert word in zh, word
    assert "sample too small" in en and "cannot tell" in en
    assert not [c for c in ANY_CELL.findall(zh) if CODE.search(c)], "code name in a Chinese table"
    num = re.compile(r"-?\d+(\.\d+)?%?")
    nums = lambda page: [c for c in ANY_CELL.findall(page) if num.fullmatch(c)]
    assert nums(en) == nums(zh) and len(nums(en)) > 100


def test_verdict_counts_and_formats():
    rows = fake_rows()
    html = report_part1._verdict_counts(rows, "en")
    rated = rows.filter(pl.col("verdict").is_not_null())
    for v in ("replicated", "not replicated", "cannot tell", "sample too small"):
        n = rated.filter((pl.col("window") == "gao_overlap") & (pl.col("verdict") == v)).height
        assert n > 0
    row = re.search(r"<td>Gao overlap</td>((?:<td>\d+</td>){4})", html).group(1)
    counts = [int(x) for x in re.findall(r"\d+", row)]
    assert counts == [
        rated.filter((pl.col("window") == "gao_overlap") & (pl.col("verdict") == v)).height
        for v in ("replicated", "not replicated", "cannot tell", "sample too small")
    ]
    # full_window rows are shown but not counted
    extra = rows.filter(pl.col("item") == "gao_slope").with_columns(
        variant=pl.lit("full_window"), verdict=pl.lit("replicated")
    )
    assert report_part1._verdict_counts(pl.concat([rows, extra]), "en") == html
    assert report_part1._num("slope", 0.069) == "0.069"
    assert report_part1._num("r2_os", 0.0169) == "1.69"
    assert report_part1._num("sharpe", 1.0801) == "1.08"
    assert report_part1._num("success", 0.5437) == "54.37"
    assert report_part1._num("slope", None) == ""


def test_labels_fail_loud():
    bad = fake_rows().with_columns(
        pl.when(pl.col("item") == "gao_slope").then(pl.lit("new_term")).otherwise(pl.col("term"))
        .alias("term")
    )  # fmt: skip
    with pytest.raises(AssertionError, match="no zh label for term 'new_term'"):
        report_part1._comparison(bad, "zh")
    with pytest.raises(AssertionError, match="no zh label for window"):
        report_part1._lab(report_part1.WINDOW, "gao_extra", "zh", "window")
    with pytest.raises(AssertionError, match="rows.parquet lacks"):
        report_part1._render("en", {}, fake_rows().drop("verdict"), {})
    two = pl.concat([fake_rows(), fake_rows().head(1)])
    with pytest.raises(AssertionError, match="two rows in one window kind"):
        report_part1._comparison(two, "en")


def test_figures_are_written_and_check_inputs(tmp_path):
    figures(tmp_path)
    assert (tmp_path / "figures" / "part1_cumulative.png").stat().st_size > 5000
    assert (tmp_path / "figures" / "part1_rolling_slope.png").stat().st_size > 5000
    assert config.PART1_REPORT_ROLLING_WINDOW == 756
    cum = pl.DataFrame({"session": [date(2011, 1, 3)], "rule": ["eta_r1"], "cost_bp": [0.0],
                        "wealth": [1.0]})  # fmt: skip
    with pytest.raises(AssertionError, match="rules differ"):
        report_part1.plot_cumulative(cum, tmp_path / "x.png", date(2013, 12, 31), date(2020, 5, 1))


def test_table_shows_no_skill_note_and_loo_min():
    rows = fake_rows().with_columns(
        null_value=pl.when(pl.col("metric") == "r2").then(0.087).otherwise(pl.col("null_value")),
        null_label=pl.when(pl.col("metric") == "r2").then(pl.lit("no-skill R² k/(n-1)"))
        .otherwise(pl.col("null_label")),
    )  # fmt: skip
    en = report_part1._comparison(rows, "en")
    zh = report_part1._comparison(rows, "zh")
    assert "8.70 (no-skill R² k/(n-1))" in en and "8.70 (無技能 R² k/(n-1))" in zh
    assert "same regression as the 0.062 row (B1)" in en and "和 0.062 那一列" in zh
    assert "leave-one-out min" in en and ">1.00<" in en  # loo_min 0.01 shown as a percent
    assert "cut-offs from earlier sessions only" in report_part1.VARIANT["past_only"][0]
    with pytest.raises(AssertionError, match="no en label for note"):
        report_part1._comparison(rows.with_columns(note=pl.lit("a new note")), "en")


def test_size_note_on_far_replicated_rows():
    from first_half_hour import report_part1

    base = {"verdict": "replicated", "paper_value": -0.145}
    assert report_part1._size_note(base | {"ours": -0.663}) == "larger"
    assert report_part1._size_note(base | {"ours": -0.05}) == "smaller"
    assert report_part1._size_note(base | {"ours": -0.2}) is None
    assert report_part1._size_note(base | {"ours": -0.663, "verdict": "cannot tell"}) is None
