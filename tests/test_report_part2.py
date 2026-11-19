"""Part 2 figures and report on fake frames (the column contract of `part2.run_all`)."""

import json
import re
from datetime import date, timedelta

import polars as pl
import pytest

from first_half_hour import config, report_part2

TAUS = ["10:00", "12:00", "14:00", "15:00", "15:30"]
CS = [0, 2, 4]
MODELS = ["logit", "hgb"]
ANY_CELL = re.compile(r"<t[dh][^>]*>([^<]*)</t[dh]>")
CODE = re.compile(r"[a-z]+_[a-z0-9_]+")
BARE = re.compile(r"(?<![A-Za-z0-9_])(r1|r12|r13|rod)(?![A-Za-z0-9_])")
FEATURES = list(report_part2.FEAT)[:22]


def fake_frames(skill: bool = True) -> dict[str, pl.DataFrame]:
    grid = pl.DataFrame([
        {"tau": t, "c_bp": c, "model": m, "auc": 0.52, "top20_minus_af": 0.0001 * (i + j + 1),
         "se": 0.00004 * (k + 1), "d_af": 2.5}
        for i, t in enumerate(TAUS) for j, c in enumerate(CS) for k, m in enumerate(MODELS)
    ])  # fmt: skip
    rules = []
    for t in TAUS:
        for c in CS:
            for m in MODELS:
                for split in report_part2.SPLITS:
                    for ce in CS:
                        for rule, q in [("always_follow", 1.0)] + [
                            (r, q)
                            for q in (0.1, 0.2, 0.5)
                            for r in ("model", "big_move", "vol_only")
                        ]:
                            rules.append({
                                "tau": t, "c_bp": c, "model": m, "rule": rule, "q": q,
                                "c_eval_bp": ce, "split": split, "n_trades": int(3000 * q),
                                "mean_net": 0.0002 * (1 + q), "se": 0.00005, "hit": 0.52,
                                "hit_se": 0.01, "sharpe": 0.4, "sharpe_se": 0.2,
                                "diff_vs_af": 0.0001, "diff_vs_af_se": 0.00004,
                                "diff_vs_bm": 0.0001, "diff_vs_bm_se": 0.00004,
                                "diff_vs_vol": 0.0001, "diff_vs_vol_se": 0.00004, "coverage": q,
                            })  # fmt: skip
    quint = pl.DataFrame([
        {"tau": t, "c_bp": c, "model": m, "group": g, "n": 600, "mean_net": 0.0001 * g,
         "se": 0.00005, "hit": 0.5, "hit_se": 0.01, "af_mean_net": 0.0002}
        for t in TAUS for c in CS for m in MODELS for g in range(1, 6)
    ])  # fmt: skip
    groups = pl.DataFrame([
        {"step": s, "features_added": f, "auc": 0.5 + 0.01 * i, "top20_mean_net": 0.0002, "se": 0.0001}
        for i, (s, f) in enumerate([("vol", "volatility"), ("+signal", "signal"), ("+path", "path")])
    ]) if skill else pl.DataFrame(schema={"step": pl.String, "features_added": pl.String,
                                          "auc": pl.Float64, "top20_mean_net": pl.Float64,
                                          "se": pl.Float64})  # fmt: skip
    picked = pl.DataFrame({
        "feature": FEATURES, "mean_picked": [0.1 * i for i in range(22)],
        "mean_other": [0.05] * 22, "std_diff": [(-1) ** i * 0.05 * i for i in range(22)],
    })  # fmt: skip
    per_year = pl.DataFrame({"year": list(range(2014, 2027)), "n_picked": [50] * 13,
                             "n_days": [250] * 13})  # fmt: skip
    return {"grid": grid, "rules": pl.DataFrame(rules), "quintiles": quint, "groups": groups,
            "picked": picked, "per_year": per_year}  # fmt: skip


def write_inputs(out, verdict="partly amplified", skill=True, cumulative=True):
    d = out / "part2"
    d.mkdir()
    for k, v in fake_frames(skill).items():
        v.write_parquet(d / f"{k}.parquet")
    if cumulative:
        days = [date(2014, 1, 2) + timedelta(days=i) for i in range(30)]
        pl.DataFrame([
            {"session": x, "rule": r, "wealth": 1 + 0.001 * k * (1 + n)}
            for n, r in enumerate(("model", "always_follow", "big_move")) for k, x in enumerate(days)
        ]).write_parquet(d / "cumulative.parquet")  # fmt: skip
    summary = {
        "verdict": verdict, "d_af": 2.6, "d_bm": 1.4, "d_0": 3.1, "skill": skill,
        "headline_model": "logit", "n_fits": 1234, "n_cells": 90, "synthetic": True,
        "data_window": ["2011-01-03", "2026-08-31"], "count_line": "part 2: 90 cells, 500 runs",
    }  # fmt: skip
    (out / "part2_summary.json").write_text(json.dumps(summary))
    report_part2.make_figures(out)


def pages(out):
    en, zh = report_part2.write_part2_report(out)
    return en.read_text(), zh.read_text()


def test_report_renders_both_languages_with_d50_labels_only(tmp_path):
    write_inputs(tmp_path)
    en, zh = pages(tmp_path)
    for page in (en, zh):
        assert "part 2: 90 cells, 500 runs" in page and "1234" in page
        assert page.count("data:image/png;base64,") == 4
        assert "http://" not in page and "<script" not in page
        text = re.sub(r"<style>.*?</style>|<img[^>]*>", "", page, flags=re.DOTALL)
        assert not BARE.search(text), BARE.search(text)
    assert "previous close to 15:30" in en and "前一天收盤到 15:30" in zh
    assert "previous close to 12:00" in en and "前一天收盤到 14:00" in zh
    assert "15:30 to close" in en and "15:30 到收盤" in zh
    assert "partly amplified" in en and "部分放大" in zh
    assert not [c for c in ANY_CELL.findall(zh) if CODE.search(c)], "code name in a Chinese table"
    assert not [c for c in ANY_CELL.findall(en) if CODE.search(c)], "code name in an English table"
    num = re.compile(r"-?\d+(\.\d+)?")
    nums = lambda page: [c for c in ANY_CELL.findall(page) if num.fullmatch(c)]
    assert nums(en) == nums(zh) and len(nums(en)) > 20


@pytest.mark.parametrize(
    ("verdict", "zh_word"),
    [("amplified", "放得大"), ("partly amplified", "部分放大"), ("not amplified", "放不大")],
)
def test_three_verdicts(tmp_path, verdict, zh_word):
    write_inputs(tmp_path, verdict=verdict)
    en, zh = pages(tmp_path)
    assert f"<b>{verdict}</b>" in en and f"<b>{zh_word}</b>" in zh


def test_empty_groups_says_skipped_and_no_cumulative_is_noted(tmp_path):
    write_inputs(tmp_path, skill=False, cumulative=False)
    en, zh = pages(tmp_path)
    assert (
        "input-group comparison was skipped because the headline cell has no predictive power" in en
    )
    assert "輸入群組的比較被略過" in zh
    assert en.count("data:image/png;base64,") == 3 and "cumulative series was not available" in en
    assert not (tmp_path / "figures" / "part2_cumulative.png").exists()


def test_every_grid_cell_shows_value_and_se(tmp_path):
    write_inputs(tmp_path)
    en, _ = pages(tmp_path)
    grid = fake_frames()["grid"]
    html = report_part2._grid_table(grid, "en")
    assert html.count("<tr>") == 1 + len(MODELS) * len(TAUS)
    for r in grid.iter_rows(named=True):
        assert f"{r['top20_minus_af'] * 1e4:.1f} ({r['se'] * 1e4:.1f})" in html
    assert html in en
    for stem in config.PART2_FIGURES:
        assert (tmp_path / "figures" / f"{stem}.png").stat().st_size > 5000


def test_rules_table_values_and_baselines(tmp_path):
    rules = fake_frames()["rules"]
    html = report_part2._rules_table(rules, "logit", "en")
    # always follow once per cost; 3 q x (model, big-move, vol-only) per cost; header row
    assert html.count("<tr>") == 1 + 3 * (1 + 9)
    # mean_net 0.0002 * (1 + 0.2) = 2.4 bp, SE 0.5 bp -> "2.4 ± 1.0" (checked by hand)
    assert "model, top 20% of days</td><td>2</td><td>600</td><td>2.4 ± 1.0" in html
    assert "always follow the signal</td><td>2</td><td>3000</td><td>4.0 ± 1.0" in html


def test_labels_fail_loud(tmp_path):
    write_inputs(tmp_path)
    f = fake_frames()
    f["picked"] = f["picked"].with_columns(feature=pl.lit("new_feature"))
    with pytest.raises(AssertionError, match="no zh label for feature 'new_feature'"):
        report_part2._picked_table(f["picked"], "zh")
    with pytest.raises(AssertionError, match="no en label for verdict"):
        report_part2._lab(report_part2.VERDICT, "maybe", "en", "verdict")
    with pytest.raises(AssertionError, match="unknown decision time"):
        report_part2._signal("11:00", "en")
    two = f["grid"].filter(pl.col("tau") == "10:00")
    with pytest.raises(AssertionError, match="grid cell"):
        report_part2._grid_table(pl.concat([f["grid"], two.head(1)]), "en")
