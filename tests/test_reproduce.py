"""`fhh reproduce --synthetic` end to end: both chapters, the count lines, the summaries with their
baselines and the two-language reports."""

import html as html_lib
import json
import re

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from first_half_hour import chapter2, cli, config, report, report_ch2, reproduce

CELL = re.compile(r"<td>([^<]*)</td>")
ANY_CELL = re.compile(r"<t[dh][^>]*>([^<]*)</t[dh]>")
SCI = re.compile(r"\d[eE][+-]?\d")
# a code name (snake_case, r1/r12/r13/rod, or the variant keys) must not show in Chinese tables
CODE = re.compile(
    r"[a-z]+_[a-z0-9_]+|(?<![A-Za-z0-9])(r1|r12|r13|rod|lr1|inter|const)(?![A-Za-z0-9])"
)
NUM = re.compile(r"-?\d+(\.\d+)?(e[+-]?\d+)?")


def _numeric_cells(html: str) -> list[list[str]]:
    """Per table, the cells that are pure numbers (labels differ by language, numbers must not)."""
    tables = html.split("<table>")[1:]
    return [[c for c in CELL.findall(t) if NUM.fullmatch(c)] for t in tables]


@pytest.mark.slow
def test_reproduce_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    before = (config.DATA_START, config.SAMPLES, config.BREAK_DATES, config.CH2_VALID)
    cli.main(["reproduce", "--synthetic", "--out", str(tmp_path)])
    lines = capsys.readouterr().out.strip().splitlines()
    printed, printed2 = lines[-2], lines[-1]
    assert (config.DATA_START, config.SAMPLES, config.BREAK_DATES, config.CH2_VALID) == before, (
        "config not restored"
    )

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlruns.db').resolve()}")
    exp = mlflow.get_experiment_by_name("chapter1-synthetic")
    runs = MlflowClient().search_runs([exp.experiment_id], max_results=50_000)
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    assert len(parents) == 9
    assert sorted(int(r.data.params["item"]) for r in parents) == list(range(1, 10))
    assert printed == (f"chapter 1: 9 items, {len(runs)} runs ({len(runs) - 9} nested)"), (
        f"printed {printed!r} but MLflow holds {len(runs)}"
    )
    assert (tmp_path / "figures" / "rolling_slope.png").exists()
    assert all((tmp_path / "chapter1" / f"item{n}.parquet").exists() for n in range(1, 10))

    summary = json.loads((tmp_path / "chapter1_summary.json").read_text())
    for h in ("H1", "H2"):
        assert summary[h]["baseline_slope"] == 0.0 and summary[h]["baseline_r2"] == 0.0
    # planted slope (SYNTHETIC_BETA) recovered on the OVERLAP-equivalent sample
    assert abs(summary["H1"]["slope"] - reproduce.SYNTHETIC_BETA) < 0.08
    assert all("baseline_r2_os" in v for v in summary["H3"]["specs"].values())
    assert 0 <= summary["H4"]["joint_p"] <= 1 and "baseline" in summary["H4"]
    h5 = summary["H5"]
    assert h5["baseline_zero_sharpe"] == 0.0 and "baseline_always_long_sharpe" in h5
    assert h5["ci_lo"] <= h5["sharpe_net"] <= h5["ci_hi"]

    html = (tmp_path / "chapter1_report.html").read_text()
    assert html.count('<h2 id="item') == 9
    assert printed in html and "data:image/png;base64," in html
    assert re.findall(r'src="([^"]*)"', html) and all(
        u.startswith("data:image/png;base64,") for u in re.findall(r'src="([^"]*)"', html)
    )
    assert "http://" not in html and "https://" not in html and "<script" not in html

    zh = (tmp_path / "chapter1_report_zh.html").read_text()
    assert zh.count('<h2 id="item') == 9 and printed in zh and 'lang="zh-Hant-TW"' in zh
    assert "data:image/png;base64," in zh and "http://" not in zh and "<script" not in zh
    en_tables, zh_tables = _numeric_cells(html), _numeric_cells(zh)
    assert len(en_tables) == len(zh_tables) >= 9 and any(en_tables)
    assert en_tables == zh_tables, "numbers differ between the English and Chinese reports"
    zh_cells = " ".join(CELL.findall(zh))
    assert not re.search(r"OVERLAP|POST|FULL|\bnw\b|cluster_date|driscoll_kraay", zh_cells)
    for page in (html, zh):
        box = page.split('<div class="box">')[1].split("</div>")[0]
        assert not re.search(r"\d\.\d{4,}", box) and not SCI.search(box), "raw number in box"
        t9 = page.split('<h2 id="item9">')[1]
        assert "<th colspan=" in t9 and t9.count('class="sub"') >= 8, "no rule sub-headings"
        assert max(len(c) for c in ANY_CELL.findall(t9)) <= 45, "long cell in item 9"
        assert '<p class="fn">¹' in t9
    # every table sits in a horizontally scrolling box
    for page in (html, zh):
        assert page.count("<table>") == page.count('style="overflow-x:auto"><table>') >= 9
        assert not [c for c in ANY_CELL.findall(page) if SCI.search(c)], "e-notation in a cell"
    zh_all = ANY_CELL.findall(zh)
    assert not [c for c in zh_all if CODE.search(c)], "code name shown in a Chinese table"

    _check_chapter2(tmp_path, printed2)


def _check_chapter2(out, printed):
    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlruns.db').resolve()}")
    exp = mlflow.get_experiment_by_name("chapter2-synthetic")
    runs = MlflowClient().search_runs([exp.experiment_id], max_results=50_000)
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    assert sorted(r.info.run_name for r in parents) == ["chapter2", "register final model"]
    nested = len(runs) - 2  # the parent and the registry run are the only two top-level runs
    assert printed == f"chapter 2: {nested + 1} runs ({nested} nested)", printed
    # 25 primary models + 3 descriptive targets x (4 baselines + the chosen tier's configs + 1)
    assert nested in (25 + 3 * 6, 25 + 3 * 14, 25 + 3 * 15), nested
    model = MlflowClient().get_registered_model(config.CH2_REGISTERED_MODEL)
    assert model.latest_versions, "chosen model not registered"

    s = json.loads((out / "chapter2_summary.json").read_text())
    p = s["primary"]
    assert p["baseline_bss"] == 0.0 and p["alpha"] == 0.05
    assert p["primary_pass"] == (p["judging_bss_vs_calibrated"] > 0 and p["dm_p_one_sided"] < 0.05)
    if not p["primary_pass"]:
        assert s["stated_result"] == chapter2.STATED_FAIL
    assert set(s["validation_bss_cal_by_tier"]) == {"tier:logit", "tier:enet", "tier:hgb"}
    assert s["spa"]["n_models"] == 24 and "23 distinct" in s["spa"]["note"]
    assert s["deflated_sharpe"]["n_trials"] == 20 and s["runs"]["count_line"] == printed
    assert {r["rule"] for r in s["economics_1bp"]} >= {"always_buy", "zero"}
    assert all((out / "figures" / f"{f}.png").exists() for f in
               ("reliability", "permutation_importance", "partial_dependence"))  # fmt: skip

    en = (out / "chapter2_report.html").read_text()
    zh = (out / "chapter2_report_zh.html").read_text()
    for page in (en, zh):
        assert printed in page and page.count('<h2 id="s') == 11
        assert "http://" not in page and "<script" not in page
        assert page.count("<table>") == page.count('style="overflow-x:auto"><table>') >= 9
        assert not [c for c in ANY_CELL.findall(page) if SCI.search(c)], "e-notation in a cell"
        assert page.count("data:image/png;base64,") == 3 and '<p class="fn">¹' in page
        assert page.count('class="sub"') >= 4  # a sub-heading per economics rule
    assert "adds nothing beyond a one-line formula" in en or p["primary_pass"]
    assert "auditor's advice before any real-data result" in html_lib.unescape(en)
    assert _numeric_cells(en) == _numeric_cells(zh), "numbers differ between the languages"
    assert not [c for c in ANY_CELL.findall(zh) if CODE.search(c)], "code name in a Chinese table"


def test_chinese_label_missing_fails_loud():
    assert report._label("zh", "sample", "POST") == "2014–26（發表後）"
    assert report._label("zh", "variant", "break_2014-01-02").startswith("2014-01-02")
    with pytest.raises(AssertionError, match="no Chinese label"):
        report._label("zh", "variant", "trim_top_7%")
    assert report._label("en", "variant", "trim_top_7%") == "trim_top_7%"


def test_cell_formats():
    f = report._fmt
    assert f(0.0004321, "p") == "&lt;0.001"  # HTML-escaped and f(1.0, "p_holm") == "1.000"
    assert f(0.0235, "r2") == "2.35" and f(-0.00001, "r2") == "0.00"
    assert f(1.23456, "t") == "1.23" and f(-0.0004, "sharpe") == "0.00"
    assert f(0.00000123, "slope") == "0.00000123" and f(123456.0, "chi2") == "123456"
    assert f(0.0002, "coef", bp=True) == "2" and f(0.0002, "coef") == "0.0002"
    assert f(float("nan"), "t") == "" and f(None, "p") == ""


def test_chapter2_labels_fail_loud_and_targets_are_named():
    assert report._label("zh", "model", "cfg:enet:C=0.1,l1_ratio=0.5") == (
        "彈性網路（正則化強度 C 0.1，L1 比例 0.5）"
    )
    with pytest.raises(AssertionError, match="no Chinese label"):
        report._label("zh", "model", "cfg:svm:C=1")
    with pytest.raises(AssertionError, match="no Chinese label"):
        report._label("zh", "feature", "new_feature")
    assert list(chapter2.targets()) == ["up_0.4", "up_0.2", "up_0.6", "down_0.4"]
    prim = {"primary_pass": True}
    spa = {"p_consistent": 0.01, "n_models": 24}
    assert "SPA p = 0.010 over 24 models" in report_ch2.stated_result(
        prim, spa, "chapter 2: 1 runs"
    )
    assert report_ch2.stated_result({"primary_pass": False}, spa, "x") == chapter2.STATED_FAIL
