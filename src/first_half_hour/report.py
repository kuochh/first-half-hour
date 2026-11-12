"""One self-contained HTML page for chapter 1: a table per analysis item, baselines beside the numbers.

Reads `<out>/chapter1/item1..9.parquet`, `<out>/chapter1_summary.json` and the rolling-slope
figure; writes `<out>/chapter1_report.html` (inline CSS, the figure embedded as base64, no JS and
no external assets). Nothing is recomputed here: it only lays out what `fhh reproduce` wrote.
"""

import base64
import json
import math
from html import escape
from pathlib import Path

import polars as pl

CSS = """
body{font:15px/1.55 -apple-system,Helvetica,Arial,sans-serif;max-width:1100px;margin:20px auto;
padding:0 14px;color:#222}
h1{font-size:1.5em}h2{margin-top:2em;border-bottom:1px solid #ddd;font-size:1.2em}
.t{overflow-x:auto}table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 7px;text-align:right;white-space:nowrap}
th{background:#f4f4f4}.box{background:#f7f9fc;border:1px solid #dde5f0;padding:8px 16px;
border-radius:6px}.warn{color:#8a4b00}img{max-width:100%;height:auto}
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


def _fmt(v) -> str:
    """Cell text: None/NaN blank, floats to 4 significant digits, everything HTML-escaped."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return escape(f"{v:.4g}")
    return escape(str(v))


def _table(df: pl.DataFrame, cols: list[str]) -> str:
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for row in df.select(cols).iter_rows()
    )
    return f'<div class="t"><table><tr>{head}</tr>{body}</table></div>'


def _section(n: int, title: str, text: str, *parts: str) -> str:
    return f'<h2 id="item{n}">Item {n}: {escape(title)}</h2><p>{text}</p>' + "".join(parts)


def _box(s: dict, count_line: str) -> str:
    h1, h2, h3, h4, h5 = (s[k] for k in ("H1", "H2", "H3", "H4", "H5"))
    spec = "r13~r1"
    o = h3["specs"][spec]
    return f"""<div class="box"><p><b>{escape(count_line)}</b></p><ul>
<li><b>H1</b> (2011-13 slope of last half-hour return on first half-hour return): slope {h1["slope"]:.4g},
Newey-West t {h1["nw_t"]:.3g}, p {h1["p"]:.3g}, R2 {h1["r2"]:.3g}, T {h1["T"]}. Baseline: slope 0, R2 0.</li>
<li><b>H2</b> (same slope, 2014-2026): slope {h2["slope"]:.4g}, t {h2["nw_t"]:.3g}, p {h2["p"]:.3g},
R2 {h2["r2"]:.3g}, T {h2["T"]}. Baseline: slope 0, R2 0.</li>
<li><b>H3</b> (out-of-sample, {escape(spec)}): R2_OS vs the expanding mean {o["r2_os_vs_mean"]:.3g}, vs zero
{o["r2_os_vs_zero"]:.3g}, Clark-West p {o["cw_p_vs_mean"]:.3g}. Baseline R2_OS: 0.</li>
<li><b>H4</b> (break at {escape(h4["break_date"])}, intercept and slope jointly): chi2 {h4["joint_chi2"]:.3g},
p {h4["joint_p"]:.3g}. Baseline: {escape(h4["baseline"])}.</li>
<li><b>H5</b> (timing rule, {h5["cost_bp_per_side"]:g} bp per side): net Sharpe {h5["sharpe_net"]:.3g},
95% CI [{h5["ci_lo"]:.3g}, {h5["ci_hi"]:.3g}]. Baselines: always-long Sharpe
{h5["baseline_always_long_sharpe"]:.3g}; zero (no trade) {h5["baseline_zero_sharpe"]:g}.</li>
</ul></div>"""


def write_chapter1_report(out: Path, count_line: str) -> Path:
    """Write `<out>/chapter1_report.html` from the files `fhh reproduce` already wrote."""
    s = json.loads((out / "chapter1_summary.json").read_text())
    t = {n: pl.read_parquet(out / "chapter1" / f"item{n}.parquet") for n in range(1, 10)}
    png = (out / "figures" / "rolling_slope.png").read_bytes()
    img = base64.b64encode(png).decode()

    starts = [v[0] for v in s["samples"].values()]
    ends = [v[1] for v in s["samples"].values()]
    samples = "; ".join(f"{k} {v[0]} to {v[1]}" for k, v in s["samples"].items())
    kind = "SYNTHETIC data (a pipeline check, not a result)" if s["synthetic"] else "SPY"
    r1 = pl.col("term") == "r1"

    r3 = t[3]
    excl = r3.select(((pl.col("lo") > 0) | (pl.col("hi") < 0)).mean()).item()
    s3 = (
        f"{r3.height} rolling windows of 252 sessions: mean slope {r3['slope'].mean():.4g} "
        f"(baseline 0), min {r3['slope'].min():.4g}, max {r3['slope'].max():.4g}; "
        f"the confidence band excludes 0 in {excl:.1%} of windows."
    )
    ctrl = t[7].filter((pl.col("variant") == "all") & ~r1 & (pl.col("term") != "const"))
    t9 = t[9].filter(pl.col("cost_bp").is_in([0.0, 1.0]))

    body = [
        _section(
            1,
            "Baseline regressions",
            "Last half-hour return (r13) regressed on the first half-hour return (r1, previous close "
            "to 10:00) and on the second-to-last half-hour return (r12, 15:00 to 15:30), by sample. Read coef against "
            "baseline_coef (0) and r2 against baseline_r2 (0). t uses the standard error named in "
            "se_kind (nw = Newey-West); T is the number of sessions.",
            _table(t[1], REG),
        ),
        _section(
            2,
            "Robust variants",
            "The same slope with outliers trimmed, half days or ex-dividend days dropped, log "
            "returns, and Huber and median regressions. Same reading as item 1; se_kind says "
            "which standard error each t uses.",
            _table(t[2], REG),
        ),
        _section(
            3,
            "Rolling slope",
            "The slope of r13 on r1 in rolling windows of 252 sessions, with a 95% band. A slope "
            "that is really there should stay above the zero line. " + escape(s3),
            f'<img alt="rolling slope" src="data:image/png;base64,{img}">',
        ),
        _section(
            4,
            "Structural breaks",
            "Tests whether intercept (D) and slope (D*r1) change after each candidate break date; "
            "the joint row (D+D*r1) gives a chi-square test. Baseline: no break (all zero).",
            _table(t[4], [*REG, "chi2"]),
        ),
        _section(
            5,
            "Out-of-sample forecasts",
            "Forecasts are made one session at a time from earlier data only. R2_OS compares the "
            "squared error with a expanding-mean forecast and with a zero forecast; positive means "
            "better than that baseline, and the no-skill value is 0. Clark-West p tests the same "
            "comparison.",
            _table(
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
        _section(
            6,
            "Decomposition of the signal",
            "Splits the first-half-hour return into the overnight part (r_on) and the opening "
            "half-hour (r_fh), and uses the return from the previous close to 15:30 (rod). Same reading "
            "as item 1.",
            _table(t[6], REG),
        ),
        _section(
            7,
            "Controls known by 10:00",
            "The r1 slope again after adding one control at a time, then all together; every "
            "control uses only information available by 10:00. Compare each r1 coef with "
            "base_same_rows (the baseline slope on the identical rows). Second table: the control "
            "coefficients in the all-controls regression.",
            _table(t[7].filter(r1), [*REG, "n_dropped"]),
            "<p>Control coefficients, all-controls regression:</p>",
            _table(ctrl, ["sample", "term", "coef", "baseline_coef", "t", "se_kind", "p", "T"]),
        ),
        _section(
            8,
            "ETF panel",
            "The same slope for other ETFs, one by one and pooled by group. p_holm is the p-value "
            "corrected for the number of ETFs tested; t_dk is the Driscoll-Kraay t for pooled "
            "rows. Baseline: slope 0, R2 0.",
            _table(
                t[8],
                [
                    "group",
                    "sample",
                    "symbol",
                    "slope",
                    "baseline_slope",
                    "t",
                    "se_kind",
                    "p",
                    "p_holm",
                    "t_dk",
                    "r2",
                    "baseline_r2",
                    "T",
                    "sessions_in_sample",
                    "rows_excluded",
                ],
            ),
        ),
        _section(
            9,
            "Timing rules",
            "Trade the last half-hour in the direction of the first half-hour, net of cost, "
            "against two baselines: always long (sharpe_always_long) and not trading at all "
            "(Sharpe 0, the zero rule). Only costs of 0 and 1 bp per side are shown here; "
            "the others are in the item 9 table. Sharpe intervals are bootstrap intervals.",
            _table(
                t9,
                [
                    "sample",
                    "rule",
                    "cost_bp",
                    "n_active",
                    "sharpe",
                    "sharpe_ci_lo",
                    "sharpe_ci_hi",
                    "sharpe_always_long",
                    "hit_rate",
                    "baseline_hit",
                    "break_even_bp",
                    "note",
                ],
            ),
        ),
    ]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chapter 1: does the first half-hour predict the last?</title><style>{CSS}</style></head><body>
<h1>Chapter 1: does the first half-hour predict the last half-hour? ({escape(kind)})</h1>
<p class="warn">Data window {min(starts)} to {max(ends)}; samples: {escape(samples)}.
Bootstrap reps: {s["bootstrap_reps"]}. Not yet fact-checked.</p>
{_box(s, count_line)}
{"".join(body)}
</body></html>"""
    path = out / "chapter1_report.html"
    path.write_text(html)
    return path
