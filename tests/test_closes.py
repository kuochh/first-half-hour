"""The official close on the bars' basis: factor, per-period constant, and every fail-loud check."""

from datetime import date

import polars as pl
import pytest

from first_half_hour import closes

# 2019-03-08 (Fri, prior session) and 2019-03-11..22. A $1.00 dividend goes ex on Mon 2019-03-18.
DAYS = [date(2019, 3, d) for d in (8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22)]
EX = date(2019, 3, 18)
CLOSE = {d: 100.0 + i for i, d in enumerate(DAYS)}  # raw Close: 100 .. 110
# f = 1 - 1.00 / Close(2019-03-15) = 1 - 1/105 before the ex-date (worked by hand), 1 after.
F0 = 1 - 1 / 105
RATIO0 = [0.9998, 0.9999, 1.0000, 1.0001, 1.0002]  # bar / (Close * f), 03-11..03-15: median 1.0000
RATIO1 = [1.0000, 1.0001, 1.0001, 1.0002, 1.0003]  # 03-18..03-22: median 1.0001


def _inputs(tmp_path, **tweak):
    ratios = dict(zip(DAYS[1:], RATIO0 + RATIO1, strict=True))
    ratios.update(tweak.get("ratio", {}))
    f = {d: F0 if d < EX else 1.0 for d in DAYS}
    table = pl.DataFrame(
        {"session": DAYS[1:], "p_close": [CLOSE[d] * f[d] * ratios[d] for d in DAYS[1:]]}
    )
    csv = pl.DataFrame(
        {
            "date": DAYS,
            "Close": [CLOSE[d] for d in DAYS],
            "Dividends": [1.0 if d == EX else 0.0 for d in DAYS],
            "Stock Splits": [0.0] * len(DAYS),
        }
    )
    csv = csv.with_columns(tweak.get("csv", lambda x: x)(csv))
    div = pl.DataFrame({"ex_date": [EX], "amount": [1.0]})
    div = tweak.get("div", div)
    csv.write_csv(tmp_path / "y.csv")
    div.write_csv(tmp_path / "d.csv")
    return table, tmp_path / "y.csv", tmp_path / "d.csv"


def test_factor_and_per_period_constant(tmp_path):
    out = closes.official_close(*_inputs(tmp_path)).sort("session")
    c0, c1 = 1.0000, 1.0001  # the periods' medians, read off RATIO0 / RATIO1 by hand
    row = {r["session"]: r for r in out.iter_rows(named=True)}
    assert row[date(2019, 3, 11)]["off_close"] == pytest.approx(CLOSE[date(2019, 3, 11)] * F0 * c0)
    assert row[date(2019, 3, 15)]["off_close"] == pytest.approx(CLOSE[date(2019, 3, 15)] * F0 * c0)
    assert row[date(2019, 3, 18)]["off_close"] == pytest.approx(CLOSE[EX] * 1.0 * c1)
    # the first session's previous close comes from the CSV row of 03-08, same period as 03-11
    assert row[date(2019, 3, 11)]["prev_off"] == pytest.approx(100.0 * F0 * c0)
    # the ex-date's previous close is the pre-dividend period's, on the same basis as its bars
    assert row[EX]["prev_off"] == pytest.approx(CLOSE[date(2019, 3, 15)] * F0 * c0)


@pytest.mark.parametrize(
    ("tweak", "message"),
    [
        ({"ratio": {date(2019, 3, 20): 1.02}}, "from the 15:59 bar"),  # 200 bp gap on one day
        ({"ratio": {d: 1.0 + 4e-4 for d in DAYS[6:]}}, "period constants differ"),  # 4 bp step
        ({"csv": lambda c: c.select(**{"Stock Splits": pl.when(pl.col("date") == EX).then(2.0).otherwise(0.0)})}, "split"),
        ({"div": pl.DataFrame({"ex_date": [EX], "amount": [1.01]})}, "amounts differ"),
        ({"div": pl.DataFrame({"ex_date": [date(2019, 3, 19)], "amount": [1.0]})}, "dividend dates differ"),
    ],
)  # fmt: skip
def test_each_check_fires(tmp_path, tweak, message):
    with pytest.raises(AssertionError, match=message):
        closes.official_close(*_inputs(tmp_path, **tweak))


def test_missing_csv_row_fires(tmp_path):
    table, csv, div = _inputs(tmp_path)
    pl.read_csv(csv).filter(pl.col("date") != "2019-03-13").write_csv(csv)
    with pytest.raises(AssertionError, match="without an official close"):
        closes.official_close(table, csv, div)
