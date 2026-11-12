"""Controls known by 10:00: no look-ahead, VIX lag, terciles, and the calendar-driven flags."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from first_half_hour import calendar, config, controls

START, END = date(2018, 1, 2), date(2019, 12, 31)  # about 500 sessions, past the 252 warm-up


@pytest.fixture(scope="module")
def cal():
    return calendar.sessions(START, END)


def _table(cal, seed=0):
    rng = np.random.default_rng(seed)
    n = cal.height
    return cal.select("session", "is_half_day").with_columns(
        rv_fh=pl.Series(rng.uniform(1e-6, 1e-4, n)),
        rv_day=pl.Series(rng.uniform(1e-5, 1e-3, n)),
        vol_fh=pl.Series(rng.integers(1_000, 90_000, n)),
    )


def _write_daily(path, vix_shift=0.0, after=None):
    """FRED-style files on every weekday of 2017-2020 (value = a smooth function of the date,
    plus `vix_shift` for dates >= `after`), one empty holiday-style value, and event files."""
    days = [d for i in range(1500) if (d := date(2017, 1, 2) + timedelta(i)).weekday() < 5]
    for name, base in (("VIXCLS", 15.0), ("VXVCLS", 18.0)):
        vals = [
            ""
            if d == date(2019, 7, 4)
            else f"{base + d.timetuple().tm_yday / 10 + (vix_shift if after and d >= after else 0):.2f}"
            for d in days
        ]
        text = f"observation_date,{name}\n" + "".join(
            f"{d},{v}\n" for d, v in zip(days, vals, strict=True)
        )
        (path / f"{name}.csv").write_text(text)
    # 2019-01-29 Tue, 2019-03-19 Tue, 2019-06-14 Fri (announcement is the next session: Mon 17th)
    (path / "fomc_meetings.csv").write_text(
        "date\n2018-06-12\n2019-01-29\n2019-03-19\n2019-06-14\n"
    )
    (path / "fomc_emergency.csv").write_text("date\n2019-03-04\n")
    (path / "fomc_cancelled.csv").write_text("date,cancelled_on\n2019-05-01,2019-04-20\n")
    (path / "release_dates.csv").write_text(
        "date,release\n2019-04-19,cpi\n2019-03-12,cpi\n2019-03-08,payroll\n2019-07-04,payroll\n"
    )


@pytest.fixture
def daily(tmp_path):
    _write_daily(tmp_path)
    return tmp_path


def test_controls_for_rows_up_to_k_ignore_later_rows(cal, daily, tmp_path_factory):
    k = 400
    table = _table(cal)
    base = controls.controls(table, cal, daily)

    cut = cal["session"][k]
    rng = np.random.default_rng(1)
    after = pl.col("session") > cut
    bumped = table.with_columns(
        rv_fh=pl.when(after).then(pl.col("rv_fh") * 50).otherwise(pl.col("rv_fh")),
        rv_day=pl.when(after).then(pl.col("rv_day") * 50).otherwise(pl.col("rv_day")),
        vol_fh=pl.when(after).then(rng.integers(1, 10**7)).otherwise(pl.col("vol_fh")),
    )
    other = tmp_path_factory.mktemp("daily_perturbed")
    _write_daily(other, vix_shift=40.0, after=cut)  # VIX dated on/after session k changes
    perturbed = controls.controls(bumped, cal, other)

    assert base.height == perturbed.height == table.height
    head, head_p = base.head(k + 1), perturbed.head(k + 1)
    assert head.equals(head_p)  # bit-identical, nulls included
    assert not base.tail(5).equals(perturbed.tail(5))  # the perturbation did reach later rows


def test_vix_at_t_is_the_value_dated_before_t(cal, daily):
    out = controls.controls(_table(cal), cal, daily)
    row = {r["session"]: r for r in out.iter_rows(named=True)}

    def vix(d):  # same formula the fixture wrote; no code under test involved
        return 15.0 + round(d.timetuple().tm_yday / 10, 2)

    assert row[date(2019, 1, 7)]["vix_lag"] == pytest.approx(vix(date(2019, 1, 4)))  # Mon: Friday
    assert row[date(2019, 1, 8)]["vix_lag"] == pytest.approx(vix(date(2019, 1, 7)))  # never t
    # 2019-07-04 has an empty FRED value: 2019-07-05 takes the latest earlier value (07-03).
    assert row[date(2019, 7, 5)]["vix_lag"] == pytest.approx(vix(date(2019, 7, 3)))
    ratio = (15.0 + round(date(2019, 1, 4).timetuple().tm_yday / 10, 2)) / (
        18.0 + round(date(2019, 1, 4).timetuple().tm_yday / 10, 2)
    )
    assert row[date(2019, 1, 7)]["vix_ratio_lag"] == pytest.approx(ratio)


def test_tercile_cutoffs_use_only_earlier_sessions(cal, daily):
    table = _table(cal)
    out = controls.controls(table, cal, daily)
    v = table["rv_fh"].to_list()
    w = config.TERCILE_WARMUP
    assert out["rv_fh_cut"][:w].null_count() == w and out["rv_fh_cut"][w] is not None
    for i in (w, w + 17, 450):
        past = sorted(v[:i])  # linear-interpolated 2/3 quantile worked by hand
        pos = (len(past) - 1) * 2 / 3
        lo = int(pos)
        want = past[lo] + (past[lo + 1] - past[lo]) * (pos - lo)
        assert out["rv_fh_cut"][i] == pytest.approx(want, rel=1e-12)
        assert out["rv_fh_top"][i] == float(v[i] > want)


def test_relative_volume_uses_previous_20_kept_sessions(cal, daily):
    table = (
        _table(cal)
        .with_columns(vol_fh=pl.lit(100))
        .with_columns(vol_fh=pl.when(pl.int_range(pl.len()) == 25).then(500).otherwise(100))
    )
    out = controls.controls(table, cal, daily)
    assert out["relvol"][:20].null_count() == 20 and out["relvol"][20] == 1.0
    assert out["relvol"][25] == 5.0  # 500 against twenty sessions of 100
    assert out["relvol"][26] == pytest.approx(100 / ((19 * 100 + 500) / 20))  # includes day 25


def test_event_flags_follow_the_calendar(cal, daily):
    out = controls.controls(_table(cal), cal, daily)
    on = lambda col: set(out.filter(pl.col(col))["session"].to_list())
    # Meeting file lists the FIRST day; announcement is the next session (Fri 06-14 -> Mon 06-17).
    # The 2019-03-04 emergency date and the cancelled 2019-05-01 meeting are never flagged.
    assert on("fomc") == {
        date(2018, 6, 13),
        date(2019, 1, 30),
        date(2019, 3, 20),
        date(2019, 6, 17),
    }
    # Good Friday 2019-04-19 was a market holiday: that CPI release maps to Monday 04-22.
    assert on("cpi") == {date(2019, 3, 12), date(2019, 4, 22)}
    assert on("payroll") == {date(2019, 3, 8), date(2019, 7, 5)}  # Jul 4 holiday -> Jul 5
    assert on("news") == on("fomc") | on("cpi") | on("payroll")


def test_expiry_month_end_and_era_flags(cal, daily):
    out = controls.controls(_table(cal), cal, daily)
    on = lambda col: set(out.filter(pl.col(col))["session"].to_list())
    monthly = on("opex_monthly")
    assert len(monthly) == 24  # one per month, Jan 2018 .. Dec 2019
    assert date(2019, 3, 15) in monthly and date(2018, 6, 15) in monthly  # third Fridays
    assert date(2019, 4, 18) in monthly and date(2019, 4, 19) not in monthly  # Good Friday -> Thu
    assert date(2019, 3, 15) in on("opex_quarterly") and date(2019, 4, 18) not in on(
        "opex_quarterly"
    )
    assert len(on("opex_quarterly")) == 8
    assert date(2019, 3, 29) in on("month_end") and date(2019, 4, 30) in on("month_end")
    assert date(2019, 3, 28) not in on("month_end") and len(on("month_end")) == 24
    assert not on("daily_expiry_era")  # window ends before 2022-05-11
    assert on("is_half_day") == set(cal.filter(pl.col("is_half_day"))["session"].to_list())
