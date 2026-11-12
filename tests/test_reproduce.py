"""`fhh reproduce --synthetic` end to end: nine MLflow items, the count line, H1-H5 with baselines."""

import json

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from first_half_hour import cli, config, reproduce


@pytest.mark.slow
def test_reproduce_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    before = (config.DATA_START, config.SAMPLES, config.BREAK_DATES)
    cli.main(["reproduce", "--synthetic", "--out", str(tmp_path)])
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert (config.DATA_START, config.SAMPLES, config.BREAK_DATES) == before, "config not restored"

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
