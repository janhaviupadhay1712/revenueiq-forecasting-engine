import pandas as pd
from forecasting.analytics import clean_series, decompose, detect_anomalies, scenarios
from forecasting.engine import ALL_MODELS, RevenueForecaster, generate_demo_data, prepare_series
from forecasting.store import list_runs, log_run, realized_accuracy
import pytest


def _y():
    return prepare_series(generate_demo_data())


def test_decompose_strengths_in_range():
    d = decompose(_y())
    assert 0 <= d.attrs["trend_strength"] <= 1 and 0 <= d.attrs["seasonal_strength"] <= 1


def test_injected_spike_is_flagged_and_cleaned():
    y = _y().copy()
    y.iloc[30] *= 3
    a = detect_anomalies(y)
    assert a["is_anomaly"].iloc[30]
    assert clean_series(y, a).iloc[30] < y.iloc[30]


def test_model_subset_and_unknown_model():
    res = RevenueForecaster(horizon=3, models=["Theta", "Seasonal Naive + Drift"]).run(_y())
    assert set(res.weights) == {"Theta", "Seasonal Naive + Drift"}
    with pytest.raises(ValueError):
        RevenueForecaster(models=["nope"])


def test_all_models_run():
    assert len(RevenueForecaster(horizon=3).run(_y()).leaderboard) == len(ALL_MODELS)


def test_scenarios_ordered():
    f = RevenueForecaster(horizon=6).run(_y()).forecast
    t = scenarios(f)["Total"]
    assert t[0] <= t[1] <= t[2]


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("REVENUEIQ_DB", str(tmp_path / "t.db"))
    y = _y()
    log_run(y.iloc[:-3], RevenueForecaster(horizon=3).run(y.iloc[:-3]))
    log_run(y.iloc[:-3], RevenueForecaster(horizon=3).run(y.iloc[:-3]))   # idempotent
    assert len(list_runs()) == 1
    assert len(realized_accuracy(y)) == 1
