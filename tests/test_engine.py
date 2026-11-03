import numpy as np
import pytest

from forecasting.engine import RevenueForecaster, generate_demo_data, prepare_series


def test_pipeline_shapes_and_accuracy():
    res = RevenueForecaster(horizon=6).run(prepare_series(generate_demo_data()))
    assert len(res.forecast) == 6
    assert (res.forecast["lower"] <= res.forecast["forecast"]).all()
    assert (res.forecast["forecast"] <= res.forecast["upper"]).all()
    assert res.ensemble_mape < 15
    assert np.isclose(sum(res.weights.values()), 1)


def test_rejects_short_history():
    with pytest.raises(ValueError):
        prepare_series(generate_demo_data(months=10))
