from fastapi.testclient import TestClient

from api import app
from forecasting.engine import generate_demo_data

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_forecast_roundtrip():
    df = generate_demo_data()
    payload = {
        "data":
            [{"date": d.strftime("%Y-%m-%d"), "revenue": r}
             for d, r in zip(
                df["date"],
                df["revenue"],
                strict=True
            )
             ],
               "horizon": 6
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    assert len(r.json()["points"]) == 6


def test_short_history_rejected():
    r = client.post("/forecast", json={"data": [{"date": "2024-01-01", "revenue": 1}]})
    assert r.status_code == 422


def test_demo():
    assert client.get("/forecast/demo?horizon=3").status_code == 200
