"""RevenueIQ REST API. Run: uvicorn api:app --reload  ->  docs at http://localhost:8000/docs"""
from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from datetime import date

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from forecasting.analytics import decompose, detect_anomalies
from forecasting.engine import ALL_MODELS, RevenueForecaster, generate_demo_data, prepare_series
from forecasting.store import list_runs, log_run
from forecasting.insights import narrative

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("revenueiq.api")

app = FastAPI(title="RevenueIQ API", version="1.0.0",
              description="Ensemble ML revenue forecasting with backtested confidence intervals.")


API_KEY = os.getenv("REVENUEIQ_API_KEY")  # unset => auth disabled (local dev only)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid, t0 = request.headers.get("x-request-id", uuid.uuid4().hex[:12]), time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info("%s %s -> %s in %.0fms [%s]", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000, rid)
    return response


class Record(BaseModel):
    date: date
    revenue: float


class ForecastRequest(BaseModel):
    data: list[Record] = Field(min_length=24, description="Monthly revenue history, >= 24 rows")
    horizon: int = Field(12, ge=1, le=36)
    confidence: float = Field(0.90, ge=0.5, le=0.99)
    scenario_pct: float = Field(0.0, ge=-50, le=50, description="What-if uplift/drag in %")
    include_insights: bool = False
    models: list[str] | None = Field(None, description="Subset of GET /models; default = all")


class ForecastPoint(BaseModel):
    month: date
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    points: list[ForecastPoint]
    leaderboard: list[dict]
    ensemble_mape_pct: float | None
    total_forecast: float
    insights: str | None = None
    insights_engine: str | None = None


def _run(df: pd.DataFrame, req: ForecastRequest) -> ForecastResponse:
    try:
        series = prepare_series(df)
        res = RevenueForecaster(horizon=req.horizon, confidence=req.confidence, models=req.models).run(series)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        log.exception("Forecast failed")
        raise HTTPException(status_code=500, detail="Forecast failed")from exc

    try:
        log_run(series, res)
    except Exception:
        log.warning("run logging failed", exc_info=True)
    mult = 1 + req.scenario_pct / 100
    f = res.forecast[["forecast", "lower", "upper"]] * mult
    board = res.leaderboard.astype(object).where(res.leaderboard.notna(), None).to_dict("records")
    text, engine = (None, None)
    if req.include_insights:
        text, engine = narrative(res, req.scenario_pct)
    return ForecastResponse(
        points=[ForecastPoint(month=i.date(), **r) for i, r in zip(f.index, f.to_dict("records"))],
        leaderboard=board,
        ensemble_mape_pct=None if pd.isna(res.ensemble_mape) else round(res.ensemble_mape, 3),
        total_forecast=float(f["forecast"].sum()),
        insights=text, insights_engine=engine,
    )


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"], dependencies=[Depends(require_key)])
def forecast(req: ForecastRequest) -> ForecastResponse:
    df = pd.DataFrame([r.model_dump() for r in req.data])
    return _run(df, req)


@app.get("/forecast/demo", response_model=ForecastResponse, tags=["forecast"], dependencies=[Depends(require_key)])
def forecast_demo(horizon: int = 12) -> ForecastResponse:
    req = ForecastRequest.model_construct(horizon=horizon, confidence=0.9, scenario_pct=0.0,
                                          include_insights=False)
    return _run(generate_demo_data(), req)


class SeriesRequest(BaseModel):
    data: list[Record] = Field(min_length=24)


def _series(req: SeriesRequest) -> pd.Series:
    try:
        return prepare_series(pd.DataFrame([r.model_dump() for r in req.data]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/models", tags=["forecast"], dependencies=[Depends(require_key)])
def models() -> dict:
    return {"models": list(ALL_MODELS)}


@app.post("/anomalies", tags=["diagnostics"], dependencies=[Depends(require_key)])
def anomalies(req: SeriesRequest, threshold: float = 3.5) -> dict:
    a = detect_anomalies(_series(req), threshold)
    hits = a[a.is_anomaly]
    return {"count": len(hits), "anomalies": [
        {"month": i.date(), "revenue": float(r.revenue), "expected": float(r.expected), "score": float(r.score)}
        for i, r in hits.iterrows()]}


@app.post("/decompose", tags=["diagnostics"], dependencies=[Depends(require_key)])
def decompose_endpoint(req: SeriesRequest) -> dict:
    d = decompose(_series(req))
    return {"trend_strength": d.attrs["trend_strength"], "seasonal_strength": d.attrs["seasonal_strength"],
            "components": [{"month": i.date(), "trend": float(r.trend), "seasonal": float(r.seasonal),
                            "residual": float(r.resid)} for i, r in d.iterrows()]}


@app.get("/runs", tags=["monitoring"], dependencies=[Depends(require_key)])
def runs(limit: int = 20) -> list[dict]:
    return list_runs(limit).to_dict("records")
