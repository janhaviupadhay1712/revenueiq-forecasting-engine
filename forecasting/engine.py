"""Core forecasting engine: ensemble of statistical + ML models with rolling-origin backtesting."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

log = logging.getLogger(__name__)
MIN_POINTS = 24


# --------------------------------------------------------------------------- data
def generate_demo_data(months: int = 48, seed: int = 42) -> pd.DataFrame:
    """Synthetic SaaS-style revenue: trend + yearly seasonality + Q4 spike + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=months, freq="MS")
    t = np.arange(months)
    rev = (
        250_000 + 6_500 * t
        + 40_000 * np.sin(2 * np.pi * (idx.month - 3) / 12)
        + 55_000 * (idx.month == 12)
        + rng.normal(0, 9_000, months)
    )
    return pd.DataFrame({"date": idx, "revenue": rev.round(2)})


def prepare_series(df: pd.DataFrame, date_col: str = "date", value_col: str = "revenue") -> pd.Series:
    """Validate, aggregate to monthly, and gap-fill a raw revenue table."""
    if date_col not in df or value_col not in df:
        raise ValueError(f"Data must contain '{date_col}' and '{value_col}' columns.")
    d = df[[date_col, value_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        raise ValueError("No valid rows after parsing dates/revenue.")
    s = d.set_index(date_col)[value_col].sort_index().resample("MS").sum(min_count=1)
    s = s.interpolate(limit_direction="both")
    if len(s) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} months of data; got {len(s)}.")
    return s


# --------------------------------------------------------------------------- models
class HoltWintersModel:
    name = "Holt-Winters (ETS)"

    def fit(self, y: pd.Series):
        self.m = ExponentialSmoothing(
            y.values, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.asarray(self.m.forecast(h))


class RegressionModel:
    name = "Trend + Seasonality (Ridge)"

    @staticmethod
    def _X(dates: pd.DatetimeIndex, t0: int) -> np.ndarray:
        t = np.arange(t0, t0 + len(dates))
        month = np.eye(12)[dates.month - 1]
        return np.column_stack([t, month])

    def fit(self, y: pd.Series):
        self.n, self.last = len(y), y.index[-1]
        self.m = Ridge(alpha=1.0).fit(self._X(y.index, 0), y.values)
        return self

    def predict(self, h: int) -> np.ndarray:
        fut = pd.date_range(self.last, periods=h + 1, freq="MS")[1:]
        return self.m.predict(self._X(fut, self.n))


class GradientBoostModel:
    """Predicts month-over-month growth from lag/rolling/calendar features; forecasts recursively."""
    name = "Gradient Boosting (ML)"
    LAGS = (1, 2, 3, 6, 12)

    def _make(self):
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=7)

    def _row(self, hist: list[float], date: pd.Timestamp) -> list[float]:
        scale = hist[-1] or 1.0
        lags = [hist[-l] / scale for l in self.LAGS]
        roll = [np.mean(hist[-3:]) / scale, np.mean(hist[-6:]) / scale]
        ang = 2 * np.pi * date.month / 12
        return lags + roll + [np.sin(ang), np.cos(ang)]

    def fit(self, y: pd.Series):
        v, idx = list(y.values), y.index
        X, t = [], []
        for i in range(12, len(v)):
            X.append(self._row(v[:i], idx[i]))
            t.append(v[i] / (v[i - 1] or 1.0) - 1.0)
        self.m = self._make().fit(np.array(X), np.array(t))
        self.hist, self.last = v, y.index[-1]
        return self

    def predict(self, h: int) -> np.ndarray:
        hist, out = list(self.hist), []
        for date in pd.date_range(self.last, periods=h + 1, freq="MS")[1:]:
            g = float(self.m.predict(np.array([self._row(hist, date)]))[0])
            hist.append(hist[-1] * (1 + g))
            out.append(hist[-1])
        return np.array(out)


class ExtraTreesModel(GradientBoostModel):
    """Same lag features, bagged randomised trees — lower variance than boosting."""
    name = "Extra Trees (ML)"

    def _make(self):
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=7)


class ThetaForecaster:
    """Theta method — strong, simple benchmark that wins many forecasting competitions."""
    name = "Theta"

    def fit(self, y: pd.Series):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self.m = ThetaModel(y, period=12, deseasonalize=True).fit()
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.asarray(self.m.forecast(h))


class SeasonalNaiveDrift:
    """Baseline: repeat last year's pattern plus year-over-year growth."""
    name = "Seasonal Naive + Drift"

    def fit(self, y: pd.Series):
        v = y.values
        self.last12, self.growth = v[-12:], float(v[-12:].mean() - v[-24:-12].mean())
        return self

    def predict(self, h: int) -> np.ndarray:
        k = np.arange(h)
        return self.last12[k % 12] + self.growth * (k // 12 + 1)


MODEL_REGISTRY = [HoltWintersModel, RegressionModel, GradientBoostModel, ExtraTreesModel,
                  ThetaForecaster, SeasonalNaiveDrift]
ALL_MODELS = {M.name: M for M in MODEL_REGISTRY}


# --------------------------------------------------------------------------- metrics
def mape(a: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100)


def rmse(a: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - f) ** 2)))


# --------------------------------------------------------------------------- orchestration
@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.DataFrame
    leaderboard: pd.DataFrame
    ensemble_mape: float
    weights: dict[str, float] = field(default_factory=dict)


class RevenueForecaster:
    def __init__(self, horizon: int = 12, folds: int = 3, confidence: float = 0.90,
                 models: list[str] | None = None):
        self.horizon, self.folds, self.confidence = horizon, folds, confidence
        unknown = set(models or []) - set(ALL_MODELS)
        if unknown:
            raise ValueError(f"Unknown models {sorted(unknown)}. Available: {sorted(ALL_MODELS)}")
        self.registry = [ALL_MODELS[n] for n in models] if models else MODEL_REGISTRY

    def run(self, y: pd.Series) -> ForecastResult:
        bh = min(self.horizon, 6)
        errs: dict[str, list[float]] = {}
        rm: dict[str, list[float]] = {}
        preds: dict[str, list[np.ndarray]] = {}
        actuals: list[np.ndarray] = []

        for f in range(self.folds):
            cut = len(y) - bh * (f + 1)
            if cut < MIN_POINTS - 6:
                break
            train, test = y.iloc[:cut], y.iloc[cut:cut + bh].values
            actuals.append(test)
            for M in self.registry:
                try:
                    p = M().fit(train).predict(bh)
                except Exception as exc:
                    log.warning("%s failed in backtest: %s", M.name, exc)
                    p = np.full(bh, train.iloc[-1])
                preds.setdefault(M.name, []).append(p)
                errs.setdefault(M.name, []).append(mape(test, p))
                rm.setdefault(M.name, []).append(rmse(test, p))

        names = [M.name for M in self.registry]
        if actuals:
            m_mape = {n: float(np.mean(errs[n])) for n in names}
            inv = {n: 1 / max(m_mape[n], 0.1) ** 2 for n in names}
            w = {n: inv[n] / sum(inv.values()) for n in names}
            ens_err = [
                np.sum([w[n] * preds[n][i] for n in names], axis=0) - actuals[i] for i in range(len(actuals))
            ]
            sigma = float(np.std(np.concatenate(ens_err)))
            ens_mape = float(np.mean([mape(actuals[i], actuals[i] + ens_err[i]) for i in range(len(actuals))]))
        else:
            m_mape = {n: float("nan") for n in names}
            w = {n: 1 / len(names) for n in names}
            sigma, ens_mape = float(y.std() * 0.1), float("nan")

        fc = {M.name: M().fit(y).predict(self.horizon) for M in self.registry}
        point = np.sum([w[n] * fc[n] for n in names], axis=0)
        z = norm.ppf(0.5 + self.confidence / 2)
        width = z * sigma * np.sqrt(1 + 0.15 * np.arange(self.horizon))
        dates = pd.date_range(y.index[-1], periods=self.horizon + 1, freq="MS")[1:]
        out = pd.DataFrame({"forecast": point, "lower": np.maximum(point - width, 0), "upper": point + width,
                            **fc}, index=dates)

        board = pd.DataFrame({
            "Model": names,
            "MAPE %": [m_mape[n] for n in names],
            "RMSE": [float(np.mean(rm[n])) if n in rm else np.nan for n in names],
            "Weight": [w[n] for n in names],
        }).sort_values("MAPE %")
        return ForecastResult(y, out, board, ens_mape, w)
