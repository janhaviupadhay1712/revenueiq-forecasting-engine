"""Diagnostics: STL decomposition, robust anomaly detection, scenario analysis."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import STL


def decompose(y: pd.Series) -> pd.DataFrame:
    """STL decomposition + trend/seasonal strength (0-1, Hyndman definition)."""
    r = STL(y, period=12, robust=True).fit()
    d = pd.DataFrame({"trend": r.trend, "seasonal": r.seasonal, "resid": r.resid})
    d.attrs["trend_strength"] = float(max(0, 1 - d.resid.var() / (d.trend + d.resid).var()))
    d.attrs["seasonal_strength"] = float(max(0, 1 - d.resid.var() / (d.seasonal + d.resid).var()))
    return d


def detect_anomalies(y: pd.Series, threshold: float = 3.5) -> pd.DataFrame:
    """Robust z-score (median/MAD) on STL residuals — resistant to the outliers it is hunting."""
    d = decompose(y)
    med = d.resid.median()
    mad = float((d.resid - med).abs().median()) or 1e-9
    score = 0.6745 * (d.resid - med) / mad
    return pd.DataFrame({"revenue": y, "expected": d.trend + d.seasonal, "score": score,
                         "is_anomaly": score.abs() > threshold})


def clean_series(y: pd.Series, anomalies: pd.DataFrame) -> pd.Series:
    """Replace flagged months with their expected (trend + seasonal) value."""
    return y.where(~anomalies["is_anomaly"], anomalies["expected"])


def scenarios(fc: pd.DataFrame) -> pd.DataFrame:
    """Bear / Base / Bull totals from the lower band, point forecast and upper band."""
    return pd.DataFrame({"Scenario": ["Bear (lower band)", "Base", "Bull (upper band)"],
                         "Total": [fc["lower"].sum(), fc["forecast"].sum(), fc["upper"].sum()]})
