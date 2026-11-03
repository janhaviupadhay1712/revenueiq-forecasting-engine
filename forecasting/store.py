"""Run history + forecast monitoring (SQLite). Tracks every run and scores old forecasts vs new actuals."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .engine import ForecastResult


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(os.getenv("REVENUEIQ_DB", "runs.db"))
    c.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY, key TEXT UNIQUE, ts TEXT, n_months INT, horizon INT,
        mape REAL, total REAL, best_model TEXT, forecast TEXT)""")
    return c


def log_run(series: pd.Series, res: ForecastResult) -> None:
    key = hashlib.sha1(series.values.tobytes() + ",".join(res.weights).encode()
                       + str(len(res.forecast)).encode()).hexdigest()[:16]
    fc = {str(d.date()): float(v) for d, v in res.forecast["forecast"].items()}
    with closing(_conn()) as c, c:
        c.execute("INSERT OR IGNORE INTO runs(key,ts,n_months,horizon,mape,total,best_model,forecast) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (key, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), len(series), len(fc),
                   None if np.isnan(res.ensemble_mape) else res.ensemble_mape,
                   float(res.forecast["forecast"].sum()), res.leaderboard.iloc[0]["Model"], json.dumps(fc)))


def list_runs(limit: int = 20) -> pd.DataFrame:
    with closing(_conn()) as c:
        df = pd.read_sql_query("SELECT id, ts, n_months, horizon, mape, total, best_model FROM runs "
                               "ORDER BY id DESC LIMIT ?", c, params=(limit,))
    return df.astype(object).where(df.notna(), None)


def realized_accuracy(series: pd.Series) -> pd.DataFrame:
    """Score every stored forecast against actuals that have since arrived (model monitoring)."""
    rows = []
    with closing(_conn()) as c:
        for rid, ts, raw in c.execute("SELECT id, ts, forecast FROM runs"):
            f = pd.Series(json.loads(raw))
            f.index = pd.to_datetime(f.index)
            common = f.index.intersection(series.index)
            if len(common):
                err = np.abs((series[common] - f[common]) / series[common]).mean() * 100
                rows.append({"run": rid, "created": ts, "months_scored": len(common), "realized MAPE %": float(err)})
    return pd.DataFrame(rows)
