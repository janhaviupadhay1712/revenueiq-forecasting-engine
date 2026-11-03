"""Executive insights: LLM-written narrative (if ANTHROPIC_API_KEY set) with deterministic fallback."""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

from .engine import ForecastResult

log = logging.getLogger(__name__)


def build_facts(res: ForecastResult, scenario_pct: float = 0.0) -> dict:
    mult = 1 + scenario_pct / 100
    fc = res.forecast["forecast"] * mult
    last12 = res.history.iloc[-12:].sum()
    seasonal = res.history.groupby(res.history.index.month).mean()
    spread = ((res.forecast["upper"] - res.forecast["lower"]) / res.forecast["forecast"]).mean() * 100
    return {
        "next_12m_forecast": round(float(fc.sum()), 0),
        "last_12m_actual": round(float(last12), 0),
        "yoy_growth_pct": round(float(fc.sum() / last12 - 1) * 100, 1),
        "peak_forecast_month": fc.idxmax().strftime("%B %Y"),
        "trough_forecast_month": fc.idxmin().strftime("%B %Y"),
        "strongest_seasonal_month": pd.Timestamp(2000, int(seasonal.idxmax()), 1).strftime("%B"),
        "weakest_seasonal_month": pd.Timestamp(2000, int(seasonal.idxmin()), 1).strftime("%B"),
        "avg_interval_width_pct": round(float(spread), 1),
        "ensemble_mape_pct": None if pd.isna(res.ensemble_mape) else round(res.ensemble_mape, 2),
        "top_model": res.leaderboard.iloc[0]["Model"],
        "scenario_adjustment_pct": scenario_pct,
    }


def rule_based_narrative(f: dict) -> str:
    tone = "growth" if f["yoy_growth_pct"] >= 0 else "contraction"
    risk = "high" if f["avg_interval_width_pct"] > 25 else "moderate" if f["avg_interval_width_pct"] > 12 else "low"
    return (
        f"**Outlook:** Revenue is projected at **₹{f['next_12m_forecast']:,.0f}** over the forecast window, "
        f"a **{f['yoy_growth_pct']:+.1f}%** {tone} versus the trailing 12 months.\n\n"
        f"**Seasonality:** Demand peaks around **{f['peak_forecast_month']}** and dips in "
        f"**{f['trough_forecast_month']}**. Historically {f['strongest_seasonal_month']} is the strongest month and "
        f"{f['weakest_seasonal_month']} the weakest — align hiring, inventory and campaigns accordingly.\n\n"
        f"**Confidence:** Forecast uncertainty is **{risk}** (avg. band ±{f['avg_interval_width_pct']/2:.1f}%). "
        f"Backtest MAPE is {f['ensemble_mape_pct']}%, led by *{f['top_model']}*.\n\n"
        f"**Action:** Plan on the lower band for cost commitments and the central line for targets."
    )


def llm_narrative(facts: dict) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=600,
            system="You are a CFO-level FP&A analyst. Write a crisp 4-part executive summary "
                   "(Outlook, Seasonality, Risks, Recommended actions) from the JSON facts. Use only given numbers. "
                   "Currency is INR (₹). Markdown, under 180 words.",
            messages=[{"role": "user", "content": json.dumps(facts)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception as exc:
        log.warning("LLM insight failed, using fallback: %s", exc)
        return None


def narrative(res: ForecastResult, scenario_pct: float = 0.0) -> tuple[str, str]:
    facts = build_facts(res, scenario_pct)
    text = llm_narrative(facts)
    return (text, "Claude") if text else (rule_based_narrative(facts), "Rule engine")
