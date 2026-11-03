"""RevenueIQ — AI Revenue Forecasting Engine (Streamlit dashboard). Run: streamlit run app.py"""
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from forecasting.analytics import clean_series, decompose, detect_anomalies, scenarios
from forecasting.engine import ALL_MODELS, RevenueForecaster, generate_demo_data, prepare_series
from forecasting.store import list_runs, log_run, realized_accuracy
from forecasting.insights import narrative

import os

st.set_page_config(page_title="RevenueIQ · AI Forecasting", page_icon="📈", layout="wide")
if os.path.exists("docs/logo.svg"):
    st.logo("docs/logo.svg", icon_image="docs/icon.svg")

st.markdown("""
<style>
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#2a1f6b55,transparent),
       radial-gradient(900px 500px at 100% 0%,#0ea5e930,transparent),#0B0F1A;}
.hero{font-size:2.4rem;font-weight:800;letter-spacing:-1px;
      background:linear-gradient(90deg,#A78BFA,#38BDF8,#34D399);-webkit-background-clip:text;color:transparent;}
.sub{color:#8FA0C0;margin-bottom:1.2rem}
.kpi{background:linear-gradient(145deg,#161e36cc,#0f1526cc);border:1px solid #2a3558;border-radius:18px;
     padding:18px 20px;box-shadow:0 8px 30px #00000055;transition:.2s}
.kpi:hover{transform:translateY(-3px);border-color:#7C5CFF}
.kpi .l{color:#8FA0C0;font-size:.78rem;text-transform:uppercase;letter-spacing:1.2px}
.kpi .v{font-size:1.75rem;font-weight:700;color:#fff;margin-top:4px}
.kpi .d{font-size:.85rem;margin-top:2px}
.up{color:#34D399}.down{color:#F87171}
.card{background:#131A2Bcc;border:1px solid #2a3558;border-radius:18px;padding:22px}
#MainMenu,footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero">RevenueIQ</div><div class="sub">AI-powered revenue forecasting · '
            'ensemble ML + statistics · backtested · explainable</div>', unsafe_allow_html=True)


def inr(x: float) -> str:
    return f"₹{x/1e7:.2f} Cr" if abs(x) >= 1e7 else f"₹{x/1e5:.2f} L" if abs(x) >= 1e5 else f"₹{x:,.0f}"


@st.cache_data(show_spinner=False)
def run_pipeline(series: pd.Series, horizon: int, conf: float, models: tuple | None):
    return RevenueForecaster(horizon=horizon, confidence=conf, models=list(models) if models else None).run(series)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    up = st.file_uploader("Upload CSV (date, revenue)", type="csv")
    horizon = st.slider("Forecast horizon (months)", 3, 24, 12)
    conf = st.select_slider("Confidence band", [0.80, 0.90, 0.95], value=0.90, format_func=lambda v: f"{int(v*100)}%")
    st.subheader("🎯 What-if scenario")
    scen = st.slider("Revenue uplift / drag (%)", -30, 30, 0)
    st.subheader("🧪 Engine")
    chosen = st.multiselect("Models in ensemble", list(ALL_MODELS), default=list(ALL_MODELS))
    clean = st.toggle("Auto-correct anomalies before forecasting", value=False)
    target = st.number_input("Revenue target for horizon (₹, 0 = off)", min_value=0.0, value=0.0, step=100000.0)
    st.caption("Optionally set ANTHROPIC_API_KEY to enable Claude-written insights.")

try:
    raw = pd.read_csv(up) if up else generate_demo_data()
    series = prepare_series(raw)
    anomalies = detect_anomalies(series)
    if clean:
        series = clean_series(series, anomalies)
    res = run_pipeline(series, horizon, conf, tuple(chosen) or None)
    log_run(series, res)
except Exception as exc:
    st.error(f"Could not process data: {exc}")
    st.stop()

f = res.forecast.copy()
mult = 1 + scen / 100
for c in f.columns:
    f[c] *= mult
total, last12 = f["forecast"].sum(), series.iloc[-12:].sum()
growth = (total / last12 - 1) * 100
acc = 100 - res.ensemble_mape if pd.notna(res.ensemble_mape) else float("nan")

# ------------------------------------------------------------------ KPIs
k = st.columns(4)
cards = [
    (f"Next {horizon}M Revenue", inr(total), f"{growth:+.1f}% vs last 12M", growth >= 0),
    ("Next Month", inr(f['forecast'].iloc[0]), f['forecast'].index[0].strftime("%b %Y"), True),
    ("Peak Month", f['forecast'].idxmax().strftime("%b %Y"), inr(f['forecast'].max()), True),
    ("Backtest Accuracy", f"{acc:.1f}%", f"MAPE {res.ensemble_mape:.2f}%", acc > 90),
]
for col, (l, v, d, good) in zip(k, cards):
    col.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>'
                 f'<div class="d {"up" if good else "down"}">{d}</div></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Forecast", "🏆 Models", "🧠 AI Insights", "🗂 Data & Export",
                                                    "🔬 Diagnostics", "🎯 Scenarios", "🛰 Monitoring"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index, y=f["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.index, y=f["lower"], fill="tonexty", fillcolor="rgba(124,92,255,.22)",
                             line=dict(width=0), name=f"{int(conf*100)}% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Actual", line=dict(color="#38BDF8", width=3)))
    fig.add_trace(go.Scatter(x=f.index, y=f["forecast"], name="Forecast",
                             line=dict(color="#A78BFA", width=3, dash="dash"), mode="lines+markers"))
    fig.update_layout(template="plotly_dark", height=470, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08), yaxis=dict(gridcolor="#1f2947", tickprefix="₹"),
                      xaxis=dict(gridcolor="#1f2947"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    board = res.leaderboard.copy()
    board["Weight %"] = board.pop("Weight") * 100
    st.dataframe(
        board, use_container_width=True, hide_index=True,
        column_config={
            "MAPE %": st.column_config.NumberColumn("MAPE %", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.0f"),
            "Weight %": st.column_config.ProgressColumn("Weight %", format="%.1f", min_value=0, max_value=100),
        },
    )
    mf = go.Figure([go.Scatter(x=f.index, y=f[m], name=m) for m in res.weights] +
                   [go.Scatter(x=f.index, y=f["forecast"], name="Ensemble", line=dict(color="white", width=4))])
    mf.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(mf, use_container_width=True)
    st.caption("Weights ∝ 1/MAPE² from rolling-origin backtests — better models earn more influence.")

with tab3:
    text, engine = narrative(res, scen)
    st.markdown(text)
    st.caption(f"Generated by: {engine}")

with tab4:
    out = f.round(0).rename_axis("month").reset_index()
    st.dataframe(out, use_container_width=True, hide_index=True)
    buf = io.StringIO(); out.to_csv(buf, index=False)
    st.download_button("⬇️ Download forecast CSV", buf.getvalue(), "revenue_forecast.csv", "text/csv")

    xb = io.BytesIO()
    with pd.ExcelWriter(xb, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="Forecast", index=False)
        res.leaderboard.to_excel(w, sheet_name="Leaderboard", index=False)
        anomalies.reset_index().to_excel(w, sheet_name="Anomalies", index=False)
    st.download_button("⬇️ Download Excel report", xb.getvalue(), "revenueiq_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

with tab5:
    dec = decompose(series)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trend strength", f"{dec.attrs['trend_strength']:.2f}")
    c2.metric("Seasonal strength", f"{dec.attrs['seasonal_strength']:.2f}")
    c3.metric("Anomalies found", int(anomalies["is_anomaly"].sum()))
    d5 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonality", "Residual"))
    for i, (c, col) in enumerate(zip(["trend", "seasonal", "resid"], ["#38BDF8", "#A78BFA", "#F87171"]), 1):
        d5.add_trace(go.Scatter(x=dec.index, y=dec[c], line=dict(color=col), showlegend=False), row=i, col=1)
    d5.update_layout(height=520, **DARK)
    st.plotly_chart(d5, use_container_width=True)
    hits = anomalies[anomalies["is_anomaly"]]
    if hits.empty:
        st.success("No anomalies detected (robust z-score on STL residuals, |z| > 3.5).")
    else:
        st.warning("Unusual months — consider one-off deals, outages or data errors.")
        st.dataframe(hits.round(1).reset_index(), use_container_width=True, hide_index=True)

with tab6:
    sc = scenarios(f)
    sf = go.Figure(go.Bar(x=sc["Scenario"], y=sc["Total"], marker_color=["#F87171", "#A78BFA", "#34D399"],
                          text=[inr(v) for v in sc["Total"]], textposition="outside"))
    sf.update_layout(height=380, yaxis=dict(gridcolor="#1f2947"), **DARK)
    st.plotly_chart(sf, use_container_width=True)
    if target > 0:
        st.progress(min(total / target, 1.0), text=f"Base forecast covers {total / target:.0%} of target")
        gap = total - target
        st.metric("Gap to target (base)", inr(gap), delta="ahead" if gap >= 0 else "shortfall",
                  delta_color="normal" if gap >= 0 else "inverse")
        st.caption(f"Probability-style view: target is {'within' if sc.Total[0] <= target <= sc.Total[2] else 'outside'} "
                   "the bear–bull range.")
    else:
        st.info("Set a revenue target in the sidebar to track the gap.")

with tab7:
    st.subheader("Run history")
    st.dataframe(list_runs(), use_container_width=True, hide_index=True)
    st.subheader("Realized accuracy of past forecasts")
    acc_df = realized_accuracy(series)
    if acc_df.empty:
        st.info("No stored forecast overlaps your data yet. Re-upload the file with newer months to score past forecasts.")
    else:
        st.dataframe(acc_df.round(2), use_container_width=True, hide_index=True)
