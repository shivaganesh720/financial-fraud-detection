"""
dashboard.py — SentinelShield AI Streamlit Dashboard.

Features
--------
- Live KPI bar: total transactions, fraud count, fraud rate, avg confidence
- Real-time transaction feed table with colour-coded risk rows
- Risk breakdown donut chart (Plotly)
- Fraud probability time-series chart (Plotly)
- Manual prediction form: paste feature values → instant result + risk badge
- Auto-refreshes every 5 seconds via st.rerun()

Run
---
    streamlit run api/dashboard.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:5000")
REFRESH_INTERVAL_S = 5   # seconds between auto-refreshes
HISTORY_LIMIT      = 100 # most recent transactions to show

V_FEATURES = [f"V{i}" for i in range(1, 29)]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; }
    .risk-high   { color: #FF4B4B; font-weight: 600; }
    .risk-medium { color: #FFA500; font-weight: 600; }
    .risk-low    { color: #00C853; font-weight: 600; }
    .fraud-badge { background:#FF4B4B; color:white; padding:3px 10px;
                   border-radius:12px; font-size:0.85rem; }
    .safe-badge  { background:#00C853; color:white; padding:3px 10px;
                   border-radius:12px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_INTERVAL_S)
def fetch_statistics() -> dict:
    """Call GET /statistics and return the JSON dict."""
    try:
        resp = requests.get(f"{API_URL}/statistics", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {
            "total_predictions": 0,
            "total_frauds": 0,
            "total_legitimate": 0,
            "fraud_rate": 0.0,
            "avg_probability": 0.0,
            "risk_breakdown": {"high": 0, "medium": 0, "low": 0},
        }


@st.cache_data(ttl=REFRESH_INTERVAL_S)
def fetch_history(limit: int = HISTORY_LIMIT) -> list[dict]:
    """Call GET /history?limit=N and return the records list."""
    try:
        resp = requests.get(f"{API_URL}/history", params={"limit": limit}, timeout=3)
        resp.raise_for_status()
        return resp.json().get("records", [])
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_model_info() -> dict:
    """Call GET /model-info (cached for 60 s — changes rarely)."""
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def check_api_health() -> tuple[bool, bool]:
    """Return (api_ok, mongo_ok)."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        data = resp.json()
        return data.get("model_loaded", False), data.get("mongodb_connected", False)
    except Exception:
        return False, False


def post_predict(payload: dict) -> dict | None:
    """POST a single transaction to /predict. Returns JSON or None on error."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Prediction request failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ SentinelShield AI")
    st.caption("Real-time fraud detection dashboard")
    st.divider()

    model_loaded, mongo_ok = check_api_health()

    st.markdown("**System status**")
    col_a, col_b = st.columns(2)
    col_a.metric("API model",  "✅ Ready"   if model_loaded else "❌ Offline")
    col_b.metric("MongoDB",    "✅ Online"  if mongo_ok     else "⚠️ Offline")
    st.divider()

    info = fetch_model_info()
    if info.get("model_type"):
        st.markdown(f"**Model:** `{info['model_type']}`")
    if info.get("training_metrics"):
        tm = info["training_metrics"].get("test_metrics", {})
        if tm:
            st.markdown("**Test metrics**")
            st.markdown(f"- AUPRC: `{tm.get('AUPRC', '–')}`")
            st.markdown(f"- ROC-AUC: `{tm.get('ROC-AUC', '–')}`")
            st.markdown(f"- Recall: `{tm.get('Recall', '–')}`")
            st.markdown(f"- F1: `{tm.get('F1', '–')}`")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (5 s)", value=True)
    manual_refresh = st.button("🔄 Refresh now")
    st.caption(f"API: `{API_URL}`")


# ---------------------------------------------------------------------------
# Main layout — tabs
# ---------------------------------------------------------------------------

tab_live, tab_predict, tab_history = st.tabs([
    "📊 Live Monitor", "🔍 Manual Predict", "📋 History"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    stats   = fetch_statistics()
    records = fetch_history()

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total transactions", f"{stats['total_predictions']:,}")
    k2.metric("Fraud alerts",        f"{stats['total_frauds']:,}")
    k3.metric("Legitimate",          f"{stats['total_legitimate']:,}")
    k4.metric("Fraud rate",          f"{stats['fraud_rate']:.2f}%")
    k5.metric("Avg confidence",      f"{stats.get('avg_probability', 0):.3f}")

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────
    chart_col, donut_col = st.columns([2, 1])

    with chart_col:
        st.subheader("Fraud probability over time")
        if records:
            df_hist = pd.DataFrame(records)
            df_hist["timestamp"] = pd.to_datetime(
                df_hist.get("timestamp", pd.Series(dtype=str)), errors="coerce"
            )
            df_hist = df_hist.sort_values("timestamp")
            df_hist["probability"] = pd.to_numeric(
                df_hist.get("probability", 0), errors="coerce"
            ).fillna(0)
            df_hist["is_fraud"] = df_hist.get("prediction", 0).astype(bool)

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["probability"],
                mode="lines+markers",
                name="Fraud probability",
                line=dict(color="#00CED1", width=1.5),
                marker=dict(
                    color=["#FF4B4B" if f else "#00CED1"
                           for f in df_hist["is_fraud"]],
                    size=6,
                ),
            ))
            fig_ts.add_hline(y=0.75, line_dash="dash", line_color="#FF4B4B",
                             annotation_text="High-risk threshold (0.75)")
            fig_ts.add_hline(y=0.40, line_dash="dot", line_color="#FFA500",
                             annotation_text="Medium-risk threshold (0.40)")
            fig_ts.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No transaction data yet — start the simulator to populate.")

    with donut_col:
        st.subheader("Risk breakdown")
        rb = stats.get("risk_breakdown", {"high": 0, "medium": 0, "low": 0})
        fig_donut = go.Figure(go.Pie(
            labels=["High", "Medium", "Low"],
            values=[rb["high"], rb["medium"], rb["low"]],
            hole=0.55,
            marker_colors=["#FF4B4B", "#FFA500", "#00C853"],
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── Recent transaction feed ────────────────────────────────────────────
    st.subheader("Recent transactions")
    if records:
        rows = []
        for rec in records[:30]:
            risk = rec.get("risk_level", "low")
            prob = rec.get("probability", 0.0)
            pred = rec.get("prediction", 0)
            ts   = rec.get("timestamp", "")
            amt  = rec.get("transaction_data", {}).get("Amount", "—")

            risk_label = {
                "high":   "🔴 High",
                "medium": "🟡 Medium",
                "low":    "🟢 Low",
            }.get(risk, risk)

            rows.append({
                "Time":        str(ts)[:19] if ts else "—",
                "Amount":      f"${float(amt):,.2f}" if amt != "—" else "—",
                "Fraud prob":  f"{prob:.4f}",
                "Prediction":  "⚠️ FRAUD" if pred else "✅ OK",
                "Risk":        risk_label,
            })

        df_feed = pd.DataFrame(rows)
        st.dataframe(df_feed, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions logged yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.subheader("Manual transaction prediction")
    st.caption(
        "Enter transaction feature values and click **Predict** to get an "
        "instant fraud score. V1–V28 are the PCA-transformed components."
    )

    with st.expander("📋 Quick-fill: sample fraud transaction", expanded=False):
        st.code(
            "Amount=2500, V3=-4.0, V12=-5.0, V14=-6.5, V17=-7.0, "
            "all other V features=-1.8",
            language="text",
        )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=1.0)
        with col2:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        st.markdown("**PCA features V1 – V28**")

        # Render V features in 4 columns
        n_cols = 4
        v_cols = st.columns(n_cols)
        v_values: dict[str, float] = {}
        for idx, feat in enumerate(V_FEATURES):
            default = -6.5 if feat == "V14" else (-7.0 if feat == "V17" else 0.0)
            v_values[feat] = v_cols[idx % n_cols].number_input(
                feat, value=default, format="%.4f", key=f"inp_{feat}"
            )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        payload = {"Amount": amount, **v_values}
        with st.spinner("Running inference…"):
            result = post_predict(payload)

        if result:
            prob      = result.get("fraud_probability", 0.0)
            is_fraud  = result.get("is_fraud", False)
            risk      = result.get("risk_level", "low")
            ts        = result.get("timestamp", "")

            verdict_col, gauge_col = st.columns([1, 2])

            with verdict_col:
                if is_fraud:
                    st.error("⚠️ FRAUD DETECTED")
                else:
                    st.success("✅ TRANSACTION APPROVED")

                risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                st.markdown(
                    f"**Risk level:** {risk_colors.get(risk, '')} `{risk.upper()}`"
                )
                st.markdown(f"**Fraud probability:** `{prob:.6f}`")
                st.caption(f"Evaluated at {str(ts)[:19]}")

            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "valueformat": ".2f"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#FF4B4B" if is_fraud else "#00CED1"},
                        "steps": [
                            {"range": [0, 40],  "color": "rgba(0,200,83,0.2)"},
                            {"range": [40, 75], "color": "rgba(255,165,0,0.2)"},
                            {"range": [75, 100], "color": "rgba(255,75,75,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": 75,
                        },
                    },
                    title={"text": "Fraud confidence score"},
                ))
                fig_gauge.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY TABLE
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader("Full prediction history")

    limit_h = st.slider("Show last N predictions", 10, 500, 100, step=10)
    records_h = fetch_history(limit=limit_h)

    if records_h:
        rows_h = []
        for rec in records_h:
            td   = rec.get("transaction_data", {})
            prob = rec.get("probability", 0.0)
            pred = rec.get("prediction", 0)
            risk = rec.get("risk_level", "low")
            ts   = rec.get("timestamp", "")
            amt  = td.get("Amount", "—")

            rows_h.append({
                "Timestamp":  str(ts)[:19] if ts else "—",
                "Amount ($)": f"{float(amt):,.2f}" if amt != "—" else "—",
                "Prob":       round(prob, 6),
                "Fraud":      bool(pred),
                "Risk":       risk.capitalize(),
            })

        df_h = pd.DataFrame(rows_h)

        # Colour-code the Fraud column
        def _colour_fraud(val):
            return "color: #FF4B4B; font-weight:600" if val else "color: #00C853"

        def _colour_risk(val):
            colours = {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#00C853"}
            return f"color: {colours.get(val, 'inherit')}; font-weight:600"

        st.dataframe(
            df_h.style
                .map(_colour_fraud, subset=["Fraud"])
                .map(_colour_risk,  subset=["Risk"]),
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv_bytes = df_h.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_bytes,
            file_name="sentinel_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No prediction history found. Run the simulator or make predictions first.")


# ---------------------------------------------------------------------------
# Auto-refresh (must be last — triggers a full script re-run)
# ---------------------------------------------------------------------------
if auto_refresh or manual_refresh:
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL_S)
        st.rerun()