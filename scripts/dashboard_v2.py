"""
PRADHAN Dashboard — Beautiful Solar Flare Monitoring
=====================================================
Modern Streamlit dashboard with custom CSS styling.

Usage:
    streamlit run scripts/dashboard_v2.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, FIGURES_DIR, FLUX_THRESHOLDS

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRADHAN — Solar Flare Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #2d1b3d 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .header-container h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-container p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Alert cards */
    .alert-green {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #d1fae5;
    }
    .alert-yellow {
        background: linear-gradient(135deg, #713f12, #854d0e);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #fef3c7;
    }
    .alert-red {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #fee2e2;
    }
    .alert-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
        margin: 1.5rem 0 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ────────────────────────────────────────────────────────
@st.cache_data
def load_goes_data():
    """Load GOES parquet data."""
    goes_dir = DATA_DIR / "goes"
    parquet_files = sorted(goes_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    dfs = [pd.read_parquet(f) for f in parquet_files]
    goes = pd.concat(dfs, ignore_index=False).sort_index()
    if "xrsa" in goes.columns and "xrs_a_flux" not in goes.columns:
        goes = goes.rename(columns={"xrsa": "xrs_a_flux", "xrsb": "xrs_b_flux"})
    goes = goes.loc[:, ~goes.columns.duplicated()]
    return goes


@st.cache_data
def load_results():
    """Load training results."""
    results = {}
    for name in ["best_model_results", "multi_config_results", "training_results"]:
        path = Path(f"results/{name}.json")
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


# ── Plotly Theme ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    hoverlabel=dict(bgcolor="#1e293b", font_size=13),
)


# ── Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>☀️ PRADHAN — Solar Flare Forecasting</h1>
    <p>Predictive Real-time Analysis of Data from Heliospheric Aditya-Navigation • XGBoost Nowcasting Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    data_source = st.selectbox("Data Source", ["GOES XRS-B", "SoLEXS SDD2", "HEL1OS"])
    time_range = st.select_slider(
        "Time Range",
        options=["Last 24h", "Last 7 days", "Last 30 days", "All Data"],
        value="All Data",
    )
    threshold_class = st.selectbox("Alert Threshold", ["C-class", "M-class", "X-class"], index=1)
    sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("## 📊 Model Info")
    st.markdown("""
    - **Algorithm:** XGBoost
    - **Features:** 19 statistical proxies
    - **Best Config:** 1h C-class
    - **TSS:** 0.7931
    - **AUC:** 0.9611
    """)

    st.markdown("---")
    st.markdown("## 🔗 Links")
    st.markdown("[GitHub](https://github.com) • [NOAA SWPC](https://www.swpc.noaa.gov)")
    st.caption("Built for ISRO Aditya-L1 Mission")


# ── Load Data ───────────────────────────────────────────────────────────
goes = load_goes_data()
results = load_results()


# ── Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Monitor", "📈 Light Curves", "🎯 Model Performance",
    "🔬 Feature Analysis", "📋 Configuration Compare"
])


# ── Tab 1: Live Monitor ────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Real-Time Flare Monitoring</p>', unsafe_allow_html=True)

    if goes is not None:
        latest = goes.iloc[-1]
        flux_b = float(latest["xrs_b_flux"])
        flux_a = float(latest["xrs_a_flux"])
        last_time = goes.index[-1]

        # Determine alert level
        if flux_b >= FLUX_THRESHOLDS["X"]:
            alert_class, alert_color, alert_text = "X", "red", "EXTREME — X-class flare detected"
        elif flux_b >= FLUX_THRESHOLDS["M"]:
            alert_class, alert_color, alert_text = "M", "yellow", "WARNING — M-class flare active"
        elif flux_b >= FLUX_THRESHOLDS["C"]:
            alert_class, alert_color, alert_text = "C", "green", "MODERATE — C-class activity"
        else:
            alert_class, alert_color, alert_text = "B", "green", "QUIET — Low solar activity"

        # Alert banner
        if alert_color == "red":
            st.markdown(f'<div class="alert-red"><div class="alert-title">🚨 {alert_text}</div>'
                        f'Flux: {flux_b:.2e} W/m² • Class: {alert_class} • Time: {last_time}</div>',
                        unsafe_allow_html=True)
        elif alert_color == "yellow":
            st.markdown(f'<div class="alert-yellow"><div class="alert-title">⚠️ {alert_text}</div>'
                        f'Flux: {flux_b:.2e} W/m² • Class: {alert_class} • Time: {last_time}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green"><div class="alert-title">✅ {alert_text}</div>'
                        f'Flux: {flux_b:.2e} W/m² • Class: {alert_class} • Time: {last_time}</div>',
                        unsafe_allow_html=True)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{flux_b:.1e}</div>'
                        f'<div class="metric-label">XRS-B Flux (W/m²)</div></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{flux_a:.1e}</div>'
                        f'<div class="metric-label">XRS-A Flux (W/m²)</div></div>',
                        unsafe_allow_html=True)
        with col3:
            ratio = flux_b / max(flux_a, 1e-12)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{ratio:.2f}</div>'
                        f'<div class="metric-label">Hard/Soft Ratio</div></div>',
                        unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{alert_class}</div>'
                        f'<div class="metric-label">Current Class</div></div>',
                        unsafe_allow_html=True)

        # Real-time mini chart (last 24h)
        st.markdown('<p class="section-header">Last 24 Hours</p>', unsafe_allow_html=True)
        last_24h = goes.last("24H")
        if len(last_24h) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=last_24h.index, y=last_24h["xrs_b_flux"],
                name="XRS-B", line=dict(color="#ef4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"
            ))
            fig.add_trace(go.Scatter(
                x=last_24h.index, y=last_24h["xrs_a_flux"],
                name="XRS-A", line=dict(color="#3b82f6", width=1)
            ))
            for cls, thresh in [("C", 1e-6), ("M", 1e-5), ("X", 1e-4)]:
                fig.add_hline(y=thresh, line_dash="dot", line_color="#f59e0b",
                              annotation_text=cls, annotation_position="right")
            fig.update_layout(**PLOTLY_LAYOUT, height=350, yaxis_type="log",
                              yaxis_title="Flux (W/m²)", xaxis_title="Time")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No GOES data available. Run data pipeline first.")


# ── Tab 2: Light Curves ────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">X-Ray Light Curves</p>', unsafe_allow_html=True)

    if goes is not None:
        col1, col2 = st.columns([3, 1])
        with col2:
            year = st.selectbox("Year", sorted(goes.index.year.unique()), index=len(goes.index.year.unique())-1)
            show_quality = st.checkbox("Show quality flags", value=False)
            show_flares = st.checkbox("Highlight flares", value=True)

        with col1:
            year_data = goes[goes.index.year == year]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.08, row_heights=[0.7, 0.3])

            fig.add_trace(go.Scatter(
                x=year_data.index, y=year_data["xrs_b_flux"].clip(lower=1e-9),
                name="XRS-B", line=dict(color="#ef4444", width=0.5),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=year_data.index, y=year_data["xrs_a_flux"].clip(lower=1e-9),
                name="XRS-A", line=dict(color="#3b82f6", width=0.5),
            ), row=1, col=1)

            for cls, thresh, color in [("C", 1e-6, "#f59e0b"), ("M", 1e-5, "#ec4899"), ("X", 1e-4, "#a855f7")]:
                fig.add_hline(y=thresh, line_dash="dot", line_color=color,
                              annotation_text=cls, row=1, col=1)

            if show_quality and "xrs_b_quality" in year_data.columns:
                fig.add_trace(go.Scatter(
                    x=year_data.index, y=year_data["xrs_b_quality"],
                    name="Quality", line=dict(color="#64748b", width=0.5),
                ), row=2, col=1)

            fig.update_layout(**PLOTLY_LAYOUT, height=550,
                              yaxis_type="log", yaxis_title="Flux (W/m²)",
                              yaxis2_title="Quality", xaxis2_title="Time")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No GOES data available.")


# ── Tab 3: Model Performance ───────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)

    if "best_model_results" in results:
        best = results["best_model_results"]
        m = best["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["tss"]:.4f}</div>'
                        f'<div class="metric-label">TSS</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["auc"]:.4f}</div>'
                        f'<div class="metric-label">AUC-ROC</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["hss"]:.4f}</div>'
                        f'<div class="metric-label">HSS</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["pod"]:.4f}</div>'
                        f'<div class="metric-label">POD (Hit Rate)</div></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            # TSS comparison chart
            if "multi_config_results" in results:
                fig = go.Figure()
                cfgs = sorted(results["multi_config_results"],
                              key=lambda x: x["metrics"]["tss"], reverse=True)
                labels = [c["label"] for c in cfgs]
                tss_vals = [c["metrics"]["tss"] for c in cfgs]
                colors = ["#4CAF50" if t >= 0.65 else "#FF9800" if t >= 0.50 else "#F44336" for t in tss_vals]

                fig.add_trace(go.Bar(
                    y=labels, x=tss_vals, orientation="h",
                    marker_color=colors, text=[f"{v:.4f}" for v in tss_vals],
                    textposition="outside"
                ))
                fig.add_vline(x=0.65, line_dash="dash", line_color="#3b82f6",
                              annotation_text="ISRO Target")
                fig.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="TSS",
                                  title="TSS Across Configurations")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature importance
            imp = best.get("feature_importance", {})
            if imp:
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=[x[0] for x in sorted_imp][::-1],
                    x=[x[1] for x in sorted_imp][::-1],
                    orientation="h",
                    marker_color="#a78bfa"
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="Importance",
                                  title="Feature Importance (Best Model)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model results found. Run training first.")


# ── Tab 4: Feature Analysis ────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Feature Analysis</p>', unsafe_allow_html=True)

    if goes is not None:
        from src.data.features import compute_features, get_feature_names

        soft = goes["xrs_a_flux"].values
        hard = goes["xrs_b_flux"].values
        features = compute_features(soft, hard, cadence_seconds=60.0)
        feature_names = get_feature_names()
        features.index = goes.index

        col1, col2 = st.columns(2)
        with col1:
            feat = st.selectbox("Feature", feature_names, index=feature_names.index("hard_mean_5m"))
            sample = features[feat].dropna().sample(min(50000, len(features)), random_state=42)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=sample, nbinsx=80, marker_color="#60a5fa",
                                       opacity=0.8, name=feat))
            fig.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title=feat, yaxis_title="Count",
                              title=f"Distribution: {feat}")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=features["hard_mean_5m"].dropna().sample(20000, random_state=42),
                y=features["soft_hard_corr"].dropna().sample(20000, random_state=42),
                mode="markers", marker=dict(size=3, color="#f472b6", opacity=0.4)
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350,
                              xaxis_title="hard_mean_5m", yaxis_title="soft_hard_corr",
                              title="Feature Correlation")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for feature analysis.")


# ── Tab 5: Configuration Compare ───────────────────────────────────────
with tab5:
    st.markdown('<p class="section-header">Configuration Comparison</p>', unsafe_allow_html=True)

    if "multi_config_results" in results:
        cfgs = results["multi_config_results"]

        # Comparison table
        table_data = []
        for r in sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True):
            m = r["metrics"]
            table_data.append({
                "Config": r["label"],
                "TSS": f"{m['tss']:.4f}",
                "HSS": f"{m['hss']:.4f}",
                "AUC": f"{m['auc']:.4f}",
                "POD": f"{m['pod']:.4f}",
                "CSI": f"{m['csi']:.4f}",
                "Brier": f"{m['brier']:.4f}",
                "Event Rate": f"{r['event_rate']:.1%}",
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Radar comparison
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            metrics_keys = ["tss", "hss", "auc", "pod", "csi"]
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#a855f7", "#06b6d4"]
            for idx, r in enumerate(sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True)[:4]):
                vals = [r["metrics"][k] for k in metrics_keys] + [r["metrics"]["tss"]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics_keys + [metrics_keys[0]],
                    name=r["label"], fill="toself",
                    line=dict(color=colors[idx % len(colors)])
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=400, polar=dict(
                bgcolor="rgba(15,23,42,0.8)",
                radialaxis=dict(gridcolor="rgba(148,163,184,0.15)")
            ), title="Metrics Radar (Top 4)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            for r in sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True):
                fig.add_trace(go.Scatter(
                    x=[r["metrics"]["pofd"]], y=[r["metrics"]["pod"]],
                    mode="markers+text", text=[r["label"]], textposition="top center",
                    marker=dict(size=15, color=colors[cfgs.index(r) % len(colors)])
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#64748b", dash="dash"), showlegend=False
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=400,
                              xaxis_title="POFD (False Alarm Rate)",
                              yaxis_title="POD (Hit Rate)",
                              title="ROC Space")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No multi-config results found.")


# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem 0;">
    PRADHAN v1.0 • Built for ISRO Aditya-L1 SoLEXS/HEL1OS Mission •
    Data: GOES XRS (2003-2024) • Model: XGBoost (19 features, 1h C-class)
</div>
""", unsafe_allow_html=True)
