"""
PRADHAN Dashboard v3 — Solar Flare Monitoring
===============================================
Self-contained: works with 2024 data + JSON results.
Anti-AI-slop: light canvas, sky-blue accent, no cards, no emoji.

Usage:
    streamlit run scripts/dashboard_v3.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(page_title="PRADHAN", layout="wide", initial_sidebar_state="expanded")

# ── Load Data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    goes_path = Path("data/goes/goes_2024.parquet")
    goes = pd.read_parquet(goes_path) if goes_path.exists() else None
    if goes is not None and "xrsa" in goes.columns:
        goes = goes.rename(columns={"xrsa": "xrs_a_flux", "xrsb": "xrs_b_flux"})
        goes = goes.loc[:, ~goes.columns.duplicated()]

    results = {}
    for name in ["best_model_results", "multi_config_results"]:
        p = Path(f"results/{name}.json")
        if p.exists():
            with open(p) as f:
                results[name] = json.load(f)
    return goes, results

goes, results = load_data()

# ── Thresholds ─────────────────────────────────────────────────────────
THRESHOLDS = {"C": 1e-6, "M": 1e-5, "X": 1e-4}

# ── Colors ─────────────────────────────────────────────────────────────
C = {
    "canvas": "#fafbfc", "surface": "#ffffff", "ink": "#1a1a2e",
    "muted": "#5c6370", "faint": "#9ca3af", "border": "#e5e7eb",
    "accent": "#0ea5e9", "danger": "#dc2626", "warn": "#d97706", "ok": "#16a34a",
    "navy": "#0f172a", "navy_fg": "#e2e8f0", "navy_muted": "#94a3b8",
}

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background: {C['canvas']}; }}
    .main .block-container {{ padding: 1.5rem 2rem 2rem; max-width: 1400px; }}

    .hdr {{
        background: {C['navy']}; padding: 1.5rem 2rem;
        margin: -1.5rem -2rem 1.5rem -2rem;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .hdr h1 {{ color: {C['navy_fg']}; font-size: 1.4rem; font-weight: 600; margin: 0; letter-spacing: -0.03em; }}
    .hdr .sub {{ color: {C['navy_muted']}; font-size: 0.78rem; margin-top: 0.2rem; }}
    .hdr .pill {{
        background: rgba(14,165,233,0.15); color: {C['accent']};
        padding: 0.3rem 0.7rem; border-radius: 4px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    }}

    .alert {{ padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }}
    .alert-ok {{ background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }}
    .alert-w {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
    .alert-d {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
    .dot-g {{ background: {C['ok']}; }} .dot-y {{ background: {C['warn']}; }} .dot-r {{ background: {C['danger']}; }}

    .mstrip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 1px; background: {C['border']}; border: 1px solid {C['border']}; border-radius: 4px; margin-bottom: 1.5rem; overflow: hidden; }}
    .mcell {{ background: {C['surface']}; padding: 0.9rem 1rem; text-align: center; }}
    .mcell .v {{ font-size: 1.4rem; font-weight: 700; color: {C['ink']}; font-family: 'JetBrains Mono', monospace; line-height: 1.2; }}
    .mcell .v.ac {{ color: {C['accent']}; }}
    .mcell .l {{ font-size: 0.68rem; color: {C['muted']}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; font-weight: 500; }}

    .sec {{ margin-top: 1.5rem; margin-bottom: 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid {C['border']}; }}
    .sec h3 {{ font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: {C['muted']}; margin: 0; }}

    [data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {C['border']}; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 0; padding: 0.5rem 1rem; font-size: 0.82rem; font-weight: 500; color: {C['muted']}; border-bottom: 2px solid transparent; }}
    .stTabs [aria-selected="true"] {{ color: {C['ink']}; border-bottom-color: {C['accent']}; }}
    .stPlotlyChart {{ border: 1px solid {C['border']}; border-radius: 4px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
    <div>
        <h1>PRADHAN</h1>
        <div class="sub">Predictive Real-time Analysis of Data from Heliospheric Aditya-Navigation</div>
    </div>
    <div class="pill">XGBoost Nowcasting</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    source = st.selectbox("Data Source", ["GOES XRS-B", "SoLEXS SDD2", "HEL1OS"])
    threshold = st.selectbox("Alert Threshold", ["C-class", "M-class", "X-class"], index=1)
    st.markdown("---")
    st.markdown("### Model")
    if "best_model_results" in results:
        m = results["best_model_results"]["metrics"]
        st.markdown(f"""
        **Algorithm:** XGBoost  \n
        **Features:** 19 proxies  \n
        **Config:** 1h C-class  \n
        **TSS:** {m['tss']:.4f}  \n
        **AUC:** {m['auc']:.4f}
        """)
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/legendaryashwin17-dev/pradhan-solar-flare)")

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Live Monitor", "Light Curves", "Model Performance", "Config Compare"])

# ── Plotly theme ───────────────────────────────────────────────────────
PL = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0)",
          font=dict(family="Inter, sans-serif", color="#1a1a2e", size=12),
          margin=dict(l=50, r=20, t=40, b=50),
          xaxis=dict(gridcolor="#f0f0f0", linecolor="#e5e7eb"),
          yaxis=dict(gridcolor="#f0f0f0", linecolor="#e5e7eb"),
          hoverlabel=dict(bgcolor="#fff", font_size=12, bordercolor="#e5e7eb"),
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    if goes is not None and len(goes) > 0:
        latest = goes.iloc[-1]
        fb = float(latest["xrs_b_flux"])
        fa = float(latest["xrs_a_flux"])
        ts = goes.index[-1]

        if fb >= THRESHOLDS["X"]:
            ac, dc, am, as_ = "X", "dot-r", "EXTREME — X-class flare active", "alert-d"
        elif fb >= THRESHOLDS["M"]:
            ac, dc, am, as_ = "M", "dot-y", "WARNING — M-class flare active", "alert-w"
        elif fb >= THRESHOLDS["C"]:
            ac, dc, am, as_ = "C", "dot-g", "MODERATE — C-class activity", "alert-ok"
        else:
            ac, dc, am, as_ = "B", "dot-g", "QUIET — Low solar activity", "alert-ok"

        st.markdown(f'<div class="alert {as_}"><span class="dot {dc}"></span>{am} — {fb:.2e} W/m&sup2; — {ts.strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)

        ratio = fb / max(fa, 1e-12)
        st.markdown(f"""
        <div class="mstrip">
            <div class="mcell"><div class="v ac">{fb:.1e}</div><div class="l">XRS-B Flux</div></div>
            <div class="mcell"><div class="v">{fa:.1e}</div><div class="l">XRS-A Flux</div></div>
            <div class="mcell"><div class="v">{ratio:.2f}</div><div class="l">Hard/Soft Ratio</div></div>
            <div class="mcell"><div class="v">{ac}</div><div class="l">Current Class</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec"><h3>Last 24 Hours</h3></div>', unsafe_allow_html=True)
        cutoff = goes.index.max() - pd.Timedelta(hours=24)
        last24 = goes[goes.index >= cutoff]
        if len(last24) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last24.index, y=last24["xrs_b_flux"], name="XRS-B",
                                     line=dict(color="#dc2626", width=1.2), fill="tozeroy",
                                     fillcolor="rgba(220,38,38,0.05)"))
            fig.add_trace(go.Scatter(x=last24.index, y=last24["xrs_a_flux"], name="XRS-A",
                                     line=dict(color="#0ea5e9", width=1)))
            for cls, th in [("C", 1e-6), ("M", 1e-5), ("X", 1e-4)]:
                fig.add_hline(y=th, line_dash="dot", line_color="#d1d5db",
                              annotation_text=cls, annotation_position="right",
                              annotation_font_size=9)
            fig.update_layout(**PL, height=300, yaxis_type="log", yaxis_title="Flux (W/m\u00b2)")
            st.plotly_chart(fig, use_container_width=True)

        # Monthly stats from 2024
        st.markdown('<div class="sec"><h3>Monthly Flare Counts (2024)</h3></div>', unsafe_allow_html=True)
        monthly = goes["xrs_b_flux"].resample("ME").agg(
            c_count=lambda x: (x >= THRESHOLDS["C"]).sum(),
            m_count=lambda x: (x >= THRESHOLDS["M"]).sum(),
            x_count=lambda x: (x >= THRESHOLDS["X"]).sum(),
            mean_flux="mean"
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["c_count"], name="C-class",
                             marker_color="#0ea5e9", opacity=0.8))
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["m_count"], name="M-class",
                             marker_color="#d97706", opacity=0.8))
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["x_count"], name="X-class",
                             marker_color="#dc2626", opacity=0.8))
        fig.update_layout(**PL, height=280, barmode="stack", yaxis_title="Flare Count", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No GOES data available.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: LIGHT CURVES
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    if goes is not None and len(goes) > 0:
        col1, col2 = st.columns([4, 1])
        with col2:
            st.markdown("### Filters")
            month = st.selectbox("Month", ["All"] + [f"{m:02d}" for m in range(1, 13)], index=0)
            show_quality = st.checkbox("Quality flags", value=False)
            log_scale = st.checkbox("Log scale", value=True)

        with col1:
            data = goes.copy()
            if month != "All":
                data = data[data.index.month == int(month)]

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.06, row_heights=[0.75, 0.25])
            fig.add_trace(go.Scatter(x=data.index, y=data["xrs_b_flux"].clip(lower=1e-9),
                                     name="XRS-B", line=dict(color="#dc2626", width=0.5)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["xrs_a_flux"].clip(lower=1e-9),
                                     name="XRS-A", line=dict(color="#0ea5e9", width=0.5)),
                          row=1, col=1)
            for cls, th, c in [("C", 1e-6, "#d97706"), ("M", 1e-5, "#dc2626"), ("X", 1e-4, "#7c3aed")]:
                fig.add_hline(y=th, line_dash="dot", line_color=c, annotation_text=cls,
                              row=1, col=1, annotation_font_size=9)
            if show_quality and "xrs_b_quality" in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data["xrs_b_quality"],
                                         name="Quality", line=dict(color="#9ca3af", width=0.5)),
                              row=2, col=1)
            yaxis = dict(type="log", title="Flux (W/m\u00b2)") if log_scale else dict(title="Flux (W/m\u00b2)")
            fig.update_layout(**PL, height=480, yaxis=yaxis, yaxis2_title="Quality", xaxis2_title="Time")
            st.plotly_chart(fig, use_container_width=True)

        # Flux distribution
        st.markdown('<div class="sec"><h3>Flux Distribution</h3></div>', unsafe_allow_html=True)
        flux = goes["xrs_b_flux"].clip(lower=1e-9)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.log10(flux), nbinsx=120, marker_color="#0ea5e9", opacity=0.7))
        for cls, th in [("C", -6), ("M", -5), ("X", -4)]:
            fig.add_vline(x=th, line_dash="dash", line_color="#d1d5db",
                          annotation_text=cls, annotation_position="top")
        fig.update_layout(**PL, height=250, xaxis_title="log10(Flux)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No GOES data available.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    if "best_model_results" in results:
        m = results["best_model_results"]["metrics"]

        st.markdown(f"""
        <div class="mstrip">
            <div class="mcell"><div class="v ac">{m['tss']:.4f}</div><div class="l">TSS (target 0.65)</div></div>
            <div class="mcell"><div class="v">{m['auc']:.4f}</div><div class="l">AUC-ROC (target 0.80)</div></div>
            <div class="mcell"><div class="v">{m['hss']:.4f}</div><div class="l">HSS</div></div>
            <div class="mcell"><div class="v">{m['pod']:.4f}</div><div class="l">POD (target 0.80)</div></div>
            <div class="mcell"><div class="v">{m['pofd']:.4f}</div><div class="l">POFD</div></div>
            <div class="mcell"><div class="v">{m['brier']:.4f}</div><div class="l">Brier Score</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec"><h3>TSS Across Configurations</h3></div>', unsafe_allow_html=True)
            if "multi_config_results" in results:
                cfgs = sorted(results["multi_config_results"], key=lambda x: x["metrics"]["tss"], reverse=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=[c["label"] for c in cfgs], x=[c["metrics"]["tss"] for c in cfgs],
                    orientation="h",
                    marker_color=["#16a34a" if c["metrics"]["tss"] >= 0.65 else "#d97706" for c in cfgs],
                    text=[f'{c["metrics"]["tss"]:.4f}' for c in cfgs], textposition="outside"
                ))
                fig.add_vline(x=0.65, line_dash="dash", line_color="#0ea5e9", annotation_text="ISRO Target")
                fig.update_layout(**PL, height=340, xaxis_title="TSS")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="sec"><h3>Feature Importance (Top 10)</h3></div>', unsafe_allow_html=True)
            imp = results["best_model_results"].get("feature_importance", [])
            if imp:
                if isinstance(imp, list):
                    srt = sorted(imp, key=lambda x: x.get("importance", 0), reverse=True)[:10]
                    names, vals = [x["feature"] for x in srt], [x["importance"] for x in srt]
                else:
                    srt = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                    names, vals = [x[0] for x in srt], [x[1] for x in srt]
                fig = go.Figure()
                fig.add_trace(go.Bar(y=names[::-1], x=vals[::-1], orientation="h",
                                     marker_color="#0ea5e9",
                                     text=[f"{v:.3f}" for v in vals[::-1]], textposition="outside"))
                fig.update_layout(**PL, height=340, xaxis_title="Importance")
                st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        st.markdown('<div class="sec"><h3>Confusion Matrix</h3></div>', unsafe_allow_html=True)
        tp = int(m.get("pod", 0) * 1000)
        fn = 1000 - tp
        fp = int(m.get("pofd", 0) * 1000)
        tn = 1000 - fp
        fig = go.Figure(data=go.Heatmap(
            z=[[tp, fn], [fp, tn]], x=["Pred Flare", "Pred Quiet"], y=["Actual Flare", "Actual Quiet"],
            colorscale=[[0, "#f0f0f0"], [1, "#0ea5e9"]], text=[[str(tp), str(fn)], [str(fp), str(tn)]],
            texttemplate="%{text}", textfont_size=16, showscale=False
        ))
        fig.update_layout(**PL, height=280, margin=dict(l=100, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model results found.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: CONFIG COMPARE
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    if "multi_config_results" in results:
        st.markdown('<div class="sec"><h3>All Configurations</h3></div>', unsafe_allow_html=True)
        cfgs = results["multi_config_results"]
        rows = []
        for r in sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True):
            mm = r["metrics"]
            rows.append({"Config": r["label"], "TSS": f"{mm['tss']:.4f}", "HSS": f"{mm['hss']:.4f}",
                         "AUC": f"{mm['auc']:.4f}", "POD": f"{mm['pod']:.4f}", "POFD": f"{mm['pofd']:.4f}",
                         "CSI": f"{mm['csi']:.4f}", "Brier": f"{mm['brier']:.4f}",
                         "Event Rate": f"{r['event_rate']:.1%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec"><h3>Metrics Radar (Top 4)</h3></div>', unsafe_allow_html=True)
            fig = go.Figure()
            keys = ["tss", "hss", "auc", "pod", "csi"]
            palette = ["#0ea5e9", "#16a34a", "#d97706", "#dc2626"]
            top4 = sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True)[:4]
            for i, r in enumerate(top4):
                vals = [r["metrics"][k] for k in keys] + [r["metrics"]["tss"]]
                fig.add_trace(go.Scatterpolar(r=vals, theta=keys + [keys[0]], name=r["label"],
                                              fill="toself", line=dict(color=palette[i])))
            fig.update_layout(**PL, height=360, polar=dict(bgcolor="rgba(255,255,255,0)",
                                                           radialaxis=dict(gridcolor="#f0f0f0")))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="sec"><h3>ROC Space</h3></div>', unsafe_allow_html=True)
            fig = go.Figure()
            for i, r in enumerate(sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True)):
                fig.add_trace(go.Scatter(x=[r["metrics"]["pofd"]], y=[r["metrics"]["pod"]],
                                         mode="markers+text", text=[r["label"]], textposition="top center",
                                         marker=dict(size=12, color=palette[i % 4])))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="#d1d5db", dash="dash"), showlegend=False))
            fig.update_layout(**PL, height=360, xaxis_title="POFD", yaxis_title="POD")
            st.plotly_chart(fig, use_container_width=True)

        # Event rate vs TSS
        st.markdown('<div class="sec"><h3>Event Rate vs TSS</h3></div>', unsafe_allow_html=True)
        fig = go.Figure()
        for r in sorted(cfgs, key=lambda x: x["metrics"]["tss"], reverse=True):
            fig.add_trace(go.Scatter(
                x=[r["event_rate"]], y=[r["metrics"]["tss"]],
                mode="markers+text", text=[r["label"]], textposition="top center",
                marker=dict(size=14, color="#0ea5e9")))
        fig.add_hline(y=0.65, line_dash="dash", line_color="#d1d5db", annotation_text="ISRO Target")
        fig.update_layout(**PL, height=300, xaxis_title="Event Rate", yaxis_title="TSS",
                          xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No multi-config results found.")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:{C['faint']};font-size:0.72rem;padding:1rem 0;letter-spacing:0.02em;">
    PRADHAN v1.0 &mdash; ISRO Aditya-L1 &mdash; GOES XRS (2024) &mdash; XGBoost 19 features
</div>
""", unsafe_allow_html=True)
