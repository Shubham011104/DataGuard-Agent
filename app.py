# =============================================================================
# app.py — DataGuard Agent: Streamlit Dashboard
# =============================================================================
# Run with:  streamlit run app.py
#
# Tabs:
#   1. 📊 Data Overview & Health Check  — Statistical profiling + anomaly charts
#   2. 🤖 Autonomous Agent Audit        — ReAct agent investigating anomalies
#   3. 💬 NL-to-SQL Copilot             — Chat interface for plain-English queries
# =============================================================================

from __future__ import annotations

import sys
import time
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

import config
import database as db
import profiling as prof
from agent.agent_workflow import (
    NL_TO_SQL_PROMPT,
    SYSTEM_PROMPT,
    build_anomaly_prompt,
    run_agent_with_steps,
)
from agent.tools import set_engine


# =============================================================================
# Page Config & Global CSS
# =============================================================================

st.set_page_config(
    page_title="DataGuard Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS -------------------------------------------------------
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Dark gradient background ---- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #131d3a 50%, #0d1b2a 100%);
    color: #e2e8f0;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f3a 0%, #0f1629 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ---- Metric cards ---- */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 16px;
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-2px); }

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8;
}

/* ---- Section headers ---- */
.section-header {
    background: linear-gradient(90deg, rgba(99,102,241,0.2), transparent);
    border-left: 4px solid #6366f1;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0 12px 0;
}

/* ---- Anomaly severity badges ---- */
.badge-HIGH   { background:#ef4444; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-MEDIUM { background:#f59e0b; color:#000; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-LOW    { background:#3b82f6; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }

/* ---- Agent step cards ---- */
.agent-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
}
.agent-step-final {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}

/* ---- Chat messages ---- */
.chat-user {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 85%;
    margin-left: auto;
}
.chat-agent {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 85%;
}

/* ---- Code blocks ---- */
code, pre { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }

/* ---- Scrollable SQL box ---- */
.sql-box {
    background: #0d1117;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    overflow-x: auto;
    color: #a5f3fc;
    white-space: pre;
}

/* ---- Status pill ---- */
.status-ok   { color: #4ade80; font-weight: 600; }
.status-warn { color: #fbbf24; font-weight: 600; }
.status-err  { color: #f87171; font-weight: 600; }

/* ---- Hero banner ---- */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.25) 0%, rgba(139,92,246,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-title  { font-size: 2.4rem; font-weight: 700; color: #e0e7ff; margin: 0; }
.hero-sub    { font-size: 1.0rem; color: #94a3b8; margin-top: 6px; }

/* ---- Expander ---- */
details summary { color: #818cf8 !important; font-weight: 500; }

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.5) !important;
    border-radius: 10px !important;
    padding: 8px;
}

/* ---- Progress text ---- */
.progress-label { font-size: 0.82rem; color: #64748b; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialisation
# =============================================================================

def _init_state():
    defaults = {
        "df":              None,      # Active DataFrame
        "table_name":      None,      # Active table name
        "report":          None,      # ProfilingReport
        "engine":          None,      # SQLAlchemy engine
        "agent_steps":     [],        # List[AgentStep]
        "agent_answer":    "",        # Final agent answer
        "chat_history":    [],        # List[{"role": "user"|"agent", "content": str}]
        "api_key_ok":      False,     # API key validated flag
        "active_anomaly":  None,      # Currently selected anomaly index
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 🛡️ DataGuard Agent")
    st.markdown("---")

    # --- Provider Selection ---
    st.markdown("### 🤖 LLM Provider")
    provider_choice = st.selectbox(
        "Select Provider",
        ["Groq (100% FREE Cloud)", "OpenRouter (FREE Models)", "Ollama (100% FREE Local)", "OpenAI (Paid)"],
        index=0,
        help="Groq and OpenRouter provide 100% FREE cloud API keys with zero credit card required!"
    )
    config.LLM_PROVIDER = provider_choice

    # --- Key & Model configuration based on provider ---
    if "Groq" in provider_choice:
        st.markdown("💡 **Get a 100% FREE key:** [console.groq.com/keys](https://console.groq.com/keys)")
        has_sys_key = bool(config.GROQ_API_KEY)
        if has_sys_key:
            st.info("🔒 Key configured via Streamlit Secrets")
        groq_key_input = st.text_input(
            "Groq API Key (gsk_...)",
            type="password",
            placeholder="[Secrets Key Loaded]" if has_sys_key else "Paste Groq API Key...",
            key="groq_key_field",
        )
        if groq_key_input:
            config.GROQ_API_KEY = groq_key_input
            config.OPENAI_API_KEY = groq_key_input
            st.session_state.api_key_ok = True
            st.success("✅ Overrode with new key", icon="🔐")
        elif has_sys_key:
            st.session_state.api_key_ok = True

        config.LLM_MODEL = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )

    elif "OpenRouter" in provider_choice:
        st.markdown("💡 **Get a FREE key:** [openrouter.ai/keys](https://openrouter.ai/keys)")
        has_sys_key = bool(config.OPENROUTER_API_KEY)
        if has_sys_key:
            st.info("🔒 Key configured via Streamlit Secrets")
        openrouter_key_input = st.text_input(
            "OpenRouter API Key (sk-or-v1-...)",
            type="password",
            placeholder="[Secrets Key Loaded]" if has_sys_key else "Paste OpenRouter API Key...",
            key="openrouter_key_field",
        )
        if openrouter_key_input:
            config.OPENROUTER_API_KEY = openrouter_key_input
            config.OPENAI_API_KEY = openrouter_key_input
            st.session_state.api_key_ok = True
            st.success("✅ Overrode with new key", icon="🔐")
        elif has_sys_key:
            st.session_state.api_key_ok = True

        config.LLM_MODEL = st.selectbox(
            "Model (Free)",
            ["meta-llama/llama-3.1-8b-instruct:free", "google/gemini-2.0-flash-lite-preview:free"],
            index=0,
        )

    elif "Ollama" in provider_choice:
        st.info("ℹ️ Requires Ollama installed & running on localhost:11434")
        st.session_state.api_key_ok = True  # No key needed
        config.LLM_MODEL = st.selectbox(
            "Local Model",
            ["llama3.1", "mistral", "qwen2.5", "llama3"],
            index=0,
        )

    else: # OpenAI
        st.markdown("### 🔑 OpenAI API Key")
        has_sys_key = bool(config.OPENAI_API_KEY)
        if has_sys_key:
            st.info("🔒 Key configured via Streamlit Secrets")
        api_key_input = st.text_input(
            "Paste key (sk-…)",
            type="password",
            placeholder="[Secrets Key Loaded]" if has_sys_key else "Paste OpenAI Key...",
            key="api_key_field",
        )
        if api_key_input:
            config.OPENAI_API_KEY = api_key_input
            st.session_state.api_key_ok = True
            st.success("✅ Overrode with new key", icon="🔐")
        elif has_sys_key:
            st.session_state.api_key_ok = True

        config.LLM_MODEL = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )

    st.markdown("---")
    st.markdown("### ⚙️ Agent Tuning")
    config.LLM_TEMPERATURE = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    config.MAX_CORRECTION_ATTEMPTS = st.slider("Max Self-Correction Attempts", 1, 5, 3)

    st.markdown("---")

    # --- Threshold Configuration ---
    st.markdown("### 📐 Quality Thresholds")
    config.NULL_RATE_THRESHOLD = st.slider(
        "Null Rate Threshold", 0.01, 0.50, config.NULL_RATE_THRESHOLD, 0.01,
        format="%.0f%%", help="Flag columns with null % above this."
    )
    config.ZSCORE_THRESHOLD = st.slider("Z-Score Threshold", 1.5, 5.0, config.ZSCORE_THRESHOLD, 0.1)
    config.IQR_MULTIPLIER   = st.slider("IQR Multiplier", 1.0, 3.0, config.IQR_MULTIPLIER, 0.1)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#475569;text-align:center'>"
        "DataGuard Agent v1.0 · Final Year Project<br>"
        "Built with LangChain + Streamlit</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Hero Banner
# =============================================================================

st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🛡️ DataGuard Agent</p>
    <p class="hero-sub">Autonomous Data Quality &amp; Anomaly Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Data Source Panel (shared across all tabs)
# =============================================================================

with st.expander("📂 Data Source — Upload CSV or Load Sample Dataset", expanded=True):
    col_a, col_b = st.columns([2, 1])

    with col_a:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Your CSV will be loaded into an in-memory SQLite database.",
            key="file_uploader",
        )

    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample = st.button(
            "🧪 Load Sample E-Commerce Dataset",
            width="stretch",
            help="Generates 500 rows with intentional quality issues for demo.",
        )
        pk_col = st.text_input(
            "Primary Key Column (optional)",
            value="transaction_id",
            help="Used for duplicate detection on a specific key column.",
        )

    # Handle sample dataset
    if load_sample:
        with st.spinner("Generating sample dataset…"):
            engine = db.get_engine()
            sample_df = db.seed_sample_database(engine=engine)
            st.session_state.df         = sample_df
            st.session_state.table_name = config.SAMPLE_TABLE_NAME
            st.session_state.engine     = engine
            set_engine(engine)
            st.session_state.report     = None  # Reset old report
        st.success(f"✅ Sample dataset loaded: {len(sample_df):,} rows × {len(sample_df.columns)} columns")

    # Handle file upload
    if uploaded_file is not None:
        with st.spinner("Reading CSV…"):
            try:
                df = pd.read_csv(uploaded_file)
                table_name = Path(uploaded_file.name).stem.lower().replace(" ", "_")
                engine = db.get_engine()
                db.load_dataframe_to_db(df, table_name, engine=engine)
                st.session_state.df         = df
                st.session_state.table_name = table_name
                st.session_state.engine     = engine
                set_engine(engine)
                st.session_state.report     = None
                st.success(f"✅ '{uploaded_file.name}' loaded: {len(df):,} rows × {len(df.columns)} columns")
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")

    # Quick preview
    if st.session_state.df is not None:
        with st.expander("👁️ Data Preview (first 10 rows)", expanded=False):
            st.dataframe(st.session_state.df.head(10), width="stretch")


# =============================================================================
# Main Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Data Overview & Health Check",
    "🤖 Autonomous Agent Audit",
    "💬 NL-to-SQL Copilot",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Data Overview & Health Check
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if st.session_state.df is None:
        st.info("ℹ️ Load a dataset from the panel above to begin profiling.", icon="📂")
        st.stop()

    df: pd.DataFrame = st.session_state.df
    table_name: str  = st.session_state.table_name

    # --- Run Profiling ---
    col_run, _ = st.columns([1, 3])
    with col_run:
        run_profile = st.button(
            "🔍 Run Full Data Profile",
            width="stretch",
            type="primary",
            key="run_profile_btn",
        )

    if run_profile or st.session_state.report is not None:
        if run_profile or st.session_state.report is None:
            with st.spinner("Running statistical profiling and anomaly detection…"):
                pk = pk_col.strip() if pk_col.strip() in df.columns else None
                report = prof.run_full_profiling(df, table_name=table_name, pk_column=pk)
                st.session_state.report = report
        else:
            report = st.session_state.report

        # ---- KPI Cards -------------------------------------------------------
        st.markdown('<div class="section-header"><b>📈 Dataset Overview</b></div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Rows",     f"{report.total_rows:,}")
        k2.metric("Total Columns",  f"{report.total_columns}")
        k3.metric("Anomalies Found",f"{report.anomaly_count}", delta=f"{report.high_severity_count} HIGH", delta_color="inverse")
        k4.metric("Duplicates",     f"{report.duplicate_count:,}")
        k5.metric("HIGH Severity",  f"{report.high_severity_count}", delta_color="inverse")

        # ---- Column Profiles -------------------------------------------------
        st.markdown('<div class="section-header"><b>🗂️ Column-Level Statistics</b></div>', unsafe_allow_html=True)
        profile_rows = []
        for cp in report.column_profiles:
            row = {
                "Column":       cp.column_name,
                "Type":         cp.dtype,
                "Null %":       f"{cp.null_rate*100:.1f}%",
                "Unique":       cp.unique_count,
                "Mean":         f"{cp.mean:.2f}" if cp.mean is not None else "—",
                "Std Dev":      f"{cp.std:.2f}"  if cp.std  is not None else "—",
                "Min":          f"{cp.min_val:.2f}" if cp.min_val is not None else "—",
                "Max":          f"{cp.max_val:.2f}" if cp.max_val is not None else "—",
                "Skewness":     f"{cp.skewness:.2f}" if cp.skewness is not None else "—",
            }
            profile_rows.append(row)
        st.dataframe(pd.DataFrame(profile_rows), width="stretch", height=280)

        # ---- Null Rate Heatmap -----------------------------------------------
        st.markdown('<div class="section-header"><b>🕳️ Null Rate by Column</b></div>', unsafe_allow_html=True)
        null_data = {
            "Column":    [cp.column_name for cp in report.column_profiles],
            "Null Rate": [cp.null_rate * 100 for cp in report.column_profiles],
        }
        null_df = pd.DataFrame(null_data).sort_values("Null Rate", ascending=False)
        fig_null = px.bar(
            null_df, x="Column", y="Null Rate",
            color="Null Rate",
            color_continuous_scale=["#1e3a5f", "#f59e0b", "#ef4444"],
            labels={"Null Rate": "Null %"},
            title="Null Rate per Column",
            template="plotly_dark",
        )
        fig_null.add_hline(
            y=config.NULL_RATE_THRESHOLD * 100,
            line_dash="dash", line_color="#f87171",
            annotation_text=f"Threshold ({config.NULL_RATE_THRESHOLD*100:.0f}%)",
            annotation_position="top right",
        )
        fig_null.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            coloraxis_showscale=False,
            height=340,
        )
        st.plotly_chart(fig_null, width="stretch")

        # ---- Numeric Distributions + Outlier Scatter -------------------------
        numeric_cols = [
            cp.column_name for cp in report.column_profiles
            if cp.mean is not None
        ]
        if numeric_cols:
            st.markdown('<div class="section-header"><b>📉 Numeric Distributions & Outliers</b></div>', unsafe_allow_html=True)
            sel_col = st.selectbox("Select numeric column to visualise", numeric_cols, key="dist_col")

            col_left, col_right = st.columns(2)

            with col_left:
                fig_hist = px.histogram(
                    df, x=sel_col, nbins=40,
                    title=f"Distribution — {sel_col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#6366f1"],
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8"), height=320,
                )
                st.plotly_chart(fig_hist, width="stretch")

            with col_right:
                fig_box = px.box(
                    df, y=sel_col,
                    title=f"Box Plot — {sel_col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#818cf8"],
                )
                fig_box.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8"), height=320,
                )
                st.plotly_chart(fig_box, width="stretch")

        # ---- Anomaly Table ---------------------------------------------------
        st.markdown('<div class="section-header"><b>🚨 Detected Anomalies</b></div>', unsafe_allow_html=True)

        if not report.anomalies:
            st.success("✅ No anomalies detected — your dataset is clean!")
        else:
            # Summary donuts
            severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            type_counts: dict[str, int] = {}
            for a in report.anomalies:
                severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
                type_counts[a.anomaly_type]  = type_counts.get(a.anomaly_type, 0) + 1

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                fig_sev = px.pie(
                    names=list(severity_counts.keys()),
                    values=list(severity_counts.values()),
                    title="Anomalies by Severity",
                    color=list(severity_counts.keys()),
                    color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#3b82f6"},
                    hole=0.55, template="plotly_dark",
                )
                fig_sev.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8"), height=300,
                )
                st.plotly_chart(fig_sev, width="stretch")

            with dcol2:
                fig_type = px.pie(
                    names=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    title="Anomalies by Type",
                    hole=0.55, template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_type.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8"), height=300,
                )
                st.plotly_chart(fig_type, width="stretch")

            # Detailed anomaly cards
            for i, anomaly in enumerate(report.anomalies):
                badge_html = f'<span class="badge-{anomaly.severity}">{anomaly.severity}</span>'
                with st.expander(
                    f"[{i+1}] {anomaly.anomaly_type} · {anomaly.column} {badge_html} "
                    f"— {anomaly.affected_rows:,} rows",
                    expanded=(i < 2),
                ):
                    st.markdown(f"**Description:** {anomaly.description}")
                    if anomaly.metric_value is not None:
                        st.markdown(f"**Metric Value:** `{anomaly.metric_value}`")
                    if anomaly.threshold_used is not None:
                        st.markdown(f"**Threshold:** `{anomaly.threshold_used}`")
                    if anomaly.suggested_sql:
                        st.markdown("**Suggested Diagnostic SQL:**")
                        st.code(anomaly.suggested_sql, language="sql")

    else:
        st.info("Click **Run Full Data Profile** to begin analysis.", icon="🔍")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Autonomous Agent Audit
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    if st.session_state.df is None:
        st.info("ℹ️ Load a dataset first (panel above).", icon="📂")
        st.stop()

    has_key = st.session_state.api_key_ok or config.OPENAI_API_KEY or config.GROQ_API_KEY or config.OPENROUTER_API_KEY or "Ollama" in config.LLM_PROVIDER
    if not has_key:
        st.warning("⚠️ Enter an API key in the sidebar. Select **Groq (100% FREE Cloud)** for a completely free key!", icon="🔑")
        st.stop()

    st.markdown('<div class="section-header"><b>🤖 Autonomous Agent Audit Mode</b></div>', unsafe_allow_html=True)
    st.markdown(
        "The ReAct agent will investigate a selected anomaly: it will "
        "**think → generate SQL → execute → self-correct → summarise**."
    )

    # --- Anomaly selector ---
    report = st.session_state.report
    if report is None or not report.anomalies:
        st.info("Run the **Data Profile** in Tab 1 first so anomalies are available.", icon="📊")
        st.stop()

    anomaly_labels = [
        f"[{i+1}] {a.severity} | {a.anomaly_type} | {a.column} ({a.affected_rows} rows)"
        for i, a in enumerate(report.anomalies)
    ]
    selected_idx = st.selectbox("Select anomaly to investigate:", range(len(anomaly_labels)), format_func=lambda i: anomaly_labels[i])
    selected_anomaly = report.anomalies[selected_idx]

    st.markdown(f"**Selected:** {selected_anomaly.description}")

    col_inv, col_cust = st.columns([1, 1])
    with col_inv:
        investigate_btn = st.button(
            "🚀 Launch Agent Investigation",
            type="primary",
            width="stretch",
            key="launch_agent_btn",
        )
    with col_cust:
        custom_query = st.text_input(
            "Or enter a custom investigation query",
            placeholder="e.g., Why do revenue values go negative for Electronics?",
            key="custom_query",
        )
        custom_btn = st.button("▶️ Run Custom Query", width="stretch", key="custom_btn")

    # --- Build prompt ---
    def _launch_investigation(prompt_text: str):
        st.session_state.agent_steps  = []
        st.session_state.agent_answer = ""
        status_box = st.empty()
        steps_container = st.container()
        answer_container = st.empty()

        with st.spinner("🤖 Agent is reasoning… (this may take 30–90 seconds)"):
            status_box.markdown('<p class="progress-label">Agent initialising tools…</p>', unsafe_allow_html=True)
            try:
                final_answer, steps = run_agent_with_steps(
                    query=prompt_text,
                    engine=st.session_state.engine,
                )
                st.session_state.agent_steps  = steps
                st.session_state.agent_answer = final_answer
            except Exception as exc:
                st.error(f"Agent Error: {exc}")
                st.code(traceback.format_exc())
                return

        status_box.empty()

        # Display steps
        with steps_container:
            st.markdown("#### 🧠 Agent Reasoning Steps")
            for i, step in enumerate(steps):
                if step.is_final:
                    continue
                with st.expander(
                    f"Step {i+1}: **{step.action or 'Thinking'}**",
                    expanded=(i == 0),
                ):
                    if step.thought:
                        st.markdown(f"**Thought:**\n```\n{step.thought.strip()}\n```")
                    if step.action:
                        st.markdown(f"**Action:** `{step.action}`")
                    if step.action_input:
                        st.code(step.action_input, language="sql")
                    if step.observation:
                        st.markdown("**Observation:**")
                        st.text(step.observation[:2000])

        # Display final answer
        st.markdown("---")
        st.markdown("#### ✅ Agent Root-Cause Analysis & SQL Fix")
        st.markdown(
            f'<div class="agent-step-final">{final_answer}</div>',
            unsafe_allow_html=True,
        )

    if investigate_btn:
        schema_hint = db.get_schema_as_text(engine=st.session_state.engine)
        prompt = build_anomaly_prompt(selected_anomaly, table_name, schema_hint)
        _launch_investigation(prompt)

    elif custom_btn and custom_query.strip():
        _launch_investigation(custom_query.strip())

    # Show cached results
    elif st.session_state.agent_steps:
        st.markdown("#### 🧠 Last Investigation Steps (cached)")
        for i, step in enumerate(st.session_state.agent_steps):
            if step.is_final:
                continue
            with st.expander(f"Step {i+1}: **{step.action or 'Thinking'}**", expanded=False):
                if step.thought:
                    st.markdown(f"**Thought:**\n```\n{step.thought.strip()}\n```")
                if step.action:
                    st.markdown(f"**Action:** `{step.action}`")
                if step.action_input:
                    st.code(step.action_input, language="sql")
                if step.observation:
                    st.text(step.observation[:2000])

        if st.session_state.agent_answer:
            st.markdown("---")
            st.markdown("#### ✅ Last Answer")
            st.markdown(st.session_state.agent_answer)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: NL-to-SQL Copilot
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    if st.session_state.df is None:
        st.info("ℹ️ Load a dataset first (panel above).", icon="📂")
        st.stop()

    has_key = st.session_state.api_key_ok or config.OPENAI_API_KEY or config.GROQ_API_KEY or config.OPENROUTER_API_KEY or "Ollama" in config.LLM_PROVIDER
    if not has_key:
        st.warning("⚠️ Enter an API key in the sidebar. Select **Groq (100% FREE Cloud)** for a completely free key!", icon="🔑")
        st.stop()

    st.markdown('<div class="section-header"><b>💬 Natural Language → SQL Copilot</b></div>', unsafe_allow_html=True)
    st.markdown(
        "Ask plain-English questions about your data. The agent will inspect the schema, "
        "write SQL, run it, and return results — all automatically."
    )

    # --- Example chips ---
    st.markdown("**Quick example questions:**")
    ex_cols = st.columns(3)
    examples = [
        "What are the top 5 categories by total revenue?",
        "Show me all transactions where revenue is negative.",
        "How many orders were placed per region?",
        "Which payment methods have the highest average order value?",
        "Find all records where customer_email is null.",
        "What is the monthly revenue trend?",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i % 3].button(ex, key=f"ex_{i}", width="stretch"):
            st.session_state["nl_input_prefill"] = ex

    st.markdown("---")

    # --- Render chat history ---
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-agent">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # --- Chat input ---
    prefill = st.session_state.pop("nl_input_prefill", "")
    user_question = st.chat_input(
        "Ask a question about your data…",
        key="nl_chat_input",
    )

    # Handle prefill from example buttons
    if prefill and not user_question:
        user_question = prefill

    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.markdown(
            f'<div class="chat-user">🧑 {user_question}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("🤖 Generating SQL and querying database…"):
            try:
                final_answer, steps = run_agent_with_steps(
                    query=user_question,
                    system_prompt=NL_TO_SQL_PROMPT,
                    engine=st.session_state.engine,
                )
            except Exception as exc:
                final_answer = f"❌ Agent error: {exc}"
                steps = []

        # Show reasoning summary (collapsed)
        if steps:
            tool_calls = [s for s in steps if not s.is_final and s.action]
            if tool_calls:
                with st.expander(f"🔍 Agent used {len(tool_calls)} tool call(s) — click to inspect", expanded=False):
                    for s in tool_calls:
                        st.markdown(f"**Tool:** `{s.action}`")
                        if s.action_input:
                            st.code(s.action_input, language="sql")
                        if s.observation:
                            st.text(s.observation[:1500])
                        st.divider()

        # Display agent response
        st.markdown(
            f'<div class="chat-agent">🤖 {final_answer}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.chat_history.append({"role": "agent", "content": final_answer})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
