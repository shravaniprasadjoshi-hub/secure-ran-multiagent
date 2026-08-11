"""
Secure Multi-Agent AI Framework for RAN Control Loops — Interactive Dashboard
Launch: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chatbot.rag_chatbot import RAGChatbot
from src.config import load_config

st.set_page_config(
    page_title="Secure Multi-Agent RAN Control",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NOKIA_BLUE = "#124191"
NOKIA_TEAL = "#00C9FF"
ACCENT = "#FF6B35"

st.markdown(
    f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {NOKIA_BLUE} 0%, #1a5cb8 50%, {NOKIA_TEAL} 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }}
    .main-header h1 {{ margin: 0; font-size: 1.8rem; }}
    .main-header p {{ margin: 0.3rem 0 0; opacity: 0.9; }}
    .metric-card {{
        background: #f8fafc; border-left: 4px solid {NOKIA_BLUE};
        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #eef2f7; border-radius: 8px; padding: 8px 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["telemetry_csv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, cfg


@st.cache_data
def load_metrics():
    cfg = load_config()
    p = Path(cfg["paths"]["metrics_dir"]) / "all_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


@st.cache_resource
def get_chatbot():
    cfg = load_config()
    return RAGChatbot(cfg["paths"]["rag_corpus"])


def plot_image_if_exists(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Run `python run_pipeline.py` to generate: {path.name}")


def page_overview(df, metrics, cfg):
    st.markdown(
        '<div class="main-header"><h1>📡 Secure Multi-Agent AI Framework for RAN Control Loops</h1>'
        "<p>AI-Native 6G Open RAN | Multi-Agent MARL | Consensus-Secured RRC | Digital Twin</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Telemetry Records", f"{len(df):,}")
    c2.metric("AI Agents", "8")
    c3.metric("Cells / gNBs", "48 / 12")
    consensus = metrics.get("coordinator", {}).get("consensus_accept_rate", 0)
    c4.metric("Consensus Accept Rate", f"{consensus:.1%}" if consensus else "—")

    st.markdown("### Architecture Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **Layers:**
        1. **Multi-Agent Layer** — Mobility, Security, Resource, Energy, Trust, Beamforming, QoS, Policy
        2. **RAN Control Layer** — RRC decisions, scheduling, mobility (TS 38.331)
        3. **Security Layer** — Trust engine, anomaly detection, consensus, PQC
        4. **O-RAN Integration** — Near-RT RIC, E2 interface, xApps/rApps
        5. **Digital Twin** — Network simulation, attack injection, KPI monitoring
        """)
    with col2:
        plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "architecture" / "01_overall_evaluation.png",
                             "Overall Architecture Evaluation")

    st.markdown("### Scenario Distribution")
    sc = df["scenario_type"].value_counts().reset_index()
    sc.columns = ["Scenario", "Count"]
    fig = px.pie(sc, values="Count", names="Scenario", color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def page_data_exploration(df, cfg):
    st.header("📊 Data Exploration & CDF Analysis")
    tabs = st.tabs(["Distributions", "Correlation Heatmap", "CDFs", "Scenario Analysis"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        for col, feat, title in [(col1, "rsrp_dbm", "RSRP (dBm)"), (col2, "sinr_db", "SINR (dB)")]:
            fig = px.histogram(df.sample(10000), x=feat, nbins=50, color_discrete_sequence=[NOKIA_BLUE])
            fig.update_layout(title=title, height=350)
            col.plotly_chart(fig, use_container_width=True)
        plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "data" / "01_dataset_overview.png")

    with tabs[1]:
        num_cols = ["rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "prb_utilization_pct", "trust_score", "attack_probability"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r", aspect="auto")
        fig.update_layout(title="Feature Correlation Heatmap", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("**Cumulative Distribution Functions**")
        cdf_feat = st.selectbox("Select KPI", ["latency_ms", "dl_throughput_mbps", "trust_score", "sinr_db", "packet_loss_pct"])
        sample = df[cdf_feat].dropna().sort_values()
        cdf_y = (sample.rank(method="first") / len(sample)).values
        fig = go.Figure(go.Scatter(x=sample, y=cdf_y, mode="lines", line=dict(color=NOKIA_TEAL, width=2)))
        fig.update_layout(title=f"CDF — {cdf_feat}", xaxis_title=cdf_feat, yaxis_title="CDF", height=400)
        st.plotly_chart(fig, use_container_width=True)
        plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "data" / "03_cdf_plots.png")

    with tabs[3]:
        fig = px.box(df, x="scenario_type", y="sinr_db", color="scenario_type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="SINR by Scenario", height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_agents(metrics, cfg):
    st.header("🤖 Individual Agent Outcomes & Evaluation")
    if not metrics.get("agents"):
        st.warning("No trained models found. Run `python run_pipeline.py` first.")
        return

    agent_names = list(metrics["agents"].keys())
    selected = st.selectbox("Select Agent", agent_names)

    ag = metrics["agents"][selected]
    c1, c2, c3 = st.columns(3)
    task = ag["task"]
    key = "f1" if task == "classification" else "r2"
    c1.metric("Train", f"{ag['train'].get(key, 0):.3f}")
    c2.metric("Validation", f"{ag['val'].get(key, 0):.3f}")
    c3.metric("Test", f"{ag['test'].get(key, 0):.3f}")

    st.markdown(f"**Model:** {ag['model_type']} | **Task:** {task}")

    # Train/Val/Test bar chart
    splits_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        key.upper(): [ag["train"].get(key, 0), ag["val"].get(key, 0), ag["test"].get(key, 0)],
    })
    fig = px.bar(splits_df, x="Split", y=key.upper(), color="Split",
                 color_discrete_sequence=[NOKIA_BLUE, NOKIA_TEAL, ACCENT], text_auto=".3f")
    fig.update_layout(title=f"{selected} — {key.upper()} across Train/Val/Test", height=400)
    st.plotly_chart(fig, use_container_width=True)

    plots_dir = Path(cfg["paths"]["plots_dir"]) / "agents"
    col1, col2 = st.columns(2)
    with col1:
        plot_image_if_exists(plots_dir / f"{selected}_confusion_matrix.png", "Confusion Matrix (Test)")
    with col2:
        plot_image_if_exists(plots_dir / f"{selected}_roc_curve.png", "ROC Curve (Test)")
    plot_image_if_exists(plots_dir / f"training/{selected}_learning_curve.png", "Learning Curve")
    plot_image_if_exists(plots_dir / "03_agent_performance_heatmap.png", "All Agents Performance Heatmap")


def page_training(metrics, cfg):
    st.header("📈 Training, Validation & Testing")
    st.markdown("""
    All 8 agents follow **70% train / 15% validation / 15% test** stratified split by scenario type.
    Models are selected per agent function per project architecture.
    """)

    if not metrics.get("agents"):
        st.warning("Run pipeline first.")
        return

    rows = []
    for name, ag in metrics["agents"].items():
        task = ag["task"]
        key = "f1" if task == "classification" else "r2"
        for split in ["train", "val", "test"]:
            rows.append({"Agent": name.replace("_agent", ""), "Split": split.capitalize(),
                         "Metric": key.upper(), "Score": ag[split].get(key, 0)})
    mdf = pd.DataFrame(rows)
    fig = px.bar(mdf, x="Agent", y="Score", color="Split", barmode="group",
                 color_discrete_sequence=[NOKIA_BLUE, NOKIA_TEAL, ACCENT], text_auto=".2f")
    fig.update_layout(title="All Agents — Train / Val / Test Performance", height=500)
    st.plotly_chart(fig, use_container_width=True)

    plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "agents" / "01_classification_f1_comparison.png")
    plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "agents" / "02_regression_r2_comparison.png")

    st.markdown("### Model Details")
    model_table = pd.DataFrame([
        {"Agent": k, "Model": v["model_type"], "Task": v["task"],
         "Test F1/R²": v["test"].get("f1", v["test"].get("r2", 0))}
        for k, v in metrics["agents"].items()
    ])
    st.dataframe(model_table, use_container_width=True, hide_index=True)


def page_digital_twin(cfg):
    st.header("🌐 Digital Twin Visualization")
    twin_dir = Path(cfg["paths"]["twin_dir"])
    hist_path = twin_dir / "twin_simulation_history.csv"
    cell_path = twin_dir / "twin_cell_state.csv"

    if not hist_path.exists():
        st.warning("Run `python run_pipeline.py` to generate digital twin data.")
        return

    history = pd.read_csv(hist_path)
    cells = pd.read_csv(cell_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulation Steps", len(history))
    c2.metric("Final Mean SINR", f"{history['mean_sinr'].iloc[-1]:.1f} dB")
    c3.metric("Total Handovers", int(history["total_ho"].iloc[-1]))
    c4.metric("Active Attacks", int(history["attacks_active"].iloc[-1]))

    tabs = st.tabs(["Time Series", "Cell Map", "gNB Heatmap"])
    with tabs[0]:
        fig = go.Figure()
        for col, name, color in [("mean_sinr", "SINR", NOKIA_TEAL), ("mean_throughput", "Throughput", NOKIA_BLUE),
                                  ("mean_trust", "Trust", ACCENT)]:
            fig.add_trace(go.Scatter(x=history.index, y=history[col], name=name, line=dict(color=color)))
        fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="Attack")
        fig.update_layout(title="Digital Twin KPIs over Time", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = px.scatter(cells, x="rsrp_dbm", y="sinr_db", size="active_ues", color="prb_util_pct",
                         hover_data=["cell_id", "gnb_id", "attack_flag"], color_continuous_scale="Viridis",
                         title="Cell State Map — RSRP vs SINR")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        pivot = cells.groupby("gnb_id")[["prb_util_pct", "throughput_mbps", "trust_score"]].mean().reset_index()
        fig = px.imshow(pivot.set_index("gnb_id"), text_auto=".1f", color_continuous_scale="RdYlBu_r", aspect="auto")
        fig.update_layout(title="gNB-Level KPI Heatmap", height=400)
        st.plotly_chart(fig, use_container_width=True)

    plot_image_if_exists(Path(cfg["paths"]["plots_dir"]) / "digital_twin" / "01_twin_time_series.png")


def page_chatbot():
    st.header("💬 RAG Knowledge Chatbot")
    st.markdown("Ask about 3GPP RRC, O-RAN RIC, multi-agent consensus, security, digital twin, and project architecture.")

    bot = get_chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['title']}** ({s['source']}) — score: {s['score']:.3f}")
                        st.caption(s["content"][:300])

    prompt = st.chat_input("Ask about RAN control, agents, security, 3GPP...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        result = bot.query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        st.rerun()

    st.markdown("**Suggested questions:**")
    for q in ["How does consensus prevent rogue agents?", "What are RRC measurement events A2 and A3?",
              "Explain O-RAN Near-RT RIC control loops", "What attacks does the security agent detect?",
              "How does the digital twin work?"]:
        if st.button(q, key=q):
            result = bot.query(q)
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            st.rerun()


def main():
    df, cfg = load_data()
    metrics = load_metrics()

    st.sidebar.markdown(f"## 📡 RAN Multi-Agent AI")
    st.sidebar.markdown("**NBUC Project 03**")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📊 Data Exploration", "🤖 Agent Evaluation",
         "📈 Train/Val/Test", "🌐 Digital Twin", "💬 Chatbot"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**References:** TS 38.331, TS 38.215, O-RAN E2GAP, NBUC Problem Statement")

    pages = {
        "🏠 Overview": lambda: page_overview(df, metrics, cfg),
        "📊 Data Exploration": lambda: page_data_exploration(df, cfg),
        "🤖 Agent Evaluation": lambda: page_agents(metrics, cfg),
        "📈 Train/Val/Test": lambda: page_training(metrics, cfg),
        "🌐 Digital Twin": lambda: page_digital_twin(cfg),
        "💬 Chatbot": page_chatbot,
    }
    pages[page]()


if __name__ == "__main__":
    main()
