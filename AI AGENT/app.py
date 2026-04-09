import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Department",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .kpi-label { color: #8b949e; font-size: 12px; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 4px; }
    .kpi-value { color: #f0f6fc; font-size: 22px; font-weight: 700; }
    .kpi-sub   { color: #3fb950; font-size: 12px; margin-top: 4px; }

    /* Chat bubbles */
    .chat-user {
        background: #1f6feb;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 80px;
        font-size: 14px;
        line-height: 1.5;
    }
    .chat-ai {
        background: #21262d;
        border: 1px solid #30363d;
        color: #e6edf3;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 80px 8px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .agent-badge {
        font-size: 11px;
        font-weight: 600;
        color: #3fb950;
        margin-bottom: 6px;
        letter-spacing: 0.3px;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    /* Input box tweak */
    [data-testid="stChatInput"] textarea {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 10px !important;
    }

    /* Scrollable chat area */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 6px;
    }

    /* Divider */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── Load agents (cached so they only init once) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agents():
    from Data_Agent import DataAgent
    from Manager import ManagerAgent
    from Sales_Analyst import SalesAnalyst

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail_small.csv")
    d_agent = DataAgent(file_name)
    df = d_agent.get_data()

    if df is None:
        return None, None, None

    manager = ManagerAgent(df)
    sales = SalesAnalyst(df)
    return df, manager, sales


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 AI Data Department")
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df, manager, sales = load_agents()

    if df is None:
        st.error("Could not load data. Check that `online_retail_small.csv` is in the project folder.")
        st.stop()

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown("#### Key Metrics")

    total_rev = sales.get_total_revenue()
    total_orders = sales.get_total_orders()
    total_items = sales.get_total_items_sold()
    aov = sales.get_average_order_value()
    refund = sales.get_refund_rate()
    trend = sales.get_sales_trend()

    trend_color = "#3fb950" if "Up" in trend else ("#f85149" if "Down" in trend else "#d29922")

    kpis = [
        ("Total Revenue", f"£{total_rev:,.0f}", "All-time sales"),
        ("Total Orders", f"{total_orders:,}", "Unique invoices"),
        ("Items Sold", f"{total_items:,}", "Units dispatched"),
        ("Avg Order Value", f"£{aov:,.2f}", "Revenue per order"),
        ("Refund Rate", f"{refund:.1f}%", "Of all transactions"),
    ]

    for label, value, sub in kpis:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Sales Trend</div>
        <div class="kpi-value" style="color:{trend_color}; font-size:18px;">{trend}</div>
        <div class="kpi-sub">Month-over-month</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Agent selector ───────────────────────────────────────────────────────
    st.markdown("#### Active Agents")
    agents_info = {
        "Alex — Sales Analyst": "💼",
        "Dana — Product Analyst": "📦",
        "Maya — Customer Analyst": "👤",
    }
    for name, icon in agents_info.items():
        st.markdown(f"{icon} &nbsp; {name}", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"Dataset: {len(df):,} records loaded")


# ── Main Area ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <span style="font-size:32px;">📊</span>
    <div>
        <div style="color:#f0f6fc; font-size:22px; font-weight:700;">AI Data Department</div>
        <div style="color:#8b949e; font-size:13px;">Ask anything about your sales, products, or customers</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Charts row ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    monthly_data = sales.get_monthly_revenue()
    if monthly_data:
        monthly_df = pd.DataFrame(list(monthly_data.items()), columns=["Month", "Revenue"])
        fig = px.area(
            monthly_df, x="Month", y="Revenue",
            title="Monthly Revenue",
            color_discrete_sequence=["#1f6feb"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#8b949e",
            title_font_color="#f0f6fc",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor="#21262d", showgrid=True),
            yaxis=dict(gridcolor="#21262d", showgrid=True, tickprefix="£"),
            showlegend=False,
        )
        fig.update_traces(fill='tozeroy', line_color="#1f6feb", fillcolor="rgba(31,111,235,0.15)")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    top_countries = sales.get_top_countries_by_revenue(limit=5)
    if top_countries:
        countries_df = pd.DataFrame(list(top_countries.items()), columns=["Country", "Revenue"])
        fig2 = px.bar(
            countries_df, x="Revenue", y="Country",
            orientation="h",
            title="Top 5 Countries by Revenue",
            color="Revenue",
            color_continuous_scale=["#1c2128", "#1f6feb"],
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#8b949e",
            title_font_color="#f0f6fc",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor="#21262d", tickprefix="£"),
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Chat Interface ──────────────────────────────────────────────────────────────
st.markdown("#### Ask Your AI Analyst Team")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 &nbsp; {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            agent = msg.get("agent", "Manager")
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="chat-ai">
                <div class="agent-badge">🤖 {agent}</div>
                {content}
            </div>""", unsafe_allow_html=True)

# Suggested questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    suggestions = [
        "Who is my top customer?",
        "What are the top 5 products by revenue?",
        "Show me the monthly revenue trend",
        "What is the refund rate?",
        "Which country generates the most revenue?",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"sugg_{i}"):
            st.session_state.pending_input = suggestion
            st.rerun()

# Handle suggested-question click
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing your data..."):
        response = manager.handle_request(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response, "agent": "AI Analyst"})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about your sales, products, or customers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Analyzing your data..."):
        response = manager.handle_request(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response, "agent": "AI Analyst"})
    st.rerun()

# Clear chat button
if st.session_state.messages:
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
