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
    /* ── Base ── */
    .stApp { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 8px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        color: #8b949e;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #1f6feb !important;
        color: #f0f6fc !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

    /* ── Native metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] > div {
        font-size: 11px !important;
        color: #8b949e !important;
    }

    /* ── Agent status cards ── */
    .agent-card {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .agent-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #3fb950;
        box-shadow: 0 0 6px #3fb950;
        flex-shrink: 0;
    }
    .agent-name { color: #e6edf3; font-size: 13px; font-weight: 500; }
    .agent-role { color: #8b949e; font-size: 11px; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #1f6feb 0%, #8b5cf6 50%, #3fb950 100%);
    }

    /* ── Suggestion buttons ── */
    .stButton > button {
        border-radius: 20px !important;
        border: 1px solid #30363d !important;
        background: #161b22 !important;
        color: #8b949e !important;
        font-size: 12px !important;
        padding: 6px 14px !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #1f6feb !important;
        color: #79c0ff !important;
        background: rgba(31, 111, 235, 0.08) !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 1px rgba(31, 111, 235, 0.3) !important;
    }

    /* ── Section headers ── */
    .section-label {
        color: #8b949e;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 16px 0 10px 2px;
    }

    /* ── Chart subsection title ── */
    .chart-title {
        color: #8b949e;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    hr { border-color: #21262d !important; }
    .stSpinner > div { border-top-color: #1f6feb !important; }
</style>
""", unsafe_allow_html=True)

# ── Shared chart style ──────────────────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font_color="#8b949e",
    title_font_color="#f0f6fc",
    title_font_size=13,
    margin=dict(l=10, r=10, t=38, b=10),
    showlegend=False,
    hoverlabel=dict(bgcolor="#21262d", font_color="#e6edf3", bordercolor="#30363d"),
)


# ── Load agents ─────────────────────────────────────────────────────────────────
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
    st.markdown("""
    <div style="padding: 6px 0 14px 0;">
        <div style="font-size: 17px; font-weight: 700; color: #f0f6fc; letter-spacing: -0.3px;">
            📊 AI Data Dept.
        </div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 3px; letter-spacing: 0.3px;">
            Retail Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner("Initializing agents..."):
        df, manager, sales = load_agents()

    if df is None:
        st.error("Could not load data. Check that `online_retail_small.csv` is in the project folder.")
        st.stop()

    st.markdown('<div class="section-label">Active Agents</div>', unsafe_allow_html=True)

    agents_info = [
        ("Alex", "Sales Analyst", "💼"),
        ("Dana", "Product Analyst", "📦"),
        ("Maya", "Customer Analyst", "👤"),
    ]
    for name, role, icon in agents_info:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-dot"></div>
            <div>
                <div class="agent-name">{icon}&nbsp; {name}</div>
                <div class="agent-role">{role}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: #8b949e; font-size: 12px; line-height: 2;">
        📄 &nbsp;online_retail_small.csv<br>
        🗂 &nbsp;{len(df):,} records loaded<br>
        🌍 &nbsp;UK Online Retail
    </div>""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div style="position: relative; z-index: 1;">
        <div style="color: #f0f6fc; font-size: 22px; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 4px;">
            AI Data Department
        </div>
        <div style="color: #8b949e; font-size: 12px; letter-spacing: 0.2px;">
            Real-time retail analytics powered by autonomous AI agents
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab_dash, tab_chat = st.tabs(["📊  Dashboard", "💬  AI Chat"])


# ════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════
with tab_dash:

    # ── KPI row ──────────────────────────────────────────────────────────────────
    total_rev   = sales.get_total_revenue()
    total_orders = sales.get_total_orders()
    total_items  = sales.get_total_items_sold()
    aov          = sales.get_average_order_value()
    refund       = sales.get_refund_rate()
    mom          = sales.get_mom_growth_rate()
    trend_str    = f"↑ {mom:.1f}% MoM" if mom > 0 else (f"↓ {abs(mom):.1f}% MoM" if mom < 0 else "Stable MoM")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue",    f"£{total_rev:,.0f}",   trend_str)
    k2.metric("Total Orders",     f"{total_orders:,}",    "Unique invoices")
    k3.metric("Items Sold",       f"{total_items:,}",     "Units dispatched")
    k4.metric("Avg Order Value",  f"£{aov:,.2f}",         "Per invoice")
    k5.metric("Refund Rate",      f"{refund:.1f}%",       "Of all transactions")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────────
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
                **CHART_BASE,
                xaxis=dict(gridcolor="#21262d", showgrid=True, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#21262d", showgrid=True, tickprefix="£", tickfont=dict(size=10)),
            )
            fig.update_traces(
                fill="tozeroy",
                line_color="#1f6feb",
                fillcolor="rgba(31,111,235,0.10)",
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
            )
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
                color_continuous_scale=["#21262d", "#1f6feb"],
                template="plotly_dark",
            )
            fig2.update_layout(
                **CHART_BASE,
                xaxis=dict(gridcolor="#21262d", tickprefix="£", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#21262d", tickfont=dict(size=11)),
                coloraxis_showscale=False,
            )
            fig2.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────────
    col3, col4 = st.columns([2, 3])

    with col3:
        top_products = sales.get_top_products_by_revenue(limit=8)
        if top_products:
            prod_df = pd.DataFrame(list(top_products.items()), columns=["Product", "Revenue"])
            prod_df["Label"] = prod_df["Product"].str[:28].str.strip()
            fig3 = px.bar(
                prod_df, x="Revenue", y="Label",
                orientation="h",
                title="Top 8 Products by Revenue",
                color="Revenue",
                color_continuous_scale=["#21262d", "#8b5cf6"],
                template="plotly_dark",
            )
            fig3.update_layout(
                **CHART_BASE,
                height=340,
                xaxis=dict(gridcolor="#21262d", tickprefix="£", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#21262d", tickfont=dict(size=9)),
                coloraxis_showscale=False,
            )
            fig3.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        hourly = sales.get_hourly_sales_distribution()
        if hourly and "error" not in hourly:
            hourly_df = pd.DataFrame(list(hourly.items()), columns=["Hour", "Revenue"])
            hourly_df["HourNum"] = hourly_df["Hour"].str.replace(":00", "").astype(int)
            hourly_df = hourly_df.sort_values("HourNum")
            fig4 = px.bar(
                hourly_df, x="Hour", y="Revenue",
                title="Sales by Hour of Day",
                color="Revenue",
                color_continuous_scale=["#21262d", "#3fb950"],
                template="plotly_dark",
            )
            fig4.update_layout(
                **CHART_BASE,
                height=340,
                xaxis=dict(
                    gridcolor="#21262d",
                    categoryorder="array",
                    categoryarray=hourly_df["Hour"].tolist(),
                    tickfont=dict(size=10),
                    title="",
                ),
                yaxis=dict(gridcolor="#21262d", tickprefix="£", tickfont=dict(size=10), title=""),
                coloraxis_showscale=False,
            )
            fig4.update_traces(hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>")
            st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — AI CHAT
# ════════════════════════════════════════════════════════
with tab_chat:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Suggestion chips ──────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="color: #8b949e; font-size: 13px; margin-bottom: 14px; line-height: 1.6;">
            Ask your AI analyst team anything about your data. Try one of these:
        </div>""", unsafe_allow_html=True)

        suggestions = [
            "Who is my top customer?",
            "Top 5 products by revenue",
            "Show monthly revenue trend",
            "What is the refund rate?",
            "Which country earns the most?",
            "Weekend vs weekday sales",
        ]
        c1, c2, c3 = st.columns(3)
        cols_cycle = [c1, c2, c3]
        for i, s in enumerate(suggestions):
            if cols_cycle[i % 3].button(s, key=f"sugg_{i}"):
                st.session_state.pending_input = s
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.caption(f"🤖 AI Analyst")
                st.markdown(msg["content"])

    # ── Handle suggestion click ───────────────────────────────────────────────────
    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing your data..."):
            response = manager.handle_request(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response, "agent": "AI Analyst"})
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your sales, products, or customers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing your data..."):
            response = manager.handle_request(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "agent": "AI Analyst"})
        st.rerun()

    # ── Clear button ──────────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
