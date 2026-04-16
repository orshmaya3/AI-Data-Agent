import base64
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
        border: 1px solid #484f58 !important;
        background: #21262d !important;
        color: #c9d1d9 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #1f6feb !important;
        color: #79c0ff !important;
        background: rgba(31, 111, 235, 0.12) !important;
    }

    /* ── Chat message bubbles ── */
    [data-testid="stChatMessage"] {
        background: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
    }
    /* Force all text inside chat bubbles to be fully visible */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] code {
        color: #e6edf3 !important;
    }
    /* User bubble: distinct blue tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #1a2d4a !important;
        border-color: #1f6feb !important;
    }
    /* AI bubble: neutral dark card */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #1c2128 !important;
        border-color: #30363d !important;
    }
    /* Caption / agent badge */
    [data-testid="stChatMessage"] .stCaption,
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
        color: #3fb950 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 6px !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background-color: #161b22 !important;
        border: 1.5px solid #484f58 !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: none !important;
        color: #e6edf3 !important;
        border-radius: 14px !important;
        font-size: 14px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6e7681 !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 2px rgba(31, 111, 235, 0.2) !important;
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
# Bump this string whenever Manager.py / analyst files change — forces Streamlit
# to discard the cached ManagerAgent and rebuild from the current code.
_AGENT_VERSION = "v14"  # bump when Manager.py / analyst files change

def _csv_mtime() -> float:
    """Return the modification time of the CSV so the cache key tracks file changes."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixed_online_retail.csv")
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0

@st.cache_resource(show_spinner=False)
def load_agents(_version: str = _AGENT_VERSION, _mtime: float = 0.0):
    import importlib
    import sys

    # Force-reload all agent modules so Streamlit's soft-reload (which keeps
    # sys.modules alive between script reruns) never serves stale bytecode.
    for mod_name in ("Manager", "Data_Agent", "Sales_Analyst", "Product_Analyst",
                     "Customer_Analyst", "Prediction_Analyst", "Code_Executor"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from Data_Agent import DataAgent
    from Manager import ManagerAgent
    from Sales_Analyst import SalesAnalyst

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixed_online_retail.csv")
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
        df, manager, sales = load_agents(_AGENT_VERSION, _csv_mtime())

    if df is None or manager is None:
        st.error("Could not load data or initialize agents. Check that `mixed_online_retail.csv` is in the project folder.")
        st.stop()

    st.markdown('<div class="section-label">Active Agents</div>', unsafe_allow_html=True)

    agents_info = [
        ("Alex", "Sales Analyst", "💼"),
        ("Dana", "Product Analyst", "📦"),
        ("Maya", "Customer Analyst", "👤"),
        ("Rey",  "Prediction Analyst", "🔮"),
        ("Aria", "General Analyst · Code", "🧠"),
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
        📄 &nbsp;mixed_online_retail.csv<br>
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
tab_dash, tab_chat, tab_pred = st.tabs(["📊  Dashboard", "💬  AI Chat", "🔮  Prediction"])


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
    mom_data     = sales.get_mom_growth_rate()
    mom          = mom_data.get('latest', 0.0) if isinstance(mom_data, dict) else float(mom_data)
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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig2, width='stretch')

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
            st.plotly_chart(fig3, width='stretch')

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
            st.plotly_chart(fig4, width='stretch')


# ════════════════════════════════════════════════════════
# TAB 2 — AI CHAT
# ════════════════════════════════════════════════════════
with tab_chat:

    MAX_HISTORY = 50

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Auto-greeting injected once when the chat is first opened
        st.session_state.messages.append({
            "role": "assistant",
            "agent": "Manager Agent",
            "content": (
                "👋 Hello! I'm your **AI Data Department** — a team of specialised analysts "
                "ready to dig into your retail data.\n\n"
                "**Meet the team:**\n"
                "- 💼 **Alex** — Sales Analyst *(revenue, orders, trends)*\n"
                "- 📦 **Dana** — Product Analyst *(product performance, rankings, lifecycle)*\n"
                "- 👤 **Maya** — Customer Analyst *(profiles, loyalty, segmentation)*\n"
                "- 🔮 **Rey** — Prediction Analyst *(forecasts, churn risk, CLV, demand trends)*\n"
                "- 🧠 **Aria** — General Analyst *(cross-domain questions & custom code)*\n\n"
                "What would you like to analyse today?"
            ),
            "charts": [],
        })

    # Trim history to prevent unbounded memory growth in long sessions
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # ── Suggestion chips (shown until the first user message) ────────────────────
    if not any(m["role"] == "user" for m in st.session_state.messages):
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

    # ── Helper: render one assistant message (text + any saved charts) ─────────
    def _render_assistant_msg(msg: dict) -> None:
        agent_label = msg.get("agent", "AI Analyst")
        with st.chat_message("assistant"):
            st.caption(f"🤖 {agent_label}")
            st.markdown(msg["content"])
            for b64 in msg.get("charts", []):
                img_bytes = base64.b64decode(b64)
                st.image(img_bytes, width='stretch')

    # ── Chat history ──────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            _render_assistant_msg(msg)

    # ── Shared helper: run a request and show live routing steps ────────────────
    def _process_request(user_input: str) -> None:
        """Consume the handle_request generator, display routing steps via st.status,
        then persist the assistant reply to session_state."""
        response = ""
        agent_label = "AI Analyst"
        history = st.session_state.messages[:-1]  # exclude the just-appended user msg

        with st.status("🧠 Manager Agent is analyzing your request...", expanded=True) as status_box:
            for step in manager.handle_request(user_input, history=history):
                if step["type"] == "status":
                    status_box.update(label=step["message"])
                elif step["type"] == "routing":
                    st.write(step["message"])
                    status_box.update(label=step["message"])
                    agent_label = step["agent_label"]
                elif step["type"] == "result":
                    response = step["content"]
                    agent_label = step.get("agent_label", agent_label)
            status_box.update(
                label=f"✅ {agent_label} responded",
                state="complete",
                expanded=False,
            )

        charts = manager.get_pending_charts()
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "agent": agent_label, "charts": charts}
        )

    # ── Handle suggestion click ───────────────────────────────────────────────────
    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")
        st.session_state.messages.append({"role": "user", "content": user_input})
        _process_request(user_input)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your sales, products, customers, or request custom analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        _process_request(prompt)
        st.rerun()

    # ── Clear button ──────────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════
# TAB 3 — PREDICTION AGENT (Rey)
# ════════════════════════════════════════════════════════
with tab_pred:

    # ── Pre-compute all prediction data once per session ─────────────────────────
    pa = manager.prediction_analyst

    @st.cache_data(show_spinner=False)
    def _pred_data(_version: str, _mtime: float):
        """Cached computation of all prediction dashboard data."""
        churn     = pa.get_churn_risk_summary(days_inactive=90)
        repeat    = pa.get_repeat_purchase_probability()
        forecast  = pa.get_revenue_forecast(horizon_months=3)
        growth    = pa.get_high_growth_products(lookback_months=3, top_n=6)
        slow      = pa.get_slow_movers(lookback_months=3, top_n=6)
        return churn, repeat, forecast, growth, slow

    churn_data, repeat_data, forecast_data, growth_data, slow_data = _pred_data(
        _AGENT_VERSION, _csv_mtime()
    )

    # ── Section header ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Live Prediction Metrics</div>', unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────────
    churn_pct    = churn_data.get("churn_risk_pct", 0.0) if "error" not in churn_data else None
    at_risk_n    = churn_data.get("at_risk_customers", 0)  if "error" not in churn_data else None
    healthy_n    = churn_data.get("healthy_customers", 0)  if "error" not in churn_data else None
    total_cust   = churn_data.get("total_customers", 0)    if "error" not in churn_data else None
    repeat_pct   = repeat_data.get("repeat_purchase_probability_pct", 0.0) if "error" not in repeat_data else None
    avg_orders   = repeat_data.get("avg_orders_per_returning_customer", 0.0) if "error" not in repeat_data else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Churn Risk",
        f"{churn_pct:.1f}%" if churn_pct is not None else "N/A",
        f"{at_risk_n:,} of {total_cust:,} customers" if at_risk_n is not None else "",
    )
    k2.metric(
        "At-Risk Customers",
        f"{at_risk_n:,}" if at_risk_n is not None else "N/A",
        "Inactive ≥ 90 days",
    )
    k3.metric(
        "Repeat Purchase Rate",
        f"{repeat_pct:.1f}%" if repeat_pct is not None else "N/A",
        f"avg {avg_orders:.1f} orders / returner" if avg_orders is not None else "",
    )
    k4.metric(
        "Healthy Customers",
        f"{healthy_n:,}" if healthy_n is not None else "N/A",
        "Active within 90 days",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1: Revenue Forecast + Churn Donut ─────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="chart-title">Revenue Forecast</div>', unsafe_allow_html=True)
        if "error" not in forecast_data:
            # Historical monthly revenue (all months)
            hist_raw = sales.get_monthly_revenue()
            if isinstance(hist_raw, dict) and "error" not in hist_raw:
                hist_df = (
                    pd.DataFrame(list(hist_raw.items()), columns=["month", "revenue"])
                    .sort_values("month")
                )
                # Keep last 12 months for readability
                hist_df = hist_df.tail(12).copy()

                # Build forecast rows
                fcast_dict = forecast_data.get("forecast", {})
                fcast_df = pd.DataFrame(
                    [{"month": m, "revenue": v} for m, v in fcast_dict.items()]
                ).sort_values("month")

                # Bridge point: duplicate last historical point into forecast series
                bridge = hist_df.iloc[[-1]].copy()

                fig_fc = go.Figure()

                # Historical area trace
                fig_fc.add_trace(go.Scatter(
                    x=hist_df["month"], y=hist_df["revenue"],
                    mode="lines",
                    name="Historical",
                    line=dict(color="#1f6feb", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(31,111,235,0.10)",
                    hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Historical</extra>",
                ))

                # Forecast dashed trace (bridge → forecast months)
                fcast_x = pd.concat([bridge["month"], fcast_df["month"]])
                fcast_y = pd.concat([bridge["revenue"], fcast_df["revenue"]])
                fig_fc.add_trace(go.Scatter(
                    x=fcast_x, y=fcast_y,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#8b5cf6", width=2.5, dash="dash"),
                    marker=dict(size=8, color="#8b5cf6", symbol="circle"),
                    hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Forecast</extra>",
                ))

                # Annotate each forecast point with its value
                for _, row in fcast_df.iterrows():
                    fig_fc.add_annotation(
                        x=row["month"], y=row["revenue"],
                        text=f"£{row['revenue']:,.0f}",
                        showarrow=False,
                        yshift=14,
                        font=dict(size=10, color="#c9d1d9"),
                    )

                # Vertical divider between historical and forecast
                # add_vline chokes on string/category axes (tries int+str arithmetic
                # internally), so use add_shape + add_annotation instead.
                last_hist_month = hist_df["month"].iloc[-1]
                fig_fc.add_shape(
                    type="line",
                    xref="x", yref="paper",
                    x0=last_hist_month, x1=last_hist_month,
                    y0=0, y1=1,
                    line=dict(dash="dot", color="#484f58", width=1.5),
                )
                fig_fc.add_annotation(
                    x=last_hist_month, y=1,
                    xref="x", yref="paper",
                    text="Forecast →",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=10, color="#8b949e"),
                )

                slope = forecast_data.get("monthly_slope_gbp", 0)
                trend_label = (
                    f"↑ Upward trend (+£{slope:,.0f}/mo)" if slope > 0
                    else f"↓ Downward trend (£{slope:,.0f}/mo)" if slope < 0
                    else "→ Flat trend"
                )

                fig_fc.update_layout(
                    **{**CHART_BASE, "showlegend": True},
                    height=320,
                    legend=dict(
                        orientation="h", x=0, y=1.12,
                        font=dict(size=11, color="#8b949e"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    xaxis=dict(gridcolor="#21262d", tickfont=dict(size=10), title=""),
                    yaxis=dict(gridcolor="#21262d", tickprefix="£", tickfont=dict(size=10), title=""),
                    title=dict(text=f"Last 12 months + 3-month linear forecast  ·  {trend_label}", font=dict(size=11, color="#8b949e")),
                )
                st.plotly_chart(fig_fc, width='stretch')

                if forecast_data.get("outlier_warning"):
                    st.warning(forecast_data["outlier_warning"], icon="⚠️")
        else:
            st.info(forecast_data.get("error", "Forecast unavailable."), icon="ℹ️")

    with col_b:
        st.markdown('<div class="chart-title">Churn Risk Breakdown</div>', unsafe_allow_html=True)
        if "error" not in churn_data:
            fig_donut = go.Figure(go.Pie(
                labels=["At-Risk", "Healthy"],
                values=[churn_data["at_risk_customers"], churn_data["healthy_customers"]],
                hole=0.62,
                marker=dict(
                    colors=["#f85149", "#3fb950"],
                    line=dict(color="#161b22", width=3),
                ),
                textinfo="percent",
                textfont=dict(size=13, color="#e6edf3"),
                hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
                sort=False,
            ))
            fig_donut.add_annotation(
                text=f"<b>{churn_data['churn_risk_pct']}%</b><br><span style='font-size:11px'>at risk</span>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="#f0f6fc"),
                align="center",
            )
            fig_donut.update_layout(
                **CHART_BASE,
                height=320,
                showlegend=True,
                legend=dict(
                    orientation="h", x=0.15, y=-0.08,
                    font=dict(size=11, color="#8b949e"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title=dict(
                    text=f"90-day inactivity threshold  ·  ref date {churn_data.get('reference_date', '')}",
                    font=dict(size=11, color="#8b949e"),
                ),
            )
            st.plotly_chart(fig_donut, width='stretch')
        else:
            st.info(churn_data.get("error", "Churn data unavailable."), icon="ℹ️")

    # ── Charts row 2: High-Growth + Slow Movers ───────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="chart-title">Top High-Growth Products (last 3 mo. vs prior 3 mo.)</div>', unsafe_allow_html=True)
        if growth_data and "error" not in growth_data[0]:
            g_df = pd.DataFrame(growth_data).sort_values("growth_pct")
            g_df["label"] = g_df["product"].str[:30].str.strip()
            fig_growth = go.Figure(go.Bar(
                x=g_df["growth_pct"],
                y=g_df["label"],
                orientation="h",
                marker=dict(
                    color=g_df["growth_pct"],
                    colorscale=[[0, "#1a3a2a"], [1, "#2ecc71"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"+{v:.0f}%" for v in g_df["growth_pct"]],
                textposition="outside",
                textfont=dict(size=10, color="#c9d1d9"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Growth: +%{x:.1f}%<br>"
                    "Recent: %{customdata[0]:,} units  |  Prior: %{customdata[1]:,} units"
                    "<extra></extra>"
                ),
                customdata=g_df[["recent_units", "prior_units"]].values,
            ))
            fig_growth.update_layout(
                **CHART_BASE,
                height=300,
                xaxis=dict(gridcolor="#21262d", ticksuffix="%", tickfont=dict(size=10), title=""),
                yaxis=dict(gridcolor="#21262d", tickfont=dict(size=9), title=""),
            )
            st.plotly_chart(fig_growth, width='stretch')
        elif growth_data and "error" in growth_data[0]:
            st.info(growth_data[0]["error"], icon="ℹ️")

    with col_d:
        st.markdown('<div class="chart-title">Top Slow Movers (last 3 mo. vs prior 3 mo.)</div>', unsafe_allow_html=True)
        if slow_data and "error" not in slow_data[0]:
            s_df = pd.DataFrame(slow_data).sort_values("decline_pct")
            s_df["label"] = s_df["product"].str[:30].str.strip()
            fig_slow = go.Figure(go.Bar(
                x=s_df["decline_pct"],
                y=s_df["label"],
                orientation="h",
                marker=dict(
                    color=s_df["decline_pct"],
                    colorscale=[[0, "#1a0a0a"], [1, "#f85149"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"-{v:.0f}%" for v in s_df["decline_pct"]],
                textposition="outside",
                textfont=dict(size=10, color="#c9d1d9"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Decline: -%{x:.1f}%<br>"
                    "Recent: %{customdata[0]:,} units  |  Prior: %{customdata[1]:,} units"
                    "<extra></extra>"
                ),
                customdata=s_df[["recent_units", "prior_units"]].values,
            ))
            fig_slow.update_layout(
                **CHART_BASE,
                height=300,
                xaxis=dict(gridcolor="#21262d", ticksuffix="%", tickfont=dict(size=10), title=""),
                yaxis=dict(gridcolor="#21262d", tickfont=dict(size=9), title=""),
            )
            st.plotly_chart(fig_slow, width='stretch')
        elif slow_data and "error" in slow_data[0]:
            st.info(slow_data[0]["error"], icon="ℹ️")

    # ── Divider between dashboard and chat ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Ask Rey — Predictive AI Chat</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #8b949e; font-size: 12px; margin-bottom: 14px; line-height: 1.6;">
        Ask deeper questions: CLV for a specific customer, product demand trends, custom forecasts, churn lists.
    </div>""", unsafe_allow_html=True)

    # ── Prediction chat state ────────────────────────────────────────────────────
    MAX_PRED_HISTORY = 50

    if "pred_messages" not in st.session_state:
        st.session_state.pred_messages = []
        st.session_state.pred_messages.append({
            "role": "assistant",
            "agent": "Prediction Agent (Rey)",
            "content": (
                "👋 Hi, I'm **Rey** — your Predictive Analytics Specialist.\n\n"
                "I can help you with:\n"
                "- 📉 **Churn risk** — which customers are likely to leave\n"
                "- 📈 **Revenue forecasts** — what the next months might look like\n"
                "- 🚀 **High-growth products** — what's taking off\n"
                "- 🐢 **Slow movers** — what's declining and should be reviewed\n"
                "- 💰 **Customer CLV** — projected lifetime value per customer\n"
                "- 🔄 **Repeat purchase probability** — how sticky your buyers are\n\n"
                "What would you like to predict today?"
            ),
            "charts": [],
        })

    if len(st.session_state.pred_messages) > MAX_PRED_HISTORY:
        st.session_state.pred_messages = st.session_state.pred_messages[-MAX_PRED_HISTORY:]

    # ── Suggestion chips (shown until first user message) ────────────────────────
    if not any(m["role"] == "user" for m in st.session_state.pred_messages):
        pred_suggestions = [
            "Who are our top at-risk customers?",
            "Which products are declining in demand?",
            "Forecast revenue for the next 6 months",
            "What is the CLV of customer 17850?",
            "Show product demand trend for WHITE HANGING HEART T-LIGHT HOLDER",
            "What is the repeat purchase probability?",
        ]
        c1, c2, c3 = st.columns(3)
        cols_cycle = [c1, c2, c3]
        for i, s in enumerate(pred_suggestions):
            if cols_cycle[i % 3].button(s, key=f"pred_sugg_{i}"):
                st.session_state.pred_pending = s
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Render helper ────────────────────────────────────────────────────────────
    def _render_pred_msg(msg: dict) -> None:
        agent_label = msg.get("agent", "Prediction Agent (Rey)")
        with st.chat_message("assistant"):
            st.caption(f"🔮 {agent_label}")
            st.markdown(msg["content"])
            for b64 in msg.get("charts", []):
                img_bytes = base64.b64decode(b64)
                st.image(img_bytes, width='stretch')

    # ── Chat history ─────────────────────────────────────────────────────────────
    for msg in st.session_state.pred_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            _render_pred_msg(msg)

    # ── Request handler ───────────────────────────────────────────────────────────
    def _process_pred_request(user_input: str) -> None:
        history = st.session_state.pred_messages[:-1]
        response = ""
        agent_label = "Prediction Agent (Rey)"

        with st.status("🔮 Rey is analysing your request...", expanded=True) as status_box:
            for step in manager.handle_prediction_request(user_input, history=history):
                if step["type"] == "status":
                    status_box.update(label=step["message"])
                elif step["type"] == "result":
                    response = step["content"]
                    agent_label = step.get("agent_label", agent_label)
            status_box.update(
                label="✅ Prediction Agent (Rey) responded",
                state="complete",
                expanded=False,
            )

        charts = manager.get_pending_charts()
        st.session_state.pred_messages.append(
            {"role": "assistant", "content": response, "agent": agent_label, "charts": charts}
        )

    # ── Handle suggestion click ───────────────────────────────────────────────────
    if "pred_pending" in st.session_state:
        user_input = st.session_state.pop("pred_pending")
        st.session_state.pred_messages.append({"role": "user", "content": user_input})
        _process_pred_request(user_input)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask Rey about forecasts, churn, CLV, product trends...", key="pred_input"):
        st.session_state.pred_messages.append({"role": "user", "content": prompt})
        _process_pred_request(prompt)
        st.rerun()

    # ── Clear button ──────────────────────────────────────────────────────────────
    if st.session_state.pred_messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  Clear conversation", key="clear_pred"):
            st.session_state.pred_messages = []
            st.rerun()
