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

# ── Custom CSS  (Monday.com light theme) ───────────────────────────────────────
st.markdown("""
<style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    header    { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ── Base ── */
    .stApp { background-color: #F7F5FF; }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #EDE9FE !important;
        border-right: 1px solid #DDD6FE;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p { color: #4C1D95 !important; }

    /* ── Split-pane columns: let Streamlit handle spacing natively ── */

    /* ── Artifact canvas card ── */
    .canvas-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 20px 24px;
        min-height: calc(100vh - 120px);
        position: sticky;
        top: 0.5rem;
        box-shadow: 0 1px 4px rgba(109,40,217,0.06);
    }
    .canvas-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #EDE9FE;
    }
    .canvas-title {
        color: #1E1B4B;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    .canvas-badge {
        background: #EDE9FE;
        border: 1px solid #DDD6FE;
        color: #6D28D9;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-radius: 20px;
        padding: 3px 10px;
    }
    .canvas-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: calc(100vh - 250px);
        color: #C4B5FD;
        text-align: center;
        gap: 12px;
    }
    .canvas-empty-icon { font-size: 48px; opacity: 0.5; }
    .canvas-empty-text { font-size: 13px; line-height: 1.6; max-width: 260px; color: #9CA3AF; }

    /* ── Chat pane scroll ── */
    .chat-scroll {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
        padding-right: 4px;
        scrollbar-width: thin;
        scrollbar-color: #DDD6FE transparent;
    }

    /* ── Chat pane divider ── */
    .chat-pane {
        border-right: 1px solid #E5E7EB;
        padding-right: 24px;
        min-height: calc(100vh - 80px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        color: #9CA3AF;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #7C3AED !important;
        color: #1E1B4B !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }
    [data-testid="stTabs"] { margin-top: 20px !important; }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] + [data-testid="stHorizontalBlock"] { margin-top: 20px !important; }

    /* ── Native metric cards ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(109,40,217,0.06);
    }
    [data-testid="stMetricLabel"] p {
        color: #6B7280 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        color: #1E1B4B !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] svg { width: 14px !important; height: 14px !important; }
    [data-testid="stMetricDelta"] > div { font-size: 11px !important; font-weight: 600 !important; }
    [data-testid="stMetricDelta"][data-direction="increase"] > div { color: #059669 !important; }
    [data-testid="stMetricDelta"][data-direction="decrease"] > div { color: #DC2626 !important; }

    /* ── Agent status cards ── */
    .agent-card {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #FFFFFF;
        border: 1px solid #DDD6FE;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(109,40,217,0.05);
    }
    .agent-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #059669;
        box-shadow: 0 0 6px #059669;
        flex-shrink: 0;
    }
    .agent-name { color: #1E1B4B; font-size: 13px; font-weight: 500; }
    .agent-role { color: #6B7280; font-size: 11px; }

    /* ── Compact page header ── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 14px 0;
        border-bottom: 1px solid #DDD6FE;
        margin-bottom: 16px;
    }
    .page-header-title {
        color: #1E1B4B;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .page-header-sub {
        color: #6B7280;
        font-size: 11px;
        margin-top: 1px;
    }

    /* ── Main header (dashboard only) ── */
    .main-header {
        background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 60%, #E0E7FF 100%);
        border: 1px solid #DDD6FE;
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #7C3AED 0%, #6D28D9 50%, #4F46E5 100%);
        border-radius: 16px 16px 0 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #DDD6FE !important;
        background: #FFFFFF !important;
        color: #4C1D95 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        text-align: left !important;
        line-height: 1.5 !important;
        min-height: 40px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(109,40,217,0.06) !important;
    }
    .stButton > button:hover {
        border-color: #7C3AED !important;
        color: #6D28D9 !important;
        background: #F5F3FF !important;
    }

    /* ── Chat message bubbles ── */
    [data-testid="stChatMessage"] {
        background: #FAFAFA !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] code {
        color: #1E1B4B !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #F5F3FF !important;
        border-color: #DDD6FE !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #FFFFFF !important;
        border-color: #E5E7EB !important;
    }
    [data-testid="stChatMessage"] .stCaption,
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
        color: #7C3AED !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 6px !important;
    }

    /* ── Chat input bottom bar ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        background: #F7F5FF !important;
        border-top: 1px solid #E5E7EB !important;
    }

    /* ── Chat input container ── */
    [data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 1.5px solid #DDD6FE !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 4px rgba(109,40,217,0.07) !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        /* -webkit-text-fill-color overrides Streamlit's CSS variable for actual rendering */
        color: #1E1B4B !important;
        -webkit-text-fill-color: #1E1B4B !important;
        caret-color: #7C3AED !important;
        font-size: 14px !important;
        font-family: inherit !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important;
        -webkit-text-fill-color: #9CA3AF !important;
        font-size: 14px !important;
    }
    /* Send button only — identified by aria-label, not all buttons */
    [data-testid="stChatInput"] button[aria-label="Submit message"],
    [data-testid="stChatInputSubmitButton"] {
        background: #7C3AED !important;
        border: none !important;
        border-radius: 8px !important;
        transition: background 0.15s !important;
    }
    [data-testid="stChatInput"] button[aria-label="Submit message"]:hover,
    [data-testid="stChatInputSubmitButton"]:hover {
        background: #6D28D9 !important;
    }
    [data-testid="stChatInput"] button[aria-label="Submit message"] svg path,
    [data-testid="stChatInputSubmitButton"] svg path {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* ── Section headers ── */
    .section-label {
        color: #4B5563;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 16px 0 10px 2px;
    }

    /* ── Chart subsection title ── */
    .chart-title {
        color: #4B5563;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    hr { border-color: #E5E7EB !important; }
    .stSpinner > div { border-top-color: #7C3AED !important; }

    /* ── Sidebar navigation radio ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px; flex-direction: column;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: #5B21B6 !important;
        font-size: 13px;
        font-weight: 500;
        padding: 8px 12px;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s;
        display: flex; align-items: center; gap: 6px;
    }
    /* Hide the radio circle indicator */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label > div:first-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
        background: rgba(124,58,237,0.10);
        border-color: #DDD6FE;
        color: #4C1D95 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:has(input:checked) {
        background: rgba(124,58,237,0.12);
        border-color: #7C3AED;
        border-left: 3px solid #7C3AED !important;
        padding-left: 9px;
        color: #6D28D9 !important;
        font-weight: 600;
    }

    /* ── Expander chrome ── */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #4B5563 !important; font-size: 12px !important; font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover { color: #1E1B4B !important; }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: 1px solid #EDE9FE !important; padding-top: 12px !important;
    }

    /* ── Dataframe inside canvas ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Status widget ── */
    [data-testid="stStatusWidget"] {
        background: #FFFFFF !important;
        border: 1px solid #DDD6FE !important;
        border-radius: 10px !important;
        color: #4C1D95 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Page navigation state ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "📊 Dashboard"

# ── Shared chart style ──────────────────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font_color="#6B7280",
    title_font_color="#1E1B4B",
    title_font_size=13,
    margin=dict(l=10, r=10, t=38, b=10),
    showlegend=False,
    hoverlabel=dict(bgcolor="#F5F3FF", font_color="#1E1B4B", bordercolor="#DDD6FE"),
)


# ── Load agents ─────────────────────────────────────────────────────────────────
_AGENT_VERSION = "v18"

def _csv_mtime() -> float:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixed_online_retail.csv")
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0

@st.cache_resource(show_spinner=False)
def load_agents(_version: str = _AGENT_VERSION, _mtime: float = 0.0):
    import importlib
    import sys

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
        <div style="font-size: 17px; font-weight: 700; color: #1E1B4B; letter-spacing: -0.3px;">
            📊 AI Data Dept.
        </div>
        <div style="font-size: 11px; color: #7C3AED; margin-top: 3px; letter-spacing: 0.3px;">
            Retail Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    _nav_options = ["📊 Dashboard", "💬 AI Chat", "🔮 Prediction"]
    _selected = st.radio(
        "nav", _nav_options,
        index=_nav_options.index(st.session_state.get("page", "📊 Dashboard")),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.page = _selected
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner("Initializing agents..."):
        df, manager, sales = load_agents(_AGENT_VERSION, _csv_mtime())

    if df is None or manager is None:
        st.error("Could not load data or initialize agents. Check that `mixed_online_retail.csv` is in the project folder.")
        st.stop()

    st.markdown('<div class="section-label">Active Agents</div>', unsafe_allow_html=True)

    _AGENT_COLORS = {
        "Alex": "#7C3AED",
        "Dana": "#2563EB",
        "Maya": "#059669",
        "Rey":  "#6D28D9",
        "Aria": "#0891B2",
    }
    agents_info = [
        ("Alex", "Sales Analyst", "💼"),
        ("Dana", "Product Analyst", "📦"),
        ("Maya", "Customer Analyst", "👤"),
        ("Rey",  "Prediction Analyst", "🔮"),
        ("Aria", "General Analyst · Code", "🧠"),
    ]
    for name, role, icon in agents_info:
        color = _AGENT_COLORS.get(name, "#7C3AED")
        st.markdown(f"""
        <div class="agent-card" style="border-left: 3px solid {color};">
            <div class="agent-dot" style="background: {color}; box-shadow: 0 0 6px {color};"></div>
            <div>
                <div class="agent-name">{icon}&nbsp; {name}</div>
                <div class="agent-role">{role}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: #5B21B6; font-size: 12px; line-height: 2;">
        📄 &nbsp;mixed_online_retail.csv<br>
        🗂 &nbsp;{len(df):,} records loaded<br>
        🌍 &nbsp;UK Online Retail
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# CANVAS RENDER HELPER
# ════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if st.session_state.page == "📊 Dashboard":

    st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index: 1;">
            <div style="color: #1E1B4B; font-size: 20px; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 3px;">
                📊 Dashboard
            </div>
            <div style="color: #7C3AED; font-size: 12px; letter-spacing: 0.2px;">
                Real-time retail analytics powered by autonomous AI agents
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

    dash_t1, dash_t2, dash_t3, dash_t4 = st.tabs([
        "📈 Revenue Trend", "🌍 Top Countries", "📦 Top Products", "🕐 Hourly Sales",
    ])

    with dash_t1:
        monthly_data = sales.get_monthly_revenue()
        if monthly_data:
            monthly_df = pd.DataFrame(list(monthly_data.items()), columns=["Month", "Revenue"])
            fig = px.area(
                monthly_df, x="Month", y="Revenue",
                title="Monthly Revenue",
                color_discrete_sequence=["#7C3AED"],
            )
            fig.update_layout(
                **CHART_BASE,
                height=380,
                xaxis=dict(gridcolor="#E5E7EB", showgrid=True, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#E5E7EB", showgrid=True, tickprefix="£", tickfont=dict(size=10)),
            )
            fig.update_traces(
                fill="tozeroy",
                line_color="#7C3AED",
                fillcolor="rgba(124,58,237,0.08)",
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

    with dash_t2:
        top_countries = sales.get_top_countries_by_revenue(limit=5)
        if top_countries:
            countries_df = pd.DataFrame(list(top_countries.items()), columns=["Country", "Revenue"])
            fig2 = px.bar(
                countries_df, x="Revenue", y="Country",
                orientation="h",
                title="Top 5 Countries by Revenue",
                color="Revenue",
                color_continuous_scale=["#EDE9FE", "#6D28D9"],
            )
            fig2.update_layout(
                **CHART_BASE,
                height=380,
                xaxis=dict(gridcolor="#E5E7EB", tickprefix="£", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=11)),
                coloraxis_showscale=False,
            )
            fig2.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

    with dash_t3:
        top_products = sales.get_top_products_by_revenue(limit=8)
        if top_products:
            prod_df = pd.DataFrame(list(top_products.items()), columns=["Product", "Revenue"])
            prod_df["Label"] = prod_df["Product"].str[:28].str.strip()
            fig3 = px.bar(
                prod_df, x="Revenue", y="Label",
                orientation="h",
                title="Top 8 Products by Revenue",
                color="Revenue",
                color_continuous_scale=["#EDE9FE", "#7C3AED"],
            )
            fig3.update_layout(
                **CHART_BASE,
                height=380,
                xaxis=dict(gridcolor="#E5E7EB", tickprefix="£", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=9)),
                coloraxis_showscale=False,
            )
            fig3.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

    with dash_t4:
        hourly = sales.get_hourly_sales_distribution()
        if hourly and "error" not in hourly:
            hourly_df = pd.DataFrame(list(hourly.items()), columns=["Hour", "Revenue"])
            hourly_df["HourNum"] = hourly_df["Hour"].str.replace(":00", "").astype(int)
            hourly_df = hourly_df.sort_values("HourNum")
            fig4 = px.bar(
                hourly_df, x="Hour", y="Revenue",
                title="Sales by Hour of Day",
                color="Revenue",
                color_continuous_scale=["#D1FAE5", "#059669"],
            )
            fig4.update_layout(
                **CHART_BASE,
                height=380,
                xaxis=dict(
                    gridcolor="#E5E7EB",
                    categoryorder="array",
                    categoryarray=hourly_df["Hour"].tolist(),
                    tickfont=dict(size=10),
                    title="",
                ),
                yaxis=dict(gridcolor="#E5E7EB", tickprefix="£", tickfont=dict(size=10), title=""),
                coloraxis_showscale=False,
            )
            fig4.update_traces(hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})


# ════════════════════════════════════════════════════════
# PAGE 2 — AI CHAT  (split-pane)
# ════════════════════════════════════════════════════════
elif st.session_state.page == "💬 AI Chat":

    MAX_HISTORY = 50

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "agent": "Manager Agent",
            "content": "👋 Ready to analyse your retail data. Try a suggestion below or ask anything.",
            "charts": [],
        })

    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # ── Helper: render one assistant message with inline charts ─────────────────
    def _render_assistant_msg(msg: dict) -> None:
        agent_label = msg.get("agent", "AI Analyst")
        with st.chat_message("assistant"):
            st.caption(f"🤖 {agent_label}")
            st.markdown(msg["content"])
            for b64 in msg.get("charts", []):
                img_bytes = base64.b64decode(b64)
                st.image(img_bytes, use_container_width=True)

    # ── Shared helper: run a request and push result to canvas ──────────────────
    def _process_request(user_input: str) -> None:
        response = ""
        agent_label = "AI Analyst"
        history = st.session_state.messages[:-1]

        with st.status("🧠 Manager Agent is analyzing your request...", expanded=False) as status_box:
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

    st.markdown("""
    <div class="page-header">
        <div>
            <div class="page-header-title">💬 AI Chat</div>
            <div class="page-header-sub">Ask your analyst team anything about your retail data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Suggestion chips ──────────────────────────────────────────────────────
    if not any(m["role"] == "user" for m in st.session_state.messages):
        st.markdown("""
        <div style="color: #6B7280; font-size: 12px; margin-bottom: 10px; line-height: 1.6;">
            Try a suggestion:
        </div>""", unsafe_allow_html=True)
        suggestions = [
            "Who is my top customer?",
            "Top 5 products by revenue",
            "Show monthly revenue trend",
            "What is the refund rate?",
            "Which country earns the most?",
            "Weekend vs weekday sales",
        ]
        chip_cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if chip_cols[i % 3].button(s, key=f"sugg_{i}"):
                st.session_state.pending_input = s
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            _render_assistant_msg(msg)

    # ── Handle suggestion click ───────────────────────────────────────────────
    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")
        st.session_state.messages.append({"role": "user", "content": user_input})
        _process_request(user_input)
        st.rerun()

    # ── Clear button ──────────────────────────────────────────────────────────
    if len(st.session_state.messages) > 1:
        if st.button("🗑  Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your sales, products, customers, or request custom analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        _process_request(prompt)
        st.rerun()


# ════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION AGENT (Rey)  (split-pane chat)
# ════════════════════════════════════════════════════════
elif st.session_state.page == "🔮 Prediction":

    pa = manager.prediction_analyst

    @st.cache_data(show_spinner=False)
    def _pred_data(_version: str, _mtime: float):
        churn     = pa.get_churn_risk_summary(days_inactive=90)
        repeat    = pa.get_repeat_purchase_probability()
        forecast  = pa.get_revenue_forecast(horizon_months=3)
        growth    = pa.get_high_growth_products(lookback_months=3, top_n=6)
        slow      = pa.get_slow_movers(lookback_months=3, top_n=6)
        ml_churn  = pa.get_churn_probability_scores(churn_threshold_days=90, top_n=20)
        ml_seg    = pa.get_customer_segments(n_clusters=4)
        return churn, repeat, forecast, growth, slow, ml_churn, ml_seg

    churn_data, repeat_data, forecast_data, growth_data, slow_data, ml_churn_data, ml_seg_data = _pred_data(
        _AGENT_VERSION, _csv_mtime()
    )

    pred_t1, pred_t2, pred_t3 = st.tabs(["📊 Live Metrics", "💬 Ask Rey", "🤖 Live ML Models"])

    # ════════════════════════
    # TAB 2 — Ask Rey (split-pane)
    # ════════════════════════
    with pred_t2:

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

        def _render_pred_msg(msg: dict) -> None:
            agent_label = msg.get("agent", "Prediction Agent (Rey)")
            with st.chat_message("assistant"):
                st.caption(f"🔮 {agent_label}")
                st.markdown(msg["content"])
                for b64 in msg.get("charts", []):
                    img_bytes = base64.b64decode(b64)
                    st.image(img_bytes, use_container_width=True)

        def _process_pred_request(user_input: str) -> None:
            history = st.session_state.pred_messages[:-1]
            response = ""
            agent_label = "Prediction Agent (Rey)"

            with st.status("🔮 Rey is analysing your request...", expanded=False) as status_box:
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

        st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-header-title">🔮 Ask Rey</div>
                <div class="page-header-sub">Predictive AI — forecasts, churn, segmentation, CLV</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Suggestion chips ──────────────────────────────────────────────────
        if not any(m["role"] == "user" for m in st.session_state.pred_messages):
            pred_suggestions = [
                "Who are our top at-risk customers?",
                "Which products are declining?",
                "Forecast next 6 months",
                "CLV of customer 17850",
                "Repeat purchase probability",
                "Show customer segments",
            ]
            pred_chip_cols = st.columns(3)
            for i, s in enumerate(pred_suggestions):
                if pred_chip_cols[i % 3].button(s, key=f"pred_sugg_{i}"):
                    st.session_state.pred_pending = s
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Chat history ──────────────────────────────────────────────────────
        for msg in st.session_state.pred_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                _render_pred_msg(msg)

        # ── Handle suggestion click ───────────────────────────────────────────
        if "pred_pending" in st.session_state:
            user_input = st.session_state.pop("pred_pending")
            st.session_state.pred_messages.append({"role": "user", "content": user_input})
            _process_pred_request(user_input)
            st.rerun()

        # ── Clear button ──────────────────────────────────────────────────────
        if len(st.session_state.pred_messages) > 1:
            if st.button("🗑  Clear conversation", key="clear_pred"):
                st.session_state.pred_messages = []
                st.rerun()

        # ── Chat input ────────────────────────────────────────────────────────
        if prompt := st.chat_input("Ask Rey about forecasts, churn, CLV, product trends...", key="pred_input"):
            st.session_state.pred_messages.append({"role": "user", "content": prompt})
            _process_pred_request(prompt)
            st.rerun()

    # ════════════════════════
    # TAB 1 — Live Metrics
    # ════════════════════════
    with pred_t1:

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

        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.markdown('<div class="chart-title">Revenue Forecast</div>', unsafe_allow_html=True)
            if "error" not in forecast_data:
                hist_raw = sales.get_monthly_revenue()
                if isinstance(hist_raw, dict) and "error" not in hist_raw:
                    hist_df = (
                        pd.DataFrame(list(hist_raw.items()), columns=["month", "revenue"])
                        .sort_values("month")
                    )
                    hist_df = hist_df.tail(12).copy()

                    fcast_dict = forecast_data.get("forecast", {})
                    fcast_df = pd.DataFrame(
                        [{"month": m, "revenue": float(v["predicted_gbp"] if isinstance(v, dict) else v)}
                         for m, v in fcast_dict.items()]
                    ).sort_values("month")

                    bridge = hist_df.iloc[[-1]].copy()

                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(
                        x=hist_df["month"], y=hist_df["revenue"],
                        mode="lines",
                        name="Historical",
                        line=dict(color="#7C3AED", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(124,58,237,0.08)",
                        hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Historical</extra>",
                    ))

                    fcast_x = pd.concat([bridge["month"], fcast_df["month"]])
                    fcast_y = pd.concat([bridge["revenue"], fcast_df["revenue"]])
                    fig_fc.add_trace(go.Scatter(
                        x=fcast_x, y=fcast_y,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#6D28D9", width=2.5, dash="dash"),
                        marker=dict(size=8, color="#6D28D9", symbol="circle"),
                        hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Forecast</extra>",
                    ))

                    for _, row in fcast_df.iterrows():
                        fig_fc.add_annotation(
                            x=row["month"], y=row["revenue"],
                            text=f"£{row['revenue']:,.0f}",
                            showarrow=False,
                            yshift=14,
                            font=dict(size=10, color="#4C1D95"),
                        )

                    last_hist_month = hist_df["month"].iloc[-1]
                    fig_fc.add_shape(
                        type="line",
                        xref="x", yref="paper",
                        x0=last_hist_month, x1=last_hist_month,
                        y0=0, y1=1,
                        line=dict(dash="dot", color="#DDD6FE", width=1.5),
                    )
                    fig_fc.add_annotation(
                        x=last_hist_month, y=1,
                        xref="x", yref="paper",
                        text="Forecast →",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=10, color="#9CA3AF"),
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
                        xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=10), title=""),
                        yaxis=dict(gridcolor="#E5E7EB", tickprefix="£", tickfont=dict(size=10), title=""),
                        title=dict(text=f"Last 12 months + 3-month linear forecast  ·  {trend_label}", font=dict(size=11, color="#8b949e")),
                    )
                    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

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
                        colors=["#F87171", "#34D399"],
                        line=dict(color="#FFFFFF", width=3),
                    ),
                    textinfo="percent",
                    textfont=dict(size=13, color="#1E1B4B"),
                    hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
                    sort=False,
                ))
                fig_donut.add_annotation(
                    text=f"<b>{churn_data['churn_risk_pct']}%</b><br><span style='font-size:11px'>at risk</span>",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20, color="#1E1B4B"),
                    align="center",
                )
                fig_donut.update_layout(
                    **{**CHART_BASE, "showlegend": True},
                    height=320,
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
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})
            else:
                st.info(churn_data.get("error", "Churn data unavailable."), icon="ℹ️")

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
                        colorscale=[[0, "#D1FAE5"], [1, "#059669"]],
                        line=dict(color="rgba(0,0,0,0)"),
                    ),
                    text=[f"+{v:.0f}%" for v in g_df["growth_pct"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#374151"),
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
                    xaxis=dict(gridcolor="#E5E7EB", ticksuffix="%", tickfont=dict(size=10), title=""),
                    yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=9), title=""),
                )
                st.plotly_chart(fig_growth, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})
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
                        colorscale=[[0, "#FEE2E2"], [1, "#DC2626"]],
                        line=dict(color="rgba(0,0,0,0)"),
                    ),
                    text=[f"-{v:.0f}%" for v in s_df["decline_pct"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#374151"),
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
                    xaxis=dict(gridcolor="#E5E7EB", ticksuffix="%", tickfont=dict(size=10), title=""),
                    yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=9), title=""),
                )
                st.plotly_chart(fig_slow, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})
            elif slow_data and "error" in slow_data[0]:
                st.info(slow_data[0]["error"], icon="ℹ️")

    # ════════════════════════
    # TAB 3 — Live ML Models
    # ════════════════════════
    with pred_t3:

        st.markdown('<div class="section-label">Live ML Model Status</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown('<div class="chart-title">Prophet Forecaster</div>', unsafe_allow_html=True)
            if "error" not in forecast_data:
                st.metric("Model", "Prophet")
                st.metric("Training Period",
                          f"{forecast_data.get('training_start', 'N/A')} → {forecast_data.get('training_end', 'N/A')}")
                st.metric("Training Days", f"{forecast_data.get('training_days', 'N/A'):,}" if forecast_data.get('training_days') else "N/A")
                st.caption("Multiplicative seasonality · UK holidays · 95% CI")
            else:
                st.warning(forecast_data.get("error", "Model unavailable"), icon="⚠️")
                st.caption("Falling back to linear regression")

        with col_m2:
            st.markdown('<div class="chart-title">Random Forest Churn Classifier</div>', unsafe_allow_html=True)
            if "error" not in ml_churn_data:
                meta = ml_churn_data.get("model_metadata", {})
                st.metric("Accuracy", f"{meta.get('accuracy', 'N/A')}%")
                st.metric("Recall (Churned)", f"{meta.get('recall_churned', 'N/A')}%")
                st.metric("Training Samples", f"{meta.get('training_samples', 0):,}")
                st.caption(f"High-risk customers: {ml_churn_data.get('high_risk_count', 'N/A'):,}")
            else:
                st.warning(ml_churn_data.get("error", "Model unavailable"), icon="⚠️")

        with col_m3:
            st.markdown('<div class="chart-title">KMeans Segmentation</div>', unsafe_allow_html=True)
            if "error" not in ml_seg_data:
                meta = ml_seg_data.get("model_metadata", {})
                st.metric("Clusters", meta.get("n_clusters", "N/A"))
                st.metric("Silhouette Score", f"{meta.get('silhouette_score', 0):.4f}")
                total_seg = sum(ml_seg_data.get("all_segments_count", {}).values())
                st.metric("Total Customers", f"{total_seg:,}")
                st.caption("RFM-based · Champions / Loyal / At-Risk / Hibernating")
            else:
                st.warning(ml_seg_data.get("error", "Model unavailable"), icon="⚠️")

        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-title">Prophet Revenue Forecast — 3-Month Horizon with 95% Confidence Intervals</div>', unsafe_allow_html=True)

        if "error" not in forecast_data and "forecast" in forecast_data:
            fcast_dict = forecast_data["forecast"]
            rows = []
            for month_str, vals in fcast_dict.items():
                if isinstance(vals, dict):
                    rows.append({
                        "month":     month_str,
                        "predicted": vals.get("predicted_gbp", 0),
                        "lower":     vals.get("lower_bound_gbp", 0),
                        "upper":     vals.get("upper_bound_gbp", 0),
                    })
                else:
                    rows.append({
                        "month":     month_str,
                        "predicted": float(vals),
                        "lower":     float(vals) * 0.9,
                        "upper":     float(vals) * 1.1,
                    })

            if rows:
                fcast_df_ml = pd.DataFrame(rows).sort_values("month")
                hist_raw = sales.get_monthly_revenue()
                if isinstance(hist_raw, dict) and "error" not in hist_raw:
                    hist_df_ml = (
                        pd.DataFrame(list(hist_raw.items()), columns=["month", "revenue"])
                        .sort_values("month")
                        .tail(12)
                    )

                    fig_prophet = go.Figure()
                    fig_prophet.add_trace(go.Scatter(
                        x=hist_df_ml["month"], y=hist_df_ml["revenue"],
                        mode="lines",
                        name="Historical",
                        line=dict(color="#7C3AED", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(124,58,237,0.06)",
                        hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Historical</extra>",
                    ))

                    bridge_ml = hist_df_ml.iloc[[-1]].copy()
                    fcast_x_all = pd.concat([bridge_ml["month"], fcast_df_ml["month"]])
                    fcast_y_all = pd.concat([bridge_ml["revenue"], fcast_df_ml["predicted"]])
                    fcast_lo    = pd.concat([bridge_ml["revenue"], fcast_df_ml["lower"]])
                    fcast_hi    = pd.concat([bridge_ml["revenue"], fcast_df_ml["upper"]])

                    fig_prophet.add_trace(go.Scatter(
                        x=pd.concat([fcast_x_all, fcast_x_all.iloc[::-1]]),
                        y=pd.concat([fcast_hi, fcast_lo.iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(109,40,217,0.10)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="95% CI",
                        hoverinfo="skip",
                    ))

                    fig_prophet.add_trace(go.Scatter(
                        x=fcast_x_all, y=fcast_y_all,
                        mode="lines+markers",
                        name="Prophet Forecast",
                        line=dict(color="#6D28D9", width=2.5, dash="dash"),
                        marker=dict(size=8, color="#6D28D9"),
                        hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra>Forecast</extra>",
                    ))

                    fig_prophet.update_layout(
                        **{**CHART_BASE, "showlegend": True},
                        height=360,
                        legend=dict(
                            orientation="h", x=0, y=1.12,
                            font=dict(size=11, color="#8b949e"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=10), title=""),
                        yaxis=dict(gridcolor="#E5E7EB", tickprefix="£", tickfont=dict(size=10), title=""),
                        title=dict(
                            text=f"{forecast_data.get('model', 'Prophet')} · Shaded = 95% confidence interval",
                            font=dict(size=11, color="#8b949e"),
                        ),
                    )
                    st.plotly_chart(fig_prophet, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})
        else:
            st.info(forecast_data.get("error", "Forecast unavailable."), icon="ℹ️")

        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown('<div class="chart-title">ML Churn Probability Distribution</div>', unsafe_allow_html=True)
            if "error" not in ml_churn_data and ml_churn_data.get("high_risk_customers"):
                hr_df = pd.DataFrame(ml_churn_data["high_risk_customers"])

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=hr_df["churn_probability"],
                    nbinsx=15,
                    marker=dict(color="#F87171", line=dict(color="#FFFFFF", width=0.5)),
                    name="Churn Probability",
                    hovertemplate="Score: %{x:.0f}%<br>Customers: %{y}<extra></extra>",
                ))
                fig_hist.update_layout(
                    **CHART_BASE,
                    height=260,
                    xaxis=dict(gridcolor="#E5E7EB", title="Churn Probability (%)", tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#E5E7EB", title="# Customers", tickfont=dict(size=10)),
                    title=dict(
                        text=f"Top {len(hr_df)} at-risk · Random Forest · "
                             f"Accuracy {ml_churn_data.get('model_metadata', {}).get('accuracy', '?')}%",
                        font=dict(size=11, color="#8b949e"),
                    ),
                )
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

                fi = ml_churn_data.get("feature_importances", [])
                if fi:
                    fi_df = pd.DataFrame(fi).sort_values("importance_pct")
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["importance_pct"],
                        y=fi_df["feature"],
                        orientation="h",
                        marker=dict(
                            color=fi_df["importance_pct"],
                            colorscale=[[0, "#FEE2E2"], [1, "#EF4444"]],
                            line=dict(color="rgba(0,0,0,0)"),
                        ),
                        text=[f"{v:.1f}%" for v in fi_df["importance_pct"]],
                        textposition="outside",
                        textfont=dict(size=10, color="#374151"),
                        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
                    ))
                    fig_fi.update_layout(
                        **CHART_BASE,
                        height=210,
                        xaxis=dict(gridcolor="#E5E7EB", ticksuffix="%", tickfont=dict(size=10), title="Importance %"),
                        yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=10), title=""),
                        title=dict(text="Feature Importance", font=dict(size=11, color="#8b949e")),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})
            else:
                st.info(ml_churn_data.get("error", "Churn ML model unavailable"), icon="ℹ️")

        with col_c2:
            st.markdown('<div class="chart-title">Customer Segmentation — RFM KMeans</div>', unsafe_allow_html=True)
            if "error" not in ml_seg_data and ml_seg_data.get("cluster_summary"):
                seg_df = pd.DataFrame(ml_seg_data["cluster_summary"])

                _SEG_COLORS = {
                    "Champions":       "#059669",
                    "Loyal Customers": "#6D28D9",
                    "At-Risk":         "#F59E0B",
                    "Hibernating":     "#DC2626",
                    "Lost":            "#9CA3AF",
                    "New Customers":   "#7C3AED",
                }

                fig_seg = go.Figure()
                for _, row in seg_df.iterrows():
                    color = _SEG_COLORS.get(row["label"], "#8b949e")
                    size  = max(20, min(70, row["avg_monetary_gbp"] / 80))
                    fig_seg.add_trace(go.Scatter(
                        x=[row["avg_recency_days"]],
                        y=[row["avg_frequency"]],
                        mode="markers+text",
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=0.85,
                            line=dict(color="#FFFFFF", width=1.5),
                        ),
                        text=[row["label"]],
                        textposition="top center",
                        textfont=dict(size=10, color="#1E1B4B"),
                        name=row["label"],
                        hovertemplate=(
                            f"<b>{row['label']}</b><br>"
                            f"Customers: {row['customer_count']:,} ({row['pct_of_total']}%)<br>"
                            f"Avg Recency: {row['avg_recency_days']:.0f} days<br>"
                            f"Avg Frequency: {row['avg_frequency']:.1f} orders<br>"
                            f"Avg Spend: £{row['avg_monetary_gbp']:,.0f}"
                            "<extra></extra>"
                        ),
                    ))

                sil = ml_seg_data.get("model_metadata", {}).get("silhouette_score", 0)
                n_cl = ml_seg_data.get("model_metadata", {}).get("n_clusters", 4)
                fig_seg.update_layout(
                    **{**CHART_BASE, "showlegend": False},
                    height=380,
                    xaxis=dict(gridcolor="#E5E7EB", title="Avg Recency (days — lower = more recent)", tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#E5E7EB", title="Avg Order Frequency", tickfont=dict(size=10)),
                    title=dict(
                        text=f"KMeans RFM · {n_cl} clusters · Silhouette score: {sil:.3f}  (bubble size ∝ avg spend)",
                        font=dict(size=11, color="#8b949e"),
                    ),
                )
                st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","zoomIn2d","zoomOut2d"], "displaylogo": False})

                st.markdown('<div class="chart-title">Segment Summary</div>', unsafe_allow_html=True)
                display_seg = seg_df[["label", "customer_count", "pct_of_total",
                                      "avg_recency_days", "avg_frequency", "avg_monetary_gbp"]].rename(columns={
                    "label":            "Segment",
                    "customer_count":   "Customers",
                    "pct_of_total":     "% of Base",
                    "avg_recency_days": "Avg Recency (days)",
                    "avg_frequency":    "Avg Orders",
                    "avg_monetary_gbp": "Avg Spend (£)",
                })
                st.dataframe(
                    display_seg.style.format({
                        "% of Base":          "{:.1f}%",
                        "Avg Recency (days)":  "{:.0f}",
                        "Avg Orders":          "{:.1f}",
                        "Avg Spend (£)":       "£{:,.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(ml_seg_data.get("error", "Segmentation model unavailable"), icon="ℹ️")
