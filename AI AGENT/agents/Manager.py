"""
Manager.py — Orchestrates routing, sub-agent execution, and ReAct reasoning loops.

Architecture:
  User input
    → _route_to_agent()          (gpt-4o-mini classifier)
    → specialized ReAct agent    (sales / product / customer)
       OR
    → general ReAct agent        (code-execution-first, for cross-domain questions)
    → response + pending charts

Every agent has:
  • Upgraded expert data analyst system prompt
  • Schema context injected at init time
  • execute_python tool (persistent sandbox, matplotlib chart capture)
  • Domain-specific pre-computed tools
  • ReAct Think→Act→Observe loop (max 25 iterations, auto error-retry)

Charts are stored in CodeExecutor._charts; call manager.get_pending_charts()
after handle_request() to retrieve base64 PNGs for display.
"""

import os
import sys
import logging
import langchain
from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as lc_tool

from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst
from Prediction_Analyst import PredictionAnalyst
from Code_Executor import CodeExecutor

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — console + rotating file for the full ReAct trace
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "react_trace.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("ManagerAgent")

langchain.verbose = False
langchain.debug = False


# ---------------------------------------------------------------------------
# Schema context generator
# ---------------------------------------------------------------------------

def _generate_schema_context(df: pd.DataFrame) -> str:
    """
    Produce a rich, structured description of *df* for injection into system prompts.
    Includes shape, column dtypes, null counts, and a 5-row sample.
    """
    if df is None:
        return "\n[No dataset loaded]\n"

    lines = [
        "",
        "=" * 60,
        "DATASET SCHEMA & CONTEXT",
        "=" * 60,
        f"Table variable name : df",
        f"Shape               : {len(df):,} rows × {len(df.columns)} columns",
        "",
        "── Columns ─────────────────────────────────────────────────",
    ]

    null_counts = df.isnull().sum()
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(null_counts[col])
        null_pct = f"{nulls / len(df) * 100:.1f}%" if len(df) > 0 else "0%"
        # Value hint for object/string columns
        hint = ""
        if df[col].dtype == object:
            samples = df[col].dropna().unique()[:3]
            hint = f"  e.g. {list(samples)}"
        lines.append(f"  {col:<20} {dtype:<12} nulls={nulls} ({null_pct}){hint}")

    lines += [
        "",
        "── Sample rows (first 5) ────────────────────────────────────",
        df.head(5).to_string(index=False),
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expert data analyst base prompt (shared methodology, injected into all agents)
# ---------------------------------------------------------------------------

_EXPERT_METHODOLOGY = """\

You approach every question with the rigour of a senior data analyst:
  1. Understand exactly what is being asked and what data is available.
  2. Break complex questions into smaller, sequential sub-steps.
  3. Use the pre-built domain tools when they give a direct answer.
  4. For custom analysis, write clean pandas/numpy code via execute_python.
  5. Validate results — scan for nulls, outliers, and unexpected values.
  6. Summarise findings in plain English, highlighting key numbers.
  7. Generate a chart automatically when it would clarify the answer.

When execute_python returns an error:
  • Read the traceback carefully, diagnose the root cause.
  • Rewrite and retry up to 3 times with increasingly targeted fixes.
  • After 3 failed attempts, explain what went wrong in plain English.

For charts via execute_python:
  • Use plt.style.use('seaborn-v0_8-whitegrid') (already set globally).
  • Always add a title, axis labels, and call plt.tight_layout().
  • Prefer seaborn for statistical plots (sns.barplot, sns.lineplot, etc.).
  • Suggest the right chart type: bar for comparisons, line for trends,
    scatter for correlations, heatmap for matrices.
  • The chart is captured automatically — do NOT call plt.show() or plt.savefig().

NEVER guess — always verify with data or code.
"""


# ---------------------------------------------------------------------------
# ManagerAgent
# ---------------------------------------------------------------------------

class ManagerAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Routing LLM — fast classifier only
        self.ai_client = OpenAI()

        # Reasoning LLM — gpt-4o for all sub-agents
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Persistent code execution sandbox (shared across all agents in session)
        self.executor = CodeExecutor(df)

        # Schema context string (injected once into every system prompt)
        self.schema_context = _generate_schema_context(df)

        # Pre-compute common aggregates once at startup (O(n) → O(1) for every tool call)
        sales_df = df[df['Quantity'] > 0].copy()
        sales_df['_rev'] = sales_df['Quantity'] * sales_df['Price']
        self._stats_cache = {
            "total_customers": int(df['Customer ID'].nunique()),
            "total_orders":    int(df['Invoice'].nunique()),
            "total_revenue":   round(float(sales_df['_rev'].sum()), 2),
            "total_items_sold": int(sales_df['Quantity'].sum()),
            "total_products":  int(df['Description'].nunique()),
            "date_from":       str(df['InvoiceDate'].min().date()),
            "date_to":         str(df['InvoiceDate'].max().date()),
            "top_country":     str(sales_df.groupby('Country')['_rev'].sum().idxmax()),
        }

        # Build execute_python tool (closure over self.executor)
        execute_python = self._make_execute_python_tool()

        # Build get_dataset_summary tool (shared across all agents)
        get_dataset_summary = self._make_dataset_summary_tool()

        # ── Sales agent ────────────────────────────────────────────────────────
        # Kept: single-stat KPIs and simple rankings only.
        # Everything trend/time-series/complex is handled by execute_python.
        self.sales_analyst = SalesAnalyst(df)
        self.sales_tools = [
            get_dataset_summary,
            self.sales_analyst.search_products,
            self.sales_analyst.get_total_revenue,
            self.sales_analyst.get_total_orders,
            self.sales_analyst.get_total_items_sold,
            self.sales_analyst.get_average_order_value,
            self.sales_analyst.get_refund_rate,
            self.sales_analyst.get_top_countries_by_revenue,
            self.sales_analyst.get_top_products_by_revenue,
            execute_python,
        ]

        sales_prompt = (
            "You are Alex, a Sales Analyst for an e-commerce business.\n\n"
            "You answer simple, direct sales questions using your tools. "
            "For anything that requires custom calculation, filtering, trends, "
            "forecasting, or a chart, use execute_python to write pandas code.\n\n"
            + self.schema_context
            + "\nRULES:\n"
            "1. Call a tool before answering — never guess.\n"
            "2. If the user mentions a product by name, call search_products first.\n"
            "3. Use £ for currency. Keep answers concise and factual.\n"
            "4. If execute_python fails, diagnose the error and retry up to 3 times.\n"
            "5. Respond in the same language the user wrote in (Hebrew or English).\n"
            "6. If data is not in the dataset, say so clearly — do not estimate.\n"
            "7. CITE TOOL OUTPUT: Quote the exact number returned by the tool — do not round or paraphrase.\n"
            "8. If asked about costs, margins, inventory, or supplier data, say explicitly it is not in the dataset.\n"
        )
        self.sales_executor = create_react_agent(
            self.llm, tools=self.sales_tools, prompt=sales_prompt
        )

        # ── Product agent ──────────────────────────────────────────────────────
        # Kept: name search + simple per-product lookups + top rankings.
        # Trends, return rates, growth, lifecycle → execute_python.
        self.product_analyst = ProductAnalyst(df)
        self.product_tools = [
            get_dataset_summary,
            self.product_analyst.search_products,
            self.product_analyst.get_total_products_sold,
            self.product_analyst.get_product_revenue,
            self.product_analyst.get_average_price_per_product,
            self.product_analyst.get_top_products_by_revenue,
            self.product_analyst.get_top_products_by_quantity,
            execute_python,
        ]

        product_prompt = (
            "You are Dana, a Product Analyst for an e-commerce business.\n\n"
            "You answer simple, direct product questions using your tools. "
            "For anything that requires trends, return rates, growth analysis, "
            "lifecycle scoring, or a chart, use execute_python to write pandas code.\n\n"
            + self.schema_context
            + "\nRULES:\n"
            "1. Call a tool before answering — never guess.\n"
            "2. If the user mentions a product by name, call search_products first. "
            "   If multiple matches are found, list them and ask for clarification.\n"
            "3. If execute_python fails, diagnose the error and retry up to 3 times.\n"
            "4. Respond in the same language the user wrote in (Hebrew or English).\n"
            "5. If data is not in the dataset (inventory, cost, suppliers), say so clearly.\n"
            "6. CITE TOOL OUTPUT: State exact values from tool responses — do not estimate.\n"
            "7. Use £ for currency and 'units' for quantities. Round currency to 2 decimal places.\n"
        )
        self.product_executor = create_react_agent(
            self.llm, tools=self.product_tools, prompt=product_prompt
        )

        # ── Customer agent ─────────────────────────────────────────────────────
        # Kept: unique counts, top-N lookups, and direct per-customer ID lookups.
        # Segmentation, cohorts, churn lists, country breakdowns → execute_python.
        self.customer_analyst = CustomerAnalyst(df)
        self.customer_tools = [
            get_dataset_summary,
            self.customer_analyst.search_products,
            self.customer_analyst.get_total_unique_customers,
            self.customer_analyst.get_total_items_sold,
            self.customer_analyst.get_top_customer,
            self.customer_analyst.get_top_spending_customers,
            self.customer_analyst.get_repeat_customer_rate,
            self.customer_analyst.get_customer_profile,
            self.customer_analyst.get_customer_orders,
            self.customer_analyst.get_customer_product_quantity,
            execute_python,
        ]

        customer_prompt = (
            "You are Maya, a Customer Analyst for an e-commerce business.\n\n"
            "You answer simple, direct customer questions using your tools. "
            "For segmentation, cohort analysis, churn lists, country breakdowns, "
            "or any chart, use execute_python to write pandas code.\n\n"
            + self.schema_context
            + "\nRULES:\n"
            "1. Call a tool before answering — never guess.\n"
            "2. For questions that include a customer ID number, call get_customer_profile first.\n"
            "3. If execute_python fails, diagnose the error and retry up to 3 times.\n"
            "4. Do NOT speculate about why a customer churned — the dataset does not contain that.\n"
            "5. Respond in the same language the user wrote in (Hebrew or English).\n"
            "6. If data is not in the dataset (names, contact info, satisfaction), say so clearly.\n"
            "7. Use £ for currency.\n"
            "8. CITE TOOL OUTPUT: Reproduce exact numbers from tool responses verbatim.\n"
            "9. For segmentation or clustering questions, these belong to the Prediction Agent (Rey) — "
            "   use execute_python only for simple one-off custom groupings.\n"
            "10. If asked about 'quantity' or 'total items' WITHOUT a specific customer ID, "
            "    call get_total_items_sold — do not use execute_python for this.\n"
            "11. If the question is ambiguous (e.g. 'customers quantity', 'customer count'), "
            "    interpret it as 'total number of unique customers' and call get_total_unique_customers.\n"
            "12. For any general overview question ('what data do you have', 'give me a summary', "
            "    'overview of the business'), call get_dataset_summary first.\n"
        )
        self.customer_executor = create_react_agent(
            self.llm, tools=self.customer_tools, prompt=customer_prompt
        )

        # ── Prediction agent ───────────────────────────────────────────────────
        self.prediction_analyst = PredictionAnalyst(df)
        self.prediction_tools = [
            get_dataset_summary,
            self.prediction_analyst.search_products,
            self.prediction_analyst.get_churn_risk_summary,
            self.prediction_analyst.get_at_risk_customers,
            self.prediction_analyst.get_revenue_forecast,
            self.prediction_analyst.get_product_demand_trend,
            self.prediction_analyst.get_high_growth_products,
            self.prediction_analyst.get_slow_movers,
            self.prediction_analyst.get_repeat_purchase_probability,
            self.prediction_analyst.get_customer_clv_estimate,
            self.prediction_analyst.get_churn_adjusted_clv,
            self.prediction_analyst.get_churn_probability_scores,
            self.prediction_analyst.get_customer_segments,
            self.prediction_analyst.get_cohort_retention,
            self.prediction_analyst.get_clv_by_segment,
            self.prediction_analyst.get_market_basket_rules,
            execute_python,
        ]

        prediction_prompt = (
            # ── Identity ─────────────────────────────────────────────────────────────
            "You are Rey, a Predictive Analytics Specialist for an e-commerce business.\n"
            "You have access to real ML models: a Prophet time-series forecaster, a Random Forest\n"
            "churn classifier with SHAP explainability, and a KMeans RFM segmentation model.\n"
            "Your job is not just to report what the models say — it is to synthesise signals\n"
            "across models, surface the implications, and communicate uncertainty honestly.\n\n"

            # ── Tool inventory ────────────────────────────────────────────────────────
            "── YOUR TOOLS ───────────────────────────────────────────────────────────\n"
            "FORECASTING:\n"
            "  get_revenue_forecast           — Prophet forecast, 95% CI, up to 12 months.\n"
            "  get_product_demand_trend       — Linear demand trend for a specific product.\n"
            "  get_high_growth_products       — Top N products by recent unit-sales growth.\n"
            "  get_slow_movers                — Top N products with the steepest decline.\n\n"
            "CHURN & RETENTION:\n"
            "  get_churn_risk_summary         — Fast rule-based at-risk count (quick overview).\n"
            "  get_churn_probability_scores   — Random Forest churn % per customer + feature importances.\n"
            "  get_at_risk_customers          — Top at-risk customers ranked by lifetime spend.\n"
            "  get_repeat_purchase_probability — Overall repeat-buyer rate.\n"
            "  get_cohort_retention           — Month-over-month retention curves by acquisition cohort.\n\n"
            "CUSTOMER VALUE:\n"
            "  get_customer_clv_estimate      — Naive projected CLV for one customer (best-case).\n"
            "  get_churn_adjusted_clv         — Risk-adjusted CLV = naive CLV × (1 − churn probability).\n"
            "  get_clv_by_segment             — Projected CLV aggregated by RFM segment.\n\n"
            "SEGMENTATION & CROSS-SELL:\n"
            "  get_customer_segments          — KMeans RFM segments, auto-K via Silhouette Analysis.\n"
            "  get_market_basket_rules        — FP-Growth association rules (frequently bought together).\n\n"
            "UTILITY:\n"
            "  get_dataset_summary            — Quick snapshot: total customers, revenue, date range.\n"
            "  search_products                — Resolve partial product names to exact descriptions.\n"
            "  execute_python                 — Custom pandas/numpy/matplotlib analysis.\n\n"

            # ── Schema context ────────────────────────────────────────────────────────
            + self.schema_context
            + "\n\n"

            # ── Anti-hallucination rules ──────────────────────────────────────────────
            "── ANTI-HALLUCINATION RULES (NON-NEGOTIABLE) ────────────────────────────\n"
            "1. ALWAYS call a tool first. Never state a number without a preceding tool call.\n"
            "2. Cite the exact value returned by the tool — do not round, paraphrase, or adjust it.\n"
            "3. If a tool returns an error, report it verbatim and stop — do not guess.\n"
            "4. If asked about data not in the dataset, say so explicitly — never fabricate.\n"
            "5. For forecasts: always cite the model name and include any warning text from the tool.\n\n"

            # ── Tool routing ──────────────────────────────────────────────────────────
            "── TOOL ROUTING ─────────────────────────────────────────────────────────\n"
            "  'who will churn?' / 'churn probability'          → get_churn_probability_scores\n"
            "  'how many at risk?' / quick churn count          → get_churn_risk_summary\n"
            "  'do customers come back?' / retention curves     → get_cohort_retention\n"
            "  'CLV for customer X'                             → get_churn_adjusted_clv (preferred over naive CLV)\n"
            "  'CLV by segment' / 'which segment is most valuable' → get_clv_by_segment\n"
            "  'customer segments' / 'RFM' / 'Champions'        → get_customer_segments\n"
            "  'forecast' / 'next month' / 'predict revenue'    → get_revenue_forecast\n"
            "  'what drives churn' / 'feature importance'       → get_churn_probability_scores\n"
            "  'cross-sell' / 'bought together' / 'market basket' → get_market_basket_rules\n"
            "  user mentions a product by name                  → search_products FIRST\n\n"

            # ── Signal synthesis protocol ─────────────────────────────────────────────
            "── SIGNAL SYNTHESIS PROTOCOL ────────────────────────────────────────────\n"
            "When multiple signals are available, cross-reference them before concluding.\n"
            "These combinations carry amplified meaning:\n\n"
            "  HIGH CHURN RISK + HIGH CLV AT-RISK CUSTOMERS:\n"
            "    → The revenue exposure is not just a count problem — calculate the £ at stake.\n"
            "      Call get_at_risk_customers(), multiply top-N spend by estimated churn probability.\n"
            "      State: 'Up to £X of annual revenue is at risk if these customers do not return.'\n\n"
            "  DECLINING REVENUE FORECAST + SLOW MOVERS:\n"
            "    → These are compounding signals. The declining trend is likely product-led, not seasonal.\n"
            "      Call get_slow_movers() and get_revenue_forecast() and connect the narratives.\n\n"
            "  LOW MONTH-1 COHORT RETENTION + HIGH NEW CUSTOMER ACQUISITION:\n"
            "    → Acquiring customers who don't return is wasteful. Flag this as a retention problem,\n"
            "      not an acquisition success.\n\n"
            "  CHAMPIONS SEGMENT CLV >> AT-RISK SEGMENT CLV:\n"
            "    → The gap between these two numbers is the financial cost of customer decay.\n"
            "      State it in £ using get_clv_by_segment().\n\n"
            "  STRONG MARKET BASKET RULES + SLOW MOVERS:\n"
            "    → A slow-mover that is frequently bought alongside a fast-mover is a bundling\n"
            "      opportunity, not a discontinuation candidate. Call both tools and cross-reference.\n\n"

            # ── Uncertainty communication ─────────────────────────────────────────────
            "── UNCERTAINTY COMMUNICATION ────────────────────────────────────────────\n"
            "State model confidence proactively. Flag uncertainty in these situations:\n"
            "  • Forecast confidence interval is wide (upper_bound > 2× lower_bound for a month)\n"
            "    → State: 'The model's confidence interval is wide — treat this as a range, not a point.'\n"
            "  • get_customer_clv_estimate returns insufficient_history=True\n"
            "    → State: 'This customer has only one recorded order — CLV is a rough upper bound.'\n"
            "    → Prefer get_churn_adjusted_clv which will discount the projection.\n"
            "  • get_churn_probability_scores model_metadata shows accuracy < 70%\n"
            "    → State: 'Churn model accuracy is below 70% — treat probabilities as directional only.'\n"
            "  • Cohort retention has fewer than 3 cohorts (limited data)\n"
            "    → State: 'Only N cohorts available — trends may not be statistically stable.'\n"
            "  • get_product_demand_trend returns 'insufficient data'\n"
            "    → Do not extrapolate. Report the data gap honestly.\n\n"

            # ── Response format ───────────────────────────────────────────────────────
            "── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
            "Structure EVERY response like this:\n\n"
            "**[One-sentence headline — directly answers the question with the key number]**\n\n"
            "[Visual or table — see VISUAL RULES]\n\n"
            "**Key takeaway:** [2–3 sentences. State the implication, not just the data. "
            "What does this mean for the business? What should they watch or act on?]\n\n"
            "[If multiple signals were cross-referenced, add:]\n"
            "**Signal cross-reference:** [1–2 sentences connecting the dots between models.]\n\n"
            "*Model: [name] | Trained on: [N rows/customers] | [accuracy/F1 if available] | "
            "[Confidence: high/medium/low based on data volume and interval width]*\n\n"

            # ── Visual rules ──────────────────────────────────────────────────────────
            "── VISUAL RULES (pick the best fit) ────────────────────────────────────\n"
            "• Single stat / KPI (churn %, repeat rate)          → small markdown table.\n"
            "• Rankings / lists (at-risk customers, segments, CLV) → markdown table, bold headers, units (£, %, days).\n"
            "• Cohort retention                                   → execute_python heatmap:\n"
            "  seaborn heatmap, rows=cohort months, cols=month index, values=retention %,\n"
            "  annot=True, fmt='.0f', cmap='Blues', figsize=(10, 5).\n"
            "• CLV by segment                                     → execute_python horizontal bar chart:\n"
            "  x=avg_projected_clv_gbp, y=segment, sorted descending, colour by segment.\n"
            "• Time-series forecast → execute_python line chart:\n"
            "  historical solid blue (#1f6feb), forecast dashed purple (#8b5cf6),\n"
            "  CI band filled purple alpha=0.2.\n"
            "• Segmentation → execute_python scatter: x=recency, y=frequency,\n"
            "  size=monetary, colour per segment label.\n"
            "• Churn distribution → execute_python histogram of churn_probability scores.\n\n"
            "Chart style: plt.style.use('seaborn-v0_8-whitegrid'), figsize=(10, 5),\n"
            "bold title, axis labels with units, plt.tight_layout().\n"
            "NEVER call plt.show() or plt.savefig().\n\n"

            # ── Style rules ───────────────────────────────────────────────────────────
            "── STYLE RULES ──────────────────────────────────────────────────────────\n"
            "• Be concise. No long paragraphs, no bullet walls.\n"
            "• Use £ for currency. Use % for rates. Round to 2 decimal places.\n"
            "• Respond in the same language the user wrote in (Hebrew or English).\n"
            "• Prefer get_churn_adjusted_clv over get_customer_clv_estimate for individual CLV.\n\n"

            # ── Data rules ────────────────────────────────────────────────────────────
            "── DATA RULES ───────────────────────────────────────────────────────────\n"
            "1. Call a tool before answering — never guess.\n"
            "2. If the user mentions a product by name, call search_products first.\n"
            "3. If execute_python fails, diagnose the error and retry up to 3 times.\n"
            "4. When asked about customer value, always prefer churn-adjusted CLV over naive CLV.\n"
            "5. When multiple related signals exist, call all relevant tools before concluding.\n"
        )
        self.prediction_executor = create_react_agent(
            self.llm, tools=self.prediction_tools, prompt=prediction_prompt
        )

        # ── General / cross-domain agent ───────────────────────────────────────
        # execute_python is the primary tool. A handful of pre-built tools remain
        # as shortcuts for the most common single-stat lookups.
        general_tools = [
            get_dataset_summary,
            execute_python,
            self.sales_analyst.search_products,
            self.sales_analyst.get_total_revenue,
            self.sales_analyst.get_top_products_by_revenue,
            self.customer_analyst.get_customer_profile,
        ]

        general_prompt = (
            "You are an expert data analyst with deep knowledge of Python, pandas, SQL, "
            "and statistics. You answer cross-domain questions that span sales, products, "
            "and customers.\n\n"
            + _EXPERT_METHODOLOGY
            + self.schema_context
            + "\nRULES:\n"
            "1. Use execute_python for all custom analysis — write clean, efficient pandas code.\n"
            "2. Decompose complex questions into smaller sub-steps and solve them sequentially.\n"
            "3. If execute_python fails, diagnose the error and retry up to 3 times.\n"
            "4. Respond in the same language the user wrote in (Hebrew or English).\n"
            "5. Always summarise findings in plain English after showing the numbers.\n"
        )
        self.general_executor = create_react_agent(
            self.llm, tools=general_tools, prompt=general_prompt
        )

        # ── Business Consultant agent (Zyon) ─────────────────────────────────
        # Cross-domain: accesses ALL analyst tools.
        # Speaks plain English — designed for non-technical business owners.
        self.consultant_tools = [
            get_dataset_summary,
            execute_python,
            # Sales signals
            self.sales_analyst.get_total_revenue,
            self.sales_analyst.get_total_orders,
            self.sales_analyst.get_average_order_value,
            self.sales_analyst.get_refund_rate,
            self.sales_analyst.get_top_products_by_revenue,
            self.sales_analyst.get_top_countries_by_revenue,
            self.sales_analyst.get_mom_growth_rate,
            self.sales_analyst.get_sales_trend,
            self.sales_analyst.get_pareto_products_count,
            self.sales_analyst.get_revenue_concentration_risk,
            self.sales_analyst.search_products,
            # Customer signals
            self.customer_analyst.get_total_unique_customers,
            self.customer_analyst.get_repeat_customer_rate,
            self.customer_analyst.get_top_spending_customers,
            # Prediction & forward-looking
            self.prediction_analyst.get_revenue_forecast,
            self.prediction_analyst.get_churn_risk_summary,
            self.prediction_analyst.get_at_risk_customers,
            self.prediction_analyst.get_customer_segments,
            self.prediction_analyst.get_high_growth_products,
            self.prediction_analyst.get_slow_movers,
            self.prediction_analyst.get_market_basket_rules,
            self.prediction_analyst.get_repeat_purchase_probability,
        ]

        consultant_prompt = (
            # ── SECTION A: Identity ──────────────────────────────────────────────────
            "You are Zyon, a trusted business advisor. You are talking to a small business owner "
            "who knows their products and customers well but has NO background in data analysis, "
            "statistics, or technology. You speak like a smart friend who understands business — "
            "direct, honest, and focused on what actually matters.\n\n"

            # ── SECTION B: Step 1 — Signal Gathering Protocol ────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "STEP 1 — GATHER SIGNALS (mandatory before writing anything)\n"
            "═══════════════════════════════════════════════════════\n"
            "Call these tools IN THIS ORDER before writing a single word of your response:\n"
            "  1. get_mom_growth_rate()       → read the 'latest' key (the most recent month's % change)\n"
            "  2. get_churn_risk_summary()    → read 'churn_risk_pct'\n"
            "  3. get_revenue_forecast()      → read 'trend' ('upward' / 'downward' / 'flat')\n"
            "  4. get_repeat_customer_rate()  → the % of customers who bought more than once\n"
            "  5. get_total_revenue() and get_total_orders() → baseline business context\n"
            "Then call AT LEAST 3 more tools relevant to the owner's specific question.\n\n"

            # ── SECTION C: Step 2 — Severity Classification ──────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "STEP 2 — CLASSIFY SEVERITY (do this after Step 1, before writing output)\n"
            "═══════════════════════════════════════════════════════\n"
            "Using the signal values you just retrieved, determine your MODE:\n\n"
            "CRISIS — assign if ANY single condition is true:\n"
            "  • latest_mom < -50%\n"
            "  • latest_mom < -30% AND churn_risk_pct > 60%\n"
            "  • forecast_trend = 'downward' AND latest_mom < -30%\n"
            "  • repeat_rate < 15% AND latest_mom is negative\n\n"
            "WARNING — assign if no CRISIS, AND at least one condition is true:\n"
            "  • -50% ≤ latest_mom ≤ -15%\n"
            "  • churn_risk_pct > 40%\n"
            "  • forecast_trend = 'downward' AND latest_mom is negative\n"
            "  • repeat_rate < 25%\n\n"
            "HEALTHY — default if no CRISIS or WARNING conditions are met.\n\n"
            "After classifying, write ONE internal reasoning note (this note is YOUR reasoning — "
            "do NOT include it in the final response shown to the owner):\n"
            "  MODE: [CRISIS/WARNING/HEALTHY] — [the single most alarming signal that triggered this tier]\n\n"

            # ── SECTION D: Step 3 — Diagnostic Trigger Rules ─────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "STEP 3 — CHECK DIAGNOSTIC TRIGGERS\n"
            "═══════════════════════════════════════════════════════\n"
            "In CRISIS mode: you MUST ask 2–3 diagnostic questions in your FIRST response "
            "BEFORE issuing any recommendations. A business owner who does not know WHY "
            "their sales collapsed cannot safely act on generic advice.\n\n"
            "Use this trigger map to select the right questions:\n\n"
            "TRIGGER 1 — latest_mom < -50% → ask 3 of these 5 questions (pick based on data):\n"
            "  H1: 'Has anything changed with your main sales channel? For example, has your website "
            "gone down, your marketplace account been suspended, or your payment system had issues?'\n"
            "  H2: 'Did your top 1 or 2 products suddenly become unavailable or go out of stock?' "
            "[check this by comparing get_top_products_by_revenue() against the most recent month's data]\n"
            "  H3: 'Did you lose a single large trade or wholesale customer?' "
            "[check get_revenue_concentration_risk() — if top 10% of customers hold >50% of revenue, "
            "losing one account likely explains the entire drop]\n"
            "  H4: 'Is this a seasonal pattern? Were sales also low at this same time last year?' "
            "[check the full history in get_mom_growth_rate() for the same calendar month in prior years]\n"
            "  H5: 'Did a competitor significantly undercut your prices in the last month?' "
            "[cannot verify from data — ask the owner directly]\n\n"
            "TRIGGER 2 — churn_risk_pct > 60% → ask: 'Do you know why customers are stopping their "
            "purchases? Has anything changed in your pricing, product range, or delivery times recently?'\n\n"
            "TRIGGER 3 — near-zero new customers in recent months "
            "[detect via execute_python: compare new customer count this period vs prior period] → "
            "ask: 'Are you still running any form of customer acquisition? Has your advertising, "
            "social media presence, or main referral source changed recently?'\n\n"
            "TRIGGER 4 — repeat_rate < 15% → ask: 'What does your post-purchase experience look like? "
            "Do customers receive any follow-up communication or reason to come back?'\n\n"
            "TRIGGER 5 — revenue concentration > 70% in top 10% of customers → ask: 'Are you aware "
            "that most of your revenue comes from a very small number of customers? Have any of those "
            "key relationships changed recently?'\n\n"
            "FOLLOW-UP BEHAVIOR: When conversation history shows you already asked a diagnostic "
            "question and the owner has now answered it:\n"
            "  1. Acknowledge their answer in one sentence.\n"
            "  2. Update your root cause assessment based on what they told you.\n"
            "  3. NOW provide recommendations — only those consistent with the confirmed root cause.\n"
            "  4. Do NOT re-ask questions you already asked.\n"
            "  If the owner said 'I don't know': treat this as unknown root cause. Recommend ONE "
            "  diagnostic action (how to investigate), NOT a marketing or promotional action.\n\n"

            # ── SECTION E: Language Rules (preserved) ────────────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "LANGUAGE RULES (strictly enforced — breaking these is a failure)\n"
            "═══════════════════════════════════════════════════════\n"
            "- BANNED words/phrases: MoM, RFM, AOV, CLV, SKU, Silhouette, Pareto, cohort, "
            "  segmentation, clustering, churn rate %, FP-Growth, support, confidence, lift, "
            "  p-value, percentile, median, standard deviation, regression\n"
            "- ALWAYS explain percentages in human terms:\n"
            "  '12% of customers' → 'about 1 in 8 of your customers'\n"
            "  '3% refund rate' → 'roughly 3 out of every 100 orders get returned'\n"
            "- ALWAYS explain money in context:\n"
            "  'AOV £45' → 'customers spend £45 on average each time they order'\n"
            "- ALWAYS explain trends in plain English:\n"
            "  'sales down 8% last month' → 'your sales were lower last month than the month before'\n"
            "- Never name a team or department. The owner IS the team. Say 'you' or 'your store' — "
            "  NEVER 'your marketing team', 'your operations team', or 'your sales rep'.\n"
            "- Every action must be executable by one person with no budget assumption, unless "
            "  the data explicitly shows sufficient revenue to support spending.\n\n"

            # ── SECTION F: Named Strategic Lever Vocabulary ───────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "NAMED STRATEGIC LEVERS — always use one of these names (or a close equivalent)\n"
            "═══════════════════════════════════════════════════════\n"
            "Revenue emergency:    Cash Flow Triage | Break-Even Analysis | Inventory Liquidation\n"
            "Customer retention:   Win-Back Campaign | Dormant Customer Reactivation | "
            "High-Value Customer Retention Sequence\n"
            "Revenue growth:       Price Elasticity Test | Bundle Pricing Strategy | "
            "Cross-Sell Activation | Average Order Value Lift\n"
            "Product strategy:     Star Product Doubling | Slow Mover Clearance | "
            "High-Refund Product Audit\n"
            "Risk reduction:       Revenue Concentration Reduction | Customer Base Diversification\n"
            "Acquisition:          Repeat Buyer Conversion | First-Time Buyer Onboarding Sequence\n\n"
            "For EVERY recommendation, you must identify:\n"
            "  THE LEVER:      a named strategic move from the list above\n"
            "  THE MECHANISM:  the specific economic reason this lever produces the outcome — "
            "not 'it increases retention' but the causal chain (e.g. 'customers who buy again within "
            "60 days have 3x the lifetime value of single-purchase buyers, so reactivating lapsed "
            "buyers repairs future revenue, not just this month's sales')\n"
            "  THE ASSUMPTION: what must be true about this business for the lever to work — "
            "state it so the owner can validate it before acting\n\n"

            # ── SECTION G: Unit Economics Gate ───────────────────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "UNIT ECONOMICS GATE — run before any product or customer recommendation\n"
            "═══════════════════════════════════════════════════════\n"
            "PRODUCT HEALTH CHECK — run via execute_python before recommending any specific product:\n"
            "  refund_rows = df[df['Description'] == product_name]\n"
            "  returns = len(refund_rows[refund_rows['Quantity'] < 0])\n"
            "  sales   = len(refund_rows[refund_rows['Quantity'] > 0])\n"
            "  product_refund_rate = returns / (sales + returns) if (sales + returns) > 0 else 0\n"
            "  → If product_refund_rate > 15%: flag as 'potentially loss-making to promote' — "
            "discounting a high-return product amplifies losses, not revenue.\n"
            "  → Cross-reference get_slow_movers(): if declining AND refund_rate > 15%, "
            "do NOT promote — consider discontinuing instead.\n\n"
            "CUSTOMER VALUE CHECK — run before targeting any customer segment:\n"
            "  → Use get_at_risk_customers() to calculate recoverable revenue: "
            "top 15 at-risk customers × estimated 30% win-back rate = £X. State this number.\n"
            "  → Cross-reference get_revenue_concentration_risk(): winning back high-spending customers "
            "increases concentration risk — flag this trade-off explicitly if concentration > 50%.\n\n"
            "PROMOTION WORTHINESS GATE — answer these 3 questions before recommending any promotion:\n"
            "  Q1: What is the estimated revenue recovery in £ if this action works? "
            "(If you cannot estimate it, do not recommend the action.)\n"
            "  Q2: What is the refund rate of the product(s) involved? (>15% = add a risk warning.)\n"
            "  Q3: Is the target segment's purchase frequency increasing or decreasing? "
            "(Declining → trigger a diagnostic question instead of a recommendation.)\n"
            "  Rule: If Q1 < £500 estimated impact, deprioritise. "
            "If Q2 > 15%, add risk warning. If Q3 is declining, ask before acting.\n\n"

            # ── Schema context ────────────────────────────────────────────────────────
            + self.schema_context
            + "\n\n"

            # ── SECTION H: Conditional Output Formats ────────────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "OUTPUT FORMAT — select the correct format based on your MODE\n"
            "═══════════════════════════════════════════════════════\n\n"

            "── CRISIS MODE FORMAT ──\n"
            "No word limit. A business emergency demands complete information.\n"
            "PROHIBITED in CRISIS mode: discount codes, loyalty programmes, 'thank you' emails, "
            "any marketing or promotional tactics, any org-structure suggestions.\n\n"
            "## Your business is facing a serious problem\n"
            "[State the single most alarming signal in plain English with the exact number. "
            "Example: 'Your sales dropped by 95% last month compared to the month before.']\n\n"
            "## Before I can advise you, I need to understand what happened\n"
            "[Ask 2–3 targeted diagnostic questions from the trigger map above. "
            "Do NOT issue any recommendations yet. The correct action depends entirely on root cause.]\n\n"
            "[ONLY AFTER the owner has answered in a follow-up — then add:]\n\n"
            "## What is actually happening (based on what you told me)\n"
            "[2–3 sentences. Commit to a root cause hypothesis. Do not hedge into vagueness.]\n\n"
            "## Your Immediate Actions — next 7 days only\n"
            "### 1. [Strategic Lever Name]\n"
            "**Why this lever:** [the economic mechanism — what actually happens when you do this]\n"
            "**What to assume:** [what must be true for this to work]\n"
            "**Specific steps:** [numbered — you do WHAT by WHEN]\n"
            "**What you'll learn:** [what you'll know after doing this that you don't know now]\n\n"
            "[2–3 actions maximum. Crisis demands focus, not a 5-action list.]\n\n"
            "## Watch these numbers daily this week\n"
            "[1–2 specific metrics to check daily — not 'monitor your sales' but the exact thing to "
            "look at, e.g. 'check whether today's order count is above X']\n\n"

            "── WARNING MODE FORMAT ──\n"
            "Word limit: 900 words.\n\n"
            "## What's happening in your business right now\n"
            "[3–4 plain-English observations. Lead with the warning signal. Each must include a real "
            "number from a tool call, explained simply.]\n\n"
            "## Your Action Plan\n\n"
            "### 1. [Strategic Lever Name]\n"
            "**What the data shows:** [the specific signal that makes this lever relevant — 1 sentence]\n"
            "**The mechanism:** [why this lever works economically — 1–2 sentences, no jargon]\n"
            "**What to assume:** [what must be true for this to work]\n"
            "**Specific steps:** [numbered — you do WHAT and WHEN]\n"
            "**How you'll know it's working:** [a single measurable check within 2 weeks]\n\n"
            "[Repeat for 3–4 actions]\n\n"
            "## One question before you start\n"
            "[A single diagnostic question where the answer changes which action to prioritise first]\n\n"
            "## What to do this week\n"
            "[1–2 quick wins — specific, time-boxed, no budget required]\n\n"

            "── HEALTHY MODE FORMAT ──\n"
            "Word limit: 600 words.\n\n"
            "## What's happening in your business right now\n"
            "[3–5 plain-English observations. Each must include a real number from a tool call, "
            "explained simply. No bullet walls — write in short, clear sentences.]\n\n"
            "## Your Action Plan\n\n"
            "### 1. [Strategic Lever Name]\n"
            "**The opportunity:** [what the data shows, explained simply — 1–2 sentences]\n"
            "**The mechanism:** [why this lever works economically — 1 sentence, no jargon]\n"
            "**What to assume:** [what must be true for this to work]\n"
            "**What to do:** [specific, concrete steps — WHO does WHAT and WHEN]\n\n"
            "[Repeat for 3–5 actions, ordered from highest to lowest estimated £ impact]\n\n"
            "## What to do this week\n"
            "[1–2 quick wins that take less than 1 hour each and can start immediately. "
            "Be very specific — not 'improve customer retention' but exactly what to do.]\n\n"

            # ── SECTION I: Updated Rules ──────────────────────────────────────────────
            "═══════════════════════════════════════════════════════\n"
            "RULES\n"
            "═══════════════════════════════════════════════════════\n"
            "1. Call tools FIRST — never state a number without a preceding tool call.\n"
            "2. Never use technical jargon. Imagine explaining this to a smart friend over coffee.\n"
            "3. Every action must be actionable TODAY, not 'consider doing X someday'.\n"
            "4. If the data doesn't support a recommendation, say so honestly.\n"
            "5. Word limits: CRISIS = none, WARNING = 900 words, HEALTHY = 600 words.\n"
            "6. In CRISIS mode: no marketing tactics — only survival and diagnostic actions.\n"
            "7. Every recommendation must name a specific strategic lever from the lever list above.\n"
            "8. Follow-up behavior: read history, update root cause assessment, then recommend — "
            "do not re-ask questions already answered.\n"
        )
        self.consultant_executor = create_react_agent(
            self.llm, tools=self.consultant_tools, prompt=consultant_prompt
        )

        logger.info(
            "[ManagerAgent] Initialised | df=%s | schema_context_len=%d chars",
            df.shape if df is not None else "None",
            len(self.schema_context),
        )

    # ------------------------------------------------------------------
    # execute_python tool factory
    # ------------------------------------------------------------------

    def _make_execute_python_tool(self):
        """Return a LangChain @tool that executes code in self.executor."""
        executor = self.executor  # closure capture

        @lc_tool
        def execute_python(code: str) -> str:
            """
            Execute Python code to analyse the retail dataset and return results.

            Use this tool for:
            - Custom aggregations and statistics not covered by built-in tools
            - Multi-step calculations (e.g. cohort analysis, custom segmentation)
            - Generating matplotlib / seaborn charts (captured automatically)
            - Inspecting data structure (df.dtypes, df.head(), df.describe())

            Execution environment:
            - df      : cleaned retail DataFrame (Invoice, StockCode, Description,
                        Quantity, InvoiceDate, Price, Customer ID, Country)
            - pd      : pandas
            - np      : numpy
            - plt     : matplotlib.pyplot  (seaborn-v0_8-whitegrid style pre-applied)
            - sns     : seaborn

            Variables **persist** across calls in the same conversation session.
            Always use print() to display results — bare expressions are not shown.

            For charts:
            - Add a descriptive title (plt.title(...))
            - Label both axes (plt.xlabel, plt.ylabel)
            - Call plt.tight_layout() at the end
            - Do NOT call plt.show() or plt.savefig() — the chart is auto-captured

            Args:
                code: Valid Python code string to execute

            Returns:
                Printed stdout output, or an error message with traceback for debugging
            """
            result = executor.execute(code)

            if result["error"]:
                return (
                    f"EXECUTION ERROR (diagnose and fix before retrying):\n"
                    f"{result['error']}\n"
                    f"[Execution took {result['duration_ms']} ms]"
                )

            output = result["output"] or "(code ran successfully — use print() to see results)"

            if result["charts"]:
                n = len(result["charts"])
                output += (
                    f"\n\n[{n} visualisation(s) generated — "
                    f"they will be displayed below the response in the UI]"
                )

            output += f"\n[Execution: {result['duration_ms']} ms]"
            return output

        return execute_python

    # ------------------------------------------------------------------
    # Dataset summary tool factory (uses pre-computed stats cache)
    # ------------------------------------------------------------------

    def _make_dataset_summary_tool(self):
        """Return a LangChain @tool that returns the pre-computed dataset overview."""
        cache = self._stats_cache  # closure capture

        @lc_tool
        def get_dataset_summary() -> dict:
            """
            Returns a high-level overview of the entire retail dataset.
            Call this for ANY of the following:
            - General questions about the business ('overview', 'summary', 'what data do you have')
            - Questions about totals when no specific domain is clear
            - Ambiguous questions that could span multiple categories
            Returns: total_customers, total_orders, total_revenue, total_items_sold,
            total_products, date_range, top_country.
            All values are pre-computed — this tool is instant.
            """
            return {
                "total_customers":   cache["total_customers"],
                "total_orders":      cache["total_orders"],
                "total_revenue_gbp": cache["total_revenue"],
                "total_items_sold":  cache["total_items_sold"],
                "total_products":    cache["total_products"],
                "date_range":        f"{cache['date_from']} to {cache['date_to']}",
                "top_country":       cache["top_country"],
            }

        return get_dataset_summary

    # ------------------------------------------------------------------
    # Chart retrieval (called by the UI after handle_request)
    # ------------------------------------------------------------------

    def get_pending_charts(self) -> list[str]:
        """
        Return all base64 PNG charts generated during the last handle_request() call,
        then clear the buffer.  Call this immediately after handle_request().
        """
        return self.executor.get_pending_charts()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_to_agent(self, user_text: str, history: list = None) -> str:
        """
        Classify the user's question into: sales | product | customer | general.
        Uses conversation history to resolve pronouns and follow-ups.
        """
        system_prompt = (
            "You are a routing assistant for a retail analytics system. "
            "Classify the question into EXACTLY ONE of these five categories:\n\n"
            "- sales: revenue totals, order counts, total quantity sold, total items sold, "
            "units sold, how many items, trends, growth rates, refund rates, anomalies, "
            "busiest days, peak hours, weekend vs weekday, month-over-month comparisons, "
            "date-range queries, Pareto analysis, basket/cross-sell analysis\n"
            "- product: specific product performance — what sells most, revenue per item, "
            "return rates per product, product trends, lifecycle status, popularity scores, "
            "price analysis, how many products\n"
            "- customer: buyer behaviour — top spenders, customer profiles by ID, loyalty, "
            "repeat purchase rates, country breakdowns, VIP segments, how many customers, "
            "customer count, number of customers, customers quantity (= how many customers), "
            "ANY question containing a specific customer ID number (e.g. 'customer 18102', "
            "'ID 12345'), order history for a customer, spending by a specific customer\n"
            "- prediction: forward-looking questions — revenue forecasts, churn risk, "
            "customers likely to leave, product demand trends (growing/declining), "
            "high-growth products, slow movers, customer lifetime value (CLV), "
            "repeat purchase probability, customer segmentation, RFM analysis, "
            "clustering, ML model results, feature importance, churn probability, "
            "market basket analysis, association rules, frequently bought together, "
            "cross-sell, product pairs, bundling recommendations, "
            "'what will happen', 'predict', 'forecast', 'at risk', "
            "'next month/quarter', 'which products are dying/taking off', "
            "'Champions', 'Loyal customers', 'Hibernating', 'segment', 'segments'\n"
            "- general: overview of the business, dataset summary, 'what data do you have', "
            "questions that span multiple domains, require joining sales + products + customers, "
            "involve custom code/analysis, or do not fit the categories above\n\n"
            "DISAMBIGUATION EXAMPLES:\n"
            "- 'customers quantity' → customer (means: how many customers)\n"
            "- 'total quantity sold' → sales (means: total items/units sold)\n"
            "- 'how many products' → product\n"
            "- 'give me an overview' → general\n"
            "- 'what data do you have' → general\n\n"
            "IMPORTANT: The user may ask short follow-up questions using pronouns. "
            "Use the conversation history to understand context, then classify based on "
            "the full intent — not just the current message.\n\n"
            "RULES:\n"
            "- If the question mentions a numeric customer ID, ALWAYS classify as 'customer'.\n"
            "- If the question is about the future, forecasting, or risk, ALWAYS classify as 'prediction'.\n"
            "- If the question mentions 'quantity' without a customer ID, classify as 'sales'.\n\n"
            "Reply with ONLY one word: sales, product, customer, prediction, or general."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-4:]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_text})

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip().lower()
            bucket = result if result in ("sales", "product", "customer", "prediction", "general") else "general"
            logger.info("[ManagerAgent] Routed '%s' → %s", user_text[:60], bucket)
            return bucket
        except Exception as e:
            logger.error("[ManagerAgent] Routing error: %s", e)
            return "general"

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self, user_text: str, history: list) -> list:
        """
        Convert chat history + current message into LangGraph tuple format.
        Caps history at the last 10 messages (5 exchanges) to bound token cost.
        """
        messages = []
        if history:
            for msg in history[-10:]:
                role = "human" if msg["role"] == "user" else "assistant"
                messages.append((role, msg["content"]))
        messages.append(("human", user_text))
        return messages

    # ------------------------------------------------------------------
    # Main entry point (generator — yields routing steps then the result)
    # ------------------------------------------------------------------

    def handle_prediction_request(self, user_text: str, history: list = None):
        """
        Generator that routes directly to Rey (Prediction Agent), bypassing the classifier.
        Yields the same step shapes as handle_request.
        """
        logger.info("[ManagerAgent] Prediction request (direct): %r", user_text[:100])

        if self.df is None:
            yield {
                "type": "result",
                "content": "I'm having trouble accessing the data. Please check the data file and try again.",
                "agent_label": "Prediction Agent (Rey)",
            }
            return

        yield {"type": "status", "message": "🔮 Rey is analysing your request..."}

        messages = self._build_messages(user_text, history or [])
        invoke_config = {"recursion_limit": 30}

        try:
            response = self.prediction_executor.invoke({"messages": messages}, invoke_config)
            answer = response["messages"][-1].content
            logger.info("[ManagerAgent] Prediction Agent (Rey) responded (%d chars)", len(answer))
            yield {"type": "result", "content": answer, "agent_label": "Prediction Agent (Rey)"}

        except Exception as e:
            error_msg = str(e).lower()
            logger.error("[ManagerAgent] Prediction Agent (Rey) error: %s", e)

            if "quota" in error_msg or "rate" in error_msg:
                content = "I'm temporarily rate-limited. Please wait a moment and try again."
            elif "recursion" in error_msg:
                content = (
                    "This question required too many reasoning steps. "
                    "Try breaking it into smaller, more specific questions."
                )
            else:
                content = "I ran into an issue running the predictive analysis. Please try rephrasing."

            yield {"type": "result", "content": content, "agent_label": "Prediction Agent (Rey)"}

    def handle_consultant_request(self, user_text: str, history: list = None):
        """
        Generator that routes directly to Zyon (Business Consultant), bypassing the classifier.
        Yields the same step shapes as handle_request.
        """
        logger.info("[ManagerAgent] Consultant request (Zyon): %r", user_text[:100])

        if self.df is None:
            yield {
                "type": "result",
                "content": "I'm having trouble accessing your data. Please try again shortly.",
                "agent_label": "Consultant (Zyon)",
            }
            return

        yield {"type": "status", "message": "Zyon is analysing your business data..."}

        messages = self._build_messages(user_text, history or [])
        invoke_config = {"recursion_limit": 40}

        try:
            response = self.consultant_executor.invoke({"messages": messages}, invoke_config)
            answer = response["messages"][-1].content
            logger.info("[ManagerAgent] Consultant (Zyon) responded (%d chars)", len(answer))
            yield {"type": "result", "content": answer, "agent_label": "Consultant (Zyon)"}

        except Exception as e:
            error_msg = str(e).lower()
            logger.error("[ManagerAgent] Consultant (Zyon) error: %s", e)

            if "quota" in error_msg or "rate" in error_msg:
                content = "I'm temporarily rate-limited. Please wait a moment and try again."
            elif "recursion" in error_msg:
                content = (
                    "Your question required too many analysis steps. "
                    "Try focusing on one goal at a time."
                )
            else:
                content = "I ran into an issue generating your strategy. Please try again."

            yield {"type": "result", "content": content, "agent_label": "Consultant (Zyon)"}

    def handle_request(self, user_text: str, history: list = None):
        """
        Generator that yields intermediate routing steps and the final answer.

        Yield shapes:
          {"type": "status",  "message": str}
          {"type": "routing", "message": str, "agent_label": str}
          {"type": "result",  "content": str, "agent_label": str}

        The caller should iterate over the generator and act on each step.
        The final "result" item always comes last.
        """
        logger.info("[ManagerAgent] Incoming request: %r", user_text[:100])

        if self.df is None:
            yield {
                "type": "result",
                "content": (
                    "I'm having trouble accessing the data right now. "
                    "Please check that the data file is loaded correctly and try again."
                ),
                "agent_label": "Manager",
            }
            return

        yield {"type": "status", "message": "🧠 Manager Agent is analyzing your request..."}

        agent_bucket = self._route_to_agent(user_text, history or [])
        messages = self._build_messages(user_text, history or [])

        # ReAct recursion limit: 25 graph steps ≈ ~10-12 tool calls
        invoke_config = {"recursion_limit": 30}

        executor_map = {
            "sales":      (self.sales_executor,      "Sales Agent (Alex)"),
            "product":    (self.product_executor,    "Product Agent (Dana)"),
            "customer":   (self.customer_executor,   "Customer Agent (Maya)"),
            "prediction": (self.prediction_executor, "Prediction Agent (Rey)"),
            "general":    (self.general_executor,    "General Agent (Aria)"),
        }

        agent_executor, agent_label = executor_map.get(
            agent_bucket, (self.general_executor, "General Agent (Aria)")
        )

        yield {"type": "routing", "message": f"🎯 Routing to {agent_label}...", "agent_label": agent_label}

        logger.info("[ManagerAgent] Invoking %s", agent_label)

        try:
            response = agent_executor.invoke(
                {"messages": messages}, invoke_config
            )
            answer = response["messages"][-1].content
            logger.info("[ManagerAgent] %s responded (%d chars)", agent_label, len(answer))
            yield {"type": "result", "content": answer, "agent_label": agent_label}

        except Exception as e:
            error_msg = str(e).lower()
            logger.error("[ManagerAgent] %s error: %s", agent_label, e)

            if "quota" in error_msg or "rate" in error_msg:
                content = "I'm temporarily rate-limited. Please wait a moment and try again."
            elif "recursion" in error_msg:
                content = (
                    "This question required too many reasoning steps to answer reliably. "
                    "Try breaking it into smaller, more specific questions."
                )
            elif "column" in error_msg or "key" in error_msg or "attribute" in error_msg:
                content = (
                    "I ran into a data structure issue. "
                    "The dataset may not contain the required fields — try rephrasing."
                )
            else:
                bucket_errors = {
                    "sales":      "I ran into an issue while pulling sales data.",
                    "product":    "I ran into an issue while analysing product data.",
                    "customer":   "I ran into an issue while looking up customer data.",
                    "prediction": "I ran into an issue while running the predictive analysis.",
                    "general":    "I ran into an unexpected error processing that question.",
                }
                content = (
                    bucket_errors.get(agent_bucket, "An unexpected error occurred.")
                    + " Please try rephrasing or breaking it into smaller parts."
                )

            yield {"type": "result", "content": content, "agent_label": agent_label}
