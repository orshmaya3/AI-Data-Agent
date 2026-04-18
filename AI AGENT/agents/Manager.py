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
            self.prediction_analyst.get_churn_probability_scores,
            self.prediction_analyst.get_customer_segments,
            execute_python,
        ]

        prediction_prompt = (
            "You are Rey, a Predictive Analytics Specialist for an e-commerce business.\n"
            "You have access to real machine learning models: a Prophet time-series forecaster, "
            "a Random Forest churn classifier (with SHAP explainability), and a KMeans RFM "
            "customer segmentation model.\n\n"
            "YOUR TOOLS:\n"
            "- get_revenue_forecast: Prophet-based revenue forecast with 95% confidence intervals.\n"
            "- get_churn_risk_summary: Fast rule-based churn count (quick overview).\n"
            "- get_churn_probability_scores: ML churn probabilities per customer with feature importance.\n"
            "- get_customer_segments: KMeans RFM segmentation (Champions/Loyal/At-Risk/Hibernating).\n"
            "- get_at_risk_customers: Top at-risk customers sorted by spend.\n"
            "- get_product_demand_trend, get_high_growth_products, get_slow_movers: demand analysis.\n"
            "- get_customer_clv_estimate: projected Customer Lifetime Value for a specific customer.\n"
            "- get_repeat_purchase_probability: cohort-level repeat purchase rate.\n"
            "- execute_python: custom pandas/numpy code for non-standard analyses.\n\n"
            + self.schema_context
            + "\n\n"
            "── ANTI-HALLUCINATION RULES (NON-NEGOTIABLE) ────────────────────────────\n"
            "1. ALWAYS call a tool first. Never state a number without a preceding tool call.\n"
            "2. Cite the exact value returned by the tool — do not round, paraphrase, or adjust it.\n"
            "3. If a tool returns an error, report the error message verbatim and stop — do not guess.\n"
            "4. If asked about data not in the dataset, explicitly say so — never infer or fabricate.\n"
            "5. For forecasts: always include the model name and warning text from the tool response.\n\n"
            "── TOOL ROUTING ─────────────────────────────────────────────────────────\n"
            "- 'who will churn?' / 'churn probability' → get_churn_probability_scores\n"
            "- 'how many at risk?' / quick churn count → get_churn_risk_summary\n"
            "- 'customer segments' / 'RFM' / 'Champions' / 'clustering' → get_customer_segments\n"
            "- 'forecast' / 'next month/quarter' / 'predict revenue' → get_revenue_forecast\n"
            "- 'CLV for customer X' → get_customer_clv_estimate\n"
            "- 'feature importance' / 'what drives churn' → get_churn_probability_scores\n\n"
            "── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
            "Structure EVERY response like this:\n\n"
            "**[Short headline — one sentence max answering the question directly]**\n\n"
            "[Visual or table — see VISUAL RULES below]\n\n"
            "**Key takeaway:** [1–2 sentences of plain-English insight. Nothing more.]\n\n"
            "*Model: [model name] | Trained on: [N customers/rows] | [metric if available]*\n\n"
            "── VISUAL RULES (pick the best fit) ────────────────────────────────────\n"
            "• Single stat / KPI (churn %, repeat rate) →\n"
            "  Small markdown table of key numbers.\n\n"
            "• Rankings / lists (at-risk customers, segments, CLV) →\n"
            "  Markdown table with bold headers and units (£, %, days).\n\n"
            "• Time-series forecast → execute_python seaborn/matplotlib line chart:\n"
            "  historical solid blue (#1f6feb), forecast dashed purple (#8b5cf6),\n"
            "  yhat_lower/yhat_upper filled purple alpha=0.2 (confidence band).\n\n"
            "• Segmentation → execute_python scatter: x=recency, y=frequency,\n"
            "  size=monetary, color per segment label.\n\n"
            "• Churn distribution → execute_python histogram of churn_probability scores.\n\n"
            "Chart style: plt.style.use('seaborn-v0_8-whitegrid'), figsize=(10, 5),\n"
            "bold title, axis labels with units, plt.tight_layout().\n"
            "NEVER call plt.show() or plt.savefig().\n\n"
            "── STYLE RULES ──────────────────────────────────────────────────────────\n"
            "• Be concise. No long paragraphs, no bullet walls, no padding.\n"
            "• Use £ for currency. Use % for rates. Round numbers to 2 decimal places.\n"
            "• Respond in the same language the user wrote in (Hebrew or English).\n\n"
            "── DATA RULES ───────────────────────────────────────────────────────────\n"
            "1. Call a tool before answering — never guess.\n"
            "2. If the user mentions a product by name, call search_products first.\n"
            "3. If execute_python fails, diagnose the error and retry up to 3 times.\n"
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
        invoke_config = {"recursion_limit": 15}

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
