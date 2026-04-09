import os
import langchain
from dotenv import load_dotenv
from openai import OpenAI

langchain.verbose = False
langchain.debug = False

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Our analyst classes
from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, df):
        self.df = df
        # Used for the initial routing step only — mini is sufficient for classification
        self.ai_client = OpenAI()

        # Shared LLM for all ReAct sub-agents — gpt-4o for accurate multi-step reasoning
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # --- Sales agent ---
        self.sales_analyst = SalesAnalyst(df)

        self.sales_tools = [
            self.sales_analyst.search_products,
            self.sales_analyst.get_total_revenue,
            self.sales_analyst.get_total_orders,
            self.sales_analyst.get_total_items_sold,
            self.sales_analyst.get_average_order_value,
            self.sales_analyst.get_top_countries_by_revenue,
            self.sales_analyst.get_monthly_revenue,
            self.sales_analyst.get_top_products_by_revenue,
            self.sales_analyst.get_refund_rate,
            self.sales_analyst.get_revenue_by_date_range,
            self.sales_analyst.get_busiest_days_of_week,
            self.sales_analyst.get_mom_growth_rate,
            self.sales_analyst.get_pareto_products_count,
            self.sales_analyst.get_sales_anomalies,
            self.sales_analyst.get_frequently_bought_together,
            self.sales_analyst.get_simple_sales_forecast,
            self.sales_analyst.get_sales_trend,
            self.sales_analyst.detect_revenue_drops,
            self.sales_analyst.get_repeat_customers_stats,
            self.sales_analyst.get_hourly_sales_distribution,
            self.sales_analyst.get_weekend_vs_weekday_sales,
            self.sales_analyst.get_churn_risk_customers,
            self.sales_analyst.get_revenue_concentration_risk,
            self.sales_analyst.get_average_days_between_purchases,
            self.sales_analyst.get_product_family_revenue,
        ]

        sales_prompt = (
            "You are Alex, a sharp and friendly Sales Analyst for an e-commerce business. "
            "Your job is to answer sales questions using the tools available to you. "
            "RULES: "
            "1. Always call the relevant tool first to get the real data before answering. "
            "2. If the user mentions a product by name, call search_products first to find the exact "
            "product description, then use that exact string in any follow-up tool calls. "
            "3. Present numbers clearly — use bullet points, bold key figures, and add currency symbols (£) where relevant. "
            "4. Only add a business insight if the data clearly supports one. Do NOT speculate or invent benchmarks. "
            "   If the data is ambiguous or incomplete, say so honestly instead of guessing. "
            "5. Keep a warm, professional tone — like a trusted analyst talking to a colleague. "
            "6. Always respond in the same language the user wrote in (Hebrew or English). "
            "7. If a question requires information not available in the dataset (e.g. cost data, profit margins, "
            "   web traffic, marketing spend), clearly state that this data is not available rather than estimating."
        )
        self.sales_executor = create_react_agent(self.llm, tools=self.sales_tools, prompt=sales_prompt)

        # --- Product agent ---
        self.product_analyst = ProductAnalyst(df)

        self.product_tools = [
            self.product_analyst.search_products,
            self.product_analyst.get_total_products_sold,
            self.product_analyst.get_product_revenue,
            self.product_analyst.get_average_price_per_product,
            self.product_analyst.get_product_sales_trend,
            self.product_analyst.get_top_products_by_revenue,
            self.product_analyst.get_top_products_by_quantity,
            self.product_analyst.get_product_return_rate,
            self.product_analyst.get_product_revenue_share,
            self.product_analyst.get_product_growth_rate,
            self.product_analyst.get_product_popularity_score,
            self.product_analyst.get_product_purchase_frequency,
            self.product_analyst.get_product_lifecycle_status
        ]

        product_prompt = (
            "You are Dana, a friendly and detail-oriented Product Analyst for an e-commerce business. "
            "Your job is to answer product-related questions using the tools available to you. "
            "RULES: "
            "1. Always call the relevant tool first to get the real data before answering. "
            "2. If the user mentions a product by name, call search_products first to find the exact "
            "product description, then use that exact string in any follow-up tool calls. "
            "   If multiple matches are found, list them and ask the user to clarify rather than silently picking one. "
            "3. Present product data in a clean, readable format — use bullet points and highlight top performers. "
            "4. Only add a business insight if the data clearly supports one. Do NOT speculate or invent benchmarks. "
            "   If the data is ambiguous or incomplete, say so honestly instead of guessing. "
            "5. Keep a warm, professional tone — like a trusted analyst talking to a colleague. "
            "6. Always respond in the same language the user wrote in (Hebrew or English). "
            "7. If a question requires information not available in the dataset (e.g. inventory, cost, supplier data), "
            "   clearly state that this data is not available rather than estimating."
        )
        self.product_executor = create_react_agent(self.llm, tools=self.product_tools, prompt=product_prompt)

        # --- Customer agent ---
        self.customer_analyst = CustomerAnalyst(df)

        self.customer_tools = [
            self.customer_analyst.search_products,
            self.customer_analyst.get_total_revenue,
            self.customer_analyst.get_total_unique_customers,
            self.customer_analyst.get_top_country,
            self.customer_analyst.get_total_items_sold,
            self.customer_analyst.get_average_item_price,
            self.customer_analyst.get_top_customer,
            self.customer_analyst.get_top_spending_customers,
            self.customer_analyst.get_revenue_by_country,
            self.customer_analyst.get_revenue_by_single_country,
            self.customer_analyst.get_most_popular_product,
            self.customer_analyst.get_refund_rate,
            self.customer_analyst.get_repeat_customer_rate,
            self.customer_analyst.get_best_selling_product_per_country,
            self.customer_analyst.get_average_order_value,
            self.customer_analyst.get_monthly_revenue_trend,
            self.customer_analyst.get_high_value_loyal_customers,
            self.customer_analyst.get_customer_profile,
            self.customer_analyst.get_new_customers_by_month,
            self.customer_analyst.get_churn_risk_customer_list,
        ]

        customer_prompt = (
            "You are Maya, a warm and insightful Customer Analyst for an e-commerce business. "
            "Your job is to answer customer-related questions using the tools available to you. "
            "RULES: "
            "1. Always call the relevant tool first to get the real data before answering. "
            "2. If the user mentions a product by name, call search_products first to find the exact "
            "product description, then use that exact string in any follow-up tool calls. "
            "3. Present customer data clearly — use bullet points, highlight key customer IDs or countries. "
            "4. Only add a behavioral insight if the data clearly supports one. Do NOT speculate about why a customer "
            "   stopped buying, changed habits, or churned — the dataset does not contain that information. "
            "5. Keep a warm, professional tone — like a trusted analyst talking to a colleague. "
            "6. Always respond in the same language the user wrote in (Hebrew or English). "
            "7. If a question requires information not available in the dataset (e.g. customer names, contact info, "
            "   churn reasons, satisfaction scores), clearly state it is unavailable."
        )
        self.customer_executor = create_react_agent(self.llm, tools=self.customer_tools, prompt=customer_prompt)

    def _route_to_agent(self, user_text: str) -> str:
        """Classifies the user's question into one of four buckets: sales, product, customer, or general."""
        system_prompt = (
            "You are a routing assistant for a retail analytics system. "
            "Classify the question into EXACTLY ONE of these four categories:\n\n"
            "- sales: revenue totals, order counts, trends, forecasts, growth rates, refund rates, "
            "anomalies, busiest days, peak hours, weekend vs weekday, month-over-month comparisons, "
            "date-range queries, Pareto analysis, basket/cross-sell analysis\n"
            "- product: specific product performance — what sells most, revenue per item, "
            "return rates per product, product trends, lifecycle status, popularity scores, price analysis\n"
            "- customer: buyer behaviour — top spenders, customer profiles by ID, loyalty, "
            "repeat purchase rates, country breakdowns, VIP segments, churn risk\n"
            "- general: questions that clearly span multiple domains, require joining insights from "
            "sales + products + customers together, or do not fit the above\n\n"
            "Reply with ONLY one word: sales, product, customer, or general."
        )

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().lower()
            return result if result in ("sales", "product", "customer", "general") else "general"
        except Exception as e:
            print(f"[Manager Agent] ❌ Routing Error: {e}")
            return "general"

    def _build_messages(self, user_text: str, history: list) -> list:
        """Converts chat history + current message into LangGraph's tuple message format.
        Caps history at the last 10 messages (5 exchanges) to keep token cost bounded."""
        messages = []
        if history:
            recent = history[-10:]
            for msg in recent:
                role = "human" if msg["role"] == "user" else "assistant"
                messages.append((role, msg["content"]))
        messages.append(("human", user_text))
        return messages

    def handle_request(self, user_text: str, history: list = None) -> str:
        print(f"\n[Manager Agent] 🧠 Analyzing request: '{user_text}'")
        agent_bucket = self._route_to_agent(user_text)
        print(f"[Manager Agent] 🎯 Routing to: '{agent_bucket}' agent")

        if self.df is None:
            return "Hmm, it looks like I'm having trouble accessing the data right now. Please check that the data file is loaded correctly and try again."

        messages = self._build_messages(user_text, history or [])

        if agent_bucket == "sales":
            print("[Manager Agent] 🚀 Handing over to Sales Agent (Alex)...")
            try:
                response = self.sales_executor.invoke({"messages": messages})
                return response["messages"][-1].content
            except Exception:
                return "I ran into an issue while pulling the sales data. Please try rephrasing your question or try again in a moment."

        elif agent_bucket == "product":
            print("[Manager Agent] 🚀 Handing over to Product Agent (Dana)...")
            try:
                response = self.product_executor.invoke({"messages": messages})
                return response["messages"][-1].content
            except Exception:
                return "I ran into an issue while analyzing the product data. Please try rephrasing your question or try again in a moment."

        elif agent_bucket == "customer":
            print("[Manager Agent] 🚀 Handing over to Customer Agent (Maya)...")
            try:
                response = self.customer_executor.invoke({"messages": messages})
                return response["messages"][-1].content
            except Exception:
                return "I ran into an issue while looking up the customer data. Please try rephrasing your question or try again in a moment."

        else:
            # General / cross-domain: let the pandas agent handle it freely
            print("[Manager Agent] 🔀 Cross-domain question — routing to General Agent...")
            try:
                from langchain_experimental.agents import create_pandas_dataframe_agent

                pandas_agent = create_pandas_dataframe_agent(
                    self.llm,
                    self.df,
                    verbose=False,
                    allow_dangerous_code=True
                )

                # Inject recent history as a structured prefix so the agent has context
                context_lines = []
                if history:
                    for msg in (history or [])[-6:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        context_lines.append(f"{role}: {msg['content']}")
                context_lines.append(f"User: {user_text}")
                full_input = "\n".join(context_lines)

                response = pandas_agent.invoke({"input": full_input})
                return response['output']

            except KeyError:
                return "I couldn't compute that — the data doesn't seem to contain the columns needed for this question. Could you rephrase or be more specific?"
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate" in error_msg:
                    return "I'm temporarily rate-limited. Please wait a moment and try again."
                if "column" in error_msg or "key" in error_msg or "attribute" in error_msg:
                    return "I ran into a data structure issue with that question. The dataset may not contain the required fields. Try rephrasing."
                print(f"[Manager Agent] ❌ General agent error: {e}")
                return "I ran into an unexpected error processing that question. Please try rephrasing or breaking it into smaller parts."
