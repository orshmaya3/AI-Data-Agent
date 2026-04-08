import os
from dotenv import load_dotenv
from openai import OpenAI

# ייבוא ספריות LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

# ייבוא הסוכנים שלנו
from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, df):
        self.df = df
        # הקליינט הישן עבור מחלקת הלקוחות
        self.ai_client = OpenAI() 
        
        # --- הגדרה משותפת ל-LangChain ---
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # הורדת תבנית הנחיות (Prompt) מוכנה של מנהל מסד הנתונים של LangChain
        prompt = hub.pull("hwchase17/openai-functions-agent")

        # ==========================================
        # 1. הגדרות LangChain למחלקת מכירות (Sales)
        # ==========================================
        self.sales_analyst = SalesAnalyst(df)
        
        self.sales_tools = [
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
            self.sales_analyst.get_repeat_customers_stats
        ]
        
        sales_agent = create_openai_functions_agent(self.llm, self.sales_tools, prompt)
        self.sales_executor = AgentExecutor(agent=sales_agent, tools=self.sales_tools, verbose=True)

        # ==========================================
        # 2. הגדרות LangChain למחלקת מוצרים (Products)
        # ==========================================
        self.product_analyst = ProductAnalyst(df)
        
        self.product_tools = [
            self.product_analyst.get_total_products_sold,
            self.product_analyst.get_product_revenue,
            self.product_analyst.get_average_price_per_product,
            self.product_analyst.get_product_sales_trend,
            self.product_analyst.get_top_products_by_revenue,
            self.product_analyst.get_top_products_by_quantity,
            self.product_analyst.get_low_stock_indicator,
            self.product_analyst.get_product_conversion_rate,
            self.product_analyst.get_product_return_rate,
            self.product_analyst.get_product_revenue_share,
            self.product_analyst.get_product_growth_rate,
            self.product_analyst.get_product_popularity_score,
            self.product_analyst.get_product_profit_estimate,
            self.product_analyst.get_product_purchase_frequency,
            self.product_analyst.get_product_lifecycle_status
        ]
        
        product_agent = create_openai_functions_agent(self.llm, self.product_tools, prompt)
        self.product_executor = AgentExecutor(agent=product_agent, tools=self.product_tools, verbose=True)

    def _translate_to_command(self, user_text):
        """
        המוח של המנהל - מסווג את המחלקה (Sales/Product/Customer)
        """
        system_prompt = """
        You are the translation brain of a data system. Map the question to ONE command.
        
        SALES COMMANDS:
        - 'total_revenue' (total sales, money earned)
        - 'total_orders' (how many orders/invoices)
        - 'total_items_sold' (quantity of items sold)
        - 'average_order_value' (AOV, average spend)
        - 'top_countries_revenue' (best countries by money)
        - 'monthly_revenue' (sales by month, revenue over time)
        - 'top_products_revenue' (best selling products by money)
        - 'refund_rate' (percentage of returns/refunds)
        - 'busiest_days' (which days have most sales)
        - 'growth_rate' (monthly growth, MOM)
        - 'pareto_analysis' (80/20 rule, top products contributing to 80 percent revenue)
        - 'sales_anomalies' (unusual sales spikes)
        - 'bought_together' (frequently bought together)
        - 'sales_forecast' (prediction for next week)
        - 'sales_trend' (is business up or down, trend, overall direction)
        - 'revenue_drops' (detect drops, significant losses, bad months)

        PRODUCT COMMANDS:
        - 'product_revenue' (how much money a product made, revenue per item)
        - 'product_quantity' (units sold per product, volume)
        - 'product_avg_price' (average selling price of a product)
        - 'product_trend' (is a product selling better or worse, product growth)
        - 'product_return_rate' (which products are returned most, refund rate per item)
        - 'product_share' (contribution of a product to total sales, revenue share)
        - 'product_popularity' (most popular items by score, weighted popularity)
        - 'product_frequency' (how often is a product in a basket, purchase frequency)
        - 'product_lifecycle' (is a product growing, stable or declining, lifecycle status)
        - 'top_product' (most popular item overall)
        - 'total_unique_products' (variety of items)

        CUSTOMER COMMANDS:
        - 'total_unique_customers' (how many total distinct customers)
        - 'top_customer' (who spent the most)
        - 'top_spending_customers' (list of top spenders)
        - 'top_country' (where most orders come from)
        - 'revenue_by_country' (revenue generated per country)
        - 'most_popular_product_customer' (product that sold the most units)
        - 'repeat_customer_rate' (percentage of repeat customers)
        - 'repeat_customers' (loyalty, returning buyers, how many come back stats)
        - 'best_selling_product_per_country' (top product for each country)
        - 'high_value_loyal_customers' (VIP loyal customers)
        - 'customer_average_item_price' (average item price in store)

        Reply ONLY with the exact command word. If unsure, reply 'unknown'.
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.0 
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"[Manager Agent] ❌ AI Error: {e}")
            return "unknown"

    def handle_request(self, user_text):
        print(f"\n[Manager Agent] 🧠 Analyzing request: '{user_text}'")
        request_type = self._translate_to_command(user_text)
        print(f"[Manager Agent] 🎯 AI determined command: '{request_type}'")
        
        df = self.df
        if df is None:
            return "Sorry, I couldn't get the data from the Data Agent."
        
        # --- הגדרת מחלקות ---
        sales_commands = [
            "total_revenue", "total_orders", "total_items_sold", "average_order_value",
            "top_countries_revenue", "monthly_revenue", "top_products_revenue", 
            "refund_rate", "busiest_days", "growth_rate", "pareto_analysis", 
            "sales_anomalies", "bought_together", "sales_forecast", "sales_trend", "revenue_drops"
        ]
        
        product_commands = [
            "product_revenue", "product_quantity", "product_avg_price", "product_trend",
            "product_return_rate", "product_share", "product_popularity", 
            "product_frequency", "product_lifecycle", "top_product", "total_unique_products"
        ]
        
        customer_commands = [
            "total_unique_customers", "top_customer", "top_spending_customers", 
            "top_country", "revenue_by_country", "most_popular_product_customer", 
            "repeat_customer_rate", "repeat_customers", "best_selling_product_per_country", 
            "high_value_loyal_customers", "customer_average_item_price"
        ]

        # 1. מחלקת מכירות (Sales) - מנוהל באופן אוטונומי על ידי LangChain
        if request_type in sales_commands:
            print("[Manager Agent] 🚀 Handing over to LangChain Autonomous Sales Agent...")
            try:
                response = self.sales_executor.invoke({"input": user_text})
                return response["output"]
            except Exception as e:
                return f"❌ LangChain Error: {e}"

        # 2. מחלקת מוצרים (Products) - מנוהל כעת באופן אוטונומי על ידי LangChain
        elif request_type in product_commands:
            print("[Manager Agent] 🚀 Handing over to LangChain Autonomous Product Agent...")
            try:
                # ה-AI בוחר את הפונקציות (כולל שליפת שם המוצר מהמשפט) ומרכיב תשובה
                response = self.product_executor.invoke({"input": user_text})
                return response["output"]
            except Exception as e:
                return f"❌ LangChain Error: {e}"

        # 3. מחלקת לקוחות (Customers) - פועל במצב קלאסי (if/elif)
        elif request_type in customer_commands:
            print("[Manager Agent] 📞 Routing to Customer Analyst...")
            analyst = CustomerAnalyst(df)
            
            if request_type == "total_unique_customers":
                res = analyst.get_total_unique_customers()
                return f"👥 We have {res} unique customers."
            elif request_type == "top_customer":
                res = analyst.get_top_customer()
                return f"🥇 Our top customer ID is: {res}."
            elif request_type == "top_spending_customers":
                res = analyst.get_top_spending_customers()
                return f"💰 Top spending customers: {res}"
            elif request_type == "top_country":
                res = analyst.get_top_country()
                return f"📍 Most orders come from: {res}."
            elif request_type == "revenue_by_country":
                res = analyst.get_revenue_by_country()
                return f"🌍 Revenue by Country: {res}"
            elif request_type == "most_popular_product_customer":
                res = analyst.get_most_popular_product()
                return f"🛍️ The most popular product by units sold is: {res}."
            elif request_type == "repeat_customer_rate":
                res = analyst.get_repeat_customer_rate()
                return f"🔄 Repeat Customer Rate: {res}%."
            elif request_type == "repeat_customers":
                res = analyst.get_repeat_customers_stats()
                return (f"🔄 Customer Loyalty Stats:\n"
                        f"- Repeat Customers: {res['repeat_customers_count']}\n"
                        f"- Retention Rate: {res['repeat_customers_percentage']}\n"
                        f"- Total Unique Customers: {res['total_unique_customers']}")
            elif request_type == "best_selling_product_per_country":
                res = analyst.get_best_selling_product_per_country()
                return f"🗺️ Best selling products per country: {res}"
            elif request_type == "high_value_loyal_customers":
                res = analyst.get_high_value_loyal_customers()
                return f"💎 VIP Loyal Customers (IDs): {res}"
            elif request_type == "customer_average_item_price":
                res = analyst.get_average_item_price()
                return f"🏷️ The average item price is ${res}."

        return "I'm not sure how to handle that request yet. Try asking about revenue, growth, or top products!"