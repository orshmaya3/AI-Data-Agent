import os
from dotenv import load_dotenv
from openai import OpenAI

# ייבוא הסוכנים שלנו
from Data_Agent import DataAgent
from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.ai_client = OpenAI()
    
    def _translate_to_command(self, user_text):
        """
        המוח של המנהל - מתרגם שפה חופשית לפקודות שהסוכנים מבינים
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
        - 'pareto_analysis' (80/20 rule, top products contributing to 80% revenue)
        - 'sales_anomalies' (unusual sales spikes)
        - 'bought_together' (frequently bought together)
        - 'sales_forecast' (prediction for next week)

        PRODUCT COMMANDS:
        - 'top_product' (most popular item)
        - 'total_unique_products' (variety of items)

        CUSTOMER COMMANDS:
        - 'top_customer' (who spent the most)
        - 'top_country' (where most orders come from)

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
        
        # טעינת הנתונים דרך ה-Data Agent
        d_agent = DataAgent(self.data_file_path)
        df = d_agent.get_data()
        
        if df is None:
            return "Sorry, I couldn't get the data from the Data Agent."
        
        # --- לוגיקת ניתוב (Routing) ---
        
        # 1. מחלקת מכירות (Sales)
        sales_commands = [
            "total_revenue", "total_orders", "total_items_sold", "average_order_value",
            "top_countries_revenue", "monthly_revenue", "top_products_revenue", 
            "refund_rate", "busiest_days", "growth_rate", "pareto_analysis", 
            "sales_anomalies", "bought_together", "sales_forecast"
        ]

        if request_type in sales_commands:
            print("[Manager Agent] 📞 Routing to Sales Analyst...")
            analyst = SalesAnalyst(df)
            
            if request_type == "total_revenue":
                res = analyst.get_total_revenue()
                return f"💰 The total revenue is ${res:,.2f}."
            elif request_type == "total_orders":
                res = analyst.get_total_orders()
                return f"📦 Total number of orders: {res}"
            elif request_type == "average_order_value":
                res = analyst.get_average_order_value()
                return f"📊 The Average Order Value (AOV) is ${res:.2f}."
            elif request_type == "total_items_sold":
                res = analyst.get_total_items_sold()
                return f"🛒 Total items sold: {res}"
            elif request_type == "top_countries_revenue":
                res = analyst.get_top_countries_by_revenue()
                return f"🌍 Top countries by revenue: {res}"
            elif request_type == "monthly_revenue":
                res = analyst.get_monthly_revenue()
                return f"📅 Monthly Revenue Breakdown: {res}"
            elif request_type == "top_products_revenue":
                res = analyst.get_top_products_by_revenue()
                return f"🔝 Top products by revenue contribution: {res}"
            elif request_type == "refund_rate":
                res = analyst.get_refund_rate()
                return f"⚠️ Our current refund rate is {res:.2f}%."
            elif request_type == "busiest_days":
                res = analyst.get_busiest_days_of_week()
                return f"📈 Busiest days of the week: {res}"
            elif request_type == "growth_rate":
                res = analyst.get_mom_growth_rate()
                return f"🚀 Month-over-Month Growth: {res:.2f}%"
            elif request_type == "pareto_analysis":
                res = analyst.get_pareto_products_count()
                return f"⚖️ Pareto Analysis: {res} products are responsible for 80% of your total revenue."
            elif request_type == "sales_anomalies":
                res = analyst.get_sales_anomalies()
                return f"🚨 Sales Anomalies Detected: {res if res else 'No major anomalies found.'}"
            elif request_type == "sales_forecast":
                res = analyst.get_simple_sales_forecast()
                return f"🔮 Estimated sales forecast for the next 7 days: ${res:,.2f}"
            elif request_type == "bought_together":
                # לצורך הדוגמה אנחנו שולחים מוצר קבוע, בגרסה הבאה נחלץ את שם המוצר מהשאלה
                res = analyst.get_frequently_bought_together('WHITE HANGING HEART T-LIGHT HOLDER')
                return f"💡 People who bought that also bought: {res}"

        # 2. מחלקת מוצרים (Products)
        elif request_type in ["top_product", "total_unique_products"]:
            print("[Manager Agent] 📞 Routing to Product Analyst...")
            analyst = ProductAnalyst(df)
            if request_type == "top_product":
                res = analyst.get_top_product()
                return f"🏆 Our top selling product is: {res}."
            res = analyst.get_total_unique_products()
            return f"🔢 We have {res} unique products in our catalog."

        # 3. מחלקת לקוחות (Customers)
        elif request_type in ["top_customer", "top_country"]:
            print("[Manager Agent] 📞 Routing to Customer Analyst...")
            analyst = CustomerAnalyst(df)
            if request_type == "top_customer":
                res = analyst.get_top_customer()
                return f"🥇 Our top customer ID is: {res}."
            res = analyst.get_top_country()
            return f"📍 Most orders come from: {res}."

        # במקרה שה-AI לא זיהה פקודה
        return "I'm not sure how to handle that request yet. Try asking about revenue, growth, or top products!"

# סוף הקובץ - אין פקודות הרצה מחוץ ל-Class