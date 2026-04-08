import pandas as pd
import numpy as np

class SalesAnalyst:
    def __init__(self, data_frame):
        # יצירת עותק כדי לא לפגוע בנתונים המקוריים
        self.df = data_frame.copy()
        
        # 1. ניקוי בסיסי וחישוב הכנסה לכל שורה
        self.df['Revenue'] = self.df['Quantity'] * self.df['Price']
        
        # 2. המרת עמודת התאריך לפורמט תאריך חכם של פייתון (אם קיימת)
        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])

    # --- רמה 1: קל (Easy) ---

    def get_total_revenue(self) -> float:
        """Calculates and returns the total revenue from all transactions."""
        return float(self.df['Revenue'].sum())

    def get_total_orders(self) -> int:
        """Calculates the total number of unique orders (invoices) placed."""
        return int(self.df['Invoice'].nunique())

    def get_total_items_sold(self) -> int:
        """Calculates the total quantity of all physical items sold."""
        return int(self.df['Quantity'].sum())

    def get_average_order_value(self) -> float:
        """Calculates the Average Order Value (AOV), which is total revenue divided by total orders."""
        total_rev = self.get_total_revenue()
        total_orders = self.get_total_orders()
        return float(total_rev / total_orders) if total_orders > 0 else 0.0

    def get_top_countries_by_revenue(self, limit: int = 5) -> dict:
        """Gets the top performing countries ranked by total sales revenue. Allows specifying a limit."""
        return self.df.groupby('Country')['Revenue'].sum().nlargest(limit).to_dict()

    # --- רמה 2: בינוני (Medium) ---

    def get_monthly_revenue(self) -> dict:
        """Calculates the total sales revenue grouped by month."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data available"}
        monthly_df = self.df.copy()
        monthly_df['Month'] = monthly_df['InvoiceDate'].dt.to_period('M').astype(str)
        return monthly_df.groupby('Month')['Revenue'].sum().astype(float).to_dict()

    def get_top_products_by_revenue(self, limit: int = 5, country: str = None) -> dict:
        """
        Gets the top-selling products ranked by total revenue generated. 
        Args:
            limit: How many products to return (default 5).
            country: Optional. If the user asks for a specific country, pass the FULL COUNTRY NAME here. 
                     IMPORTANT: Translate abbreviations (e.g., 'FR' -> 'France', 'UK' -> 'United Kingdom').
        """
        data = self.df
        
        # אם ה-AI העביר לנו שם של מדינה, נסנן את הטבלה קודם כל לפי המדינה הזו
        if country:
            data = data[data['Country'].str.lower() == country.lower()]
            if data.empty:
                return {"error": f"No sales data found for country: {country}"}

        # עכשיו נחשב את הטופ מוצרים על הטבלה המסוננת (או המלאה אם לא התבקשה מדינה)
        return data.groupby('Description')['Revenue'].sum().nlargest(limit).to_dict()
    def get_refund_rate(self) -> float:
        """Calculates the percentage of transactions that were refunds or returns (negative quantity)."""
        total_transactions = len(self.df)
        returns = len(self.df[self.df['Quantity'] < 0])
        return float((returns / total_transactions) * 100) if total_transactions > 0 else 0.0

    def get_revenue_by_date_range(self, start_date: str, end_date: str) -> float:
        """Calculates total revenue within a specific date range. Dates must be provided as strings in YYYY-MM-DD format."""
        mask = (self.df['InvoiceDate'] >= start_date) & (self.df['InvoiceDate'] <= end_date)
        filtered_df = self.df.loc[mask]
        return float(filtered_df['Revenue'].sum())

    def get_busiest_days_of_week(self) -> dict:
        """Counts the number of transactions for each day of the week to identify the busiest shopping days."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        days = self.df['InvoiceDate'].dt.day_name()
        return days.value_counts().to_dict()

    # --- רמה 3: קשה (Hard) ---

    def get_mom_growth_rate(self) -> float:
        """Calculates the Month-over-Month (MoM) sales revenue growth rate percentage."""
        if 'InvoiceDate' not in self.df.columns: return 0.0
        monthly_rev = self.df.set_index('InvoiceDate').resample('ME')['Revenue'].sum()
        growth = monthly_rev.pct_change() * 100 
        return float(growth.iloc[-1]) if not pd.isna(growth.iloc[-1]) else 0.0

    def get_pareto_products_count(self) -> int:
        """Calculates how many top products make up 80% of the total revenue (Pareto principle / 80-20 rule)."""
        product_rev = self.df.groupby('Description')['Revenue'].sum().sort_values(ascending=False)
        total_rev = product_rev.sum()
        cumulative_rev = product_rev.cumsum() / total_rev
        top_80_percent_products = cumulative_rev[cumulative_rev <= 0.8]
        return int(len(top_80_percent_products))

    def get_sales_anomalies(self) -> dict:
        """Detects days with unusually high sales spikes (anomalies) based on standard deviation."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        daily_rev = self.df.set_index('InvoiceDate').resample('D')['Revenue'].sum()
        mean = daily_rev.mean()
        std_dev = daily_rev.std()
        anomalies = daily_rev[daily_rev > (mean + 2 * std_dev)]
        return {str(k.date()): v for k, v in anomalies.to_dict().items()}

    def get_frequently_bought_together(self, product_desc: str, limit: int = 3) -> dict:
        """Finds other products that are most frequently bought in the same invoice as the specified product."""
        invoices_with_product = self.df[self.df['Description'] == product_desc]['Invoice']
        basket = self.df[self.df['Invoice'].isin(invoices_with_product)]
        other_products = basket[basket['Description'] != product_desc]
        return other_products['Description'].value_counts().nlargest(limit).to_dict()

    def get_simple_sales_forecast(self) -> float:
        """Forecasts the expected sales revenue for the next 7 days based on the average of the last 14 days."""
        if 'InvoiceDate' not in self.df.columns: return 0.0
        daily_rev = self.df.set_index('InvoiceDate').resample('D')['Revenue'].sum()
        last_14_days_avg = daily_rev.tail(14).mean()
        forecast_next_7_days = last_14_days_avg * 7
        return float(forecast_next_7_days)

    # --- פונקציות השלמה ל-15 (רמה: ניהול עסקי) ---

    def get_sales_trend(self) -> str:
        """Determines the current sales trend (Up, Down, or Stable) compared to the previous month."""
        growth = self.get_mom_growth_rate()
        if growth > 0:
            return f"Up (↑ {growth:.2f}%)"
        elif growth < 0:
            return f"Down (↓ {abs(growth):.2f}%)"
        return "Stable (0%)"

    def detect_revenue_drops(self, threshold: float = -15.0) -> dict:
        """Detects specific months where revenue dropped by more than the specified threshold percentage (default is -15%)."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        monthly_rev = self.df.set_index('InvoiceDate').resample('ME')['Revenue'].sum()
        growth = monthly_rev.pct_change() * 100
        drops = growth[growth < threshold]
        if drops.empty:
            return {"status": "No significant revenue drops detected."}
        return {str(k): v for k, v in drops.to_dict().items()}

    def get_repeat_customers_stats(self) -> dict:
        """Calculates statistics about repeat customers, including total count and retention rate percentage."""
        customer_counts = self.df.groupby('Customer ID')['Invoice'].nunique()
        repeat_customers = customer_counts[customer_counts > 1]
        
        total_unique = self.df['Customer ID'].nunique()
        repeat_count = len(repeat_customers)
        percentage = (repeat_count / total_unique) * 100 if total_unique > 0 else 0
        
        return {
            "repeat_customers_count": int(repeat_count),
            "repeat_customers_percentage": f"{percentage:.2f}%",
            "total_unique_customers": int(total_unique)
        }