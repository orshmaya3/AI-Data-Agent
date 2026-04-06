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

    def get_total_revenue(self):
        return self.df['Revenue'].sum()

    def get_total_orders(self):
        return self.df['Invoice'].nunique()

    def get_total_items_sold(self):
        return self.df['Quantity'].sum()

    def get_average_order_value(self):
        total_rev = self.get_total_revenue()
        total_orders = self.get_total_orders()
        return total_rev / total_orders if total_orders > 0 else 0

    def get_top_countries_by_revenue(self, limit=5):
        return self.df.groupby('Country')['Revenue'].sum().nlargest(limit).to_dict()

    # --- רמה 2: בינוני (Medium) ---

    def get_monthly_revenue(self):
        if 'InvoiceDate' not in self.df.columns: return "No date data available"
        monthly_df = self.df.copy()
        monthly_df['Month'] = monthly_df['InvoiceDate'].dt.to_period('M')
        return monthly_df.groupby('Month')['Revenue'].sum().astype(float).to_dict()

    def get_top_products_by_revenue(self, limit=5):
        return self.df.groupby('Description')['Revenue'].sum().nlargest(limit).to_dict()

    def get_refund_rate(self):
        total_transactions = len(self.df)
        returns = len(self.df[self.df['Quantity'] < 0])
        return (returns / total_transactions) * 100 if total_transactions > 0 else 0

    def get_revenue_by_date_range(self, start_date, end_date):
        mask = (self.df['InvoiceDate'] >= start_date) & (self.df['InvoiceDate'] <= end_date)
        filtered_df = self.df.loc[mask]
        return filtered_df['Revenue'].sum()

    def get_busiest_days_of_week(self):
        if 'InvoiceDate' not in self.df.columns: return None
        days = self.df['InvoiceDate'].dt.day_name()
        return days.value_counts().to_dict()

    # --- רמה 3: קשה (Hard) ---

    def get_mom_growth_rate(self):
        if 'InvoiceDate' not in self.df.columns: return None
        monthly_rev = self.df.set_index('InvoiceDate').resample('ME')['Revenue'].sum()
        growth = monthly_rev.pct_change() * 100 
        return growth.iloc[-1] if not pd.isna(growth.iloc[-1]) else 0

    def get_pareto_products_count(self):
        product_rev = self.df.groupby('Description')['Revenue'].sum().sort_values(ascending=False)
        total_rev = product_rev.sum()
        cumulative_rev = product_rev.cumsum() / total_rev
        top_80_percent_products = cumulative_rev[cumulative_rev <= 0.8]
        return len(top_80_percent_products)

    def get_sales_anomalies(self):
        if 'InvoiceDate' not in self.df.columns: return None
        daily_rev = self.df.set_index('InvoiceDate').resample('D')['Revenue'].sum()
        mean = daily_rev.mean()
        std_dev = daily_rev.std()
        anomalies = daily_rev[daily_rev > (mean + 2 * std_dev)]
        return anomalies.to_dict()

    def get_frequently_bought_together(self, product_desc, limit=3):
        invoices_with_product = self.df[self.df['Description'] == product_desc]['Invoice']
        basket = self.df[self.df['Invoice'].isin(invoices_with_product)]
        other_products = basket[basket['Description'] != product_desc]
        return other_products['Description'].value_counts().nlargest(limit).to_dict()

    def get_simple_sales_forecast(self):
        if 'InvoiceDate' not in self.df.columns: return None
        daily_rev = self.df.set_index('InvoiceDate').resample('D')['Revenue'].sum()
        last_14_days_avg = daily_rev.tail(14).mean()
        forecast_next_7_days = last_14_days_avg * 7
        return forecast_next_7_days