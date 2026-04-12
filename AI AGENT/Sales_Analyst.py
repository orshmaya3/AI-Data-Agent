import pandas as pd
import numpy as np

class SalesAnalyst:
    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame.copy()

        if 'Quantity' in self.df.columns and 'Price' in self.df.columns:
            self.df['Revenue'] = self.df['Quantity'] * self.df['Price']

        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')

    # --- Basic metrics ---

    def get_total_revenue(self) -> float:
        """Calculates and returns the total revenue from all transactions (sales only, excludes returns)."""
        return float(self.df[self.df['Quantity'] > 0]['Revenue'].sum())

    def get_total_orders(self) -> int:
        """Calculates the total number of unique orders (invoices) placed."""
        return int(self.df['Invoice'].nunique())

    def get_total_items_sold(self) -> int:
        """Calculates the total quantity of all physical items sold, excluding returns and refunds."""
        return int(self.df[self.df['Quantity'] > 0]['Quantity'].sum())

    def get_average_order_value(self) -> float:
        """Calculates the Average Order Value (AOV), which is total revenue divided by total orders."""
        total_rev = self.get_total_revenue()
        total_orders = self.get_total_orders()
        return float(total_rev / total_orders) if total_orders > 0 else 0.0

    def get_top_countries_by_revenue(self, limit: int = 5) -> dict:
        """Gets the top performing countries ranked by total sales revenue. Allows specifying a limit."""
        limit = min(limit, 50)
        return self.df.groupby('Country')['Revenue'].sum().nlargest(limit).to_dict()

    # --- Grouped & filtered metrics ---

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
        limit = min(limit, 50)
        data = self.df

        if country:
            data = data[data['Country'].str.lower() == country.lower()]
            if data.empty:
                return {"error": f"No sales data found for country: {country}"}

        return data[data['Quantity'] > 0].groupby('Description')['Revenue'].sum().nlargest(limit).to_dict()

    def get_refund_rate(self, country: str = None) -> float:
        """Calculates the percentage of transactions that were refunds or returns (negative quantity).
        Args:
            country: Optional. Filter to a specific country (e.g. 'Germany', 'France'). Leave empty for global rate.
        """
        data = self.df
        if country:
            data = self.df[self.df['Country'].str.lower() == country.lower()]
            if data.empty:
                return 0.0
        total_transactions = len(data)
        returns = len(data[data['Quantity'] < 0])
        return float((returns / total_transactions) * 100) if total_transactions > 0 else 0.0

    def get_revenue_by_date_range(self, start_date: str, end_date: str) -> dict:
        """Calculates total revenue within a specific date range. Dates must be provided as strings in YYYY-MM-DD format.
        Returns a dict with revenue and a note if the range falls outside the dataset's coverage."""
        dataset_start = str(self.df['InvoiceDate'].min().date())
        dataset_end   = str(self.df['InvoiceDate'].max().date())
        mask = (self.df['InvoiceDate'] >= start_date) & (self.df['InvoiceDate'] <= end_date)
        filtered_df = self.df.loc[mask]
        revenue = float(filtered_df['Revenue'].sum())
        if filtered_df.empty:
            return {
                "revenue": 0.0,
                "warning": f"No data found for {start_date} to {end_date}. Dataset covers {dataset_start} to {dataset_end}."
            }
        return {"revenue": revenue, "date_range": f"{start_date} to {end_date}"}

    def get_busiest_days_of_week(self) -> dict:
        """Counts the number of transactions for each day of the week to identify the busiest shopping days."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        days = self.df['InvoiceDate'].dt.day_name()
        return days.value_counts().to_dict()

    # --- Advanced time-series & business logic ---

    def get_mom_growth_rate(self) -> dict:
        """Calculates Month-over-Month (MoM) sales revenue growth rate for every month in the dataset.
        Returns a dict mapping each month to its growth rate percentage. The most recent month is labeled 'latest'.
        Use this to assess trend consistency, not just the last month's change."""
        if 'InvoiceDate' not in self.df.columns: return {}
        monthly_rev = self.df.set_index('InvoiceDate').resample('ME')['Revenue'].sum()
        growth = (monthly_rev.pct_change() * 100).dropna()
        result = {str(k.to_period('M')): round(float(v), 2) for k, v in growth.items()}
        if result:
            last_key = list(result)[-1]
            result['latest'] = result[last_key]
        return result

    def get_pareto_products_count(self) -> int:
        """Calculates how many top products make up 80% of the total revenue (Pareto principle / 80-20 rule)."""
        product_rev = self.df.groupby('Description')['Revenue'].sum().sort_values(ascending=False)
        total_rev = product_rev.sum()
        cumulative_rev = product_rev.cumsum() / total_rev
        # Use < 0.8 and add 1 to include the product that crosses the 80% threshold
        top_80_percent_products = cumulative_rev[cumulative_rev < 0.8]
        return int(len(top_80_percent_products)) + 1

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
        limit = min(limit, 50)
        sales_only = self.df[self.df['Quantity'] > 0]
        invoices_with_product = sales_only[sales_only['Description'] == product_desc]['Invoice']
        basket = sales_only[sales_only['Invoice'].isin(invoices_with_product)]
        other_products = basket[basket['Description'] != product_desc]
        return other_products['Description'].value_counts().nlargest(limit).to_dict()

    def get_simple_sales_forecast(self) -> dict:
        """Forecasts the expected sales revenue for the next 7 days based on the average of the last 14 days in the dataset.
        IMPORTANT: Returns a warning if the baseline period falls in a seasonal peak (e.g. December) which may inflate the estimate."""
        if 'InvoiceDate' not in self.df.columns: return {"forecast": 0.0, "warning": "No date data available."}
        daily_rev = self.df.set_index('InvoiceDate').resample('D')['Revenue'].sum()
        last_14 = daily_rev.tail(14)
        last_14_days_avg = last_14.mean()
        forecast_next_7_days = float(last_14_days_avg * 7)
        baseline_end = last_14.index[-1]
        warning = None
        if baseline_end.month == 12:
            warning = (
                f"Caution: The baseline period ends in December ({baseline_end.date()}), "
                "which is a seasonal peak. This forecast may significantly overestimate normal trading periods."
            )
        return {
            "forecast_7_day_gbp": round(forecast_next_7_days, 2),
            "based_on": f"14-day average ending {baseline_end.date()}",
            "warning": warning
        }

    # --- Business strategy metrics ---

    def get_sales_trend(self) -> str:
        """Determines the current sales trend (Up, Down, or Stable) compared to the previous month."""
        mom_series = self.get_mom_growth_rate()
        growth = mom_series.get('latest', 0.0) if isinstance(mom_series, dict) else float(mom_series)
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

    # --- Advanced behavioral analytics ---

    def get_hourly_sales_distribution(self) -> dict:
        """Analyzes the distribution of sales by hour of the day to find peak shopping hours."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        hourly_sales = self.df.groupby(self.df['InvoiceDate'].dt.hour)['Revenue'].sum()
        return {f"{hour}:00": float(rev) for hour, rev in hourly_sales.items()}

    def get_weekend_vs_weekday_sales(self) -> dict:
        """Compares total revenue generated on weekdays versus weekends to identify shopping patterns."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        is_weekend = self.df['InvoiceDate'].dt.dayofweek >= 5
        weekend_rev = self.df[is_weekend]['Revenue'].sum()
        weekday_rev = self.df[~is_weekend]['Revenue'].sum()
        return {"Weekday Revenue": float(weekday_rev), "Weekend Revenue": float(weekend_rev)}

    def get_churn_risk_customers(self, days_inactive: int = 90) -> dict:
        """Identifies the number of customers who are at risk of churning (have not purchased in the last X days, default 90).
        NOTE: Churn is calculated relative to the dataset's last date, not today's date, since the data is historical."""
        if 'InvoiceDate' not in self.df.columns: return {"error": "No date data"}
        last_purchase = self.df.groupby('Customer ID')['InvoiceDate'].max()
        dataset_end_date = self.df['InvoiceDate'].max()

        days_since_last_purchase = (dataset_end_date - last_purchase).dt.days
        at_risk = days_since_last_purchase[days_since_last_purchase >= days_inactive]

        return {
            "total_customers": int(len(last_purchase)),
            "at_risk_customers": int(len(at_risk)),
            "churn_risk_percentage": f"{(len(at_risk) / len(last_purchase)) * 100:.2f}%" if len(last_purchase) > 0 else "0%",
            "reference_date": str(dataset_end_date.date()),
            "note": "Churn is measured relative to the dataset's last recorded date, not today."
        }

    def get_revenue_concentration_risk(self) -> str:
        """Calculates the percentage of total revenue that comes from the top 10% of customers to assess business risk (Whale dependence)."""
        customer_rev = self.df.groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False)
        if customer_rev.empty: return "No customer data"

        top_10_percent_count = max(1, int(len(customer_rev) * 0.10))
        top_10_rev = customer_rev.head(top_10_percent_count).sum()
        total_rev = customer_rev.sum()

        concentration = (top_10_rev / total_rev) * 100
        return f"Risk Level: {concentration:.2f}% of our total revenue comes from just the top 10% of our customers."

    def search_products(self, query: str) -> list:
        """Searches the product catalog for items whose description contains the query string.
        ALWAYS call this first when the user refers to a product by a partial or approximate name,
        before calling any other tool that requires an exact product description.
        Returns up to 10 matching product descriptions (exact strings) from the dataset.
        Args:
            query: A partial product name or keyword (case-insensitive, e.g. 'heart candle').
        """
        if 'Description' not in self.df.columns:
            return []
        mask = self.df['Description'].str.contains(query, case=False, na=False, regex=False)
        return sorted(self.df[mask]['Description'].unique().tolist())[:10]

    def get_average_days_between_purchases(self) -> float:
        """Calculates the average number of days a returning customer waits before making another purchase."""
        if 'InvoiceDate' not in self.df.columns: return 0.0

        # Sort by customer and date to compute time gaps
        sorted_df = self.df.dropna(subset=['Customer ID']).sort_values(['Customer ID', 'InvoiceDate'])

        # Keep one row per invoice to avoid counting 0-day gaps between items in the same order
        invoices = sorted_df.drop_duplicates(subset=['Customer ID', 'Invoice']).copy()

        invoices['PrevPurchaseDate'] = invoices.groupby('Customer ID')['InvoiceDate'].shift(1)
        invoices['DaysBetween'] = (invoices['InvoiceDate'] - invoices['PrevPurchaseDate']).dt.days

        avg_days = invoices['DaysBetween'].mean()
        return float(avg_days) if pd.notna(avg_days) else 0.0

    def get_product_family_revenue(self, keyword: str) -> dict:
        """Aggregates total revenue across ALL products whose name contains the given keyword.
        Use this when the user asks about a category of products (e.g. 'candles', 'bags', 'heart').
        Returns total revenue, unit count, and the individual products found.
        Args:
            keyword: A partial product name or category keyword (case-insensitive, e.g. 'candle', 'bag').
        """
        if 'Description' not in self.df.columns:
            return {"error": "No product data available."}
        mask = self.df['Description'].str.contains(keyword, case=False, na=False, regex=False)
        family = self.df[mask]
        if family.empty:
            return {"error": f"No products found matching '{keyword}'."}
        total_revenue = float(family['Revenue'].sum())
        total_units   = int(family[family['Quantity'] > 0]['Quantity'].sum())
        product_breakdown = (
            family.groupby('Description')['Revenue'].sum()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
        return {
            "keyword": keyword,
            "matching_products_count": int(mask.sum() and family['Description'].nunique()),
            "total_revenue_gbp": round(total_revenue, 2),
            "total_units_sold": total_units,
            "top_10_products": product_breakdown
        }
