import pandas as pd
import numpy as np

class CustomerAnalyst:
    def __init__(self, data_frame: pd.DataFrame):
        # Work on a copy to avoid SettingWithCopyWarning
        self.df = data_frame.copy()

        # Pre-calculate revenue and parse dates
        if 'Quantity' in self.df.columns and 'Price' in self.df.columns:
            self.df['Revenue'] = self.df['Quantity'] * self.df['Price']

        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')

    # --- Basic metrics ---

    def get_total_revenue(self) -> float:
        """Calculates and returns the total revenue generated across all customer transactions."""
        return float(round(self.df['Revenue'].sum(), 2))

    def get_total_unique_customers(self) -> int:
        """Calculates the total number of unique customers in the dataset."""
        return int(self.df['Customer ID'].nunique())

    def get_top_country(self) -> str:
        """Finds and returns the name of the country with the most recorded transactions."""
        return str(self.df['Country'].value_counts().idxmax())

    def get_total_items_sold(self) -> int:
        """Calculates the total amount of physical items sold to customers, ignoring refunds."""
        valid_sales = self.df[self.df['Quantity'] > 0]
        return int(valid_sales['Quantity'].sum())

    def get_average_item_price(self) -> float:
        """Calculates the average price of a single product in the store."""
        return float(round(self.df['Price'].mean(), 2))

    # --- Grouped & filtered metrics ---

    def get_top_customer(self, limit: int = 5, country: str = None) -> int:
        """Finds the Customer ID of the single customer who generated the most total revenue, optionally filtered by country."""
        df_filtered = self.df
        if country:
            df_filtered = self.df[self.df['Country'] == country]
        top_customer = df_filtered.groupby('Customer ID')['Revenue'].sum().idxmax()
        return int(top_customer)

    def get_top_spending_customers(self, top_n: int = 5) -> dict:
        """Gets a dictionary of the top N spending customers ranked by total revenue."""
        top_customers = self.df.groupby('Customer ID')['Revenue'].sum().nlargest(top_n)
        return {int(k): round(float(v), 2) for k, v in top_customers.items()}

    def get_revenue_by_country(self, top_n: int = 5) -> dict:
        """Gets a dictionary of the top N countries ranked by total revenue generated."""
        country_rev = self.df.groupby('Country')['Revenue'].sum().nlargest(top_n)
        return country_rev.round(2).to_dict()

    def get_most_popular_product(self) -> str:
        """Finds the product description of the item that sold the most total units."""
        if 'Description' not in self.df.columns: return "Description column missing"
        popular_product = self.df.groupby('Description')['Quantity'].sum().idxmax()
        return str(popular_product)

    def get_refund_rate(self) -> float:
        """Calculates the percentage of all transactions that were refunds (negative quantity)."""
        total_rows = len(self.df)
        if total_rows == 0: return 0.0
        refund_rows = len(self.df[self.df['Quantity'] < 0])
        return float(round((refund_rows / total_rows) * 100, 2))

    # --- Advanced logic ---

    def get_repeat_customer_rate(self) -> float:
        """Calculates the percentage of customers who made more than one separate purchase (repeat buyers)."""
        if 'InvoiceNo' not in self.df.columns: return 0.0
        invoices_per_customer = self.df.groupby('Customer ID')['InvoiceNo'].nunique()
        repeat_customers = (invoices_per_customer > 1).sum()
        total_customers = invoices_per_customer.count()
        if total_customers == 0: return 0.0
        return float(round((repeat_customers / total_customers) * 100, 2))

    def get_best_selling_product_per_country(self) -> dict:
        """Finds the single best-selling product description for each country based on quantity."""
        if 'Description' not in self.df.columns: return {}
        grouped = self.df.groupby(['Country', 'Description'])['Quantity'].sum().reset_index()
        idx = grouped.groupby('Country')['Quantity'].idxmax()
        best_sellers = grouped.loc[idx].set_index('Country')['Description'].to_dict()
        return best_sellers

    def get_average_order_value(self) -> float:
        """Calculates the Average Order Value (AOV) per invoice."""
        if 'InvoiceNo' not in self.df.columns: return 0.0
        revenue_per_invoice = self.df.groupby('InvoiceNo')['Revenue'].sum()
        valid_invoices = revenue_per_invoice[revenue_per_invoice > 0]
        if valid_invoices.empty: return 0.0
        return float(round(valid_invoices.mean(), 2))

    def get_monthly_revenue_trend(self) -> dict:
        """Calculates the total revenue grouped by month to show the trend."""
        if 'InvoiceDate' not in self.df.columns: return {}
        temp_df = self.df.dropna(subset=['InvoiceDate']).copy()
        temp_df['YearMonth'] = temp_df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_trend = temp_df.groupby('YearMonth')['Revenue'].sum()
        return monthly_trend.round(2).to_dict()

    def get_customer_profile(self, customer_id: int) -> dict:
        """Returns a full profile for a specific customer ID: total spend, number of orders, items bought, favorite product, first and last purchase date, and refund count."""
        cdf = self.df[self.df['Customer ID'] == customer_id]
        if cdf.empty:
            return {"error": f"Customer ID {customer_id} not found in the dataset."}

        total_spend = round(float(cdf[cdf['Quantity'] > 0]['Revenue'].sum()), 2)
        total_orders = int(cdf['InvoiceNo'].nunique()) if 'InvoiceNo' in cdf.columns else 0
        total_items = int(cdf[cdf['Quantity'] > 0]['Quantity'].sum())
        refund_count = int((cdf['Quantity'] < 0).sum())

        favorite_product = None
        if 'Description' in cdf.columns:
            pos = cdf[cdf['Quantity'] > 0]
            if not pos.empty:
                favorite_product = str(pos.groupby('Description')['Quantity'].sum().idxmax())

        first_purchase = None
        last_purchase = None
        if 'InvoiceDate' in cdf.columns:
            dates = cdf['InvoiceDate'].dropna()
            if not dates.empty:
                first_purchase = str(dates.min().date())
                last_purchase = str(dates.max().date())

        return {
            "customer_id": customer_id,
            "total_spend_gbp": total_spend,
            "total_orders": total_orders,
            "total_items_bought": total_items,
            "refund_transactions": refund_count,
            "favorite_product": favorite_product,
            "first_purchase": first_purchase,
            "last_purchase": last_purchase,
        }

    def get_high_value_loyal_customers(self, order_threshold: int = 5, revenue_threshold: float = 1000.0) -> list:
        """Finds VIP loyal customers who have ordered more than the order_threshold AND spent over the revenue_threshold."""
        if 'InvoiceNo' not in self.df.columns: return []
        customer_stats = self.df.groupby('Customer ID').agg(
            Total_Spend=('Revenue', 'sum'),
            Total_Orders=('InvoiceNo', 'nunique')
        )
        vips = customer_stats[
            (customer_stats['Total_Orders'] >= order_threshold) &
            (customer_stats['Total_Spend'] >= revenue_threshold)
        ]
        return [int(cid) for cid in vips.index.tolist()]
