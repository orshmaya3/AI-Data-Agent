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
        """Calculates and returns the total revenue generated across all customer transactions (sales only, excludes returns)."""
        return float(round(self.df[self.df['Quantity'] > 0]['Revenue'].sum(), 2))

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
        """Gets a dictionary of the top N spending customers ranked by total revenue (sales only, excludes returns)."""
        top_n = min(top_n, 50)
        top_customers = self.df[self.df['Quantity'] > 0].groupby('Customer ID')['Revenue'].sum().nlargest(top_n)
        return {int(k): round(float(v), 2) for k, v in top_customers.items()}

    def get_revenue_by_country(self, top_n: int = 5) -> dict:
        """Gets a dictionary of the top N countries ranked by total revenue generated."""
        top_n = min(top_n, 50)
        country_rev = self.df.groupby('Country')['Revenue'].sum().nlargest(top_n)
        return country_rev.round(2).to_dict()

    def get_most_popular_product(self) -> str:
        """Finds the product description of the item that sold the most total units."""
        if 'Description' not in self.df.columns: return "Description column missing"
        popular_product = self.df.groupby('Description')['Quantity'].sum().idxmax()
        return str(popular_product)

    def get_refund_rate(self, country: str = None) -> float:
        """Calculates the percentage of all transactions that were refunds (negative quantity).
        Args:
            country: Optional. Filter to a specific country (e.g. 'Germany', 'France'). Leave empty for global rate.
        """
        data = self.df
        if country:
            data = self.df[self.df['Country'].str.lower() == country.lower()]
            if data.empty:
                return 0.0
        total_rows = len(data)
        if total_rows == 0: return 0.0
        refund_rows = len(data[data['Quantity'] < 0])
        return float(round((refund_rows / total_rows) * 100, 2))

    # --- Advanced logic ---

    def get_repeat_customer_rate(self) -> float:
        """Calculates the percentage of customers who made more than one separate purchase (repeat buyers)."""
        if 'Invoice' not in self.df.columns: return 0.0
        invoices_per_customer = self.df.groupby('Customer ID')['Invoice'].nunique()
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
        if 'Invoice' not in self.df.columns: return 0.0
        revenue_per_invoice = self.df.groupby('Invoice')['Revenue'].sum()
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
        """Returns a full profile for a specific customer ID: total spend, number of orders, items bought, favorite product, first and last purchase date, refund count, and country."""
        # Cast to float to match pandas' float64 Customer ID storage (CSV reads produce float64 when NaNs exist)
        try:
            customer_id_f = float(customer_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid customer ID: '{customer_id}'. Please provide a numeric ID."}
        cdf = self.df[self.df['Customer ID'] == customer_id_f]
        if cdf.empty:
            return {"error": f"Customer ID {customer_id} not found in the dataset."}

        total_spend = round(float(cdf[cdf['Quantity'] > 0]['Revenue'].sum()), 2)
        total_orders = int(cdf['Invoice'].nunique()) if 'Invoice' in cdf.columns else 0
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

        country = str(cdf['Country'].mode()[0]) if 'Country' in cdf.columns and not cdf['Country'].empty else None

        return {
            "customer_id": customer_id,
            "country": country,
            "total_spend_gbp": total_spend,
            "total_orders": total_orders,
            "total_items_bought": total_items,
            "refund_transactions": refund_count,
            "favorite_product": favorite_product,
            "first_purchase": first_purchase,
            "last_purchase": last_purchase,
        }

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

    def get_high_value_loyal_customers(self, order_threshold: int = 5, revenue_threshold: float = 1000.0) -> list:
        """Finds VIP loyal customers who have ordered more than the order_threshold AND spent over the revenue_threshold."""
        if 'Invoice' not in self.df.columns: return []
        customer_stats = self.df.groupby('Customer ID').agg(
            Total_Spend=('Revenue', 'sum'),
            Total_Orders=('Invoice', 'nunique')
        )
        vips = customer_stats[
            (customer_stats['Total_Orders'] >= order_threshold) &
            (customer_stats['Total_Spend'] >= revenue_threshold)
        ]
        return [int(cid) for cid in vips.index.tolist()]

    def get_new_customers_by_month(self) -> dict:
        """Counts customers who placed their very first order in each month.
        Useful for tracking customer acquisition trends over time."""
        if 'InvoiceDate' not in self.df.columns: return {}
        first_order = self.df.groupby('Customer ID')['InvoiceDate'].min().reset_index()
        first_order['Month'] = first_order['InvoiceDate'].dt.to_period('M').astype(str)
        return first_order.groupby('Month').size().to_dict()

    def get_churn_risk_customer_list(self, days_inactive: int = 90, top_n: int = 20) -> list:
        """Returns a list of specific at-risk customers — their ID, days since last purchase, and total spend.
        Useful for targeting retention campaigns. Sorted by total spend descending so you focus on the most valuable at-risk customers.
        Args:
            days_inactive: Number of days without a purchase to be considered at risk (default 90).
            top_n: How many at-risk customers to return (default 20, max 50).
        """
        if 'InvoiceDate' not in self.df.columns: return []
        top_n = min(top_n, 50)
        reference_date = self.df['InvoiceDate'].max()
        last_purchase = self.df.groupby('Customer ID')['InvoiceDate'].max()
        total_spend   = self.df.groupby('Customer ID')['Revenue'].sum()

        days_inactive_series = (reference_date - last_purchase).dt.days
        at_risk = days_inactive_series[days_inactive_series >= days_inactive]

        result = []
        for cid in at_risk.index:
            result.append({
                "customer_id": int(cid),
                "days_since_last_purchase": int(at_risk[cid]),
                "total_spend_gbp": round(float(total_spend.get(cid, 0)), 2)
            })
        result.sort(key=lambda x: x['total_spend_gbp'], reverse=True)
        return result[:top_n]

    def get_customer_orders(self, customer_id: int) -> list:
        """Returns the full order history for a specific customer: invoice ID, date, total revenue, and item count.
        Use this to answer questions like 'highest purchase', 'largest order', 'most recent order',
        'how much did customer X spend in a single visit', or any order-level question for a customer.
        Orders are sorted by total revenue descending so the highest purchase appears first.
        Args:
            customer_id: The numeric customer ID (e.g. 18102).
        """
        try:
            customer_id_f = float(customer_id)
        except (ValueError, TypeError):
            return [{"error": f"Invalid customer ID: '{customer_id}'. Please provide a numeric ID."}]

        cdf = self.df[self.df['Customer ID'] == customer_id_f]
        if cdf.empty:
            return [{"error": f"Customer ID {customer_id} not found in the dataset."}]

        orders = (
            cdf.groupby('Invoice')
            .agg(
                date=('InvoiceDate', 'min'),
                total_revenue_gbp=('Revenue', 'sum'),
                items_bought=('Quantity', 'sum')
            )
            .reset_index()
        )
        # Exclude pure-refund invoices (negative total)
        orders = orders[orders['total_revenue_gbp'] > 0]
        orders = orders.sort_values('total_revenue_gbp', ascending=False)

        result = []
        for _, row in orders.iterrows():
            result.append({
                "invoice": str(row['Invoice']),
                "date": str(row['date'].date()) if pd.notna(row['date']) else None,
                "total_revenue_gbp": round(float(row['total_revenue_gbp']), 2),
                "items_bought": int(row['items_bought']),
            })
        return result

    def get_customer_product_quantity(self, customer_id: int, product_desc: str) -> dict:
        """Returns how many units of a specific product a customer has purchased in total.
        Use this when the user asks 'how many X did customer Y buy?' or 'how much of product X did customer Y order?'.
        IMPORTANT: call search_products first if the product name might be approximate, then pass the exact description here.
        Args:
            customer_id: The numeric customer ID (e.g. 18102).
            product_desc: The exact product description string from the dataset (case-insensitive match is applied).
        """
        try:
            customer_id_f = float(customer_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid customer ID: '{customer_id}'. Please provide a numeric ID."}

        cdf = self.df[self.df['Customer ID'] == customer_id_f]
        if cdf.empty:
            return {"error": f"Customer ID {customer_id} not found in the dataset."}

        product_rows = cdf[cdf['Description'].str.lower() == product_desc.lower()]
        if product_rows.empty:
            return {
                "customer_id": customer_id,
                "product": product_desc,
                "total_units_bought": 0,
                "note": "This customer has no recorded purchases of that product."
            }

        units_bought = int(product_rows[product_rows['Quantity'] > 0]['Quantity'].sum())
        units_returned = int(abs(product_rows[product_rows['Quantity'] < 0]['Quantity'].sum()))
        net_units = units_bought - units_returned

        return {
            "customer_id": customer_id,
            "product": product_desc,
            "total_units_bought": units_bought,
            "total_units_returned": units_returned,
            "net_units_kept": net_units,
        }

    def get_revenue_by_single_country(self, country: str) -> dict:
        """Returns the total revenue for a single named country.
        Args:
            country: The full country name (e.g. 'Germany', 'France', 'EIRE'). Case-insensitive.
        """
        data = self.df[self.df['Country'].str.lower() == country.lower()]
        if data.empty:
            return {"error": f"No data found for country: '{country}'. Check the spelling or try the full name."}
        revenue = float(data['Revenue'].sum())
        return {"country": country, "total_revenue_gbp": round(revenue, 2)}
