import pandas as pd

class ProductAnalyst:
    def __init__(self, data_frame):
        # We store the dataframe here so the analyst can calculate metrics
        self.df = data_frame
    
    def get_top_product(self):
        """
        Finds the product with the highest total quantity sold.
        Filters out returns (where Quantity is less than 0).
        """
        try:
            # We only look at actual sales (Quantity > 0)
            sales_df = self.df[self.df['Quantity'] > 0]
            # Group by product description, sum the quantities, and find the highest one
            top_item = sales_df.groupby('Description')['Quantity'].sum().idxmax()
            return top_item
        except Exception as e:
            print(f"[Product Analyst] ❌ Error finding top product: {e}")
            return None

    def get_total_unique_products(self):
        """
        Calculates how many different types of products were sold.
        Uses 'StockCode' (SKU) to count unique items.
        """
        try:
            # Count unique StockCodes
            unique_products = self.df['StockCode'].nunique()
            return unique_products
        except Exception as e:
            print(f"[Product Analyst] ❌ Error calculating unique products: {e}")
            return None