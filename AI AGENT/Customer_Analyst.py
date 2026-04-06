import pandas as pd

class CustomerAnalyst:
    def __init__(self, data_frame):
        # We store the dataframe here to analyze customer behavior
        self.df = data_frame
    
    def get_top_customer(self):
        """
        Finds the Customer ID who generated the most revenue (spent the most money).
        """
        try:
            # Calculate revenue per row, group by Customer ID, sum it, and find the max
            revenue_series = self.df['Quantity'] * self.df['Price']
            top_customer_id = revenue_series.groupby(self.df['Customer ID']).sum().idxmax()
            
            # Convert to an integer to remove decimals from the ID (e.g., 13085.0 -> 13085)
            return int(top_customer_id)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error finding top customer: {e}")
            return None

    def get_top_country(self):
        """
        Finds the country with the most recorded transactions in the data.
        """
        try:
            # Count how many times each country appears and get the top one
            top_country = self.df['Country'].value_counts().idxmax()
            return top_country
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error finding top country: {e}")
            return None