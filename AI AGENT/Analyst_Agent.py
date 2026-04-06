import pandas as pd
# We are importing our previous agent so they can work together!
from Data_Agent import DataAgent

class AnalystAgent:
    def __init__(self, data_frame):
        # Store the data_frame inside the agent's memory
        self.df = data_frame
    
    def get_total_sales(self):
        """
        Calculates total sales by multiplying Quantity and Price for each row, then summing it all up.
        """
        try:
            # Calculate Revenue: (Quantity * Price) and then sum()
            total_revenue = (self.df['Quantity'] * self.df['Price']).sum()
            print(f"[Analyst Agent] 📊 Total Sales calculated: ${total_revenue:.2f}")
            return total_revenue
        
        except Exception as e:
            print(f"[Analyst Agent] ❌ Error calculating sales: {e}")
            return None









# --- Small test code to see both agents working together ---
if __name__ == "__main__":
    print("--- Starting Multi-Agent Test ---")
    
    # 1. Manager calls the Data Agent
    data_agent = DataAgent("online_retail_small.csv")
    df = data_agent.get_data()
    
    if df is not None:
        # 2. Manager passes the data to the Analyst Agent
        analyst = AnalystAgent(df)
        total = analyst.get_total_sales()
        
        print(f"\n✅ Final Result delivered to Manager: ${total:.2f}")