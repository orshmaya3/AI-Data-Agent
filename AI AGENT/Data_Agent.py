import pandas as pd

class DataAgent:
    def __init__(self, file_path):
        # Store the file path as part of the agent's configuration
        self.file_path = file_path
    
    def get_data(self):
        """
        This function reads the data from the file and returns it as a pandas DataFrame.
        """
        try:
            # Read the data using pandas
            df = pd.read_csv(self.file_path)
            print(f"[Data Agent] ✅ Data loaded successfully! Found {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        
        except FileNotFoundError:
            print(f"[Data Agent] ❌ Error: The file {self.file_path} was not found.")
            return None
        except Exception as e:
            print(f"[Data Agent] ❌ Unexpected error: {e}")
            return None


    
    if df is not None:
        # Print the first 3 rows to visually verify everything is correct
        print(df.head(3))