import pandas as pd
import os
from Manager import ManagerAgent

def main():
    print("🚀 Starting AI Data Agency...")
    
    # הגדרת נתיב הקובץ (וודא שהשם תואם למה שיש לכם בתיקייה)
    file_path = "online_retail_small.csv" 
    
    # 1. בדיקה שהקובץ קיים
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find the data file '{file_path}' in the current directory.")
        return

    # 2. טעינת הנתונים לזיכרון פעם אחת בלבד!
    print("📊 Loading data into memory...")
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"✅ Data loaded successfully! ({len(df)} rows ready).")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. יצירת המנהל והעברת הנתונים אליו
    manager = ManagerAgent(df)
    
    print("🤖 Manager Agent is ready. Type 'exit' to quit.\n")
    print("-" * 50)
    
    # 4. לולאת השיחה
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Shutting down the agency. Goodbye!")
            break
            
        # העברת הבקשה למנהל
        response = manager.handle_request(user_input)
        print(f"\n🤖 Manager: {response}")

if __name__ == "__main__":
    main()