# We import our Manager to handle the logic
from Manager import ManagerAgent

def start_app():
    """
    This function starts the interactive application loop.
    """
    print("=========================================")
    print("🤖 Welcome to the AI Data Department 🤖")
    print("============================================")
    print("Type 'exit' or 'quit' to close the app.\n")

    # Initialize the system by hiring the Manager and giving the data path
    manager = ManagerAgent("online_retail_small.csv")

    # The infinite loop that keeps the program running waiting for user input
    while True:
        # Prompt the user to type something
        user_input = input("\n👤 You: ").strip()

        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit', 'יציאה']:
            print("👋 Shutting down the Data Department. Goodbye!")
            break
        
        # Prevent empty inputs
        if not user_input:
            continue

        # Pass the input to the Manager and get the response
        response = manager.handle_request(user_input)
        
        # Print the Manager's response
        print(f"\n👔 {response}")

# --- Application entry point ---
if __name__ == "__main__":
    start_app()