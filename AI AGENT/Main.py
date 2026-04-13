from Data_Agent import DataAgent
from Manager import ManagerAgent
import os

def start_app():
    """Entry point — loads data and starts the conversation loop."""
    print("╔══════════════════════════════════════════╗")
    print("║     Welcome to your AI Data Department   ║")
    print("╚══════════════════════════════════════════╝")

    # Load the data
    file_name = "mixed_online_retail.csv"
    print(f"\nLoading your data, just a moment...")

    d_agent = DataAgent(file_name)
    df = d_agent.get_data()

    if df is None:
        print("❌ CRITICAL ERROR: Could not load data. Please check the file path.")
        return

    print(f"Data loaded successfully — {len(df):,} records ready for analysis.")

    # Initialize the manager and all sub-agents
    print("Setting up your analyst team...")
    manager = ManagerAgent(df)

    print("\n✔ Your AI Data Department is ready.")
    print("─" * 44)
    print("You can ask things like:")
    print("  • 'Who is my top customer?'")
    print("  • 'How are sales trending this month?'")
    print("  • 'Which products have the highest return rate?'")
    print("\nType 'exit' to shut down.")
    print("─" * 44)

    # Conversation history for the CLI session
    history = []

    # Main conversation loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'יציאה', 'ביי']:
                print("Goodbye! The Data Department is signing off.")
                break

            if not user_input:
                continue

            if len(user_input) > 2000:
                print("⚠️  Your message is too long (max 2000 characters). Please shorten it.")
                continue

            response = manager.handle_request(user_input, history=history)
            print(f"\n👔 Manager: {response}")

            # Grow history after each successful exchange
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nShutting down. Have a great day!")
            break
        except Exception as e:
            print(f"\nSomething went wrong on my end: {e}. Please try again.")

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set in .env before running
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables!")

    start_app()
