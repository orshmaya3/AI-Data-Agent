"""
Singleton loader for AI agents — shared across all Flask routes.
Agents are loaded once on first request and cached in memory.
Thread-safe: uses a Lock to prevent double-initialization under concurrent requests.
"""
import os
import sys
import threading

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, 'agents'))

_df      = None
_manager = None
_sales   = None
_lock    = threading.Lock()


def get_agents():
    global _df, _manager, _sales

    if _df is not None:
        return _df, _manager, _sales

    with _lock:
        # Re-check inside lock — another thread may have loaded while we waited
        if _df is not None:
            return _df, _manager, _sales

        try:
            from Data_Agent import DataAgent
            from Manager import ManagerAgent
            from Sales_Analyst import SalesAnalyst

            csv_path = os.path.join(_BASE, "data", "mixed_online_retail.csv")
            d_agent = DataAgent(csv_path)
            _df = d_agent.get_data()

            if _df is None:
                return None, None, None

            _manager = ManagerAgent(_df)
            _sales   = SalesAnalyst(_df)

        except Exception as e:
            print(f"[flask_agents] ❌ Error loading agents: {e}")
            return None, None, None

    return _df, _manager, _sales
