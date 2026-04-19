"""
Singleton loader for AI agents — shared across all Flask routes.
Agents are loaded once on first request and cached in memory.

DataAgent + SalesAnalyst are loaded first (no API key needed).
ManagerAgent is loaded separately — if it fails (e.g. missing OPENAI_API_KEY
or a package incompatibility), the dashboard still works; only chat is affected.
"""
import os
import sys
import threading
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, 'agents'))

_df      = None
_manager = None
_sales   = None
_lock    = threading.Lock()
_data_loaded   = False
_manager_error = None   # stores the exception message if ManagerAgent fails


def get_agents():
    global _df, _manager, _sales, _data_loaded, _manager_error

    if _data_loaded:
        return _df, _manager, _sales

    with _lock:
        if _data_loaded:
            return _df, _manager, _sales

        # ── Step 1: Load data + SalesAnalyst (no API key required) ────────────
        try:
            from Data_Agent import DataAgent
            from Sales_Analyst import SalesAnalyst

            csv_path = os.path.join(_BASE, "data", "mixed_online_retail.csv")
            d_agent = DataAgent(csv_path)
            _df = d_agent.get_data()

            if _df is not None:
                _sales = SalesAnalyst(_df)
                print("[flask_agents] Data and SalesAnalyst loaded successfully.")
            else:
                print("[flask_agents] ERROR: DataAgent returned None — check CSV path.")

        except Exception:
            print("[flask_agents] ERROR loading data/SalesAnalyst:")
            traceback.print_exc()
            _df    = None
            _sales = None

        # ── Step 2: Load ManagerAgent (requires OPENAI_API_KEY) ───────────────
        if _df is not None:
            try:
                from Manager import ManagerAgent
                _manager = ManagerAgent(_df)
                print("[flask_agents] ManagerAgent loaded successfully.")
            except Exception as e:
                _manager_error = str(e)
                print(f"[flask_agents] WARNING: ManagerAgent failed to load: {e}")
                traceback.print_exc()
                _manager = None   # dashboard still works; chat will show an error

        _data_loaded = True

    return _df, _manager, _sales


def get_manager_error() -> str | None:
    """Returns the ManagerAgent init error message, or None if it loaded OK."""
    return _manager_error
