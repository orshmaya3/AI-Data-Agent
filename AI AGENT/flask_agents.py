"""
Singleton loader for AI agents — shared across all Flask routes.

Data + SalesAnalyst load on first dashboard/KPI request (no API key, ~50 MB).
ManagerAgent loads on first chat/consultant request (LangChain, ~300 MB) so
the dashboard works immediately without triggering the heavy import.

Per-session agents: users who upload their own data get isolated agent
instances keyed by session_id. See register_session_data() and friends.
"""
import os
import sys
import time
import threading
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, 'agents'))

_df    = None
_sales = None
_manager = None

_data_lock    = threading.Lock()
_manager_lock = threading.Lock()

_data_loaded    = False
_manager_loaded = False
_manager_error  = None

# ── Per-session registry ───────────────────────────────────────────────────
# Each entry: {df, sales, manager, status, error, created_at}
_session_registry: dict[str, dict] = {}
_session_lock = threading.Lock()

SESSION_PENDING    = "pending"
SESSION_DATA_READY = "data_ready"
SESSION_READY      = "ready"
SESSION_ERROR      = "error"

MAX_UPLOAD_SESSIONS = int(os.environ.get('MAX_UPLOAD_SESSIONS', '10'))


def get_data_agents():
    """Return (df, sales) — loads once on first call. No API key required."""
    global _df, _sales, _data_loaded

    if _data_loaded:
        return _df, _sales

    with _data_lock:
        if _data_loaded:
            return _df, _sales

        try:
            from Data_Agent import DataAgent
            from Sales_Analyst import SalesAnalyst

            csv_path = os.path.join(_BASE, "data", "online_retail_II_sampled.parquet")
            d_agent = DataAgent(csv_path)
            _df = d_agent.get_data()

            if _df is not None:
                _sales = SalesAnalyst(_df)
                print("[flask_agents] Data + SalesAnalyst loaded successfully.")
            else:
                print("[flask_agents] ERROR: DataAgent returned None — check data path.")

        except Exception:
            print("[flask_agents] ERROR loading data/SalesAnalyst:")
            traceback.print_exc()
            _df    = None
            _sales = None

        _data_loaded = True

    return _df, _sales


def get_manager():
    """Return ManagerAgent — loads once on first call. Requires OPENAI_API_KEY."""
    global _manager, _manager_loaded, _manager_error

    if _manager_loaded:
        return _manager

    with _manager_lock:
        if _manager_loaded:
            return _manager

        df, _ = get_data_agents()

        if df is not None:
            try:
                from Manager import ManagerAgent
                _manager = ManagerAgent(df)
                print("[flask_agents] ManagerAgent loaded successfully.")
            except Exception as e:
                _manager_error = str(e)
                print(f"[flask_agents] WARNING: ManagerAgent failed to load: {e}")
                traceback.print_exc()
                _manager = None

        _manager_loaded = True

    return _manager


def get_agents():
    """Legacy helper — returns (df, manager, sales). Loads everything."""
    df, sales = get_data_agents()
    manager   = get_manager()
    return df, manager, sales


def get_manager_error() -> str | None:
    return _manager_error


# ── Per-session agent management ──────────────────────────────────────────

def register_session_data(session_id: str, df) -> None:
    """
    Store a cleaned DataFrame for this session, create a SalesAnalyst, then
    kick off ManagerAgent initialisation in a background thread.
    """
    from Sales_Analyst import SalesAnalyst

    sales = SalesAnalyst(df)
    with _session_lock:
        _session_registry[session_id] = {
            'df':         df,
            'sales':      sales,
            'manager':    None,
            'status':     SESSION_DATA_READY,
            'error':      None,
            'created_at': time.time(),
        }
    print(f"[flask_agents] Session {session_id[:8]}: data registered ({len(df):,} rows). Starting manager init…")

    t = threading.Thread(
        target=_init_session_manager,
        args=(session_id,),
        daemon=True,
        name=f"manager-init-{session_id[:8]}",
    )
    t.start()


def _init_session_manager(session_id: str) -> None:
    """Background worker: initialise ManagerAgent and update registry status."""
    with _session_lock:
        entry = _session_registry.get(session_id)
    if not entry:
        return
    df = entry.get('df')
    if df is None:
        return
    try:
        from Manager import ManagerAgent
        manager = ManagerAgent(df)
        with _session_lock:
            if session_id in _session_registry:
                _session_registry[session_id]['manager'] = manager
                _session_registry[session_id]['status']  = SESSION_READY
        print(f"[flask_agents] Session {session_id[:8]}: ManagerAgent ready.")
    except Exception as e:
        traceback.print_exc()
        with _session_lock:
            if session_id in _session_registry:
                _session_registry[session_id]['status'] = SESSION_ERROR
                _session_registry[session_id]['error']  = str(e)
        print(f"[flask_agents] Session {session_id[:8]}: manager init FAILED — {e}")


def get_session_status(session_id: str | None) -> dict:
    """Return status dict for a session."""
    if not session_id:
        return {'status': 'not_found', 'error': None, 'rows': None, 'columns': None}
    with _session_lock:
        entry = _session_registry.get(session_id)
    if not entry:
        return {'status': 'not_found', 'error': None, 'rows': None, 'columns': None}
    df = entry.get('df')
    return {
        'status':  entry.get('status', SESSION_PENDING),
        'error':   entry.get('error'),
        'rows':    int(len(df)) if df is not None else None,
        'columns': list(df.columns) if df is not None else None,
    }


def get_session_manager(session_id: str | None):
    """Return the ManagerAgent for this session, or None if not ready."""
    if not session_id:
        return None
    with _session_lock:
        entry = _session_registry.get(session_id)
    return entry.get('manager') if entry else None


def get_session_agents(session_id: str | None) -> tuple:
    """Return (df, sales) for this session, or (None, None) if not available."""
    if not session_id:
        return None, None
    with _session_lock:
        entry = _session_registry.get(session_id)
    if not entry:
        return None, None
    return entry.get('df'), entry.get('sales')


def evict_expired_sessions(max_age_seconds: int = 3600) -> int:
    """Remove sessions older than max_age_seconds. Returns eviction count."""
    now = time.time()
    to_evict = []
    with _session_lock:
        for sid, entry in _session_registry.items():
            if now - entry.get('created_at', now) > max_age_seconds:
                to_evict.append(sid)
        for sid in to_evict:
            del _session_registry[sid]
    return len(to_evict)
