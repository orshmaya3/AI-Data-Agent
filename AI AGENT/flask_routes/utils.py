from functools import wraps
from flask import session, redirect, url_for


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('username'):
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


def resolve_manager(session_id: str | None):
    """
    Return the ManagerAgent for this session if its upload is fully ready,
    otherwise fall back to the global demo manager.
    """
    if session_id:
        from flask_agents import get_session_status, get_session_manager, SESSION_READY
        if get_session_status(session_id).get('status') == SESSION_READY:
            m = get_session_manager(session_id)
            if m is not None:
                return m
    from flask_agents import get_manager
    return get_manager()


def resolve_data_agents(session_id: str | None) -> tuple:
    """
    Return (df, sales) for this session if upload data is available,
    otherwise fall back to the global demo agents.
    """
    if session_id:
        from flask_agents import get_session_status, get_session_agents, SESSION_DATA_READY, SESSION_READY
        status = get_session_status(session_id).get('status')
        if status in (SESSION_DATA_READY, SESSION_READY):
            df, sales = get_session_agents(session_id)
            if df is not None and sales is not None:
                return df, sales
    from flask_agents import get_data_agents
    return get_data_agents()

