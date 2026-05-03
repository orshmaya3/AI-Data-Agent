import math
from flask import Blueprint, render_template, session, jsonify
from flask_routes.utils import login_required

dashboard_bp = Blueprint('dashboard', __name__)


def _safe(v, default=0.0):
    """Return v if it is a finite number, otherwise default."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@dashboard_bp.route('/dashboard')
@login_required
def main():
    return render_template('dashboard.html', username=session.get('username'))


@dashboard_bp.route('/api/health')
def api_health():
    """Diagnostic endpoint — shows what loaded successfully on startup."""
    from flask_agents import get_data_agents, get_manager_error
    df, sales = get_data_agents()
    return jsonify({
        'data_loaded':    df is not None,
        'sales_loaded':   sales is not None,
        'records':        len(df) if df is not None else 0,
        'manager_error':  get_manager_error(),
    })


@dashboard_bp.route('/api/kpis')
@login_required
def api_kpis():
    try:
        from flask_routes.utils import resolve_data_agents
        df, sales = resolve_data_agents(session.get('session_id'))

        if df is None:
            return jsonify({'error': 'Data not loaded — check server logs for details.'}), 503
        if sales is None:
            return jsonify({'error': 'SalesAnalyst not initialised — check server logs.'}), 503

        mom_data = sales.get_mom_growth_rate()
        mom = mom_data.get('latest', 0.0) if isinstance(mom_data, dict) else 0.0
        return jsonify({
            'total_revenue': _safe(sales.get_total_revenue()),
            'total_orders':  int(sales.get_total_orders()),
            'total_items':   int(sales.get_total_items_sold()),
            'aov':           _safe(sales.get_average_order_value()),
            'refund_rate':   _safe(sales.get_refund_rate()),
            'mom_growth':    _safe(mom),
            'records':       int(len(df)),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'KPI calculation failed: {e}'}), 500


@dashboard_bp.route('/api/charts')
@login_required
def api_charts():
    try:
        from flask_routes.utils import resolve_data_agents
        df, sales = resolve_data_agents(session.get('session_id'))

        if df is None:
            return jsonify({'error': 'Data not loaded — check server logs for details.'}), 503
        if sales is None:
            return jsonify({'error': 'SalesAnalyst not initialised — check server logs.'}), 503

        return jsonify({
            'monthly_revenue': sales.get_monthly_revenue(),
            'top_countries':   sales.get_top_countries_by_revenue(limit=5),
            'top_products':    sales.get_top_products_by_revenue(limit=8),
            'hourly_sales':    sales.get_hourly_sales_distribution(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chart data failed: {e}'}), 500
