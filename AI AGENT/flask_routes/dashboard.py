from flask import Blueprint, render_template, session, jsonify
from flask_routes.utils import login_required

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def main():
    return render_template('dashboard.html', username=session.get('username'))


@dashboard_bp.route('/api/kpis')
@login_required
def api_kpis():
    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if df is None:
        return jsonify({'error': 'Data not available'}), 500

    mom = sales.get_mom_growth_rate()
    return jsonify({
        'total_revenue': sales.get_total_revenue(),
        'total_orders':  sales.get_total_orders(),
        'total_items':   sales.get_total_items_sold(),
        'aov':           sales.get_average_order_value(),
        'refund_rate':   sales.get_refund_rate(),
        'mom_growth':    mom,
        'records':       len(df),
    })


@dashboard_bp.route('/api/charts')
@login_required
def api_charts():
    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if df is None:
        return jsonify({'error': 'Data not available'}), 500

    return jsonify({
        'monthly_revenue': sales.get_monthly_revenue(),
        'top_countries':   sales.get_top_countries_by_revenue(limit=5),
        'top_products':    sales.get_top_products_by_revenue(limit=8),
        'hourly_sales':    sales.get_hourly_sales_distribution(),
    })
