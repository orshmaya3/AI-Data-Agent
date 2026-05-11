import math
from flask import Blueprint, render_template, session, jsonify
from flask_routes.utils import login_required

dashboard_bp = Blueprint('dashboard', __name__)

# ── Currency detection ─────────────────────────────────────────────────────
_CURRENCY_MAP = {
    'pakistan':       ('PKR', 'PKR'),
    'united kingdom': ('£',   'GBP'),
    'uk':             ('£',   'GBP'),
    'united states':  ('$',   'USD'),
    'usa':            ('$',   'USD'),
    'germany':        ('€',   'EUR'),
    'france':         ('€',   'EUR'),
    'spain':          ('€',   'EUR'),
    'italy':          ('€',   'EUR'),
    'india':          ('₹',   'INR'),
    'australia':      ('A$',  'AUD'),
    'canada':         ('C$',  'CAD'),
}

def _detect_currency(top_country: str, session_id: str | None) -> tuple[str, str]:
    key = (top_country or '').lower()
    for country_key, pair in _CURRENCY_MAP.items():
        if country_key in key:
            return pair
    # Fallback: scan original filename for country hints
    fname = (_get_upload_filename_raw(session_id) or '').lower()
    for country_key, pair in _CURRENCY_MAP.items():
        if country_key.replace(' ', '') in fname.replace(' ', ''):
            return pair
    return ('£', 'GBP')

def _get_upload_filename_raw(session_id: str | None) -> str | None:
    """Return original filename for the active upload of the current user, or None."""
    try:
        from flask import session as _s
        user_id = _s.get('user_id')
        if not user_id:
            return None
        from models import UserUpload
        rec = UserUpload.query.filter_by(user_id=user_id, is_active=True).first()
        return rec.filename_original if rec else None
    except Exception:
        return None

def _build_data_profile(df, session_id: str | None) -> dict:
    """Compute the data profile dict from a cleaned DataFrame."""
    import pandas as pd

    # Date range
    has_dates = 'InvoiceDate' in df.columns and df['InvoiceDate'].notna().any()
    date_from = df['InvoiceDate'].min().strftime('%b %Y') if has_dates else None
    date_to   = df['InvoiceDate'].max().strftime('%b %Y') if has_dates else None

    # Country / currency
    has_country = (
        'Country' in df.columns and
        df['Country'].notna().any() and
        df['Country'].nunique() > 1
    )
    top_country = (
        df['Country'].mode()[0] if has_country
        else (df['Country'].dropna().iloc[0] if 'Country' in df.columns and df['Country'].notna().any() else '')
    )
    currency_symbol, currency_code = _detect_currency(top_country, session_id)

    # Categories vs products
    desc_unique    = df['Description'].nunique() if 'Description' in df.columns else 0
    is_categorical = desc_unique > 0 and desc_unique < 100
    top_items = []
    if 'Description' in df.columns and 'Quantity' in df.columns:
        top_items = (
            df.groupby('Description')['Quantity'].sum()
              .sort_values(ascending=False).head(5).index.tolist()
        )

    filename = _get_upload_filename_raw(session_id)
    unique_customers = int(df['Customer ID'].nunique()) if 'Customer ID' in df.columns else 0

    return {
        'row_count':        len(df),
        'date_from':        date_from,
        'date_to':          date_to,
        'has_country':      has_country,
        'is_categorical':   is_categorical,
        'top_items':        top_items,
        'item_label':       'Categories' if is_categorical else 'Products',
        'currency_symbol':  currency_symbol,
        'currency_code':    currency_code,
        'filename':         filename,
        'unique_customers': unique_customers,
    }


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
        session_id = session.get('session_id')
        df, sales = resolve_data_agents(session_id)

        if df is None:
            return jsonify({'error': 'Data not loaded — check server logs for details.'}), 503
        if sales is None:
            return jsonify({'error': 'SalesAnalyst not initialised — check server logs.'}), 503

        profile = _build_data_profile(df, session_id)
        return jsonify({
            'monthly_revenue': sales.get_monthly_revenue(),
            'top_countries':   sales.get_top_countries_by_revenue(limit=5),
            'top_products':    sales.get_top_products_by_revenue(limit=8),
            'hourly_sales':    sales.get_hourly_sales_distribution(),
            'meta': {
                'has_country':     profile['has_country'],
                'is_categorical':  profile['is_categorical'],
                'item_label':      profile['item_label'],
                'currency_symbol': profile['currency_symbol'],
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chart data failed: {e}'}), 500


@dashboard_bp.route('/api/data/profile')
@login_required
def api_data_profile():
    try:
        from flask_routes.utils import resolve_data_agents
        session_id = session.get('session_id')
        df, sales  = resolve_data_agents(session_id)
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 503
        return jsonify(_build_data_profile(df, session_id))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
