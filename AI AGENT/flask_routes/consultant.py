import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, render_template, session, request, jsonify
from flask_routes.utils import login_required

consultant_bp = Blueprint('consultant', __name__)

_LOG_PATH = Path(__file__).parent.parent / 'data' / 'admin_log.json'
_log_lock = threading.Lock()


def _append_log(entry: dict):
    with _log_lock:
        _LOG_PATH.parent.mkdir(exist_ok=True)
        entries = json.loads(_LOG_PATH.read_text('utf-8')) if _LOG_PATH.exists() else []
        entries.append(entry)
        _LOG_PATH.write_text(json.dumps(entries, indent=2), encoding='utf-8')


@consultant_bp.route('/consultant')
@login_required
def consultant():
    return render_template('consultant.html', username=session.get('username'))


@consultant_bp.route('/api/consultant/profile', methods=['POST'])
@login_required
def api_consultant_profile():
    data                = request.get_json(silent=True) or {}
    name                = data.get('name', '').strip()
    email               = data.get('email', '').strip()
    business_type       = data.get('business_type', '').strip()
    business_type_other = data.get('business_type_other', '').strip()

    if not name or not email or not business_type:
        return jsonify({'error': 'Name, email, and business type are required.'}), 400

    VALID_TYPES = {'e-commerce', 'retail', 'restaurant', 'services', 'other'}
    if business_type not in VALID_TYPES:
        return jsonify({'error': 'Invalid business type.'}), 400

    if business_type == 'other' and not business_type_other:
        return jsonify({'error': 'Please describe your business type.'}), 400

    profile = {'name': name, 'email': email, 'business_type': business_type}
    if business_type == 'other':
        profile['business_type_other'] = business_type_other
    session['business_profile'] = profile
    _append_log({
        'id':            str(uuid.uuid4()),
        'event':         'profile_submitted',
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'name':          name,
        'email':         email,
        'business_type': business_type_other if business_type == 'other' else business_type,
    })
    return jsonify({'ok': True})


@consultant_bp.route('/api/consultant/profile', methods=['DELETE'])
@login_required
def api_consultant_profile_delete():
    session.pop('business_profile', None)
    return jsonify({'ok': True})


@consultant_bp.route('/api/consultant/health_preview')
@login_required
def api_consultant_health_preview():
    from flask_agents import get_manager
    manager = get_manager()

    if manager is None:
        return jsonify({'error': 'Data not available.'}), 503

    try:
        sales_analyst      = manager.sales_analyst
        customer_analyst   = manager.customer_analyst
        prediction_analyst = manager.prediction_analyst

        mom_data  = sales_analyst.get_mom_growth_rate()
        mom_val   = mom_data.get('latest', 0.0) if isinstance(mom_data, dict) else 0.0

        churn_data = prediction_analyst.get_churn_risk_summary()
        churn_pct  = churn_data.get('churn_risk_pct', 0.0) if isinstance(churn_data, dict) else 0.0

        repeat_pct = customer_analyst.get_repeat_customer_rate()

        def traffic_light(val, green_thresh, yellow_thresh, higher_is_better=True):
            if higher_is_better:
                if val >= green_thresh:  return 'green'
                if val >= yellow_thresh: return 'yellow'
                return 'red'
            else:
                if val <= green_thresh:  return 'green'
                if val <= yellow_thresh: return 'yellow'
                return 'red'

        return jsonify({
            'mom_growth': {
                'label':   'Monthly growth',
                'value':   round(mom_val, 1),
                'display': 'Growing' if mom_val > 5 else 'Stable' if mom_val > -5 else 'Declining',
                'color':   traffic_light(mom_val, 5, -5),
            },
            'churn_risk': {
                'label':   'Customers at risk',
                'value':   round(churn_pct, 1),
                'display': 'Low risk' if churn_pct < 30 else 'Moderate' if churn_pct < 55 else 'High risk',
                'color':   traffic_light(churn_pct, 30, 55, higher_is_better=False),
            },
            'repeat_rate': {
                'label':   'Repeat buyers',
                'value':   round(repeat_pct, 1),
                'display': 'Strong' if repeat_pct > 35 else 'Average' if repeat_pct > 20 else 'Needs work',
                'color':   traffic_light(repeat_pct, 35, 20),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@consultant_bp.route('/api/consultant/analyze', methods=['POST'])
@login_required
def api_consultant_analyze():
    data           = request.get_json(silent=True) or {}
    goal           = data.get('goal', '').strip()
    target         = data.get('target', '').strip()
    timeframe      = data.get('timeframe', 'the next 3 months').strip()
    goal_label     = data.get('goal_label', '').strip()
    goal_questions = data.get('goal_questions', [])

    if not goal:
        return jsonify({'error': 'No goal provided.'}), 400

    # Inject business profile context if available
    profile       = session.get('business_profile', {})
    profile_name  = profile.get('name', '')
    profile_email = profile.get('email', '')
    profile_type  = profile.get('business_type', '')
    profile_type_label = profile.get('business_type_other') or profile_type

    # Build a rich natural-language prompt for Zyon
    parts = []
    if profile_name and profile_type_label:
        parts.append(
            f"[Context: You are speaking with {profile_name}, "
            f"who runs a {profile_type_label} business.]"
        )
    parts.append(f"My business goal is: {goal}.")
    if target:
        parts.append(f"My target is: {target}.")
    parts.append(f"I want to achieve this in {timeframe}.")
    parts.append(
        "Please analyse my business data thoroughly — call at least 5 different tools "
        "to look at this from multiple angles — then give me a clear, practical action plan "
        "in plain English. Tell me exactly what to do, why it matters, and how to do it."
    )
    prompt = " ".join(parts)

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available. Please check server logs.'}), 503

    try:
        for step in manager.handle_consultant_request(prompt):
            if step['type'] == 'result':
                strategy_text = step['content']
                _append_log({
                    'id':              str(uuid.uuid4()),
                    'event':           'strategy_generated',
                    'timestamp':       datetime.now(timezone.utc).isoformat(),
                    'name':            profile_name,
                    'email':           profile_email,
                    'business_type':   profile_type,
                    'goal_label':      goal_label,
                    'goal_questions':  goal_questions,
                    'timeframe':       timeframe,
                    'target':          target,
                    'strategy_snippet': strategy_text[:300],
                })
                return jsonify({
                    'response': strategy_text,
                    'agent':    step.get('agent_label', 'Consultant (Zyon)'),
                })
        return jsonify({'error': 'No response generated.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@consultant_bp.route('/api/consultant/followup', methods=['POST'])
@login_required
def api_consultant_followup():
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'Empty message.'}), 400
    if len(message) > 2000:
        return jsonify({'error': 'Message too long.'}), 400

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available.'}), 503

    try:
        for step in manager.handle_consultant_request(message, history=history):
            if step['type'] == 'result':
                return jsonify({
                    'response': step['content'],
                    'agent':    step.get('agent_label', 'Consultant (Zyon)'),
                })
        return jsonify({'error': 'No response.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
