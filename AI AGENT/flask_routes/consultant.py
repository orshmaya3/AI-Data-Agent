from flask import Blueprint, render_template, session, request, jsonify
from flask_routes.utils import login_required

consultant_bp = Blueprint('consultant', __name__)


@consultant_bp.route('/consultant')
@login_required
def consultant():
    return render_template('consultant.html', username=session.get('username'))


@consultant_bp.route('/api/consultant/analyze', methods=['POST'])
@login_required
def api_consultant_analyze():
    data      = request.get_json(silent=True) or {}
    goal      = data.get('goal', '').strip()
    target    = data.get('target', '').strip()
    timeframe = data.get('timeframe', 'the next 3 months').strip()

    if not goal:
        return jsonify({'error': 'No goal provided.'}), 400

    # Build a rich natural-language prompt for Zyon
    parts = [f"My business goal is: {goal}."]
    if target:
        parts.append(f"My target is: {target}.")
    parts.append(f"I want to achieve this in {timeframe}.")
    parts.append(
        "Please analyse my business data thoroughly — call at least 5 different tools "
        "to look at this from multiple angles — then give me a clear, practical action plan "
        "in plain English. Tell me exactly what to do, why it matters, and how to do it."
    )
    prompt = " ".join(parts)

    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if manager is None:
        return jsonify({'error': 'Agent not available. Please check server logs.'}), 503

    try:
        for step in manager.handle_consultant_request(prompt):
            if step['type'] == 'result':
                return jsonify({
                    'response': step['content'],
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

    from flask_agents import get_agents
    df, manager, sales = get_agents()

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
