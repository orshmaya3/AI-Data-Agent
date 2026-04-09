from flask import Blueprint, render_template, session, request, jsonify
from flask_routes.utils import login_required

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/chat')
@login_required
def chat():
    return render_template('chat.html', username=session.get('username'))


@chat_bp.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if manager is None:
        return jsonify({'error': 'Agent not available. Check that the data file is loaded.'}), 500

    try:
        response = manager.handle_request(message, history=history)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
