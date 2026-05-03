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

    if len(message) > 2000:
        return jsonify({'error': 'Message too long. Please keep questions under 2000 characters.'}), 400

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available. Check that the data file is loaded.'}), 500

    try:
        # handle_request is a generator — consume it and extract the final result
        response = ""
        agent_label = ""
        for step in manager.handle_request(message, history=history):
            if step["type"] == "result":
                response = step["content"]
                agent_label = step.get("agent_label", "")
        return jsonify({'response': response, 'agent': agent_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
