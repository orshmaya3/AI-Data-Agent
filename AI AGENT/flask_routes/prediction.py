import json
import queue
import threading

from flask import Blueprint, render_template, session, request, jsonify, Response, stream_with_context
from flask_routes.utils import login_required

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html', username=session.get('username'))


@prediction_bp.route('/api/prediction/chat', methods=['POST'])
@login_required
def api_prediction_chat():
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'Empty message'}), 400
    if len(message) > 2000:
        return jsonify({'error': 'Message too long.'}), 400

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available.'}), 503

    def generate():
        q = queue.Queue()

        def worker():
            try:
                for step in manager.handle_prediction_request(message, history=history):
                    q.put(('event', step))
            except Exception as e:
                q.put(('error', str(e)))
            finally:
                q.put(('done', None))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            try:
                kind, payload = q.get(timeout=20)
            except queue.Empty:
                yield ': keep-alive\n\n'
                continue
            if kind == 'done':
                break
            elif kind == 'error':
                yield f"data: {json.dumps({'type': 'error', 'content': payload})}\n\n"
                break
            else:
                yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@prediction_bp.route('/api/prediction/metrics')
@login_required
def api_prediction_metrics():
    from flask_routes.utils import resolve_manager
    from flask_agents import get_manager_error
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': f'Prediction agent unavailable. {get_manager_error() or ""}'}), 503

    try:
        pa = manager.prediction_analyst
        churn  = pa.get_churn_risk_summary()
        repeat = pa.get_repeat_purchase_probability()
        return jsonify({'churn': churn, 'repeat': repeat})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/api/prediction/charts')
@login_required
def api_prediction_charts():
    from flask_routes.utils import resolve_manager
    from flask_agents import get_manager_error
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': f'Prediction agent unavailable. {get_manager_error() or ""}'}), 503

    try:
        pa = manager.prediction_analyst
        return jsonify({
            'forecast':    pa.get_revenue_forecast(horizon_months=3),
            'high_growth': pa.get_high_growth_products(top_n=5),
            'slow_movers': pa.get_slow_movers(top_n=8),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
