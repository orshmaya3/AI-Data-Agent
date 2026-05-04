import json
import threading
import uuid
from datetime import datetime, timezone, timedelta
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


def _build_survey_opening_prompt(plan) -> str:
    profile = json.loads(plan.business_profile)
    days_since = (datetime.now(timezone.utc) - plan.created_at.replace(tzinfo=timezone.utc)).days

    return (
        "[SURVEY MODE — WEEKLY PROGRESS CHECK-IN]\n\n"
        f"You are Zyon conducting a weekly progress check-in with {profile.get('name', 'this business owner')}.\n\n"
        "CONTEXT YOU HAVE:\n"
        f"- Business type: {profile.get('business_type', 'unknown')}\n"
        f"- Original goal: {plan.goal_label or plan.goal_text}\n"
        f"- Days since plan was created: {days_since}\n"
        "- Their previous strategy (do not repeat this verbatim):\n"
        f"{(plan.strategy_text or '')[:800]}\n\n"
        "YOUR JOB IN THIS CHECK-IN:\n"
        "1. Open with a warm but direct greeting referencing their specific goal.\n"
        "2. Ask 2-3 short, focused questions to understand what's happened:\n"
        "   - Did they try any of the recommended actions?\n"
        "   - What results have they seen (even small ones)?\n"
        "   - What got in the way?\n"
        "3. After they answer, run your standard data tools to see how the numbers have moved.\n"
        "4. Produce an UPDATED plan that incorporates both what they told you AND what the new data shows.\n"
        "5. When you write the updated plan, start it with the exact marker: [[PLAN_UPDATE]]\n\n"
        "TONE: You are a trusted advisor who remembers their situation. Build on what you already know. "
        "Be brief in the opening questions — this is a check-in, not a full consultation.\n\n"
        "Start now with your opening greeting and first question."
    )


def _build_survey_continuation_prompt(plan) -> str:
    return (
        "[SURVEY MODE — CONTINUING CHECK-IN]\n"
        f"Original goal: {plan.goal_label or plan.goal_text}\n"
        f"Original strategy snippet: {(plan.strategy_text or '')[:500]}\n\n"
        "You are mid-way through a weekly check-in. The conversation so far is in your history. "
        "Continue the check-in. When you have gathered enough information (usually after 2-3 user replies), "
        "run your data tools and produce an updated strategy. "
        "Begin the updated strategy section with [[PLAN_UPDATE]]. "
        "If you need more information first, ask one focused follow-up question."
    )


@consultant_bp.route('/consultant')
@login_required
def consultant():
    user_id = session.get('user_id')
    existing_plan = None
    survey_due = False

    if user_id:
        from models import ConsultantPlan
        plan = (ConsultantPlan.query
                .filter_by(user_id=user_id)
                .order_by(ConsultantPlan.updated_at.desc())
                .first())
        if plan:
            existing_plan = {
                'id':                   plan.id,
                'strategy_text':        plan.strategy_text,
                'goal_label':           plan.goal_label or '',
                'business_profile':     json.loads(plan.business_profile),
                'conversation_history': json.loads(plan.conversation_history or '[]'),
                'updated_at':           plan.updated_at.isoformat(),
            }
            session['active_plan_id'] = plan.id
            if plan.next_checkin_due:
                due = plan.next_checkin_due
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                survey_due = due <= datetime.now(timezone.utc)

    return render_template(
        'consultant.html',
        username=session.get('username'),
        existing_plan=existing_plan,
        survey_due=survey_due,
    )


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

    profile       = session.get('business_profile', {})
    profile_name  = profile.get('name', '')
    profile_email = profile.get('email', '')
    profile_type  = profile.get('business_type', '')
    profile_type_label = profile.get('business_type_other') or profile_type

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

                # Persist plan for logged-in users
                user_id = session.get('user_id')
                if user_id:
                    from models import db, ConsultantPlan
                    initial_history = json.dumps([
                        {'role': 'user',      'content': prompt},
                        {'role': 'assistant', 'content': strategy_text},
                    ])
                    plan = ConsultantPlan(
                        user_id=user_id,
                        business_profile=json.dumps(profile),
                        goal_label=goal_label,
                        goal_text=goal,
                        timeframe=timeframe,
                        target=target,
                        strategy_text=strategy_text,
                        conversation_history=initial_history,
                        next_checkin_due=datetime.now(timezone.utc) + timedelta(days=7),
                    )
                    db.session.add(plan)
                    db.session.commit()
                    session['active_plan_id'] = plan.id

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
                response_text = step['content']

                # Persist conversation update
                user_id  = session.get('user_id')
                plan_id  = session.get('active_plan_id')
                if user_id and plan_id:
                    from models import db, ConsultantPlan
                    plan = ConsultantPlan.query.get(plan_id)
                    if plan and plan.user_id == user_id:
                        h = json.loads(plan.conversation_history or '[]')
                        h.append({'role': 'user',      'content': message})
                        h.append({'role': 'assistant', 'content': response_text})
                        plan.conversation_history = json.dumps(h)
                        plan.updated_at = datetime.now(timezone.utc)
                        db.session.commit()

                return jsonify({
                    'response': response_text,
                    'agent':    step.get('agent_label', 'Consultant (Zyon)'),
                })
        return jsonify({'error': 'No response.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Weekly survey routes ──────────────────────────────────────────────────────

@consultant_bp.route('/api/consultant/survey/start', methods=['POST'])
@login_required
def api_survey_start():
    user_id = session.get('user_id')
    plan_id = session.get('active_plan_id')

    if not user_id or not plan_id:
        return jsonify({'error': 'No active plan.'}), 404

    from models import db, ConsultantPlan, WeeklySurvey
    plan = ConsultantPlan.query.get(plan_id)
    if not plan or plan.user_id != user_id:
        return jsonify({'error': 'Plan not found.'}), 404

    survey = WeeklySurvey(
        plan_id=plan.id,
        user_id=user_id,
        survey_conversation='[]',
    )
    db.session.add(survey)
    db.session.commit()
    session['active_survey_id'] = survey.id

    opening_prompt = _build_survey_opening_prompt(plan)

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))
    if manager is None:
        return jsonify({'error': 'Agent not available.'}), 503

    try:
        for step in manager.handle_consultant_request(opening_prompt):
            if step['type'] == 'result':
                convo = [{'role': 'assistant', 'content': step['content']}]
                survey.survey_conversation = json.dumps(convo)
                db.session.commit()
                return jsonify({'response': step['content'], 'survey_id': survey.id})
        return jsonify({'error': 'No response from Zyon.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@consultant_bp.route('/api/consultant/survey/reply', methods=['POST'])
@login_required
def api_survey_reply():
    data      = request.get_json(silent=True) or {}
    message   = data.get('message', '').strip()
    user_id   = session.get('user_id')
    survey_id = session.get('active_survey_id')

    if not message:
        return jsonify({'error': 'Empty message.'}), 400
    if not survey_id or not user_id:
        return jsonify({'error': 'No active survey.'}), 404

    from models import db, ConsultantPlan, WeeklySurvey
    survey = WeeklySurvey.query.get(survey_id)
    if not survey or survey.user_id != user_id:
        return jsonify({'error': 'Survey not found.'}), 404

    plan  = ConsultantPlan.query.get(survey.plan_id)
    convo = json.loads(survey.survey_conversation or '[]')
    convo.append({'role': 'user', 'content': message})

    continuation_prompt = _build_survey_continuation_prompt(plan)

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))
    if manager is None:
        return jsonify({'error': 'Agent not available.'}), 503

    try:
        for step in manager.handle_consultant_request(continuation_prompt, history=convo):
            if step['type'] == 'result':
                response_text = step['content']
                convo.append({'role': 'assistant', 'content': response_text})
                survey.survey_conversation = json.dumps(convo)

                if '[[PLAN_UPDATE]]' in response_text:
                    updated_strategy = response_text.split('[[PLAN_UPDATE]]', 1)[1].strip()
                    survey.updated_strategy = updated_strategy
                    survey.completed_at     = datetime.now(timezone.utc)

                    plan.strategy_text        = updated_strategy
                    plan.conversation_history = json.dumps(convo)
                    plan.updated_at           = datetime.now(timezone.utc)
                    plan.next_checkin_due     = datetime.now(timezone.utc) + timedelta(days=7)

                db.session.commit()
                return jsonify({
                    'response':        response_text,
                    'plan_updated':    '[[PLAN_UPDATE]]' in response_text,
                    'updated_strategy': survey.updated_strategy,
                })
        return jsonify({'error': 'No response.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
