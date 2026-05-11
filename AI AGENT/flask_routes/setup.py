from flask import Blueprint, render_template, redirect, url_for, session

setup_bp = Blueprint('setup', __name__)


@setup_bp.route('/setup')
def data_setup():
    if not session.get('username'):
        return redirect(url_for('auth.login'))

    if session.get('data_ready'):
        return redirect(url_for('dashboard.main'))

    session_id = session.get('session_id')
    if session_id:
        from flask_agents import get_session_status, SESSION_READY
        if get_session_status(session_id).get('status') == SESSION_READY:
            session['data_ready'] = True
            return redirect(url_for('setup.data_ready'))

    return render_template('setup.html', username=session.get('username'))


@setup_bp.route('/setup/ready')
def data_ready():
    if not session.get('username'):
        return redirect(url_for('auth.login'))
    if not session.get('data_ready'):
        return redirect(url_for('setup.data_setup'))
    return render_template('ready.html', username=session.get('username'))


@setup_bp.route('/setup/demo')
def setup_use_demo():
    if not session.get('username'):
        return redirect(url_for('auth.login'))
    session['data_ready'] = True
    return redirect(url_for('dashboard.main'))
