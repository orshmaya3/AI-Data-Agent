from flask import Blueprint, render_template, request, redirect, url_for, session
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__)

login_attempts = {}
LOCKOUT_TIME = timedelta(minutes=10)
MAX_ATTEMPTS = 5

USERS = {
    "Or":   {"password": "admin",   "role": "admin"},
    "Taeir": {"password": "admin", "role": "admin"},
}


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username'):
        return redirect(url_for('dashboard.main'))

    ip = request.remote_addr
    now = datetime.now()

    if ip in login_attempts:
        attempts, last_attempt = login_attempts[ip]
        if attempts >= MAX_ATTEMPTS and now - last_attempt < LOCKOUT_TIME:
            return render_template('login.html', error="נחסמת זמנית. נסה שוב בעוד כמה דקות.")

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = USERS.get(username)

        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            session.permanent = True
            login_attempts.pop(ip, None)
            return redirect(url_for('dashboard.main'))

        attempts = login_attempts.get(ip, (0, now))[0] + 1
        login_attempts[ip] = (attempts, now)
        attempts_left = MAX_ATTEMPTS - attempts

        if attempts_left <= 0:
            error = "הגעת למספר מקסימלי של ניסיונות. נסה שוב בעוד 10 דקות."
        else:
            error = f"שם משתמש או סיסמא שגויים. נותרו {attempts_left} ניסיונות."

        return render_template('login.html', error=error)

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
