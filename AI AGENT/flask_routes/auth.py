from flask import Blueprint, render_template, request, redirect, url_for, session
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__)

login_attempts = {}
LOCKOUT_TIME = timedelta(minutes=10)
MAX_ATTEMPTS = 5

# Plaintext passwords only for the two hardcoded admin accounts
HARDCODED_ADMIN_PASSWORDS = {
    'Or':    'admin',
    'Taeir': 'admin',
}


def _restore_user_upload(user_id, session):
    """If the user has an active upload, pre-load it into the session manager."""
    from models import UserUpload
    from pathlib import Path
    import pandas as pd

    record = UserUpload.query.filter_by(user_id=user_id, is_active=True).first()
    if not record:
        return

    base = Path(__file__).parent.parent / 'data'
    full_path = base / record.parquet_path
    if not full_path.exists():
        return

    session_id = session.get('session_id')
    if not session_id:
        return

    try:
        df = pd.read_parquet(str(full_path))
        from flask_agents import register_session_data
        register_session_data(session_id, df)
    except Exception:
        pass


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username'):
        return redirect(url_for('dashboard.main'))

    ip = request.remote_addr
    now = datetime.now()

    if ip in login_attempts:
        attempts, last_attempt = login_attempts[ip]
        if attempts >= MAX_ATTEMPTS and now - last_attempt < LOCKOUT_TIME:
            return render_template('login.html', error="Too many failed attempts. Try again in a few minutes.")

    if request.method == 'POST':
        identifier = request.form.get('username', '').strip()
        password   = request.form.get('password', '').strip()

        from models import User
        user = (User.query.filter_by(username=identifier).first() or
                User.query.filter_by(email=identifier).first())

        auth_ok = False
        if user:
            if user.password_hash is None:
                # Hardcoded admin — plaintext check
                auth_ok = HARDCODED_ADMIN_PASSWORDS.get(user.username) == password
            else:
                from flask_bcrypt import check_password_hash
                auth_ok = check_password_hash(user.password_hash, password)

        if auth_ok:
            session['username'] = user.username
            session['role']     = user.role
            session['user_id']  = user.id
            session.permanent   = True
            login_attempts.pop(ip, None)

            from models import db
            user.last_login = datetime.utcnow()
            db.session.commit()

            _restore_user_upload(user.id, session)
            return redirect(url_for('dashboard.main'))

        attempts = login_attempts.get(ip, (0, now))[0] + 1
        login_attempts[ip] = (attempts, now)
        attempts_left = MAX_ATTEMPTS - attempts

        if attempts_left <= 0:
            error = "Maximum attempts reached. Try again in 10 minutes."
        else:
            error = f"Incorrect username or password. {attempts_left} attempt(s) remaining."

        return render_template('login.html', error=error)

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('username'):
        return redirect(url_for('dashboard.main'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not username or not email or not password:
            return render_template('register.html', error='All fields are required.')
        if len(username) < 3:
            return render_template('register.html', error='Username must be at least 3 characters.')
        if len(password) < 8:
            return render_template('register.html', error='Password must be at least 8 characters.')
        if password != confirm:
            return render_template('register.html', error='Passwords do not match.')

        from models import User, db
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already taken.')
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered.')

        from flask_bcrypt import generate_password_hash
        pw_hash = generate_password_hash(password).decode('utf-8')

        user = User(username=username, email=email, password_hash=pw_hash, role='user')
        db.session.add(user)
        db.session.commit()

        session['username'] = user.username
        session['role']     = user.role
        session['user_id']  = user.id
        session.permanent   = True
        return redirect(url_for('dashboard.main'))

    return render_template('register.html')


@auth_bp.route('/demo')
def demo():
    session['username'] = 'Demo'
    session['role']     = 'viewer'
    session['user_id']  = None
    session.permanent   = True
    return redirect(url_for('dashboard.main'))


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
