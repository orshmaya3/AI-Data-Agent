import os
import sys
from datetime import timedelta
from flask import Flask, redirect, url_for
from dotenv import load_dotenv

# Make sure the AI AGENT directory is on the path before any blueprint imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(BASE_DIR, 'flask_static'),
        template_folder=os.path.join(BASE_DIR, 'flask_templates')
    )
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'ai_data_agent_dev_fallback_change_me')
    app.permanent_session_lifetime = timedelta(minutes=30)

    from flask_routes.auth import auth_bp
    from flask_routes.dashboard import dashboard_bp
    from flask_routes.chat import chat_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)

    @app.route('/')
    def index():
        return redirect(url_for('auth.login'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001)
