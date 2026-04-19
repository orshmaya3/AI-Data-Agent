import os
import sys
from datetime import timedelta
from flask import Flask, redirect, url_for
from dotenv import load_dotenv

# Make sure the AI AGENT directory and agents/ subfolder are on the path
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, os.path.join(_BASE_DIR, 'agents'))

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(BASE_DIR, 'flask_static'),
        template_folder=os.path.join(BASE_DIR, 'flask_templates')
    )
    app.secret_key = (
        os.environ.get('SECRET_KEY') or
        os.environ.get('FLASK_SECRET_KEY') or
        'ai_data_agent_dev_fallback_change_me'
    )
    app.permanent_session_lifetime = timedelta(minutes=30)

    from flask_routes.dashboard import dashboard_bp
    from flask_routes.chat import chat_bp
    from flask_routes.prediction import prediction_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(prediction_bp)

    @app.route('/')
    def index():
        return redirect(url_for('dashboard.main'))

    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import render_template
        return render_template('500.html'), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
