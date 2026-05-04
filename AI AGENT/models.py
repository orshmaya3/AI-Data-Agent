from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(254), unique=True, nullable=False)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)  # NULL for hardcoded admins
    role          = db.Column(db.String(20), nullable=False, default='user')
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login    = db.Column(db.DateTime, nullable=True)

    uploads = db.relationship('UserUpload', back_populates='user', lazy='select')
    plans   = db.relationship('ConsultantPlan', back_populates='user', lazy='select')


class UserUpload(db.Model):
    __tablename__ = 'user_uploads'

    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename_original = db.Column(db.String(255), nullable=False)
    parquet_path      = db.Column(db.String(512), nullable=False)
    row_count         = db.Column(db.Integer)
    column_mapping    = db.Column(db.Text)  # JSON string
    uploaded_at       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_active         = db.Column(db.Boolean, default=True)

    user = db.relationship('User', back_populates='uploads')


class ConsultantPlan(db.Model):
    __tablename__ = 'consultant_plans'

    id                   = db.Column(db.Integer, primary_key=True)
    user_id              = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    business_profile     = db.Column(db.Text, nullable=False)   # JSON
    goal_label           = db.Column(db.String(120))
    goal_text            = db.Column(db.Text)
    timeframe            = db.Column(db.String(80))
    target               = db.Column(db.Text)
    strategy_text        = db.Column(db.Text, nullable=False)
    conversation_history = db.Column(db.Text)                   # JSON array [{role, content}]
    created_at           = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at           = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    next_checkin_due     = db.Column(db.DateTime, nullable=True)

    user    = db.relationship('User', back_populates='plans')
    surveys = db.relationship('WeeklySurvey', back_populates='plan', lazy='select')


class WeeklySurvey(db.Model):
    __tablename__ = 'weekly_surveys'

    id                  = db.Column(db.Integer, primary_key=True)
    plan_id             = db.Column(db.Integer, db.ForeignKey('consultant_plans.id'), nullable=False)
    user_id             = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    survey_conversation = db.Column(db.Text)  # JSON array [{role, content}]
    updated_strategy    = db.Column(db.Text, nullable=True)
    completed_at        = db.Column(db.DateTime, nullable=True)
    created_at          = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    plan = db.relationship('ConsultantPlan', back_populates='surveys')
