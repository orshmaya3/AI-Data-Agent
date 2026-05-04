import io
import json
import os

from flask import Blueprint, request, jsonify, session

from flask_routes.upload_utils import (
    detect_column_mapping,
    mapping_is_complete,
    apply_mapping_and_clean,
    ALL_COLUMNS,
)

upload_bp = Blueprint('upload', __name__)

MAX_FILE_SIZE_MB  = 50
MAX_ROWS          = 500_000
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}


@upload_bp.route('/api/upload', methods=['POST'])
def api_upload():
    from flask_agents import (
        register_session_data,
        evict_expired_sessions,
        _session_registry,
        MAX_UPLOAD_SESSIONS,
    )

    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID. Please reload the page.'}), 400

    # Periodic cleanup
    evict_expired_sessions(max_age_seconds=3600)

    # Capacity guard
    with __import__('flask_agents')._session_lock:
        current_count = len(_session_registry)
    if current_count >= MAX_UPLOAD_SESSIONS:
        return jsonify({
            'error': 'Server is at capacity. Please try again in a few minutes.'
        }), 503

    # ── File validation ───────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            'error': f'Unsupported file type "{ext}". Please upload a CSV, XLSX, or XLS file.'
        }), 415

    file_bytes = f.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({
            'error': f'File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.'
        }), 413

    # ── Parse ─────────────────────────────────────────────────────────────
    try:
        import pandas as pd
        buf = io.BytesIO(file_bytes)
        if ext == '.csv':
            try:
                df_raw = pd.read_csv(buf, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                buf.seek(0)
                df_raw = pd.read_csv(buf, encoding='ISO-8859-1', low_memory=False)
        elif ext == '.xlsx':
            df_raw = pd.read_excel(buf, engine='openpyxl', nrows=MAX_ROWS + 1)
        else:  # .xls
            df_raw = pd.read_excel(buf, engine='xlrd', nrows=MAX_ROWS + 1)
    except Exception as e:
        return jsonify({'error': f'Could not parse file: {e}'}), 422

    if len(df_raw) == 0:
        return jsonify({'error': 'The uploaded file contains no data rows.'}), 422

    if len(df_raw) > MAX_ROWS:
        return jsonify({
            'error': (
                f'File has {len(df_raw):,} rows. Maximum is {MAX_ROWS:,}. '
                f'Please upload a smaller sample.'
            )
        }), 413

    # ── Column detection ──────────────────────────────────────────────────
    uploaded_cols = list(df_raw.columns)
    detected_mapping = detect_column_mapping(uploaded_cols)

    # Merge any manual mapping override sent by the client
    manual_mapping_str = request.form.get('mapping')
    if manual_mapping_str:
        try:
            manual_mapping = json.loads(manual_mapping_str)
            for k, v in manual_mapping.items():
                if k in detected_mapping and v:
                    detected_mapping[k] = v
        except (json.JSONDecodeError, TypeError):
            pass

    if not mapping_is_complete(detected_mapping):
        preview_rows = df_raw.head(5).fillna('').astype(str).values.tolist()
        dtypes = {col: str(dt) for col, dt in df_raw.dtypes.items()}
        return jsonify({
            'status':           'mapping_required',
            'columns':          uploaded_cols,
            'detected_mapping': detected_mapping,
            'preview':          preview_rows,
            'dtypes':           dtypes,
            'row_count':        len(df_raw),
        }), 200

    # ── Clean ─────────────────────────────────────────────────────────────
    try:
        df_clean, warnings = apply_mapping_and_clean(df_raw, detected_mapping)
    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    if len(df_clean) < 10:
        return jsonify({
            'error': (
                f'After cleaning, only {len(df_clean)} rows remain. '
                f'Please check that your data has valid Customer IDs, '
                f'Descriptions, and positive Prices.'
            )
        }), 422

    # ── Persist parquet for logged-in users ──────────────────────────────
    user_id = session.get('user_id')
    if user_id:
        _save_upload_parquet(user_id, df_clean, detected_mapping, f.filename)

    # ── Register + kick off background manager init ───────────────────────
    register_session_data(session_id, df_clean)

    return jsonify({
        'status':   'processing',
        'session_id': session_id,
        'rows':     len(df_clean),
        'warnings': warnings,
        'mapping':  detected_mapping,
    }), 202


@upload_bp.route('/api/upload/status', methods=['GET'])
def api_upload_status():
    from flask_agents import get_session_status
    session_id = session.get('session_id')
    return jsonify(get_session_status(session_id)), 200


@upload_bp.route('/api/upload/session', methods=['DELETE'])
def api_clear_upload():
    from flask_agents import _session_registry, _session_lock
    session_id = session.get('session_id')
    if session_id:
        with _session_lock:
            _session_registry.pop(session_id, None)
    return jsonify({'status': 'cleared'}), 200


def _save_upload_parquet(user_id, df_clean, detected_mapping, original_filename):
    """Save cleaned DataFrame as parquet and record it in the DB."""
    import uuid as _uuid
    from pathlib import Path
    from models import db, UserUpload

    upload_dir = Path(__file__).parent.parent / 'data' / 'uploads' / str(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    parquet_name = f"{_uuid.uuid4().hex}.parquet"
    parquet_full = upload_dir / parquet_name
    df_clean.to_parquet(str(parquet_full), index=False)

    # Relative path stored in DB (relative to data/)
    parquet_rel = str(Path('uploads') / str(user_id) / parquet_name)

    # Deactivate previous uploads for this user
    UserUpload.query.filter_by(user_id=user_id, is_active=True).update({'is_active': False})

    record = UserUpload(
        user_id=user_id,
        filename_original=original_filename,
        parquet_path=parquet_rel,
        row_count=len(df_clean),
        column_mapping=json.dumps(detected_mapping),
    )
    db.session.add(record)
    db.session.commit()
