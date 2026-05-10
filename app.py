import os
import joblib
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file, abort)
from ml_model.database import (init_db, save_prediction, register_user, login_user,
                                get_user_predictions, reset_password, delete_prediction,
                                get_user_info, change_password, get_conn)
from datetime import timedelta
from functools import wraps
import io
import json
from pdf_generator import generate_prediction_pdf

app = Flask(__name__)
app.secret_key = 'forensic_bone_secret_key_2024'
app.permanent_session_lifetime = timedelta(minutes=30)


@app.before_request
def check_session_timeout():
    if 'user_id' in session:
        session.permanent = True


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model')

label_encoder = joblib.load(os.path.join(MODEL_DIR, 'age_label_encoder.pkl'))

BONE_MODELS = {}
for bone_key in ['humerus', 'radius', 'femur', 'tibia', 'os_coxae']:
    BONE_MODELS[bone_key] = {
        'gender':   joblib.load(os.path.join(MODEL_DIR, f'gender_{bone_key}.pkl')),
        'age':      joblib.load(os.path.join(MODEL_DIR, f'age_{bone_key}.pkl')),
        'scaler':   joblib.load(os.path.join(MODEL_DIR, f'scaler_{bone_key}.pkl')),
        'features': joblib.load(os.path.join(MODEL_DIR, f'features_{bone_key}.pkl')),
    }

init_db()

BONE_MEASUREMENTS = {
    'Humerus':  ['LHML','LHEB','LHHD','LHMLD','LHAPD','RHML','RHEB','RHHD','RHMLD','RHAPD'],
    'Radius':   ['LRML','LRMLD','LRAPD','RRML','RRMLD','RRAPD'],
    'Femur':    ['LFML','LFBL','LFEB','LFAB','LFHD','LFMLD','LFAPD','RFML','RFBL','RFEB','RFAB','RFHD','RFMLD','RFAPD'],
    'Tibia':    ['LTML','LTPB','LTMLD','LTAPD','RTML','RTPB','RTMLD','RTAPD'],
    'Os Coxae': ['BIB','LIBL','RIBL','LAcH','RAcH'],
}


def compute_avg_features(data):
    avg_pairs = {
        'AVG_HML':  ('LHML','RHML'),  'AVG_HEB':  ('LHEB','RHEB'),
        'AVG_HHD':  ('LHHD','RHHD'),  'AVG_HMLD': ('LHMLD','RHMLD'),
        'AVG_HAPD': ('LHAPD','RHAPD'),'AVG_RML':  ('LRML','RRML'),
        'AVG_RMLD': ('LRMLD','RRMLD'),'AVG_RAPD': ('LRAPD','RRAPD'),
        'AVG_FML':  ('LFML','RFML'),  'AVG_FBL':  ('LFBL','RFBL'),
        'AVG_FEB':  ('LFEB','RFEB'),  'AVG_FAB':  ('LFAB','RFAB'),
        'AVG_FHD':  ('LFHD','RFHD'),  'AVG_FMLD': ('LFMLD','RFMLD'),
        'AVG_FAPD': ('LFAPD','RFAPD'),'AVG_TML':  ('LTML','RTML'),
        'AVG_TPB':  ('LTPB','RTPB'),  'AVG_TMLD': ('LTMLD','RTMLD'),
        'AVG_TAPD': ('LTAPD','RTAPD'),'AVG_IBL':  ('LIBL','RIBL'),
        'AVG_AcH':  ('LAcH','RAcH'),
    }
    for avg_col, (left, right) in avg_pairs.items():
        l_val = float(data.get(left, 0) or 0)
        r_val = float(data.get(right, 0) or 0)
        if l_val > 0 and r_val > 0:
            data[avg_col] = (l_val + r_val) / 2
        elif l_val > 0:
            data[avg_col] = l_val
        elif r_val > 0:
            data[avg_col] = r_val
        else:
            data[avg_col] = 0
    return data


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username          = request.form.get('username', '').strip()
        password          = request.form.get('password', '').strip()
        confirm_password  = request.form.get('confirm_password', '').strip()
        full_name         = request.form.get('full_name', '').strip()
        security_question = request.form.get('security_question', '').strip()
        security_answer   = request.form.get('security_answer', '').strip()

        if not all([username, password, confirm_password, full_name,
                    security_question, security_answer]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'danger')
            return render_template('register.html')
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')

        success, message = register_user(
            username, password, full_name, security_question, security_answer
        )
        if success:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'danger')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        success, result = login_user(username, password)
        if success:
            session['user_id']   = result['id']
            session['username']  = result['username']   # FIX: username save කරනවා
            session['full_name'] = result['full_name']
            flash(f"Welcome, {result['full_name']}!", 'success')
            return redirect(url_for('index'))
        else:
            flash(result, 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username          = request.form.get('username', '').strip()
        security_question = request.form.get('security_question', '').strip()
        security_answer   = request.form.get('security_answer', '').strip()
        new_password      = request.form.get('new_password', '').strip()
        ok, msg = reset_password(
            username, security_question, security_answer, new_password
        )
        flash(msg, 'success' if ok else 'danger')
        if ok:
            return redirect(url_for('login'))
    return render_template('forgot_password.html')


@app.route('/check-username', methods=['POST'])
def check_username():
    username = request.form.get('username', '').strip()
    if not username:
        return jsonify({'available': True})
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username=%s', (username,))
    exists = cursor.fetchone()
    cursor.close()
    conn.close()
    return jsonify({'available': exists is None})


# ── Main routes ───────────────────────────────────────────────────────────────

@app.route('/index')
@login_required
def index():
    return render_template('index.html',
        bone_measurements=BONE_MEASUREMENTS,
        full_name=session.get('full_name'),
    )


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form_data      = request.form.to_dict()
    selected_bones = request.form.getlist('bones')

    if not selected_bones:
        flash('Please select at least one bone.', 'warning')
        return redirect(url_for('index'))

    gender_probs, age_probs = [], []
    measurements = {}

    for bone in selected_bones:
        bone_key  = bone.replace(' ', '_').lower()
        model_set = BONE_MODELS.get(bone_key)
        if not model_set:
            continue

        features  = model_set['features']
        bone_vals = {}
        for f in features:
            val           = form_data.get(f, 0)
            bone_vals[f]  = float(val) if val else 0
            measurements[f] = bone_vals[f]

        bone_vals = compute_avg_features(bone_vals)
        X  = pd.DataFrame([{f: bone_vals.get(f, 0) for f in features}])

        gp = model_set['gender'].predict_proba(X)[0]
        gender_probs.append(gp)

        X_scaled = model_set['scaler'].transform(X)
        ap = model_set['age'].predict_proba(X_scaled)[0]
        age_probs.append(ap)

    avg_gender   = np.mean(gender_probs, axis=0)
    avg_age      = np.mean(age_probs, axis=0)

    gender_pred  = int(np.argmax(avg_gender))
    gender_conf  = round(float(np.max(avg_gender)) * 100, 2)
    gender_label = 'Female' if gender_pred == 1 else 'Male'

    age_pred  = int(np.argmax(avg_age))
    age_conf  = round(float(np.max(avg_age)) * 100, 2)
    age_label = label_encoder.inverse_transform([age_pred])[0]

    case_ref = save_prediction(
        session['user_id'], gender_label, gender_conf,
        age_label, age_conf, selected_bones, measurements,
    )

    return render_template('result.html',
        gender=gender_label,
        gender_conf=gender_conf,
        age=age_label,
        age_conf=age_conf,
        case_ref=case_ref,
        selected_bones=selected_bones,
        full_name=session.get('full_name'),
        measurements=measurements,
    )


@app.route('/history')
@login_required
def history():
    predictions = get_user_predictions(session['user_id'])
    return render_template('history.html',
        predictions=predictions,
        full_name=session.get('full_name'),
    )


@app.route('/export_pdf/<int:prediction_id>')
@login_required                                   # FIX: decorator add කළා
def export_pdf(prediction_id):
    # ── DB query — get_conn() use කරනවා (ඔයාගේ existing helper) ──
    conn   = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT id, case_reference, bone_type, predicted_sex, sex_confidence,
                  predicted_age, age_confidence, created_at, measurements
           FROM predictions
           WHERE id = %s AND user_id = %s""",
        (prediction_id, session['user_id']),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        abort(404)

    # ── Row → dict (column names match SELECT order above) ──────────
    prediction = {
        'id':             row[0],
        'case_reference': row[1],
        'bone_type':      row[2],
        'predicted_sex':  row[3],
        'sex_confidence': row[4],
        'predicted_age':  row[5],
        'age_confidence': row[6],
        'timestamp':      row[7],
        'measurements':   {},
    }

    # measurements — JSON string or dict
    raw_measurements = row[8]
    if isinstance(raw_measurements, str):
        try:
            prediction['measurements'] = json.loads(raw_measurements)
        except (json.JSONDecodeError, TypeError):
            prediction['measurements'] = {}
    elif isinstance(raw_measurements, dict):
        prediction['measurements'] = raw_measurements

    pdf_bytes = generate_prediction_pdf(
        prediction,
        generated_by=session.get('username', session.get('full_name', 'Unknown')),
    )

    filename = f"forensic_report_{prediction['case_reference']}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename,
    )


@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_pred(prediction_id):
    delete_prediction(prediction_id, session['user_id'])
    flash('Case deleted successfully.', 'success')
    return redirect(url_for('history'))


@app.route('/profile')
@login_required
def profile():
    info       = get_user_info(session['user_id'])
    total      = len(get_user_predictions(session['user_id']))
    created_at = info[2]
    if hasattr(created_at, 'strftime'):
        created_at = created_at.strftime('%Y-%m-%d')
    return render_template('profile.html',
        username=info[0],
        full_name=info[1],
        created_at=created_at,
        total_cases=total,
    )


@app.route('/profile/change-password', methods=['POST'])
@login_required
def change_pwd():
    current = request.form.get('current_password', '').strip()
    new_pwd = request.form.get('new_password', '').strip()
    confirm = request.form.get('confirm_password', '').strip()

    if new_pwd != confirm:
        flash('Passwords do not match!', 'danger')
        return redirect(url_for('profile'))
    if len(new_pwd) < 6:
        flash('Password must be at least 6 characters!', 'danger')
        return redirect(url_for('profile'))

    ok, msg = change_password(session['user_id'], current, new_pwd)
    flash(msg, 'success' if ok else 'danger')
    return redirect(url_for('profile'))


if __name__ == '__main__':
    app.run(debug=True)