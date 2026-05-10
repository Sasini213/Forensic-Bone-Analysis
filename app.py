import os
import joblib
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file, abort, send_from_directory)
from ml_model.database import (init_db, save_prediction, register_user, login_user,
                                get_user_predictions, reset_password, delete_prediction,
                                get_user_info, change_password, get_conn,
                                get_prediction_measurements, update_analyst_notes)
from datetime import timedelta
from functools import wraps
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import io
from reportlab.pdfgen import canvas as rl_canvas


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
            session['username']  = result.get('username') or result.get('user_name') or result.get('name', '')
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


# ── History + API routes ──────────────────────────────────────────────────────

@app.route('/history')
@login_required
def history():
    # render_template() use කරනවා — හැබැයි Jinja2 data pass කරන්නේ නෑ
    # history.html එකේ Jinja2 tags නෑ, data JS fetch() කරනවා
    return render_template('history.html')


@app.route('/api/me')
@login_required
def api_me():
    # Nav bar username get කරන්න
    return jsonify({'full_name': session.get('full_name', 'User')})


@app.route('/api/predictions')
@login_required
def api_predictions():
    # Table data JSON විදිහට return කරනවා
    rows = get_user_predictions(session['user_id'])
    predictions = []
    for r in rows:
      predictions.append({
    'id':         r[0],
    'case_ref':   r[2],
    'sex':        r[3],
    'sex_conf':   r[4],
    'age_range':  r[5],
    'age_conf':   r[6],
    'bones':      r[7],
    'created_at': str(r[8]) if r[8] else None,
    'notes':      r[9] if len(r) > 9 else '',
})
    return jsonify({'predictions': predictions})


@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_pred(prediction_id):
    delete_prediction(prediction_id, session['user_id'])
    # JSON return කරනවා — JS fetch() handle කරනවා
    return jsonify({'success': True})

@app.route('/update-notes/<int:prediction_id>', methods=['POST'])
@login_required
def update_notes(prediction_id):
    notes = request.form.get('notes', '').strip()
    update_analyst_notes(prediction_id, session['user_id'], notes)
    return jsonify({'success': True})

# ── PDF Report Generator ──────────────────────────────────────────────────────
def generate_forensic_report(output, data):
    from reportlab.pdfgen import canvas as rl_canvas
    W, H = A4
    c = rl_canvas.Canvas(output, pagesize=A4)

    DARK_BLUE   = colors.HexColor('#0d1b2a')
    MID_BLUE    = colors.HexColor('#1a3a5c')
    ACCENT      = colors.HexColor('#2e6da4')
    LIGHT_BLUE  = colors.HexColor('#d6e8f7')
    LIGHT_GREY  = colors.HexColor('#f5f5f5')
    MED_GREY    = colors.HexColor('#cccccc')
    TEXT_DARK   = colors.HexColor('#1a1a2e')
    TEXT_GREY   = colors.HexColor('#555555')
    GREEN       = colors.HexColor('#276749')
    GREEN_LIGHT = colors.HexColor('#e6f4ed')
    PINK        = colors.HexColor('#9a3a5c')
    PINK_LIGHT  = colors.HexColor('#fce8f0')
    WHITE       = colors.white
    margin      = 1.8 * cm

    # Header band
    c.setFillColor(DARK_BLUE)
    c.rect(0, H - 3.5*cm, W, 3.5*cm, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, H - 3.5*cm, W, 0.22*cm, fill=1, stroke=0)

    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 15)
    c.drawString(margin, H - 1.5*cm, 'FORENSIC BONE ANALYSIS SYSTEM')
    c.setFont('Helvetica', 9)
    c.setFillColor(colors.HexColor('#aac8e8'))
    c.drawString(margin, H - 2.1*cm, "Sri Lanka Police / Government Analyst's Department")
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 9)
    c.drawRightString(W - margin, H - 1.5*cm, f"Case Reference: {data.get('case_ref', '—')}")
    c.setFont('Helvetica', 8)
    c.setFillColor(colors.HexColor('#aac8e8'))
    c.drawRightString(W - margin, H - 2.0*cm, 'CONFIDENTIAL')

    y = H - 4.4*cm

    # Section helper
    def section_header(title):
        nonlocal y
        c.setFillColor(ACCENT)
        c.rect(margin, y - 0.05*cm, 0.35*cm, 0.45*cm, fill=1, stroke=0)
        c.setFillColor(TEXT_DARK)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(margin + 0.55*cm, y, title)
        y -= 0.5*cm
        c.setStrokeColor(MED_GREY)
        c.setLineWidth(0.5)
        c.line(margin, y, W - margin, y)
        y -= 0.35*cm

    # Report Information
    section_header('REPORT INFORMATION')
    info_left  = [('Report Generated', data.get('generated_at', '—')),
                  ('Prediction Date',  data.get('created_at', '—')),
                  ('Bone Type',        data.get('bones', '—'))]
    info_right = [('Analysis ID', str(data.get('pred_id', '—'))),
                  ('Analyst',     data.get('analyst', 'System')),
                  ('System',      'Forensic Bone Analysis v1.0')]
    col1_x = margin
    col2_x = W / 2 + 0.5*cm
    row_h  = 0.55*cm

    for (lbl, val), (lbl2, val2) in zip(info_left, info_right):
        c.setFont('Helvetica-Bold', 8); c.setFillColor(TEXT_GREY)
        c.drawString(col1_x, y, lbl + ':')
        c.setFont('Helvetica', 8); c.setFillColor(TEXT_DARK)
        c.drawString(col1_x + 3.2*cm, y, str(val))
        c.setFont('Helvetica-Bold', 8); c.setFillColor(TEXT_GREY)
        c.drawString(col2_x, y, lbl2 + ':')
        c.setFont('Helvetica', 8); c.setFillColor(TEXT_DARK)
        c.drawString(col2_x + 2.5*cm, y, str(val2))
        y -= row_h

    for lbl, val in [('Models', 'Random Forest (Scikit-learn)'), ('Dataset', '4,501 augmented records')]:
        c.setFont('Helvetica-Bold', 8); c.setFillColor(TEXT_GREY)
        c.drawString(col1_x, y, lbl + ':')
        c.setFont('Helvetica', 8); c.setFillColor(TEXT_DARK)
        c.drawString(col1_x + 3.2*cm, y, val)
        y -= row_h

    y -= 0.4*cm

    # Prediction Results
    section_header('PREDICTION RESULTS')
    sex      = data.get('sex', 'Male')
    sex_conf = float(data.get('sex_conf', 0))
    age      = data.get('age_range', '—')
    age_conf = float(data.get('age_conf', 0))

    sex_color = ACCENT if sex == 'Male' else PINK
    sex_bg    = LIGHT_BLUE if sex == 'Male' else PINK_LIGHT
    card_w    = (W - 2*margin - 0.6*cm) / 2
    card_h    = 3.2*cm
    bar_w     = card_w - 0.8*cm

    # Sex card
    c.setFillColor(sex_bg)
    c.roundRect(margin, y - card_h, card_w, card_h, 6, fill=1, stroke=0)
    c.setStrokeColor(sex_color); c.setLineWidth(1.5)
    c.roundRect(margin, y - card_h, card_w, card_h, 6, fill=0, stroke=1)
    c.setFillColor(sex_color); c.setFont('Helvetica-Bold', 8)
    c.drawString(margin + 0.4*cm, y - 0.5*cm, 'BIOLOGICAL SEX')
    c.setFont('Helvetica-Bold', 26)
    c.drawString(margin + 0.4*cm, y - 1.6*cm, sex)
    bx, by = margin + 0.4*cm, y - 2.3*cm
    c.setFillColor(MED_GREY); c.roundRect(bx, by, bar_w, 0.28*cm, 3, fill=1, stroke=0)
    c.setFillColor(sex_color); c.roundRect(bx, by, bar_w*(sex_conf/100), 0.28*cm, 3, fill=1, stroke=0)
    c.setFont('Helvetica-Bold', 9); c.setFillColor(sex_color)
    c.drawString(bx, by - 0.45*cm, f'Confidence: {sex_conf}%')

    # Age card
    age_x = margin + card_w + 0.6*cm
    c.setFillColor(GREEN_LIGHT)
    c.roundRect(age_x, y - card_h, card_w, card_h, 6, fill=1, stroke=0)
    c.setStrokeColor(GREEN); c.setLineWidth(1.5)
    c.roundRect(age_x, y - card_h, card_w, card_h, 6, fill=0, stroke=1)
    c.setFillColor(GREEN); c.setFont('Helvetica-Bold', 8)
    c.drawString(age_x + 0.4*cm, y - 0.5*cm, 'AGE GROUP')
    c.setFont('Helvetica-Bold', 22)
    c.drawString(age_x + 0.4*cm, y - 1.6*cm, age + ' years')
    bx2, by2 = age_x + 0.4*cm, y - 2.3*cm
    c.setFillColor(MED_GREY); c.roundRect(bx2, by2, bar_w, 0.28*cm, 3, fill=1, stroke=0)
    c.setFillColor(GREEN); c.roundRect(bx2, by2, bar_w*(age_conf/100), 0.28*cm, 3, fill=1, stroke=0)
    c.setFont('Helvetica-Bold', 9); c.setFillColor(GREEN)
    c.drawString(bx2, by2 - 0.45*cm, f'Confidence: {age_conf}%')

    y -= card_h + 0.7*cm

   # Measurements — 2 column layout
    measurements = data.get('measurements', {})
    if measurements:
        section_header('BONE MEASUREMENTS ENTERED')
        items     = [(k, v) for k, v in measurements.items()]
        half      = (len(items) + 1) // 2
        left_col  = items[:half]
        right_col = items[half:]

        col_block_w = (W - 2*margin - 0.6*cm) / 2
        row_ht      = 0.46*cm
        hdr_h       = 0.5*cm

        def draw_meas_header(x, label):
            c.setFillColor(MID_BLUE)
            c.rect(x, y - hdr_h, col_block_w, hdr_h, fill=1, stroke=0)
            c.setFillColor(WHITE); c.setFont('Helvetica-Bold', 8)
            c.drawString(x + 0.25*cm, y - hdr_h + 0.12*cm, label)
            c.drawString(x + col_block_w - 2.2*cm, y - hdr_h + 0.12*cm, 'mm')

        draw_meas_header(margin, 'Measurement')
        draw_meas_header(margin + col_block_w + 0.6*cm, 'Measurement')
        y -= hdr_h

        for i in range(max(len(left_col), len(right_col))):
            bg = LIGHT_GREY if i % 2 == 0 else WHITE

            # Left column
            if i < len(left_col):
                k, v = left_col[i]
                val_str = f"{float(v):.2f}" if v and float(v) > 0 else 'N/A'
                c.setFillColor(bg)
                c.rect(margin, y - row_ht, col_block_w, row_ht, fill=1, stroke=0)
                c.setStrokeColor(MED_GREY); c.setLineWidth(0.25)
                c.line(margin, y - row_ht, margin + col_block_w, y - row_ht)
                c.setFillColor(TEXT_DARK); c.setFont('Helvetica', 8)
                c.drawString(margin + 0.25*cm, y - row_ht + 0.1*cm, k)
                c.setFont('Helvetica-Bold', 8)
                c.drawRightString(margin + col_block_w - 0.25*cm, y - row_ht + 0.1*cm, val_str)

            # Right column
            if i < len(right_col):
                k, v = right_col[i]
                val_str = f"{float(v):.2f}" if v and float(v) > 0 else 'N/A'
                rx = margin + col_block_w + 0.6*cm
                c.setFillColor(bg)
                c.rect(rx, y - row_ht, col_block_w, row_ht, fill=1, stroke=0)
                c.setStrokeColor(MED_GREY); c.setLineWidth(0.25)
                c.line(rx, y - row_ht, rx + col_block_w, y - row_ht)
                c.setFillColor(TEXT_DARK); c.setFont('Helvetica', 8)
                c.drawString(rx + 0.25*cm, y - row_ht + 0.1*cm, k)
                c.setFont('Helvetica-Bold', 8)
                c.drawRightString(rx + col_block_w - 0.25*cm, y - row_ht + 0.1*cm, val_str)

            y -= row_ht

        y -= 0.5*cm

    # Analyst notes
    notes = data.get('notes', '')
    if notes:
        # Page break check for notes
        if y < 5*cm:
            c.showPage()
            c.setFillColor(DARK_BLUE)
            c.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
            c.setFillColor(WHITE); c.setFont('Helvetica-Bold', 8)
            c.drawString(margin, H - 0.75*cm, 'FORENSIC BONE ANALYSIS SYSTEM')
            c.setFont('Helvetica', 8)
            c.drawRightString(W - margin, H - 0.75*cm, f"Case: {data.get('case_ref','')}")
            y = H - 1.8*cm
        section_header('ANALYST NOTES')
        notes_h = 1.8*cm
        c.setFillColor(colors.HexColor('#fffdf0'))
        c.setStrokeColor(colors.HexColor('#e0cc80')); c.setLineWidth(0.5)
        c.roundRect(margin, y - notes_h, W - 2*margin, notes_h, 4, fill=1, stroke=1)
        c.setFillColor(TEXT_DARK); c.setFont('Helvetica', 8.5)
        words = notes.split(); lines_n, cur = [], ''
        for w in words:
            if len(cur) + len(w) + 1 <= 115:
                cur = cur + ' ' + w if cur else w
            else:
                lines_n.append(cur); cur = w
        if cur: lines_n.append(cur)
        note_y = y - 0.45*cm
        for ln in lines_n[:3]:
            c.drawString(margin + 0.4*cm, note_y, ln); note_y -= 0.45*cm

    # Footer
    footer_h = 2.2*cm
    c.setFillColor(LIGHT_GREY)
    c.rect(0, 0, W, footer_h, fill=1, stroke=0)
    c.setStrokeColor(ACCENT); c.setLineWidth(1)
    c.line(0, footer_h, W, footer_h)
    c.setFillColor(TEXT_GREY); c.setFont('Helvetica', 7)
    disclaimer = ('DISCLAIMER: This report is generated by an AI-assisted decision support system. '
                  'Predictions are based on statistical models trained on skeletal morphology data and are '
                  'intended to assist — not replace — the judgement of a qualified forensic expert. '
                  'All conclusions must be verified by a certified forensic anthropologist before use in '
                  'any legal or investigative proceeding.')
    dwords = disclaimer.split(); dlines, dcur = [], ''
    for w in dwords:
        if len(dcur) + len(w) + 1 <= 140:
            dcur = dcur + ' ' + w if dcur else w
        else:
            dlines.append(dcur); dcur = w
    if dcur: dlines.append(dcur)
    dy = footer_h - 0.5*cm
    for dl in dlines[:3]:
        c.drawString(margin, dy, dl); dy -= 0.32*cm
    c.setFont('Helvetica-Bold', 7); c.setFillColor(ACCENT)
    c.drawCentredString(W/2, 0.35*cm,
        f"Generated by Forensic Bone Analysis System  ·  {data.get('generated_at', '—')}")
    c.save()


@app.route('/download-report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    import datetime
    rows = get_user_predictions(session['user_id'])
    pred = next((r for r in rows if r[0] == prediction_id), None)
    if not pred:
        abort(404)

    data = {
        'case_ref':     pred[2],
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'created_at':   str(pred[8]) if pred[8] else '—',
        'bones':        pred[7],
        'pred_id':      pred[0],
        'analyst':      session.get('full_name', 'System'),
        'sex':          pred[3],
        'sex_conf':     float(pred[4]),
        'age_range':    pred[5],
        'age_conf':     float(pred[6]),
        'measurements': get_prediction_measurements(pred[0]),  # ← මේක change වුනා
        'notes': pred[9] if len(pred) > 9 and pred[9] else '',
}
    buffer = io.BytesIO()
    generate_forensic_report(buffer, data)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"forensic_report_{pred[2]}.pdf",
                     mimetype='application/pdf')


# ── Profile routes ────────────────────────────────────────────────────────────

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