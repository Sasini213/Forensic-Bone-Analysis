import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from ml_model.database import init_db, save_prediction, register_user, login_user, get_user_predictions

app = Flask(__name__)
app.secret_key = 'forensic_bone_secret_key_2024'

# Models load කරන්න
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model')

with open(os.path.join(MODEL_DIR,'age_label_encoder.pkl'),'rb') as f: label_encoder=pickle.load(f)

BONE_MODELS = {}
for bone_key in ['humerus','radius','femur','tibia','os_coxae']:
    BONE_MODELS[bone_key] = {
        'gender': pickle.load(open(os.path.join(MODEL_DIR,f'gender_{bone_key}.pkl'),'rb')),
        'age':    pickle.load(open(os.path.join(MODEL_DIR,f'age_{bone_key}.pkl'),'rb')),
        'scaler': pickle.load(open(os.path.join(MODEL_DIR,f'scaler_{bone_key}.pkl'),'rb')),
        'features': pickle.load(open(os.path.join(MODEL_DIR,f'features_{bone_key}.pkl'),'rb')),
    }

# Database initialize කරන්න
init_db()

# Bone measurements define කරන්න
BONE_MEASUREMENTS = {
    'Humerus': ['LHML', 'LHEB', 'LHHD', 'LHMLD', 'LHAPD',
                'RHML', 'RHEB', 'RHHD', 'RHMLD', 'RHAPD'],
    'Radius':  ['LRML', 'LRMLD', 'LRAPD',
                'RRML', 'RRMLD', 'RRAPD'],
    'Femur':   ['LFML', 'LFBL', 'LFEB', 'LFAB', 'LFHD', 'LFMLD', 'LFAPD',
                'RFML', 'RFBL', 'RFEB', 'RFAB', 'RFHD', 'RFMLD', 'RFAPD'],
    'Tibia':   ['LTML', 'LTPB', 'LTMLD', 'LTAPD',
                'RTML', 'RTPB', 'RTMLD', 'RTAPD'],
    'Os Coxae':['BIB', 'LIBL', 'RIBL', 'LAcH', 'RAcH']
}

def compute_avg_features(data):
    avg_pairs = {
        'AVG_HML': ('LHML', 'RHML'), 'AVG_HEB': ('LHEB', 'RHEB'),
        'AVG_HHD': ('LHHD', 'RHHD'), 'AVG_HMLD': ('LHMLD', 'RHMLD'),
        'AVG_HAPD': ('LHAPD', 'RHAPD'), 'AVG_RML': ('LRML', 'RRML'),
        'AVG_RMLD': ('LRMLD', 'RRMLD'), 'AVG_RAPD': ('LRAPD', 'RRAPD'),
        'AVG_FML': ('LFML', 'RFML'), 'AVG_FBL': ('LFBL', 'RFBL'),
        'AVG_FEB': ('LFEB', 'RFEB'), 'AVG_FAB': ('LFAB', 'RFAB'),
        'AVG_FHD': ('LFHD', 'RFHD'), 'AVG_FMLD': ('LFMLD', 'RFMLD'),
        'AVG_FAPD': ('LFAPD', 'RFAPD'), 'AVG_TML': ('LTML', 'RTML'),
        'AVG_TPB': ('LTPB', 'RTPB'), 'AVG_TMLD': ('LTMLD', 'RTMLD'),
        'AVG_TAPD': ('LTAPD', 'RTAPD'), 'AVG_IBL': ('LIBL', 'RIBL'),
        'AVG_AcH': ('LAcH', 'RAcH')
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

# Login required decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =====================
# Auth Routes
# =====================
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        full_name = request.form.get('full_name', '').strip()

        if not username or not password or not full_name:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        success, message = register_user(username, password, full_name)
        if success:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        success, result = login_user(username, password)
        if success:
            session['user_id'] = result['id']
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

# =====================
# Main Routes
# =====================
@app.route('/index')
@login_required
def index():
    return render_template('index.html', bone_measurements=BONE_MEASUREMENTS)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form_data = request.form.to_dict()
    selected_bones = request.form.getlist('bones')

    if not selected_bones:
        flash('Please select at least one bone.', 'warning')
        return redirect(url_for('index'))

    # Per-bone predictions collect කරන්න
    gender_probs, age_probs = [], []
    measurements = {}

    for bone in selected_bones:
        bone_key = bone.replace(' ', '_').lower()
        model_set = BONE_MODELS.get(bone_key)
        if not model_set:
            continue

        features = model_set['features']
        bone_vals = {}
        for f in features:
            val = form_data.get(f, 0)
            bone_vals[f] = float(val) if val else 0
            measurements[f] = bone_vals[f]

        # AVG features compute
        bone_vals = compute_avg_features(bone_vals)

        X = pd.DataFrame([{f: bone_vals.get(f, 0) for f in features}])

        # Gender predict
        gp = model_set['gender'].predict_proba(X)[0]
        gender_probs.append(gp)

        # Age predict
        X_scaled = model_set['scaler'].transform(X)
        ap = model_set['age'].predict_proba(X_scaled)[0]
        age_probs.append(ap)

    # Average probabilities across all bones
    avg_gender = np.mean(gender_probs, axis=0)
    avg_age    = np.mean(age_probs, axis=0)

    gender_pred  = int(np.argmax(avg_gender))
    gender_conf  = round(float(np.max(avg_gender)) * 100, 2)
    gender_label = 'Female' if gender_pred == 1 else 'Male'

    age_pred  = int(np.argmax(avg_age))
    age_conf  = round(float(np.max(avg_age)) * 100, 2)
    age_label = label_encoder.inverse_transform([age_pred])[0]

    case_ref = save_prediction(
        session['user_id'],
        gender_label, gender_conf,
        age_label, age_conf,
        selected_bones, measurements
    )

    return render_template('result.html',
        gender=gender_label,
        gender_conf=gender_conf,
        age=age_label,
        age_conf=age_conf,
        case_ref=case_ref,
        selected_bones=selected_bones,
        full_name=session.get('full_name')
    )

    if not selected_bones:
        flash('Please select at least one bone.', 'warning')
        return redirect(url_for('index'))

    # Measurements dictionary හදන්න
    measurements = {}
    for bone in selected_bones:
        for field in BONE_MEASUREMENTS.get(bone, []):
            val = form_data.get(field, '')
            measurements[field] = val if val else 0

    # Bone presence flags
    presence = {
        'LHUM': 1 if 'Humerus' in selected_bones else 0,
        'RHUM': 1 if 'Humerus' in selected_bones else 0,
        'LRAD': 1 if 'Radius' in selected_bones else 0,
        'RRAD': 1 if 'Radius' in selected_bones else 0,
        'LFEM': 1 if 'Femur' in selected_bones else 0,
        'RFEM': 1 if 'Femur' in selected_bones else 0,
        'LTIB': 1 if 'Tibia' in selected_bones else 0,
        'RTIB': 1 if 'Tibia' in selected_bones else 0,
        'OSCX': 1 if 'Os Coxae' in selected_bones else 0,
    }

    # Full data dictionary
    all_data = {**presence, **measurements}
    all_data = compute_avg_features(all_data)

    # DataFrame හදන්න
    input_df = pd.DataFrame([all_data])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols].fillna(0)
    input_df_scaled = scaler.transform(input_df)

    # Predictions
    gender_pred = gender_model.predict(input_df)[0]
    gender_proba = gender_model.predict_proba(input_df)[0]
    gender_conf = round(max(gender_proba) * 100, 2)
    gender_label = 'Female' if gender_pred == 1 else 'Male'

    age_pred = age_model.predict(input_df_scaled)[0]
    age_proba = age_model.predict_proba(input_df_scaled)[0]
    age_conf = round(max(age_proba) * 100, 2)
    age_label = label_encoder.inverse_transform([age_pred])[0]
    # Database save
    case_ref = save_prediction(
        session['user_id'],
        gender_label, gender_conf,
        age_label, age_conf,
        selected_bones, measurements
    )

    return render_template('result.html',
        gender=gender_label,
        gender_conf=gender_conf,
        age=age_label,
        age_conf=age_conf,
        case_ref=case_ref,
        selected_bones=selected_bones,
        full_name=session.get('full_name')
    )

@app.route('/history')
@login_required
def history():
    predictions = get_user_predictions(session['user_id'])
    return render_template('history.html',
        predictions=predictions,
        full_name=session.get('full_name')
    )

@app.route('/reset')
@login_required
def reset():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)